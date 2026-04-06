#!/usr/bin/env python3
"""
VITON-HD 2-Stage Training  (Simplified — fast convergence)
===========================================================
Stage 1 (warp):  WarpNet  learns cloth deformation flow
Stage 2 (tryon): TryOnNet synthesizes final try-on image

Optimised for RTX 4070 12 GB:
  batch=16, AMP, 0 data workers (Windows)
  Loss: L1 + 0.5 * VGG
  Early stopping: monitors L1 loss only

Usage:
  python train.py --stage warp
  python train.py --stage tryon --warp-ckpt checkpoints/warp_best.pth
  python train.py --stage warp  --resume checkpoints/warp_epoch_005.pth
"""

import argparse
import csv
import logging
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.discriminator import PatchDiscriminator
from model.warp_utils import warp_cloth
from shared.dataset import make_loader
from shared.metrics import ssim_metric, psnr_metric, metrics_header, metrics_separator, metrics_row

# ── Defaults ───────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
_BASE       = Path(__file__).resolve().parent
DATA_DIR    = str(_BASE / "dataset" / "train" / "tensors")
CKPT_DIR    = str(_BASE / "checkpoints")
LOG_DIR     = str(_BASE / "logs")

BATCH_SIZE  = 128         # RTX 4000 Ada 20 GB + 256x192 — 12 GB at 96, pushing to 128
LR          = 2e-4
BETAS       = (0.5, 0.999)
NUM_WORKERS = 0           # Windows + SSD; set to 0 if DataLoader errors appear
DECAY_START = 50          # late decay — model needs many epochs at full LR to learn geometric warping

LAMBDA_L1        = 1.0
LAMBDA_VGG       = 1.0    # warp stage: VGG provides the geometric alignment signal that L1 can't
                          # (L1 is dominated by appearance diff between flat cloth and on-body cloth;
                          #  VGG features are invariant to shading/wrinkles so they reward correct alignment)
LAMBDA_VGG_TRYON = 0.5    # tryon stage: forces color/texture transfer from warped cloth
LAMBDA_TV        = 0.0    # DISABLED — biases flow toward zero
LAMBDA_FLOW_REG  = 0.0    # DISABLED — biases flow toward zero
LAMBDA_SMOOTH    = 0.5    # second-order smoothness: penalizes jagged flow without zero-bias
LAMBDA_MASK      = 5.0    # mask alignment loss: warped cloth mask vs person clothing mask
                          # This is the PRIMARY geometric signal for WarpNet — clean binary shapes
                          # without texture/shading noise. Weight=5 because this is the most
                          # reliable gradient source for learning spatial deformation.

PATIENCE    = 20          # generous patience — model may plateau briefly then improve again
MIN_DELTA   = 1e-4        # meaningful improvement threshold

CLOTH_LABELS = [5, 6, 7]   # CIHP: upper-clothes, dress, coat

torch.backends.cudnn.benchmark = True

# ── Hyperparameter Change Log ──────────────────────────────────────────────────
# Add an entry here every time you change a value so the reason is recorded
# in every future training log.
#
# Format: (date, parameter, old_value, new_value, reason)
HYPERPARAM_CHANGELOG = [
    ("2026-03-24", "LR",          "2e-4",  "5e-5",
     "Model diverged after epoch 7 — loss jumped from 0.141 to 0.177. "
     "Lower LR prevents gradient instability after initial fast convergence."),
    ("2026-03-24", "LAMBDA_VGG",  "0.5",   "0.1",
     "VGG perceptual loss was competing with L1 and causing L1 to spike. "
     "Reducing weight keeps VGG as a texture guide without overpowering pixel-level alignment."),
    ("2026-03-24", "DECAY_START", "30",    "10",
     "With LR now at 5e-5, decay should begin earlier so learning rate "
     "gradually reduces before the model plateaus."),
    ("2026-03-24", "PATIENCE",    "10",    "15",
     "Extra patience to give the model time to improve under the lower LR."),
    ("2026-03-24", "MIN_DELTA",   "1e-4",  "5e-5",
     "Finer improvement threshold to match the slower, more stable convergence at 5e-5 LR."),
    ("2026-03-24", "NUM_WORKERS", "2",     "4",
     "Training data moved to HDD (11k samples). 4 workers needed to prefetch "
     "and hide HDD random-read latency so GPU stays at 100% utilization."),
    ("2026-03-24", "LAMBDA_FLOW_REG", "0", "0.01",
     "Identity bias: penalizes non-zero flow (flow.pow(2).mean()). "
     "Prevents cloth from drifting unnecessarily — model defaults to no-warp unless forced by L1."),
    ("2026-03-24", "masked_warp", "off", "on",
     "Applied cloth_mask to warped cloth (warped = warped * warped_mask). "
     "Without this, background pixels leak into the warped output causing noise outside cloth region."),
    ("2026-03-24", "flow_scale", "1.0", "0.3",
     "WarpNet tanh output was ±1, grid also ±1, so total displacement could reach ±2 (full image). "
     "Cloth was flying off-screen producing blob/spike artifacts. "
     "Scaling tanh by 0.3 limits max displacement to 30% of image — enough for cloth stretching "
     "but prevents catastrophic deformation. Fixed in warp_model.py line 74."),
    ("2026-03-24", "grad_clip", "none", "1.0",
     "Added gradient clipping (max_norm=1.0) to both warp and tryon. "
     "Prevents occasional large gradient updates from destabilizing training."),
    ("2026-03-24", "LAMBDA_TV", "0", "0.5",
     "WarpNet flow field was producing chaotic spike/blob artifacts (visible in col 4 of inference). "
     "No smoothness constraint meant flow could move pixels arbitrarily far. "
     "TV loss penalizes abrupt spatial changes in flow, forcing cloth to warp as a coherent shape."),
    ("2026-03-24", "LAMBDA_VGG_TRYON", "0.1", "0.4",
     "Inference showed TryOnNet ignoring cloth color/texture entirely — output matched "
     "original person not target cloth. Root cause: L1=0.1 VGG too weak to force texture "
     "transfer, model took L1 shortcut reconstructing original region. Higher VGG=0.4 "
     "forces perceptual similarity to target cloth."),
    ("2026-03-25", "LAMBDA_TV", "0.5", "0.0",
     "Inference showed warped cloth completely blank (gray) — WarpNet output near-zero flow. "
     "TV loss + flow_reg together were stronger than L1 signal, model learned to not warp at all. "
     "Disabled TV entirely so L1 is the only warp signal."),
    ("2026-03-25", "LAMBDA_FLOW_REG", "0.01", "0.0",
     "Same as TV — was compounding the no-warp bias. Disabled alongside TV."),
    ("2026-03-25", "flow_scale", "0.3", "0.5",
     "With TV/flow_reg removed, increase displacement budget so cloth can reach body region. "
     "0.3 (30%) was too tight for clothes that need to stretch across the torso."),
    ("2026-03-25", "warp_loss_mask", "warped*pcm", "warped_raw*pcm",
     "Critical bug: loss was computed on warped*warped_mask*pcm (double masked). "
     "When warp is even slightly off, both masks don't overlap -> product ~0 -> zero gradient -> model stuck. "
     "Fix: compute loss on raw warp before applying warped_mask. "
     "Masked warp kept for TryOnNet input only."),
    ("2026-03-25", "WarpNet ngf", "64", "32",
     "Lighter model: 4x fewer params (~1.9M vs 7.5M), ~2x faster per epoch. "
     "Enough capacity for 5k samples. Frees time for a full retrain today."),
    ("2026-03-25", "TryOnNet ngf", "64", "32",
     "Lighter model: matches WarpNet size reduction. Still has U-Net skip connections "
     "for detail recovery. Faster training needed to finish today."),
    ("2026-03-25", "LAMBDA_MASK", "0", "5.0",
     "Root cause of WarpNet plateau: L1/VGG compared warped flat cloth RGB vs on-body cloth RGB. "
     "These can never match well due to wrinkles/shadows/body shape — L1 floors at ~0.14. "
     "Mask loss (warped_mask vs pcm) provides a clean geometric signal (binary shapes, no texture noise). "
     "Weight=5.0 makes this the dominant gradient source for spatial deformation learning."),
    ("2026-03-25", "WarpNet ngf", "32", "64",
     "Restored to original capacity. 32 was insufficient — model couldn't learn geometric warping "
     "with fewer params. 10.6M params needed for dense flow prediction."),
]


# ── Logger Setup ──────────────────────────────────────────────────────────────

def setup_logger(stage: str) -> logging.Logger:
    """
    Creates a logger that writes to both console and a timestamped txt file.
    Log file: logs/<stage>_YYYYMMDD_HHMMSS.txt
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"{stage}_{timestamp}.txt")

    logger = logging.getLogger(stage)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    # Console handler — same detail
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_path}")
    return logger


def log_section(logger: logging.Logger, title: str):
    """Print a titled section divider."""
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


def log_hyperparam_changelog(logger: logging.Logger):
    """Print the full hyperparameter change history to the log."""
    log_section(logger, "HYPERPARAMETER CHANGE LOG")
    if not HYPERPARAM_CHANGELOG:
        logger.info("  (no changes recorded)")
    else:
        for date, param, old, new, reason in HYPERPARAM_CHANGELOG:
            logger.info(f"  [{date}]  {param}  {old} → {new}")
            logger.info(f"           Reason: {reason}")
            logger.info("")
    logger.info("")


# ── Dataset ───────────────────────────────────────────────────────────────────

class VITONDataset(Dataset):
    def __init__(self, root, max_samples=None):
        files = sorted(Path(root).glob("*.pt"))
        if not files:
            raise FileNotFoundError(f"No .pt files in {root}")
        self.files = files[:max_samples] if max_samples else files
        # Cache entire dataset in RAM — eliminates disk I/O bottleneck
        print(f"Caching {len(self.files)} samples in RAM...", flush=True)
        self.cache = [torch.load(f, map_location="cpu", weights_only=False) for f in self.files]
        print(f"Dataset cached ({len(self.cache)} samples)", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.cache[idx]


# ── VGG Perceptual Loss ───────────────────────────────────────────────────────

class VGGLoss(nn.Module):
    """L1 feature matching on VGG16 relu1_2 … relu4_3."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg[:4])),     # relu1_2
            nn.Sequential(*list(vgg[4:9])),    # relu2_2
            nn.Sequential(*list(vgg[9:16])),   # relu3_3
            nn.Sequential(*list(vgg[16:23])),  # relu4_3
        ])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        return ((x + 1) / 2 - self.mean) / self.std

    def forward(self, pred, target):
        p = self._norm(pred)
        with torch.no_grad():
            t = self._norm(target)
        loss = 0.0
        for sl in self.slices:
            p = sl(p)
            with torch.no_grad():
                t = sl(t)
            loss += F.l1_loss(p, t)
        return loss


# ── Helpers ───────────────────────────────────────────────────────────────────

def person_cloth_mask(parse_map):
    """(B, H, W) int64 → (B, 1, H, W) float32 clothing region mask."""
    mask = torch.zeros_like(parse_map, dtype=torch.float32)
    for lbl in CLOTH_LABELS:
        mask += (parse_map == lbl).float()
    return mask.unsqueeze(1).clamp_(0, 1)


def make_scheduler(optimizer, epochs, decay_start, steps_per_epoch):
    """Constant LR until decay_start, then linear decay to 0."""
    warmup_end = decay_start * steps_per_epoch
    total      = epochs * steps_per_epoch
    decay_len  = max(total - warmup_end, 1)

    def lr_lambda(step):
        if step < warmup_end:
            return 1.0
        return max(0.0, 1.0 - (step - warmup_end) / decay_len)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_cosine_scheduler(optimizer, epochs, decay_start, steps_per_epoch):
    """Constant LR until decay_start, then cosine annealing to 0."""
    warmup_steps = decay_start * steps_per_epoch
    total_steps  = epochs * steps_per_epoch
    cosine_steps = max(total_steps - warmup_steps, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return 1.0
        t = (step - warmup_steps) / cosine_steps
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * t)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def flow_tv_loss(flow):
    """Total Variation loss on flow field — penalizes abrupt spatial changes."""
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
    return dx + dy


def flow_second_order_loss(flow):
    """Second-order smoothness: penalizes non-smooth flow without biasing toward zero.

    Unlike TV (first-order), this allows large uniform displacement (cloth shifting)
    but penalizes jagged/discontinuous flow. This is the key regularizer for dense flow
    warping — it encourages coherent deformation without suppressing the flow itself.
    """
    # d²f/dy²  ≈  f[y+1] - 2*f[y] + f[y-1]
    d2y = (flow[:, :, 2:, :] - 2 * flow[:, :, 1:-1, :] + flow[:, :, :-2, :]).abs().mean()
    # d²f/dx²
    d2x = (flow[:, :, :, 2:] - 2 * flow[:, :, :, 1:-1] + flow[:, :, :, :-2]).abs().mean()
    return d2x + d2y


def save_ckpt(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, path)


def cleanup_old_checkpoints(stage: str, keep: int = 2):
    """Delete old periodic checkpoints, keeping only the last `keep` ones.
    Best checkpoint (warp_best.pth / tryon_best.pth) is never deleted.
    """
    ckpt_dir = Path(CKPT_DIR)
    # Periodic checkpoints match e.g. warp_epoch_005.pth
    periodics = sorted(ckpt_dir.glob(f"{stage}_epoch_*.pth"))
    to_delete = periodics[:-keep] if len(periodics) > keep else []
    for f in to_delete:
        f.unlink(missing_ok=True)


def check_disk_space(warn_gb: float = 0.5):
    """Warn if free disk space on the project drive drops below warn_gb."""
    usage = shutil.disk_usage(CKPT_DIR)
    free_gb = usage.free / 1e9
    if free_gb < warn_gb:
        print(f"\n  !! WARNING: Only {free_gb:.2f} GB free on disk — training may fail !!\n")
    return free_gb


def open_csv_log(stage: str):
    """Open (or append to) a per-epoch CSV log in logs/."""
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{stage}_metrics.csv")
    is_new = not os.path.exists(path)
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["epoch", "avg_l1", "avg_vgg", "ssim", "psnr_db",
                         "lr", "epoch_time_s", "best_l1"])
    return fh, writer


def log_images(writer, tag, images, step, n=2):
    writer.add_images(tag, ((images[:n] + 1) / 2).clamp(0, 1), step)


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ── Stage 1: WarpNet ──────────────────────────────────────────────────────────

def train_warp(args, logger: logging.Logger):
    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)

    # ── Config log ────────────────────────────────────────────────────────────
    log_section(logger, "WARPNET TRAINING  —  CONFIGURATION")
    logger.info(f"  Stage          : warp")
    logger.info(f"  Device         : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  Dataset        : {len(loader.dataset)} samples  ({args.data})")
    logger.info(f"  Batches/epoch  : {len(loader)}")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Learning rate  : {args.lr}")
    logger.info(f"  LR decay start : epoch {args.decay_start}")
    logger.info(f"  AMP (fp16)     : {args.amp}")
    logger.info(f"  Lambda L1      : {LAMBDA_L1}")
    logger.info(f"  Lambda VGG     : {LAMBDA_VGG}")
    logger.info(f"  Lambda Smooth  : {LAMBDA_SMOOTH}")
    logger.info(f"  Lambda Mask    : {LAMBDA_MASK}")
    logger.info(f"  Max epochs     : {args.epochs}")
    logger.info(f"  Early stop     : patience={args.patience}, min_delta={MIN_DELTA}")
    if args.resume:
        logger.info(f"  Resume from    : {args.resume}")
    logger.info("")

    free_gb = check_disk_space(warn_gb=0.5)
    logger.info(f"  Free disk space    : {free_gb:.2f} GB")
    logger.info("")

    log_hyperparam_changelog(logger)

    model = WarpNet(ngf=args.ngf, flow_scale=args.flow_scale).to(DEVICE)
    vgg   = VGGLoss().to(DEVICE).eval()

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    steps_per_epoch = len(loader)
    if args.scheduler == "cosine":
        sched = make_cosine_scheduler(opt, args.epochs, args.decay_start, steps_per_epoch)
    else:
        sched = make_scheduler(opt, args.epochs, args.decay_start, steps_per_epoch)
    scaler = GradScaler("cuda", enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "warp"))
    csv_fh, csv_writer = open_csv_log("warp")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}\n")

    step = (start_epoch - 1) * len(loader)
    best_l1 = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_start = time.time()

    log_section(logger, "WARPNET TRAINING  —  EPOCH LOG")
    logger.info(metrics_header())
    logger.info(metrics_separator())

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[warp] E{epoch:02d}", dynamic_ncols=True, leave=False)
        epoch_l1  = 0.0
        epoch_vgg = 0.0
        n_batches = 0
        epoch_start = time.time()

        for i, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag   = batch["agnostic"]
            cl   = batch["cloth"]
            cm   = batch["cloth_mask"].unsqueeze(1)
            pose = batch["pose_map"]
            per  = batch["person"]
            pm   = batch["parse_map"]

            inp = torch.cat([ag, pose, cl, cm], 1)  # 25ch

            with autocast("cuda", enabled=args.amp):
                flow        = model(inp)
                warped_raw  = warp_cloth(cl, flow)        # unmasked — used for loss
                warped_mask = warp_cloth(cm, flow)
                warped      = warped_raw * warped_mask    # masked — used as TryOnNet input
                pcm         = person_cloth_mask(pm)

                # Loss on raw warp (no double masking).
                l1     = F.l1_loss(warped_raw * pcm, per * pcm)
                vg     = vgg(warped_raw * pcm, per * pcm)
                smooth = flow_second_order_loss(flow)
                # Mask alignment: warped cloth mask should match person clothing mask.
                # This is a clean geometric signal — binary shapes, no texture noise.
                mask_l = F.l1_loss(warped_mask, pcm)
                loss   = (args.lambda_l1 * l1 + args.lambda_vgg * vg
                          + args.lambda_smooth * smooth + args.lambda_mask * mask_l)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            epoch_l1  += l1.item()
            epoch_vgg += vg.item()
            n_batches += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", VGG=f"{vg.item():.3f}", Mask=f"{mask_l.item():.3f}")

            if step % 100 == 0:
                writer.add_scalar("warp/l1",     l1.item(), step)
                writer.add_scalar("warp/vgg",    vg.item(), step)
                writer.add_scalar("warp/smooth", smooth.item(), step)
                writer.add_scalar("warp/mask",   mask_l.item(), step)
                writer.add_scalar("warp/lr",     sched.get_last_lr()[0], step)

        # ── Epoch summary ─────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_l1  = epoch_l1  / max(n_batches, 1)
        avg_vgg = epoch_vgg / max(n_batches, 1)
        cur_lr  = sched.get_last_lr()[0]

        writer.add_scalar("warp/epoch_l1",  avg_l1,  epoch)
        writer.add_scalar("warp/epoch_vgg", avg_vgg, epoch)

        # ── Compute quality metrics (SSIM, PSNR) on last batch ─────────────
        model.eval()
        with torch.no_grad():
            flow_v   = model(inp)
            warped_v = warp_cloth(cl, flow_v)
            pcm_v    = person_cloth_mask(pm)
            # Metrics: compare warped cloth region to person cloth region
            epoch_ssim = ssim_metric(warped_v * pcm_v, per * pcm_v).item()
            epoch_psnr = psnr_metric(warped_v * pcm_v, per * pcm_v).item()
            log_images(writer, "warp/cloth",  cl,            step)
            log_images(writer, "warp/warped", warped_v,      step)
            log_images(writer, "warp/target", per * pcm_v,   step)
        model.train()

        writer.add_scalar("warp/epoch_ssim", epoch_ssim, epoch)
        writer.add_scalar("warp/epoch_psnr", epoch_psnr, epoch)

        # ── Checkpoint every epoch (keep last 5 + best) ──────────────────────────
        save_ckpt(f"{CKPT_DIR}/warp_epoch_{epoch:03d}.pth", model, opt, epoch)
        cleanup_old_checkpoints("warp", keep=5)
        ckpt_saved = f"[ckpt]"

        # ── Early stopping ─────────────────────────────────────────────────────
        if avg_l1 < best_l1 - MIN_DELTA:
            best_l1 = avg_l1
            best_epoch = epoch
            patience_counter = 0
            save_ckpt(f"{CKPT_DIR}/warp_best.pth", model, opt, epoch)
            status = f"* NEW BEST *  {ckpt_saved}"
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{args.patience}  {ckpt_saved}"

        logger.info(metrics_row(epoch, avg_l1, avg_vgg, epoch_ssim, epoch_psnr,
                                cur_lr, fmt_time(epoch_time), best_l1, status))
        csv_writer.writerow([epoch, f"{avg_l1:.6f}", f"{avg_vgg:.6f}",
                             f"{epoch_ssim:.4f}", f"{epoch_psnr:.2f}",
                             f"{cur_lr:.2e}", f"{epoch_time:.1f}", f"{best_l1:.6f}"])
        csv_fh.flush()

        if patience_counter >= args.patience:
            logger.info(f"\n  >> Early stopping triggered at epoch {epoch}.")
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    log_section(logger, "WARPNET TRAINING  —  FINAL SUMMARY")
    logger.info(f"  Total epochs trained : {epoch - start_epoch + 1}")
    logger.info(f"  Best L1 loss         : {best_l1:.4f}  (epoch {best_epoch})")
    logger.info(f"  Total training time  : {fmt_time(total_time)}")
    logger.info(f"  Best checkpoint      : {CKPT_DIR}/warp_best.pth")
    logger.info(f"  CSV metrics log      : {LOG_DIR}/warp_metrics.csv")
    logger.info("")

    csv_fh.close()
    writer.close()


# ── Stage 2: TryOnNet ─────────────────────────────────────────────────────────

def train_tryon(args, logger: logging.Logger):
    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)

    # ── Config log ────────────────────────────────────────────────────────────
    log_section(logger, "TRYONNET TRAINING  —  CONFIGURATION")
    logger.info(f"  Stage          : tryon")
    logger.info(f"  Device         : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  Dataset        : {len(loader.dataset)} samples  ({args.data})")
    logger.info(f"  Batches/epoch  : {len(loader)}")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Learning rate  : {args.lr}")
    logger.info(f"  LR decay start : epoch {args.decay_start}")
    logger.info(f"  AMP (fp16)     : {args.amp}")
    logger.info(f"  Lambda L1      : {LAMBDA_L1}")
    logger.info(f"  Lambda VGG     : {LAMBDA_VGG}")
    logger.info(f"  Max epochs     : {args.epochs}")
    logger.info(f"  Early stop     : patience={args.patience}, min_delta={MIN_DELTA}")
    logger.info(f"  WarpNet ckpt   : {args.warp_ckpt if args.warp_ckpt else 'NONE (random weights!)'}")
    if args.resume:
        logger.info(f"  Resume from    : {args.resume}")
    logger.info("")

    # Frozen WarpNet
    warp = WarpNet(ngf=args.ngf, flow_scale=args.flow_scale).to(DEVICE).eval()
    if args.warp_ckpt:
        warp.load_state_dict(
            torch.load(args.warp_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        logger.info(f"Loaded WarpNet: {args.warp_ckpt}")
    else:
        logger.warning("WARNING: No --warp-ckpt provided — WarpNet weights are random!")
    for p in warp.parameters():
        p.requires_grad = False

    free_gb = check_disk_space(warn_gb=0.5)
    logger.info(f"  Free disk space    : {free_gb:.2f} GB")
    logger.info("")

    log_hyperparam_changelog(logger)

    gen  = TryOnNet(ngf=args.ngf).to(DEVICE)
    vgg  = VGGLoss().to(DEVICE).eval()

    # Optional PatchGAN discriminator for adversarial variants
    disc     = None
    opt_disc = None
    if args.lambda_gan > 0:
        disc     = PatchDiscriminator().to(DEVICE)
        opt_disc = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=BETAS)

    opt    = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=BETAS)
    if args.scheduler == "cosine":
        sched = make_cosine_scheduler(opt, args.epochs, args.decay_start, len(loader))
    else:
        sched = make_scheduler(opt, args.epochs, args.decay_start, len(loader))
    scaler = GradScaler("cuda", enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "tryon"))
    csv_fh, csv_writer = open_csv_log("tryon")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}\n")

    step = (start_epoch - 1) * len(loader)
    best_l1 = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_start = time.time()

    log_section(logger, "TRYONNET TRAINING  —  EPOCH LOG")
    logger.info(metrics_header())
    logger.info(metrics_separator())

    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        pbar = tqdm(loader, desc=f"[tryon] E{epoch:02d}", dynamic_ncols=True, leave=False)
        epoch_l1  = 0.0
        epoch_vgg = 0.0
        n_batches = 0
        epoch_start = time.time()

        for i, batch in enumerate(pbar):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag   = batch["agnostic"]
            cl   = batch["cloth"]
            cm   = batch["cloth_mask"].unsqueeze(1)
            pose = batch["pose_map"]
            per  = batch["person"]

            # Warp cloth with frozen WarpNet
            with torch.no_grad():
                warp_in     = torch.cat([ag, pose, cl, cm], 1)
                flow        = warp(warp_in)
                warped      = warp_cloth(cl, flow)
                warped_mask = warp_cloth(cm, flow)
                warped      = warped * warped_mask   # mask out background leakage

            # 25ch input: agnostic(3) + warped(3) + warped_mask(1) + pose(18)
            inp = torch.cat([ag, warped, warped_mask, pose], 1)

            with autocast("cuda", enabled=args.amp):
                fake = gen(inp)
                l1   = F.l1_loss(fake, per)
                vg   = vgg(fake, per)
                loss = args.lambda_l1 * l1 + args.lambda_vgg_tryon * vg

                # ── GAN generator loss (LSGAN: fool discriminator toward 1) ──
                if disc is not None:
                    pred_fake = disc(fake)
                    loss_gan  = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
                    loss = loss + args.lambda_gan * loss_gan

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            # ── Discriminator update ───────────────────────────────────────────
            if disc is not None and opt_disc is not None:
                with autocast("cuda", enabled=args.amp):
                    pred_real = disc(per)
                    pred_fake = disc(fake.detach())
                    loss_disc = 0.5 * (F.mse_loss(pred_real, torch.ones_like(pred_real))
                                       + F.mse_loss(pred_fake, torch.zeros_like(pred_fake)))
                opt_disc.zero_grad(set_to_none=True)
                scaler.scale(loss_disc).backward()
                scaler.unscale_(opt_disc)
                scaler.step(opt_disc)
                scaler.update()

            step += 1
            epoch_l1  += l1.item()
            epoch_vgg += vg.item()
            n_batches += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", VGG=f"{vg.item():.3f}")

            if step % 100 == 0:
                writer.add_scalar("tryon/l1",  l1.item(), step)
                writer.add_scalar("tryon/vgg", vg.item(), step)
                writer.add_scalar("tryon/lr",  sched.get_last_lr()[0], step)

        # ── Epoch summary ─────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_l1  = epoch_l1  / max(n_batches, 1)
        avg_vgg = epoch_vgg / max(n_batches, 1)
        cur_lr  = sched.get_last_lr()[0]

        writer.add_scalar("tryon/epoch_l1",  avg_l1,  epoch)
        writer.add_scalar("tryon/epoch_vgg", avg_vgg, epoch)

        # ── Compute quality metrics ───────────────────────────────────────────
        gen.eval()
        with torch.no_grad():
            epoch_ssim = ssim_metric(fake, per).item()
            epoch_psnr = psnr_metric(fake, per).item()
            log_images(writer, "tryon/output",  fake,    step)
            log_images(writer, "tryon/person",  per,     step)
            log_images(writer, "tryon/warped",  warped,  step)
            log_images(writer, "tryon/agnostic", ag,     step)
        gen.train()

        writer.add_scalar("tryon/epoch_ssim", epoch_ssim, epoch)
        writer.add_scalar("tryon/epoch_psnr", epoch_psnr, epoch)

        # ── Checkpoint every epoch (keep last 5 + best) ──────────────────────────
        save_ckpt(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth", gen, opt, epoch)
        cleanup_old_checkpoints("tryon", keep=5)
        ckpt_saved = "[ckpt]"

        # ── Early stopping ─────────────────────────────────────────────────────
        if avg_l1 < best_l1 - MIN_DELTA:
            best_l1 = avg_l1
            best_epoch = epoch
            patience_counter = 0
            save_ckpt(f"{CKPT_DIR}/tryon_best.pth", gen, opt, epoch)
            status = f"* NEW BEST *  {ckpt_saved}"
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{args.patience}  {ckpt_saved}"

        logger.info(metrics_row(epoch, avg_l1, avg_vgg, epoch_ssim, epoch_psnr,
                                cur_lr, fmt_time(epoch_time), best_l1, status))
        csv_writer.writerow([epoch, f"{avg_l1:.6f}", f"{avg_vgg:.6f}",
                             f"{epoch_ssim:.4f}", f"{epoch_psnr:.2f}",
                             f"{cur_lr:.2e}", f"{epoch_time:.1f}", f"{best_l1:.6f}"])
        csv_fh.flush()

        if patience_counter >= args.patience:
            logger.info(f"\n  >> Early stopping triggered at epoch {epoch}.")
            break

    # ── Final summary ──────────────────────────────────────────────────────────
    total_time = time.time() - train_start
    log_section(logger, "TRYONNET TRAINING  —  FINAL SUMMARY")
    logger.info(f"  Total epochs trained : {epoch - start_epoch + 1}")
    logger.info(f"  Best L1 loss         : {best_l1:.4f}  (epoch {best_epoch})")
    logger.info(f"  Total training time  : {fmt_time(total_time)}")
    logger.info(f"  Best checkpoint      : {CKPT_DIR}/tryon_best.pth")
    logger.info(f"  CSV metrics log      : {LOG_DIR}/tryon_metrics.csv")
    logger.info("")

    csv_fh.close()
    writer.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global CKPT_DIR, LOG_DIR

    p = argparse.ArgumentParser(
        description="VITON-HD Simplified Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage",       required=True, choices=["warp", "tryon", "both"])
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=PATIENCE)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--decay-start", type=int,   default=None, dest="decay_start")
    p.add_argument("--workers",     type=int,   default=NUM_WORKERS)
    p.add_argument("--data",        default=DATA_DIR)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      dest="amp", action="store_false")
    p.add_argument("--resume",      default="", help="Resume from checkpoint")
    p.add_argument("--warp-ckpt",   default="", dest="warp_ckpt",
                   help="WarpNet checkpoint for tryon stage")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                   help="Limit dataset size (e.g. 3000 for faster epochs)")
    # Variant-overridable hyperparams (defaults come from variant config)
    p.add_argument("--ngf",           type=int,   default=None)
    p.add_argument("--flow-scale",    type=float, default=None, dest="flow_scale")
    p.add_argument("--scheduler",     default=None, choices=["linear", "cosine"])
    p.add_argument("--lambda-l1",     type=float, default=None, dest="lambda_l1")
    p.add_argument("--lambda-vgg",    type=float, default=None, dest="lambda_vgg")
    p.add_argument("--lambda-vgg-tryon", type=float, default=None, dest="lambda_vgg_tryon")
    p.add_argument("--lambda-mask",   type=float, default=None, dest="lambda_mask")
    p.add_argument("--lambda-smooth", type=float, default=None, dest="lambda_smooth")
    p.add_argument("--lambda-gan",    type=float, default=None, dest="lambda_gan")
    args = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # hard cap: OOM before CPU spillage

    # ── Load variant config and apply defaults ────────────────────────────────
    if args.ngf           is None: args.ngf           = 64
    if args.flow_scale    is None: args.flow_scale    = 0.5
    if args.scheduler     is None: args.scheduler     = "linear"
    if args.decay_start   is None: args.decay_start   = DECAY_START
    if args.lambda_l1     is None: args.lambda_l1     = LAMBDA_L1
    if args.lambda_vgg    is None: args.lambda_vgg    = LAMBDA_VGG
    if args.lambda_vgg_tryon is None: args.lambda_vgg_tryon = LAMBDA_VGG_TRYON
    if args.lambda_mask   is None: args.lambda_mask   = LAMBDA_MASK
    if args.lambda_smooth is None: args.lambda_smooth = LAMBDA_SMOOTH
    if args.lambda_gan    is None: args.lambda_gan    = 0.0
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    if args.stage == "both":
        logger = setup_logger("warp")
        train_warp(args, logger)
        # Auto-set warp checkpoint for tryon stage
        args.warp_ckpt = str(Path(CKPT_DIR) / "warp_best.pth")
        args.resume = ""
        logger = setup_logger("tryon")
        train_tryon(args, logger)
    elif args.stage == "warp":
        logger = setup_logger("warp")
        train_warp(args, logger)
    else:
        logger = setup_logger("tryon")
        train_tryon(args, logger)


if __name__ == "__main__":
    main()
