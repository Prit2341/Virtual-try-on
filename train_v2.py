#!/usr/bin/env python3
"""
Flow-Based Virtual Try-On — Two-Stage Training (PF-AFN / HR-VITON style)
=========================================================================
Architecture type: Flow-Based (explicit cloth warping)

    Stage 1 — WARP
    ┌──────────────────────────────────────────────────────────┐
    │  Cloth features  ──┐                                     │
    │                     → Correlation layer → Flow field     │
    │  Person features ──┘     ↓                               │
    │                      Grid Sample → Warped cloth          │
    └──────────────────────────────────────────────────────────┘

    Stage 2 — GENERATE
    ┌──────────────────────────────────────────────────────────┐
    │  Person image + Warped cloth → Generator → Final output  │
    └──────────────────────────────────────────────────────────┘

Key PyTorch operations:
    # Stage 1 — appearance flow (TPS Geometric Matching Module)
    cloth_feat, person_feat = encoders(cloth, person)
    corr = correlation_layer(person_feat, cloth_feat)   # matching cost
    theta = regressor(corr)                              # control-point offsets
    grid = tps_generator(theta)                          # flow field
    warped_cloth = F.grid_sample(cloth, grid)            # warp

    # Stage 2 — generation
    output = generator(person, warped_cloth)             # synthesise try-on
    loss = L1(output, target) + VGG(output, target)

Loss:
    Stage 1 (GMM): mask_L1 + L1 + VGG + TPS_regularization
    Stage 2 (Gen): L1 + VGG perceptual + alpha regularization

Usage:
  python train_v2.py --stage gmm
  python train_v2.py --stage tryon --gmm-ckpt checkpoints/v2/gmm_best.pth
  python train_v2.py --stage both --epochs 100
"""

import argparse
import csv
import logging
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model.gmm_model import GMMNet
from model.tryon_model_v2 import TryOnNetV2
from shared.dataset import make_loader
from shared.metrics import ssim_metric, psnr_metric, metrics_header, metrics_separator, metrics_row

# ── Defaults ───────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
_BASE       = Path(__file__).resolve().parent
DATA_DIR    = str(_BASE / "dataset" / "train" / "tensors")
CKPT_DIR    = str(_BASE / "checkpoints" / "v2")
LOG_DIR     = str(_BASE / "logs" / "v2")

BATCH_SIZE  = 64
LR          = 2e-4
BETAS       = (0.5, 0.999)
NUM_WORKERS = 0
DECAY_START = 50

LAMBDA_L1        = 1.0
LAMBDA_VGG       = 0.5
LAMBDA_MASK      = 5.0    # warped mask vs person cloth mask alignment
LAMBDA_THETA_REG = 0.01   # penalizes large TPS control-point offsets

LAMBDA_L1_TRYON  = 1.0
LAMBDA_VGG_TRYON = 0.5
LAMBDA_ALPHA_TV  = 0.1    # smooth composition mask boundaries
LAMBDA_ALPHA_REG = 0.5    # pushes alpha toward 0 or 1 (crisp boundaries)

PATIENCE    = 20
MIN_DELTA   = 1e-4

CLOTH_LABELS = [5, 6, 7]   # CIHP: upper-clothes=5, skirt=6, dress=7 (matches parse-v3 labels)

torch.backends.cudnn.benchmark = True


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(stage: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"{stage}_v2_{timestamp}.txt")

    logger = logging.getLogger(f"{stage}_v2")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


def log_section(logger, title):
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


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


# ── VGG Loss ──────────────────────────────────────────────────────────────────

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg[:4])),
            nn.Sequential(*list(vgg[4:9])),
            nn.Sequential(*list(vgg[9:16])),
            nn.Sequential(*list(vgg[16:23])),
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
    mask = torch.zeros_like(parse_map, dtype=torch.float32)
    for lbl in CLOTH_LABELS:
        mask += (parse_map == lbl).float()
    return mask.unsqueeze(1).clamp_(0, 1)


def make_scheduler(optimizer, epochs, decay_start, steps_per_epoch):
    warmup_end = decay_start * steps_per_epoch
    total      = epochs * steps_per_epoch
    decay_len  = max(total - warmup_end, 1)

    def lr_lambda(step):
        if step < warmup_end:
            return 1.0
        return max(0.0, 1.0 - (step - warmup_end) / decay_len)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_ckpt(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, path)


def cleanup_old_checkpoints(stage, keep=2):
    ckpt_dir = Path(CKPT_DIR)
    periodics = sorted(ckpt_dir.glob(f"{stage}_epoch_*.pth"))
    for f in periodics[:-keep] if len(periodics) > keep else []:
        f.unlink(missing_ok=True)


def check_disk_space():
    # Use ROOT dir as fallback if CKPT_DIR doesn't exist yet
    check_path = CKPT_DIR if os.path.exists(CKPT_DIR) else str(_BASE)
    free_gb = shutil.disk_usage(check_path).free / 1e9
    if free_gb < 0.5:
        print(f"\n  !! WARNING: Only {free_gb:.2f} GB free on disk !!\n")
    return free_gb


def open_csv_log(stage):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{stage}_v2_metrics.csv")
    is_new = not os.path.exists(path)
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["epoch", "avg_l1", "avg_vgg", "lr", "epoch_time_s", "best_l1"])
    return fh, writer


def log_images(writer, tag, images, step, n=2):
    writer.add_images(tag, ((images[:n] + 1) / 2).clamp(0, 1), step)


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# ── Stage 1: GMM Training ────────────────────────────────────────────────────

def train_gmm(args, logger):
    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)

    log_section(logger, "GMM TRAINING (V2)  —  CONFIGURATION")
    logger.info(f"  Device         : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  Dataset        : {len(loader.dataset)} samples")
    logger.info(f"  Batches/epoch  : {len(loader)}")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Learning rate  : {args.lr}")
    logger.info(f"  LR decay start : epoch {args.decay_start}")
    logger.info(f"  AMP (fp16)     : {args.amp}")
    logger.info(f"  Lambda L1      : {LAMBDA_L1}")
    logger.info(f"  Lambda VGG     : {LAMBDA_VGG}")
    logger.info(f"  Lambda Mask    : {LAMBDA_MASK}")
    logger.info(f"  Lambda Theta   : {LAMBDA_THETA_REG}")
    logger.info(f"  Max epochs     : {args.epochs}")
    logger.info(f"  Early stop     : patience={args.patience}")
    logger.info(f"  Free disk      : {check_disk_space():.2f} GB")
    logger.info("")

    model = GMMNet().to(DEVICE)
    vgg   = VGGLoss().to(DEVICE).eval()

    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    sched  = make_scheduler(opt, args.epochs, args.decay_start, len(loader))
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "gmm_v2"))
    csv_fh, csv_writer = open_csv_log("gmm")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}")

    step = (start_epoch - 1) * len(loader)
    best_l1 = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_start = time.time()

    log_section(logger, "GMM TRAINING  —  EPOCH LOG")
    logger.info(metrics_header())
    logger.info(metrics_separator())

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[gmm] E{epoch:02d}", dynamic_ncols=True, leave=False)
        epoch_l1 = epoch_vgg = epoch_mask = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch in pbar:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag   = batch["agnostic"]
            cl   = batch["cloth"]
            cm   = batch["cloth_mask"].unsqueeze(1)
            pose = batch["pose_map"]
            per  = batch["person"]
            pm   = batch["parse_map"]

            with autocast(enabled=False):  # lstsq requires float32 — no fp16
                warped_cloth, warped_mask, theta = model(
                    cl.float(), cm.float(), ag.float(), pose.float())
                pcm = person_cloth_mask(pm)

                # Losses
                l1       = F.l1_loss(warped_cloth * pcm, per * pcm)
                vg       = vgg(warped_cloth * pcm, per * pcm)
                mask_l   = F.l1_loss(warped_mask, pcm)
                theta_l  = (theta ** 2).mean()

                loss = (LAMBDA_L1 * l1 + LAMBDA_VGG * vg
                        + LAMBDA_MASK * mask_l + LAMBDA_THETA_REG * theta_l)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            epoch_l1   += l1.item()
            epoch_vgg  += vg.item()
            epoch_mask += mask_l.item()
            n_batches  += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", Mask=f"{mask_l.item():.3f}")

            if step % 100 == 0:
                writer.add_scalar("gmm/l1",    l1.item(), step)
                writer.add_scalar("gmm/vgg",   vg.item(), step)
                writer.add_scalar("gmm/mask",  mask_l.item(), step)
                writer.add_scalar("gmm/theta", theta_l.item(), step)
                writer.add_scalar("gmm/lr",    sched.get_last_lr()[0], step)

        # ── Epoch summary ─────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_l1   = epoch_l1   / max(n_batches, 1)
        avg_vgg  = epoch_vgg  / max(n_batches, 1)
        avg_mask = epoch_mask / max(n_batches, 1)
        cur_lr   = sched.get_last_lr()[0]

        # ── Quality metrics on last batch ────────────────────────────────
        model.eval()
        with torch.no_grad():
            wc, wm, _ = model(cl, cm, ag, pose)
            epoch_ssim = ssim_metric(wc * pcm, per * pcm).item()
            epoch_psnr = psnr_metric(wc * pcm, per * pcm).item()
            log_images(writer, "gmm/cloth",  cl,       step)
            log_images(writer, "gmm/warped", wc,       step)
            log_images(writer, "gmm/target", per * pcm, step)
        model.train()

        writer.add_scalar("gmm/epoch_ssim", epoch_ssim, epoch)
        writer.add_scalar("gmm/epoch_psnr", epoch_psnr, epoch)

        ckpt_saved = ""
        if epoch % 10 == 0 or epoch == args.epochs:
            save_ckpt(f"{CKPT_DIR}/gmm_epoch_{epoch:03d}.pth", model, opt, epoch)
            cleanup_old_checkpoints("gmm", keep=2)
            ckpt_saved = "[ckpt]"

        if avg_l1 < best_l1 - MIN_DELTA:
            best_l1 = avg_l1
            best_epoch = epoch
            patience_counter = 0
            save_ckpt(f"{CKPT_DIR}/gmm_best.pth", model, opt, epoch)
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
            logger.info(f"\n  >> Early stopping at epoch {epoch}.")
            break

    total_time = time.time() - train_start
    log_section(logger, "GMM TRAINING  —  FINAL SUMMARY")
    logger.info(f"  Best L1 : {best_l1:.4f}  (epoch {best_epoch})")
    logger.info(f"  Time    : {fmt_time(total_time)}")
    logger.info(f"  Ckpt    : {CKPT_DIR}/gmm_best.pth")
    csv_fh.close()
    writer.close()


# ── Stage 2: TryOnNet V2 Training ────────────────────────────────────────────

def train_tryon(args, logger):
    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)

    log_section(logger, "TRYONNET V2 TRAINING  —  CONFIGURATION")
    logger.info(f"  Device         : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
    logger.info(f"  Dataset        : {len(loader.dataset)} samples")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Learning rate  : {args.lr}")
    logger.info(f"  Lambda L1      : {LAMBDA_L1_TRYON}")
    logger.info(f"  Lambda VGG     : {LAMBDA_VGG_TRYON}")
    logger.info(f"  Lambda Alpha TV: {LAMBDA_ALPHA_TV}")
    logger.info(f"  Lambda Alpha R : {LAMBDA_ALPHA_REG}")
    logger.info(f"  GMM ckpt       : {args.gmm_ckpt or 'NONE!'}")
    logger.info(f"  Free disk      : {check_disk_space():.2f} GB")
    logger.info("")

    # Frozen GMM
    gmm = GMMNet().to(DEVICE).eval()
    if args.gmm_ckpt:
        gmm.load_state_dict(
            torch.load(args.gmm_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        logger.info(f"Loaded GMM: {args.gmm_ckpt}")
    else:
        logger.warning("WARNING: No GMM checkpoint — random warp weights!")
    for p in gmm.parameters():
        p.requires_grad = False

    gen  = TryOnNetV2().to(DEVICE)
    vgg  = VGGLoss().to(DEVICE).eval()

    opt    = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=BETAS)
    sched  = make_scheduler(opt, args.epochs, args.decay_start, len(loader))
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "tryon_v2"))
    csv_fh, csv_writer = open_csv_log("tryon_v2")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    step = (start_epoch - 1) * len(loader)
    best_l1 = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_start = time.time()

    log_section(logger, "TRYONNET V2  —  EPOCH LOG")
    logger.info(metrics_header())
    logger.info(metrics_separator())

    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        pbar = tqdm(loader, desc=f"[tryon] E{epoch:02d}", dynamic_ncols=True, leave=False)
        epoch_l1 = epoch_vgg = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch in pbar:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag   = batch["agnostic"]
            cl   = batch["cloth"]
            cm   = batch["cloth_mask"].unsqueeze(1)
            pose = batch["pose_map"]
            per  = batch["person"]

            # Warp with frozen GMM
            with torch.no_grad():
                warped_cloth, warped_mask, _ = gmm(cl, cm, ag, pose)

            # TryOnNet V2 input
            inp = torch.cat([ag, warped_cloth, warped_mask, pose], dim=1)  # 25ch

            with autocast(enabled=args.amp):
                output, rendered, alpha = gen(inp, warped_cloth=warped_cloth)

                l1  = F.l1_loss(output, per)
                vg  = vgg(output, per)

                # Alpha regularization: push toward 0 or 1 (crisp composition)
                alpha_reg = (alpha * (1 - alpha)).mean()
                # Alpha TV: smooth boundaries
                alpha_tv = ((alpha[:, :, 1:, :] - alpha[:, :, :-1, :]).abs().mean()
                          + (alpha[:, :, :, 1:] - alpha[:, :, :, :-1]).abs().mean())

                loss = (LAMBDA_L1_TRYON * l1 + LAMBDA_VGG_TRYON * vg
                        + LAMBDA_ALPHA_REG * alpha_reg + LAMBDA_ALPHA_TV * alpha_tv)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step += 1
            epoch_l1  += l1.item()
            epoch_vgg += vg.item()
            n_batches += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", VGG=f"{vg.item():.3f}",
                             a=f"{alpha.mean().item():.2f}")

            if step % 100 == 0:
                writer.add_scalar("tryon/l1",    l1.item(), step)
                writer.add_scalar("tryon/vgg",   vg.item(), step)
                writer.add_scalar("tryon/alpha", alpha.mean().item(), step)
                writer.add_scalar("tryon/lr",    sched.get_last_lr()[0], step)

        epoch_time = time.time() - epoch_start
        avg_l1  = epoch_l1  / max(n_batches, 1)
        avg_vgg = epoch_vgg / max(n_batches, 1)
        cur_lr  = sched.get_last_lr()[0]

        # ── Quality metrics ──────────────────────────────────────────────
        gen.eval()
        with torch.no_grad():
            epoch_ssim = ssim_metric(output, per).item()
            epoch_psnr = psnr_metric(output, per).item()
            log_images(writer, "tryon/output",  output,  step)
            log_images(writer, "tryon/person",  per,     step)
            log_images(writer, "tryon/warped",  warped_cloth, step)
            writer.add_images("tryon/alpha", alpha[:2], step)
        gen.train()

        writer.add_scalar("tryon/epoch_ssim", epoch_ssim, epoch)
        writer.add_scalar("tryon/epoch_psnr", epoch_psnr, epoch)

        ckpt_saved = ""
        if epoch % 10 == 0 or epoch == args.epochs:
            save_ckpt(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth", gen, opt, epoch)
            cleanup_old_checkpoints("tryon", keep=2)
            ckpt_saved = "[ckpt]"

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
            logger.info(f"\n  >> Early stopping at epoch {epoch}.")
            break

    total_time = time.time() - train_start
    log_section(logger, "TRYONNET V2  —  FINAL SUMMARY")
    logger.info(f"  Best L1 : {best_l1:.4f}  (epoch {best_epoch})")
    logger.info(f"  Time    : {fmt_time(total_time)}")
    logger.info(f"  Ckpt    : {CKPT_DIR}/tryon_best.pth")
    csv_fh.close()
    writer.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="VITON V2 Training (GMM + Composition TryOn)")
    p.add_argument("--stage",       required=True, choices=["gmm", "tryon", "both"])
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=PATIENCE)
    p.add_argument("--batch",       type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--decay-start", type=int,   default=DECAY_START, dest="decay_start")
    p.add_argument("--workers",     type=int,   default=NUM_WORKERS)
    p.add_argument("--data",        default=DATA_DIR)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      dest="amp", action="store_false")
    p.add_argument("--resume",      default="")
    p.add_argument("--gmm-ckpt",    default="", dest="gmm_ckpt")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    args = p.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)  # hard cap: OOM before CPU spillage

    if args.stage == "both":
        logger = setup_logger("gmm")
        train_gmm(args, logger)
        args.gmm_ckpt = str(Path(CKPT_DIR) / "gmm_best.pth")
        args.resume = ""
        logger = setup_logger("tryon_v2")
        train_tryon(args, logger)
    elif args.stage == "gmm":
        logger = setup_logger("gmm")
        train_gmm(args, logger)
    else:
        logger = setup_logger("tryon_v2")
        train_tryon(args, logger)


if __name__ == "__main__":
    main()
