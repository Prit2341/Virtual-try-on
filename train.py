#!/usr/bin/env python3
"""
CNN-Based Virtual Try-On — Single-Stage Training (CP-VTON style)
=================================================================
Architecture type: CNN-Based (no explicit warping)

    Input Image (person + cloth)
         ↓
    [Encoder] Conv2D → BatchNorm → ReLU + MaxPool
         ↓
    [Bottleneck] deepest features
         ↓
    [Decoder] Bilinear upsample + skip connections
         ↓
    Output Image (try-on result)

Key operations:
    Conv2D → BatchNorm → ReLU  (double conv per level)
    MaxPool2d(2)               to downsample
    Bilinear upsample          to reconstruct
    L1 loss                    to train

Usage:
    python train.py --stage both        # train (stage arg accepted for compatibility)
    python train.py --epochs 50
    python train.py --batch 112 --max-samples 5000
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
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from model.tryon_model import TryOnNet
from shared.dataset import make_loader
from shared.metrics import ssim_metric, psnr_metric, metrics_header, metrics_separator, metrics_row

# ── Defaults ───────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
_BASE       = Path(__file__).resolve().parent
DATA_DIR    = str(_BASE / "dataset" / "train" / "tensors")
CKPT_DIR    = str(_BASE / "checkpoints")
LOG_DIR     = str(_BASE / "logs")

BATCH_SIZE  = 112
LR          = 2e-4
BETAS       = (0.5, 0.999)
NUM_WORKERS = 0
DECAY_START = 50

LAMBDA_L1   = 1.0       # primary loss: pixel-level reconstruction

PATIENCE    = 20
MIN_DELTA   = 1e-4

torch.backends.cudnn.benchmark = True


# ── Logger ────────────────────────────────────────────────────────────────────

def setup_logger(name: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOG_DIR, f"{name}_{timestamp}.txt")

    logger = logging.getLogger(name)
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
    bar = "=" * 60
    logger.info(bar)
    logger.info(f"  {title}")
    logger.info(bar)


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def save_ckpt(path, model, optimizer, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch}, path)


def cleanup_old_checkpoints(prefix: str, keep: int = 3):
    ckpt_dir = Path(CKPT_DIR)
    periodics = sorted(ckpt_dir.glob(f"{prefix}_epoch_*.pth"))
    for f in periodics[:-keep]:
        f.unlink(missing_ok=True)


def open_csv_log(name: str):
    os.makedirs(LOG_DIR, exist_ok=True)
    path   = os.path.join(LOG_DIR, f"{name}_metrics.csv")
    is_new = not os.path.exists(path)
    fh     = open(path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["epoch", "avg_l1", "ssim", "psnr_db",
                         "lr", "epoch_time_s", "best_l1"])
    return fh, writer


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


def log_images(writer, tag, images, step, n=2):
    writer.add_images(tag, ((images[:n] + 1) / 2).clamp(0, 1), step)


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, logger: logging.Logger):
    """
    CNN-Based single-stage training.

    Pipeline:
        person(3) + cloth(3)  →  [TryOnNet U-Net]  →  output(3)
        Loss: L1(output, person)

    No warping, no explicit flow field.  The U-Net learns to implicitly
    deform and blend the clothing through its encoder-decoder features.
    """
    loader = make_loader(args.data, args.batch, max_samples=args.max_samples,
                         num_workers=args.workers)

    log_section(logger, "CNN-BASED TRY-ON — CONFIGURATION")
    logger.info(f"  Architecture   : U-Net (BatchNorm + MaxPool + Bilinear)")
    logger.info(f"  Input channels : 6  (person 3 + cloth 3)")
    logger.info(f"  Loss           : L1 only (no explicit warping)")
    logger.info(f"  Device         : {DEVICE}")
    if DEVICE == "cuda":
        logger.info(f"  GPU            : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM           : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    logger.info(f"  Dataset        : {len(loader.dataset)} samples  ({args.data})")
    logger.info(f"  Batch size     : {args.batch}")
    logger.info(f"  Learning rate  : {args.lr}")
    logger.info(f"  Epochs         : {args.epochs}")
    logger.info(f"  AMP (fp16)     : {args.amp}")
    logger.info(f"  Early stop     : patience={args.patience}, min_delta={MIN_DELTA}")
    logger.info("")

    model  = TryOnNet(in_channels=6, ngf=args.ngf).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    sched  = make_scheduler(opt, args.epochs, args.decay_start, len(loader))
    scaler = GradScaler("cuda", enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "tryon"))
    csv_fh, csv_writer = open_csv_log("tryon")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {ckpt['epoch']}\n")

    step       = (start_epoch - 1) * len(loader)
    best_l1    = float("inf")
    best_epoch = -1
    patience_counter = 0
    train_start = time.time()

    log_section(logger, "CNN-BASED TRY-ON — EPOCH LOG")
    logger.info(metrics_header())
    logger.info(metrics_separator())

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[cnn] E{epoch:02d}", dynamic_ncols=True, leave=False)
        epoch_l1  = 0.0
        n_batches = 0
        epoch_start = time.time()

        for batch in pbar:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            per = batch["person"]          # (B, 3, H, W)
            cl  = batch["cloth"]           # (B, 3, H, W)

            # ── Forward ──────────────────────────────────────────────────────
            # CNN-Based: directly feed person + cloth, no warp step
            inp  = torch.cat([per, cl], dim=1)   # (B, 6, H, W)

            with autocast("cuda", enabled=args.amp):
                output = model(inp)              # (B, 3, H, W)
                loss   = F.l1_loss(output, per)  # L1 loss to train

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            step       += 1
            epoch_l1   += loss.item()
            n_batches  += 1
            pbar.set_postfix(L1=f"{loss.item():.4f}")

            if step % 100 == 0:
                writer.add_scalar("tryon/l1", loss.item(), step)
                writer.add_scalar("tryon/lr", sched.get_last_lr()[0], step)

        # ── Epoch summary ─────────────────────────────────────────────────────
        epoch_time = time.time() - epoch_start
        avg_l1     = epoch_l1 / max(n_batches, 1)
        cur_lr     = sched.get_last_lr()[0]

        writer.add_scalar("tryon/epoch_l1", avg_l1, epoch)

        model.eval()
        with torch.no_grad():
            epoch_ssim = ssim_metric(output, per).item()
            epoch_psnr = psnr_metric(output, per).item()
            log_images(writer, "tryon/output", output, step)
            log_images(writer, "tryon/person", per,    step)
            log_images(writer, "tryon/cloth",  cl,     step)
        model.train()

        writer.add_scalar("tryon/epoch_ssim", epoch_ssim, epoch)
        writer.add_scalar("tryon/epoch_psnr", epoch_psnr, epoch)

        save_ckpt(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth", model, opt, epoch)
        cleanup_old_checkpoints("tryon", keep=3)

        if avg_l1 < best_l1 - MIN_DELTA:
            best_l1         = avg_l1
            best_epoch      = epoch
            patience_counter = 0
            save_ckpt(f"{CKPT_DIR}/tryon_best.pth", model, opt, epoch)
            status = "* NEW BEST *  [ckpt]"
        else:
            patience_counter += 1
            status = f"wait {patience_counter}/{args.patience}  [ckpt]"

        logger.info(metrics_row(epoch, avg_l1, 0.0, epoch_ssim, epoch_psnr,
                                cur_lr, fmt_time(epoch_time), best_l1, status))
        csv_writer.writerow([epoch, f"{avg_l1:.6f}", f"{epoch_ssim:.4f}",
                             f"{epoch_psnr:.2f}", f"{cur_lr:.2e}",
                             f"{epoch_time:.1f}", f"{best_l1:.6f}"])
        csv_fh.flush()

        if patience_counter >= args.patience:
            logger.info(f"\n  >> Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - train_start
    log_section(logger, "CNN-BASED TRY-ON — FINAL SUMMARY")
    logger.info(f"  Total epochs trained : {epoch - start_epoch + 1}")
    logger.info(f"  Best L1 loss         : {best_l1:.4f}  (epoch {best_epoch})")
    logger.info(f"  Total training time  : {fmt_time(total_time)}")
    logger.info(f"  Best checkpoint      : {CKPT_DIR}/tryon_best.pth")
    logger.info("")

    csv_fh.close()
    writer.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="CNN-Based Virtual Try-On (single stage)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --stage accepted for run_all.py compatibility but ignored (single stage)
    p.add_argument("--stage",       default="both",
                   choices=["warp", "tryon", "both"],
                   help="Ignored — CNN-Based is single-stage")
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
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--ngf",         type=int, default=32)
    args = p.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,  exist_ok=True)

    logger = setup_logger("tryon")
    train(args, logger)


if __name__ == "__main__":
    main()
