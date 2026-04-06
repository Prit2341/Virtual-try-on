#!/usr/bin/env python3
"""
Single-stage virtual try-on training.

No warping stage — the network learns implicit cloth alignment directly.

Usage:
  python models/single_stage/train.py --data dataset/train/tensors
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from models.single_stage.network import SingleStageTryOn
from shared.dataset import make_loader
from shared.losses import VGGLoss
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("single_stage")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _keep_last_n(ckpt_dir: Path, prefix: str, n: int = 3) -> None:
    files = sorted(ckpt_dir.glob(f"{prefix}_epoch*.pth"))
    for f in files[:-n]:
        f.unlink(missing_ok=True)


def _unpack_batch(batch: dict) -> tuple:
    ag     = batch["agnostic"]
    cl     = batch["cloth"]
    cm     = batch["cloth_mask"]
    pose   = batch["pose_map"]
    person = batch["person"]
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)
    return ag, cl, cm, pose, person


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"dataset: {len(loader.dataset)} samples  batch={args.batch}")

    model  = SingleStageTryOn(in_channels=25, ngf=64).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "model_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person = _unpack_batch(batch)

            # 25ch input: agnostic + cloth + cloth_mask + pose
            inp = torch.cat([ag, cl, cm, pose], dim=1)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake  = model(inp)
                l_l1  = l1_fn(fake, person) * 1.0
                l_vgg = vgg(fake, person)   * 2.0
                loss  = l_l1 + l_vgg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        with torch.no_grad():
            ep_ssim = ssim_metric(fake, person).item()
            ep_psnr = psnr_metric(fake, person).item()
        logger.info(f"E{epoch:03d}  loss={avg_loss:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"best={best_loss:.4f}  time={elapsed:.1f}s")

        # Save epoch checkpoint
        ep_path = ckpt_dir / f"model_epoch{epoch:03d}.pth"
        _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                          "opt": opt.state_dict(), "loss": avg_loss}, ep_path)
        _keep_last_n(ckpt_dir, "model", n=3)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                               "loss": avg_loss}, best_path)
            logger.info(f"  => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"early stopping at epoch {epoch}")
                break

    logger.info(f"training complete. best_loss={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-stage try-on training")
    p.add_argument("--data",     default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=192)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--patience",    type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                   help="Limit dataset size (e.g. 5000)")
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "single_stage"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",  default=str(ROOT / "logs" / "single_stage"),
                   dest="log_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)  # hard cap: OOM before CPU spillage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = Path(args.log_dir) / f"train_{timestamp}.txt"
    logger    = _setup_logger(log_path)

    logger.info(f"Single-stage try-on training  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    train(args, logger)


if __name__ == "__main__":
    main()
