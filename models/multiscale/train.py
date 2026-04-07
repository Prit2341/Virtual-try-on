#!/usr/bin/env python3
"""
Two-stage multiscale training: CoarseNet → RefineNet.

Stage 1: Train CoarseNet end-to-end at half resolution (128×96).
Stage 2: Freeze CoarseNet, train RefineNet at full resolution.

Usage:
  python models/multiscale/train.py --stage both --data dataset/train/tensors
  python models/multiscale/train.py --stage coarse
  python models/multiscale/train.py --stage refine
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
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from models.multiscale.network import CoarseNet, RefineNet
from shared.dataset import make_loader
from shared.losses import VGGLoss
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Half resolution for coarse stage
COARSE_H, COARSE_W = 128, 96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multiscale")
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


def _downsample(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=True)


def _upsample_like(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=True)


# ---------------------------------------------------------------------------
# Stage 1: CoarseNet
# ---------------------------------------------------------------------------

def train_coarse(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[coarse] dataset: {len(loader.dataset)} samples  batch={args.batch}")

    model  = CoarseNet(ngf=32).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "coarse_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person = _unpack_batch(batch)

            # Downsample everything to coarse resolution
            ag_d     = _downsample(ag,     COARSE_H, COARSE_W)
            cl_d     = _downsample(cl,     COARSE_H, COARSE_W)
            cm_d     = _downsample(cm,     COARSE_H, COARSE_W)
            pose_d   = _downsample(pose,   COARSE_H, COARSE_W)
            person_d = _downsample(person, COARSE_H, COARSE_W)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                coarse, _, _ = model(ag_d, cl_d, cm_d, pose_d)
                l_l1  = l1_fn(coarse, person_d) * 1.0
                l_vgg = vgg(coarse, person_d)   * 1.0
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
            ep_ssim = ssim_metric(coarse, person_d).item()
            ep_psnr = psnr_metric(coarse, person_d).item()
        logger.info(f"[coarse] E{epoch:03d}  loss={avg_loss:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"best={best_loss:.4f}  time={elapsed:.1f}s")

        ep_path = ckpt_dir / f"coarse_epoch{epoch:03d}.pth"
        _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                          "opt": opt.state_dict(), "loss": avg_loss}, ep_path)
        _keep_last_n(ckpt_dir, "coarse", n=3)

        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                               "loss": avg_loss}, best_path)
            logger.info(f"[coarse]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[coarse] early stopping at epoch {epoch}")
                break

    logger.info(f"[coarse] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2: RefineNet
# ---------------------------------------------------------------------------

def train_refine(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[refine] dataset: {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen CoarseNet
    coarse_net  = CoarseNet(ngf=32).to(DEVICE)
    coarse_best = ckpt_dir / "coarse_best.pth"
    if coarse_best.exists():
        state = torch.load(coarse_best, map_location=DEVICE, weights_only=False)
        coarse_net.load_state_dict(state["model"])
        logger.info(f"[refine] loaded CoarseNet from {coarse_best}")
    else:
        logger.warning("[refine] CoarseNet checkpoint not found — random weights")
    coarse_net.eval()
    for p in coarse_net.parameters():
        p.requires_grad_(False)

    refine_net = RefineNet(in_channels=28, ngf=64).to(DEVICE)
    opt        = torch.optim.Adam(refine_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler     = GradScaler(enabled=(DEVICE == "cuda"))
    vgg        = VGGLoss().to(DEVICE)
    l1_fn      = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "refine_best.pth"

    for epoch in range(1, args.epochs + 1):
        refine_net.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person = _unpack_batch(batch)

            with torch.no_grad():
                # Run CoarseNet at half resolution
                ag_d   = _downsample(ag,   COARSE_H, COARSE_W)
                cl_d   = _downsample(cl,   COARSE_H, COARSE_W)
                cm_d   = _downsample(cm,   COARSE_H, COARSE_W)
                pose_d = _downsample(pose, COARSE_H, COARSE_W)

                coarse, warped_d, wm_d = coarse_net(ag_d, cl_d, cm_d, pose_d)

                # Upsample coarse outputs to full resolution
                coarse_up   = _upsample_like(coarse,   person)
                warped_full = _upsample_like(warped_d, person)
                wm_full     = _upsample_like(wm_d,     person)

            # RefineNet input: ag(3) + warped_full(3) + wm_full(1) + coarse_up(3) + pose(18) = 28
            refine_inp = torch.cat([ag, warped_full, wm_full, coarse_up, pose], dim=1)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                refined = refine_net(refine_inp)
                l_l1    = l1_fn(refined, person) * 1.0
                l_vgg   = vgg(refined, person)   * 2.0
                loss    = l_l1 + l_vgg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(refine_net.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        with torch.no_grad():
            ep_ssim = ssim_metric(refined, person).item()
            ep_psnr = psnr_metric(refined, person).item()
        logger.info(f"[refine] E{epoch:03d}  loss={avg_loss:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"best={best_loss:.4f}  time={elapsed:.1f}s")

        ep_path = ckpt_dir / f"refine_epoch{epoch:03d}.pth"
        _save_checkpoint({"epoch": epoch, "model": refine_net.state_dict(),
                          "opt": opt.state_dict(), "loss": avg_loss}, ep_path)
        _keep_last_n(ckpt_dir, "refine", n=3)

        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            _save_checkpoint({"epoch": epoch, "model": refine_net.state_dict(),
                               "loss": avg_loss}, best_path)
            logger.info(f"[refine]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[refine] early stopping at epoch {epoch}")
                break

    logger.info(f"[refine] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiscale try-on training")
    p.add_argument("--data",     default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=96)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stage",       choices=["coarse", "refine", "both"], default="both")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples",
                   help="Limit dataset size (e.g. 5000)")
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "multiscale"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",  default=str(ROOT / "logs" / "multiscale"),
                   dest="log_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)  # hard cap: OOM before CPU spillage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = Path(args.log_dir) / f"train_{timestamp}.txt"
    logger    = _setup_logger(log_path)

    logger.info(f"Multiscale try-on training  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    if args.stage in ("coarse", "both"):
        train_coarse(args, logger)

    if args.stage in ("refine", "both"):
        train_refine(args, logger)


if __name__ == "__main__":
    main()
