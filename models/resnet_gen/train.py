#!/usr/bin/env python3
"""
Two-stage training: WarpNet → ResNetGenerator.

Stage 1 (warp):  train WarpNet on cloth-to-person alignment.
Stage 2 (tryon): train ResNetGenerator on warped-cloth-to-person synthesis.

Usage:
  python models/resnet_gen/train.py --stage both --data dataset/train/tensors
  python models/resnet_gen/train.py --stage warp  --data dataset/train/tensors
  python models/resnet_gen/train.py --stage tryon --data dataset/train/tensors
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
from model.warp_model import WarpNet
from model.warp_utils import warp_cloth
from models.resnet_gen.network import ResNetGenerator
from shared.dataset import FastVITONLoader
from shared.losses import VGGLoss, smooth_loss, person_cloth_mask
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("resnet_gen")
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
    """Keep only the last n epoch checkpoints matching prefix_epochXXX.pth."""
    files = sorted(ckpt_dir.glob(f"{prefix}_epoch*.pth"))
    for f in files[:-n]:
        f.unlink(missing_ok=True)


def _unpack_batch(batch: dict) -> tuple:
    ag   = batch["agnostic"]                          # (B,3,H,W)
    cl   = batch["cloth"]                             # (B,3,H,W)
    cm   = batch["cloth_mask"]                        # (B,H,W)
    pose = batch["pose_map"]                          # (B,18,H,W)
    person = batch["person"]                          # (B,3,H,W)
    pm   = batch["parse_map"]                         # (B,H,W)
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)                         # (B,1,H,W)
    return ag, cl, cm, pose, person, pm


# ---------------------------------------------------------------------------
# Stage 1: WarpNet
# ---------------------------------------------------------------------------

def train_warp(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = FastVITONLoader(args.data, batch_size=args.batch, device=DEVICE)
    logger.info(f"[warp] dataset: {loader.n_samples} samples  batch={args.batch}")

    model  = WarpNet(in_channels=25, ngf=64, flow_scale=0.5).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "warp_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            ag, cl, cm, pose, person, pm = _unpack_batch(batch)
            pcm = person_cloth_mask(pm)

            # 25ch warp input
            warp_inp = torch.cat([ag, pose, cl, cm], dim=1)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                flow    = model(warp_inp)
                warped  = warp_cloth(cl,   flow)
                wm      = warp_cloth(cm,   flow)  # warped mask

                # Losses — compare against person clothing region
                l1_img   = l1_fn(warped * pcm, person * pcm)
                l_vgg    = vgg(warped * pcm, person * pcm)
                l_mask   = l1_fn(wm, pcm) * 5.0
                l_smooth = smooth_loss(flow) * 0.5
                loss = l1_img + l_vgg + l_mask + l_smooth

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        with torch.no_grad():
            ep_ssim = ssim_metric(warped * pcm, person * pcm).item()
            ep_psnr = psnr_metric(warped * pcm, person * pcm).item()
        logger.info(f"[warp] E{epoch:03d}  loss={avg_loss:.4f}  L1={l1_img.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"best={best_loss:.4f}  time={elapsed:.1f}s")

        # Save epoch checkpoint
        ep_path = ckpt_dir / f"warp_epoch{epoch:03d}.pth"
        _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                          "opt": opt.state_dict(), "loss": avg_loss}, ep_path)
        _keep_last_n(ckpt_dir, "warp", n=3)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            _save_checkpoint({"epoch": epoch, "model": model.state_dict(),
                               "loss": avg_loss}, best_path)
            logger.info(f"[warp]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[warp] early stopping at epoch {epoch}")
                break

    logger.info(f"[warp] training complete. best_loss={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2: ResNetGenerator
# ---------------------------------------------------------------------------

def train_tryon(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    loader = FastVITONLoader(args.data, batch_size=args.batch, device=DEVICE)
    logger.info(f"[tryon] dataset: {loader.n_samples} samples  batch={args.batch}")

    # Load WarpNet (frozen)
    warp_net  = WarpNet(in_channels=25, ngf=64, flow_scale=0.5).to(DEVICE)
    warp_best = ckpt_dir / "warp_best.pth"
    if warp_best.exists():
        state = torch.load(warp_best, map_location=DEVICE, weights_only=False)
        warp_net.load_state_dict(state["model"])
        logger.info(f"[tryon] loaded WarpNet from {warp_best}")
    else:
        logger.warning("[tryon] WarpNet checkpoint not found — using random weights!")
    warp_net.eval()
    for p in warp_net.parameters():
        p.requires_grad_(False)

    gen    = ResNetGenerator(in_channels=25, ngf=64).to(DEVICE)
    opt    = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "resnet_gen_best.pth"

    for epoch in range(1, args.epochs + 1):
        gen.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch in loader:
            ag, cl, cm, pose, person, pm = _unpack_batch(batch)

            with torch.no_grad():
                warp_inp  = torch.cat([ag, pose, cl, cm], dim=1)
                flow      = warp_net(warp_inp)
                warped    = warp_cloth(cl, flow)
                wm        = warp_cloth(cm, flow)

            # 25ch tryon input
            tryon_inp = torch.cat([ag, warped, wm, pose], dim=1)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake = gen(tryon_inp)
                l_l1  = l1_fn(fake, person) * 0.5
                l_vgg = vgg(fake, person)   * 2.0
                loss  = l_l1 + l_vgg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - t0
        with torch.no_grad():
            ep_ssim = ssim_metric(fake, person).item()
            ep_psnr = psnr_metric(fake, person).item()
        logger.info(f"[tryon] E{epoch:03d}  loss={avg_loss:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"best={best_loss:.4f}  time={elapsed:.1f}s")

        # Save epoch checkpoint
        ep_path = ckpt_dir / f"resnet_gen_epoch{epoch:03d}.pth"
        _save_checkpoint({"epoch": epoch, "model": gen.state_dict(),
                          "opt": opt.state_dict(), "loss": avg_loss}, ep_path)
        _keep_last_n(ckpt_dir, "resnet_gen", n=3)

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
            _save_checkpoint({"epoch": epoch, "model": gen.state_dict(),
                               "loss": avg_loss}, best_path)
            logger.info(f"[tryon]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[tryon] early stopping at epoch {epoch}")
                break

    logger.info(f"[tryon] training complete. best_loss={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet generator training")
    p.add_argument("--data",     default=str(ROOT / "dataset" / "train" / "tensors"),
                   help="path to .pt tensor directory")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--batch",    type=int, default=64)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stage",    choices=["warp", "tryon", "both"], default="both")
    p.add_argument("--ckpt-dir", default=str(ROOT / "checkpoints" / "resnet_gen"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",  default=str(ROOT / "logs" / "resnet_gen"),
                   dest="log_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.80)  # hard cap: OOM before CPU spillage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = Path(args.log_dir) / f"train_{timestamp}.txt"
    logger    = _setup_logger(log_path)

    logger.info(f"ResNet generator training  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    logger.info(f"  ckpt_dir={args.ckpt_dir}  log={log_path}")

    if args.stage in ("warp", "both"):
        train_warp(args, logger)

    if args.stage in ("tryon", "both"):
        train_tryon(args, logger)


if __name__ == "__main__":
    main()
