#!/usr/bin/env python3
"""
PF-AFN training — Parser-Free Appearance Flow Network.

Stage 1  — train AppearanceFlowNet (AFN): dense flow-based cloth warping
Stage 2  — train ContentFusionNet (CFN): synthesis from warped cloth + agnostic

Key: no human parsing map used at inference — only agnostic + cloth + pose.

Usage:
  python models/pfafn/train.py --stage both
  python models/pfafn/train.py --stage afn
  python models/pfafn/train.py --stage cfn
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

from models.pfafn.network import AppearanceFlowNet, ContentFusionNet
from shared.dataset import make_loader
from shared.losses import VGGLoss, smooth_loss, person_cloth_mask
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pfafn")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _save(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _keep_last_n(ckpt_dir: Path, prefix: str, n: int = 3) -> None:
    for f in sorted(ckpt_dir.glob(f"{prefix}_epoch*.pth"))[:-n]:
        f.unlink(missing_ok=True)


def _unpack(batch: dict) -> tuple:
    ag     = batch["agnostic"]
    cl     = batch["cloth"]
    cm     = batch["cloth_mask"]
    pose   = batch["pose_map"]
    person = batch["person"]
    parse  = batch["parse_map"]
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)
    return ag, cl, cm, pose, person, parse


def _make_lr_lambda(total_epochs: int, decay_start: int):
    def fn(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, total_epochs - decay_start))
    return fn


# ---------------------------------------------------------------------------
# Stage 1: AppearanceFlowNet (AFN)
# ---------------------------------------------------------------------------

def train_afn(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[afn] {len(loader.dataset)} samples  batch={args.batch}")

    afn    = AppearanceFlowNet(ngf=64).to(DEVICE)
    opt    = torch.optim.Adam(afn.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched  = torch.optim.lr_scheduler.LambdaLR(
                 opt, _make_lr_lambda(args.epochs, args.decay_start))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "afn_best.pth"

    if args.resume_afn and Path(args.resume_afn).exists():
        state = torch.load(args.resume_afn, map_location=DEVICE, weights_only=False)
        afn.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        logger.info(f"[afn] resumed from {args.resume_afn}")

    for epoch in range(1, args.epochs + 1):
        afn.train()
        ep_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person, parse = _unpack(batch)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                warped_cl, warped_cm, flow = afn(cl, cm, ag)
                warped_cl = warped_cl * warped_cm

                cloth_mask = person_cloth_mask(parse).to(DEVICE)
                # Pixel alignment loss (cloth region only)
                l_l1    = l1_fn(warped_cl * cloth_mask, person * cloth_mask)
                # Perceptual loss on full warped cloth
                l_vgg   = vgg(warped_cl, person) * 0.5
                # Mask shape loss
                l_mask  = l1_fn(warped_cm, cloth_mask) * 2.0
                # Flow smoothness (TV regularization)
                l_smooth = smooth_loss(flow) * args.lambda_smooth
                loss     = l_l1 + l_vgg + l_mask + l_smooth

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(afn.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        sched.step()
        avg = ep_loss / len(loader)
        dt  = time.time() - t0
        lr  = opt.param_groups[0]["lr"]
        with torch.no_grad():
            ep_ssim = ssim_metric(warped_cl, person).item()
            ep_psnr = psnr_metric(warped_cl, person).item()

        logger.info(f"[afn] E{epoch:03d}  loss={avg:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"lr={lr:.2e}  best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"afn_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": afn.state_dict(),
               "opt": opt.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "afn")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": afn.state_dict(), "loss": avg}, best_path)
            logger.info(f"[afn]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[afn] early stop at epoch {epoch}")
                break

    logger.info(f"[afn] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2: ContentFusionNet (CFN)
# ---------------------------------------------------------------------------

def train_cfn(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[cfn] {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen AFN
    afn = AppearanceFlowNet(ngf=64).to(DEVICE)
    afn_path = ckpt_dir / "afn_best.pth"
    if afn_path.exists():
        state = torch.load(afn_path, map_location=DEVICE, weights_only=False)
        afn.load_state_dict(state["model"])
        logger.info(f"[cfn] loaded AFN from {afn_path}")
    else:
        logger.warning("[cfn] AFN checkpoint not found — random weights")
    afn.eval()
    for p in afn.parameters():
        p.requires_grad_(False)

    cfn    = ContentFusionNet(in_channels=25, ngf=64).to(DEVICE)
    opt    = torch.optim.Adam(cfn.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched  = torch.optim.lr_scheduler.LambdaLR(
                 opt, _make_lr_lambda(args.epochs, args.decay_start))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "cfn_best.pth"

    if args.resume_cfn and Path(args.resume_cfn).exists():
        state = torch.load(args.resume_cfn, map_location=DEVICE, weights_only=False)
        cfn.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        logger.info(f"[cfn] resumed from {args.resume_cfn}")

    for epoch in range(1, args.epochs + 1):
        cfn.train()
        ep_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person, _ = _unpack(batch)

            with torch.no_grad():
                warped_cl, warped_cm, _ = afn(cl, cm, ag)
                warped_cl = warped_cl * warped_cm

            # CFN input — parser-free: no parse_map used
            cfn_inp = torch.cat([ag, warped_cl, warped_cm, pose], dim=1)  # 25ch

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                output = cfn(cfn_inp)
                l_l1   = l1_fn(output, person)
                l_vgg  = vgg(output, person)
                loss   = l_l1 + l_vgg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(cfn.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        sched.step()
        avg = ep_loss / len(loader)
        dt  = time.time() - t0
        lr  = opt.param_groups[0]["lr"]
        with torch.no_grad():
            ep_ssim = ssim_metric(output, person).item()
            ep_psnr = psnr_metric(output, person).item()

        logger.info(f"[cfn] E{epoch:03d}  loss={avg:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"lr={lr:.2e}  best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"cfn_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": cfn.state_dict(),
               "opt": opt.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "cfn")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": cfn.state_dict(), "loss": avg}, best_path)
            logger.info(f"[cfn]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[cfn] early stop at epoch {epoch}")
                break

    logger.info(f"[cfn] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PF-AFN training")
    p.add_argument("--data",           default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch",          type=int,   default=48)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--decay-start",    type=int,   default=50, dest="decay_start")
    p.add_argument("--patience",       type=int,   default=20)
    p.add_argument("--lambda-smooth",  type=float, default=2.0, dest="lambda_smooth")
    p.add_argument("--stage",          choices=["afn", "cfn", "both"], default="both")
    p.add_argument("--max-samples",    type=int,   default=None, dest="max_samples")
    p.add_argument("--ckpt-dir",       default=str(ROOT / "checkpoints" / "pfafn"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",        default=str(ROOT / "logs" / "pfafn"),
                   dest="log_dir")
    p.add_argument("--resume-afn",     default=None, dest="resume_afn")
    p.add_argument("--resume-cfn",     default=None, dest="resume_cfn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.backends.cudnn.benchmark = True

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger  = _setup_logger(log_dir / f"train_{ts}.txt")

    logger.info(f"PF-AFN  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    logger.info(f"  lambda_smooth={args.lambda_smooth}")

    if args.stage in ("afn", "both"):
        train_afn(args, logger)
    if args.stage in ("cfn", "both"):
        train_cfn(args, logger)


if __name__ == "__main__":
    main()
