#!/usr/bin/env python3
"""
CP-VITON training — GMM (TPS warp) + TOM (composition) + PatchGAN.

Stage 1  — train GMM: predicts TPS control points → warped cloth
Stage 2  — train TOM + PatchGAN:
             generator  loss = L1 + VGG + lambda_gan * GAN_g
             discriminator loss = BCE(real=1) + BCE(fake=0) + feature matching

Usage:
  python models/cp_viton/train.py --stage both
  python models/cp_viton/train.py --stage gmm
  python models/cp_viton/train.py --stage tom
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

from model.gmm_model import GMMNet
from models.cp_viton.network import TryOnModule, PatchGAN
from shared.dataset import make_loader
from shared.losses import VGGLoss, smooth_loss, person_cloth_mask
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cp_viton")
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
# Stage 1: GMM (TPS warp)
# ---------------------------------------------------------------------------

def train_gmm(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[gmm] {len(loader.dataset)} samples  batch={args.batch}")

    gmm    = GMMNet(in_h=256, in_w=192, grid_size=5, ngf=64).to(DEVICE)
    opt    = torch.optim.Adam(gmm.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched  = torch.optim.lr_scheduler.LambdaLR(
                 opt, _make_lr_lambda(args.epochs, args.decay_start))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "gmm_best.pth"

    # Resume
    if args.resume_gmm and Path(args.resume_gmm).exists():
        state = torch.load(args.resume_gmm, map_location=DEVICE, weights_only=False)
        gmm.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        logger.info(f"[gmm] resumed from {args.resume_gmm}")

    for epoch in range(1, args.epochs + 1):
        gmm.train()
        ep_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person, parse = _unpack(batch)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                warped_cl, warped_cm, theta = gmm(cl, cm, ag, pose)
                warped_cl = warped_cl * warped_cm

                cloth_mask = person_cloth_mask(parse).to(DEVICE)
                l_l1    = l1_fn(warped_cl * cloth_mask, person * cloth_mask)
                l_vgg   = vgg(warped_cl, person) * 0.5
                l_mask  = l1_fn(warped_cm, cloth_mask) * 2.0
                # TV regularization on theta (not flow — TPS is inherently smooth)
                l_tv    = theta.abs().mean() * 0.1
                loss    = l_l1 + l_vgg + l_mask + l_tv

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(gmm.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        sched.step()
        avg   = ep_loss / len(loader)
        dt    = time.time() - t0
        lr    = opt.param_groups[0]["lr"]
        with torch.no_grad():
            ep_ssim = ssim_metric(warped_cl, person).item()
            ep_psnr = psnr_metric(warped_cl, person).item()

        logger.info(f"[gmm] E{epoch:03d}  loss={avg:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"lr={lr:.2e}  best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"gmm_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": gmm.state_dict(),
               "opt": opt.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "gmm")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": gmm.state_dict(), "loss": avg}, best_path)
            logger.info(f"[gmm]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[gmm] early stop at epoch {epoch}")
                break

    logger.info(f"[gmm] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2: TOM + PatchGAN
# ---------------------------------------------------------------------------

def train_tom(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[tom] {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen GMM
    gmm = GMMNet(in_h=256, in_w=192, grid_size=5, ngf=64).to(DEVICE)
    gmm_path = ckpt_dir / "gmm_best.pth"
    if gmm_path.exists():
        state = torch.load(gmm_path, map_location=DEVICE, weights_only=False)
        gmm.load_state_dict(state["model"])
        logger.info(f"[tom] loaded GMM from {gmm_path}")
    else:
        logger.warning("[tom] GMM checkpoint not found — random weights")
    gmm.eval()
    for p in gmm.parameters():
        p.requires_grad_(False)

    # Generator (TOM) + Discriminator
    tom   = TryOnModule(in_channels=25, ngf=64).to(DEVICE)
    disc  = PatchGAN(in_channels=9, ndf=64, n_layers=3).to(DEVICE)

    opt_g = torch.optim.Adam(tom.parameters(),  lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    sched_g = torch.optim.lr_scheduler.LambdaLR(
                  opt_g, _make_lr_lambda(args.epochs, args.decay_start))
    sched_d = torch.optim.lr_scheduler.LambdaLR(
                  opt_d, _make_lr_lambda(args.epochs, args.decay_start))

    scaler_g = GradScaler(enabled=(DEVICE == "cuda"))
    scaler_d = GradScaler(enabled=(DEVICE == "cuda"))
    vgg      = VGGLoss().to(DEVICE)
    l1_fn    = nn.L1Loss()
    bce      = nn.BCEWithLogitsLoss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "tom_best.pth"

    if args.resume_tom and Path(args.resume_tom).exists():
        state = torch.load(args.resume_tom, map_location=DEVICE, weights_only=False)
        tom.load_state_dict(state["model"])
        opt_g.load_state_dict(state["opt_g"])
        logger.info(f"[tom] resumed from {args.resume_tom}")

    for epoch in range(1, args.epochs + 1):
        tom.train()
        disc.train()
        ep_loss_g = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person, parse = _unpack(batch)

            with torch.no_grad():
                warped_cl, warped_cm, _ = gmm(cl, cm, ag, pose)
                warped_cl = warped_cl * warped_cm

            tom_inp = torch.cat([ag, warped_cl, warped_cm, pose], dim=1)  # 25ch

            # ---- Discriminator update ----
            opt_d.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake_out, _, _ = tom(tom_inp, warped_cl)
                # Condition: agnostic + warped_cloth (6ch) + image (3ch) = 9ch
                cond     = torch.cat([ag, warped_cl], dim=1)
                real_in  = torch.cat([person,   cond], dim=1)
                fake_in  = torch.cat([fake_out.detach(), cond], dim=1)

                d_real = disc(real_in)
                d_fake = disc(fake_in)
                ones   = torch.ones_like(d_real)
                zeros  = torch.zeros_like(d_fake)
                loss_d = (bce(d_real, ones) + bce(d_fake, zeros)) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---- Generator update ----
            opt_g.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake_out, rendered, alpha = tom(tom_inp, warped_cl)
                fake_in_g = torch.cat([fake_out, cond], dim=1)
                d_fake_g  = disc(fake_in_g)

                l_l1      = l1_fn(fake_out, person)
                l_vgg     = vgg(fake_out, person)
                l_render  = l1_fn(rendered, person) * 0.5   # rendered person penalty
                l_gan     = bce(d_fake_g, torch.ones_like(d_fake_g)) * args.lambda_gan
                # Feature matching (discriminator intermediate layers)
                with torch.no_grad():
                    real_feats = disc(real_in)
                fake_feats = disc(fake_in_g)
                l_fm      = F.l1_loss(fake_feats, real_feats.detach()) * args.lambda_fm
                loss_g    = l_l1 + l_vgg + l_render + l_gan + l_fm

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(tom.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
            ep_loss_g += loss_g.item()

        sched_g.step()
        sched_d.step()
        avg = ep_loss_g / len(loader)
        dt  = time.time() - t0
        lr  = opt_g.param_groups[0]["lr"]
        with torch.no_grad():
            ep_ssim = ssim_metric(fake_out, person).item()
            ep_psnr = psnr_metric(fake_out, person).item()

        logger.info(f"[tom] E{epoch:03d}  loss={avg:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"lr={lr:.2e}  best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"tom_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": tom.state_dict(),
               "opt_g": opt_g.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "tom")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": tom.state_dict(), "loss": avg}, best_path)
            logger.info(f"[tom]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[tom] early stop at epoch {epoch}")
                break

    logger.info(f"[tom] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CP-VITON training")
    p.add_argument("--data",        default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch",       type=int,   default=48)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--decay-start", type=int,   default=50, dest="decay_start")
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--lambda-gan",  type=float, default=1.0, dest="lambda_gan")
    p.add_argument("--lambda-fm",   type=float, default=10.0, dest="lambda_fm")
    p.add_argument("--stage",       choices=["gmm", "tom", "both"], default="both")
    p.add_argument("--max-samples", type=int,   default=None, dest="max_samples")
    p.add_argument("--ckpt-dir",    default=str(ROOT / "checkpoints" / "cp_viton"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",     default=str(ROOT / "logs" / "cp_viton"),
                   dest="log_dir")
    p.add_argument("--resume-gmm",  default=None, dest="resume_gmm")
    p.add_argument("--resume-tom",  default=None, dest="resume_tom")
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

    logger.info(f"CP-VITON  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    logger.info(f"  lambda_gan={args.lambda_gan}  lambda_fm={args.lambda_fm}")

    if args.stage in ("gmm", "both"):
        train_gmm(args, logger)
    if args.stage in ("tom", "both"):
        train_tom(args, logger)


if __name__ == "__main__":
    main()
