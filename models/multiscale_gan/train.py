#!/usr/bin/env python3
"""
Multiscale GAN training — CoarseNet + RefineNet + PatchGAN adversarial loss.

This is the best model (multiscale, SSIM 0.9179) with GAN training added
at the refine stage to push toward sharper, more photorealistic results.

Stage 1  — train CoarseNet (identical to models/multiscale/train.py)
Stage 2  — train RefineNet + PatchGAN:
             G loss = L1 + VGG + lambda_gan * GAN_G + lambda_fm * feature_matching
             D loss = BCE(real) + BCE(fake)

Usage:
  python models/multiscale_gan/train.py --stage both
  python models/multiscale_gan/train.py --stage coarse
  python models/multiscale_gan/train.py --stage refine
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
from models.multiscale_gan.network import MultiScaleDiscriminatorWithFeatures
from shared.dataset import make_loader
from shared.losses import VGGLoss
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COARSE_H, COARSE_W = 128, 96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("multiscale_gan")
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
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)
    return ag, cl, cm, pose, person


def _down(t, h, w):
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=True)


def _up_like(t, ref):
    return F.interpolate(t, size=ref.shape[2:], mode="bilinear", align_corners=True)


def _make_lr_lambda(total_epochs: int, decay_start: int):
    def fn(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / max(1, total_epochs - decay_start))
    return fn


# ---------------------------------------------------------------------------
# Stage 1: CoarseNet (same as models/multiscale — can be shared checkpoint)
# ---------------------------------------------------------------------------

def train_coarse(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[coarse] {len(loader.dataset)} samples  batch={args.batch}")

    # Check if multiscale coarse checkpoint already exists to save time
    multiscale_coarse = ROOT / "checkpoints" / "multiscale" / "coarse_best.pth"
    coarse_net = CoarseNet(ngf=32).to(DEVICE)

    if multiscale_coarse.exists() and not args.force_coarse:
        state = torch.load(multiscale_coarse, map_location=DEVICE, weights_only=False)
        coarse_net.load_state_dict(state["model"])
        logger.info(f"[coarse] reusing existing multiscale coarse checkpoint: {multiscale_coarse}")
        # Just save a copy to our ckpt_dir
        _save(state, ckpt_dir / "coarse_best.pth")
        logger.info(f"[coarse] saved copy to {ckpt_dir / 'coarse_best.pth'}")
        return

    opt    = torch.optim.Adam(coarse_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched  = torch.optim.lr_scheduler.LambdaLR(
                 opt, _make_lr_lambda(args.epochs, args.decay_start))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best_loss  = float("inf")
    no_improve = 0
    best_path  = ckpt_dir / "coarse_best.pth"

    for epoch in range(1, args.epochs + 1):
        coarse_net.train()
        ep_loss = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person = _unpack(batch)

            ag_d     = _down(ag,     COARSE_H, COARSE_W)
            cl_d     = _down(cl,     COARSE_H, COARSE_W)
            cm_d     = _down(cm,     COARSE_H, COARSE_W)
            pose_d   = _down(pose,   COARSE_H, COARSE_W)
            person_d = _down(person, COARSE_H, COARSE_W)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                coarse, _, _ = coarse_net(ag_d, cl_d, cm_d, pose_d)
                loss = l1_fn(coarse, person_d) + vgg(coarse, person_d)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(coarse_net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()

        sched.step()
        avg = ep_loss / len(loader)
        dt  = time.time() - t0
        with torch.no_grad():
            ep_ssim = ssim_metric(coarse, person_d).item()

        logger.info(f"[coarse] E{epoch:03d}  loss={avg:.4f}  SSIM={ep_ssim:.4f}  "
                    f"best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"coarse_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": coarse_net.state_dict(),
               "opt": opt.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "coarse")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": coarse_net.state_dict(), "loss": avg}, best_path)
            logger.info(f"[coarse]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[coarse] early stop at epoch {epoch}")
                break

    logger.info(f"[coarse] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2: RefineNet + PatchGAN
# ---------------------------------------------------------------------------

def train_refine_gan(args, logger: logging.Logger) -> None:
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[refine_gan] {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen CoarseNet
    coarse_net  = CoarseNet(ngf=32).to(DEVICE)
    coarse_path = ckpt_dir / "coarse_best.pth"
    if coarse_path.exists():
        state = torch.load(coarse_path, map_location=DEVICE, weights_only=False)
        coarse_net.load_state_dict(state["model"])
        logger.info(f"[refine_gan] loaded CoarseNet from {coarse_path}")
    else:
        logger.warning("[refine_gan] CoarseNet not found — random weights")
    coarse_net.eval()
    for p in coarse_net.parameters():
        p.requires_grad_(False)

    # RefineNet (Generator) + Discriminator
    refine_net = RefineNet(in_channels=28, ngf=64).to(DEVICE)
    disc       = MultiScaleDiscriminatorWithFeatures(in_channels=9, ndf=64).to(DEVICE)

    opt_g = torch.optim.Adam(refine_net.parameters(), lr=args.lr,        betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(),       lr=args.lr * 0.5,  betas=(0.5, 0.999))
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
    best_path  = ckpt_dir / "refine_best.pth"

    if args.resume_refine and Path(args.resume_refine).exists():
        state = torch.load(args.resume_refine, map_location=DEVICE, weights_only=False)
        refine_net.load_state_dict(state["model"])
        opt_g.load_state_dict(state["opt_g"])
        logger.info(f"[refine_gan] resumed from {args.resume_refine}")

    for epoch in range(1, args.epochs + 1):
        refine_net.train()
        disc.train()
        ep_loss_g = 0.0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            ag, cl, cm, pose, person = _unpack(batch)

            with torch.no_grad():
                ag_d     = _down(ag,   COARSE_H, COARSE_W)
                cl_d     = _down(cl,   COARSE_H, COARSE_W)
                cm_d     = _down(cm,   COARSE_H, COARSE_W)
                pose_d   = _down(pose, COARSE_H, COARSE_W)
                coarse, warped_d, wm_d = coarse_net(ag_d, cl_d, cm_d, pose_d)
                coarse_up   = _up_like(coarse,   person)
                warped_full = _up_like(warped_d, person)
                wm_full     = _up_like(wm_d,     person)

            refine_inp = torch.cat([ag, warped_full, wm_full, coarse_up, pose], dim=1)  # 28ch

            # Condition for discriminator: agnostic + coarse_up (6ch)
            cond = torch.cat([ag, coarse_up], dim=1)

            # ---- Discriminator update ----
            opt_d.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                refined      = refine_net(refine_inp)
                real_in      = torch.cat([person,          cond], dim=1)  # 9ch
                fake_in      = torch.cat([refined.detach(), cond], dim=1)

                d_real, _ = disc(real_in)
                d_fake, _ = disc(fake_in)
                loss_d    = (bce(d_real, torch.ones_like(d_real)) +
                             bce(d_fake, torch.zeros_like(d_fake))) * 0.5

            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()

            # ---- Generator update ----
            opt_g.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake_in_g     = torch.cat([refined, cond], dim=1)
                d_fake_g, fake_feats = disc(fake_in_g)
                with torch.no_grad():
                    _, real_feats = disc(real_in)

                l_l1  = l1_fn(refined, person)
                l_vgg = vgg(refined, person) * 2.0   # higher VGG weight for textures
                l_gan = bce(d_fake_g, torch.ones_like(d_fake_g)) * args.lambda_gan

                # Feature matching across all 4 discriminator levels
                l_fm  = sum(
                    F.l1_loss(f_fake, f_real.detach())
                    for f_fake, f_real in zip(fake_feats, real_feats)
                ) / len(fake_feats) * args.lambda_fm

                loss_g = l_l1 + l_vgg + l_gan + l_fm

            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(refine_net.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
            ep_loss_g += loss_g.item()

        sched_g.step()
        sched_d.step()
        avg = ep_loss_g / len(loader)
        dt  = time.time() - t0
        lr  = opt_g.param_groups[0]["lr"]
        with torch.no_grad():
            ep_ssim = ssim_metric(refined, person).item()
            ep_psnr = psnr_metric(refined, person).item()

        logger.info(f"[refine_gan] E{epoch:03d}  loss={avg:.4f}  L1={l_l1.item():.4f}  "
                    f"SSIM={ep_ssim:.4f}  PSNR={ep_psnr:.2f}dB  "
                    f"lr={lr:.2e}  best={best_loss:.4f}  time={dt:.1f}s")

        ep_path = ckpt_dir / f"refine_epoch{epoch:03d}.pth"
        _save({"epoch": epoch, "model": refine_net.state_dict(),
               "opt_g": opt_g.state_dict(), "loss": avg}, ep_path)
        _keep_last_n(ckpt_dir, "refine")

        if avg < best_loss - 1e-4:
            best_loss  = avg
            no_improve = 0
            _save({"epoch": epoch, "model": refine_net.state_dict(), "loss": avg}, best_path)
            logger.info(f"[refine_gan]   => new best (loss={best_loss:.4f}  SSIM={ep_ssim:.4f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"[refine_gan] early stop at epoch {epoch}")
                break

    logger.info(f"[refine_gan] done. best={best_loss:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multiscale GAN training")
    p.add_argument("--data",          default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch",         type=int,   default=64)
    p.add_argument("--lr",            type=float, default=2e-4)
    p.add_argument("--decay-start",   type=int,   default=50, dest="decay_start")
    p.add_argument("--patience",      type=int,   default=20)
    p.add_argument("--lambda-gan",    type=float, default=1.0, dest="lambda_gan")
    p.add_argument("--lambda-fm",     type=float, default=10.0, dest="lambda_fm")
    p.add_argument("--stage",         choices=["coarse", "refine", "both"], default="both")
    p.add_argument("--force-coarse",  action="store_true", dest="force_coarse",
                   help="Retrain CoarseNet even if existing checkpoint found")
    p.add_argument("--max-samples",   type=int,   default=None, dest="max_samples")
    p.add_argument("--ckpt-dir",      default=str(ROOT / "checkpoints" / "multiscale_gan"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",       default=str(ROOT / "logs" / "multiscale_gan"),
                   dest="log_dir")
    p.add_argument("--resume-refine", default=None, dest="resume_refine")
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

    logger.info(f"Multiscale GAN  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")
    logger.info(f"  lambda_gan={args.lambda_gan}  lambda_fm={args.lambda_fm}")

    if args.stage in ("coarse", "both"):
        train_coarse(args, logger)
    if args.stage in ("refine", "both"):
        train_refine_gan(args, logger)


if __name__ == "__main__":
    main()
