#!/usr/bin/env python3
"""
VITON-HD training pipeline (3 stages) adapted for 256×192.

Stage 1 — SegGenerator: predict 7-class target segmentation
Stage 2 — GMM:          TPS cloth warping via feature correlation
Stage 3 — ALIASGenerator: misalignment-aware try-on synthesis

Usage:
  python models/viton_hd/train.py --stage seg
  python models/viton_hd/train.py --stage gmm
  python models/viton_hd/train.py --stage alias
  python models/viton_hd/train.py --stage all   # runs all three
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

from models.viton_hd.network import (
    SegGenerator, GMM, ALIASGenerator,
    make_parse_agnostic_onehot, parse_7_onehot, remap_parse_18_to_7,
    N_SEG, GRID_SIZE,
)
from shared.dataset import make_loader
from shared.losses import VGGLoss
from shared.metrics import ssim_metric, psnr_metric

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logger(name: str, log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    lg.addHandler(logging.FileHandler(log_path))
    lg.addHandler(logging.StreamHandler())
    for h in lg.handlers:
        h.setFormatter(fmt)
    return lg


def _save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _keep(ckpt_dir, prefix, n=3):
    for f in sorted(ckpt_dir.glob(f"{prefix}_epoch*.pth"))[:-n]:
        f.unlink(missing_ok=True)


def _unpack(batch):
    ag   = batch["agnostic"].to(DEVICE, non_blocking=True)
    cl   = batch["cloth"].to(DEVICE, non_blocking=True)
    cm   = batch["cloth_mask"].to(DEVICE, non_blocking=True)
    pose = batch["pose_map"].to(DEVICE, non_blocking=True)
    pers = batch["person"].to(DEVICE, non_blocking=True)
    pm   = batch["parse_map"].to(DEVICE, non_blocking=True)
    if cm.dim() == 3:
        cm = cm.unsqueeze(1)
    return ag, cl, cm, pose, pers, pm


def _make_seg_input(cm, cl, pm, pose):
    """Build 41-ch SegGenerator input from batch tensors."""
    noise = torch.randn_like(cm)                                   # (B,1,H,W)
    parse_ag = make_parse_agnostic_onehot(pm)                      # (B,18,H,W)
    cl_masked = cl * cm                                            # (B,3,H,W)
    return torch.cat([cm, cl_masked, parse_ag, pose, noise], dim=1)  # 41ch


def _make_seg_target(pm):
    """7-class one-hot target for SegGenerator."""
    return parse_7_onehot(pm)                                      # (B,7,H,W)


# ---------------------------------------------------------------------------
# Stage 1 — SegGenerator
# ---------------------------------------------------------------------------

def train_seg(args, logger):
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[seg] dataset: {len(loader.dataset)} samples  batch={args.batch}")

    model  = SegGenerator(input_nc=41, output_nc=N_SEG).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    bce    = nn.BCELoss()

    best, no_imp, best_path = float("inf"), 0, ckpt_dir / "seg_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()

        for batch in loader:
            ag, cl, cm, pose, pers, pm = _unpack(batch)
            inp  = _make_seg_input(cm, cl, pm, pose)
            tgt  = _make_seg_target(pm)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                pred = model(inp)
                loss = bce(pred, tgt)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.item()

        avg = total / len(loader)
        logger.info(f"[seg] E{epoch:03d}  loss={avg:.4f}  best={best:.4f}  "
                    f"time={time.time()-t0:.1f}s")
        _save({"epoch": epoch, "model": model.state_dict(),
               "opt": opt.state_dict(), "loss": avg},
              ckpt_dir / f"seg_epoch{epoch:03d}.pth")
        _keep(ckpt_dir, "seg")

        if avg < best - 1e-4:
            best, no_imp = avg, 0
            _save({"epoch": epoch, "model": model.state_dict(), "loss": avg}, best_path)
            logger.info(f"[seg]   => new best ({best:.4f})")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                logger.info(f"[seg] early stop at epoch {epoch}")
                break

    logger.info(f"[seg] done. best={best:.4f}")


# ---------------------------------------------------------------------------
# Stage 2 — GMM (TPS warping)
# ---------------------------------------------------------------------------

def train_gmm(args, logger):
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[gmm] dataset: {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen SegGenerator for cloth-region guidance
    seg_net = SegGenerator(input_nc=41, output_nc=N_SEG).to(DEVICE)
    seg_ckpt = ckpt_dir / "seg_best.pth"
    if seg_ckpt.exists():
        state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
        seg_net.load_state_dict(state["model"])
        logger.info(f"[gmm] loaded SegGenerator from {seg_ckpt}")
    else:
        logger.warning("[gmm] seg_best.pth not found — using random SegGenerator")
    seg_net.eval()
    for p in seg_net.parameters():
        p.requires_grad_(False)

    model  = GMM(input_nc_A=22, input_nc_B=3).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best, no_imp, best_path = float("inf"), 0, ckpt_dir / "gmm_best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, t0 = 0.0, time.time()
        last_ssim = last_psnr = 0.0

        for batch in loader:
            ag, cl, cm, pose, pers, pm = _unpack(batch)

            with torch.no_grad():
                seg_inp  = _make_seg_input(cm, cl, pm, pose)
                seg_pred = seg_net(seg_inp)                 # (B,7,H,W)  sigmoid
                # cloth channel in 7-class: index 2
                seg_cloth = seg_pred[:, 2:3]               # (B,1,H,W)

            # GMM inputA: seg_cloth(1) + pose(18) + agnostic(3)
            inp_A = torch.cat([seg_cloth, pose, ag], dim=1)   # 22ch
            inp_B = cl                                         # 3ch

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                theta, grid = model(inp_A, inp_B)
                warped_cl = F.grid_sample(cl, grid, padding_mode='border',
                                          align_corners=False)
                warped_cm = F.grid_sample(cm, grid, padding_mode='zeros',
                                          align_corners=False)

                # Losses against person's clothing region
                cloth_mask_gt = (pm == 4) | (pm == 7)         # upper or dress
                cloth_mask_gt = cloth_mask_gt.unsqueeze(1).float()

                l_l1     = l1_fn(warped_cl * cloth_mask_gt,
                                 pers        * cloth_mask_gt)
                l_vgg    = vgg(warped_cl * cloth_mask_gt,
                               pers        * cloth_mask_gt)
                l_mask   = l1_fn(warped_cm, cloth_mask_gt) * 2.0
                # TPS control-point L2 regularisation (keep warp smooth)
                l_tps    = theta.pow(2).mean() * 0.5
                loss     = l_l1 + l_vgg + l_mask + l_tps

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.item()

        avg = total / len(loader)
        with torch.no_grad():
            last_ssim = ssim_metric(warped_cl * cloth_mask_gt,
                                    pers        * cloth_mask_gt).item()
            last_psnr = psnr_metric(warped_cl * cloth_mask_gt,
                                    pers        * cloth_mask_gt).item()
        logger.info(f"[gmm] E{epoch:03d}  loss={avg:.4f}  SSIM={last_ssim:.4f}  "
                    f"PSNR={last_psnr:.2f}dB  best={best:.4f}  time={time.time()-t0:.1f}s")
        _save({"epoch": epoch, "model": model.state_dict(),
               "opt": opt.state_dict(), "loss": avg},
              ckpt_dir / f"gmm_epoch{epoch:03d}.pth")
        _keep(ckpt_dir, "gmm")

        if avg < best - 1e-4:
            best, no_imp = avg, 0
            _save({"epoch": epoch, "model": model.state_dict(), "loss": avg}, best_path)
            logger.info(f"[gmm]   => new best (loss={best:.4f}  SSIM={last_ssim:.4f})")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                logger.info(f"[gmm] early stop at epoch {epoch}")
                break

    logger.info(f"[gmm] done. best={best:.4f}")


# ---------------------------------------------------------------------------
# Stage 3 — ALIASGenerator
# ---------------------------------------------------------------------------

def train_alias(args, logger):
    ckpt_dir = Path(args.ckpt_dir)
    loader   = make_loader(args.data, args.batch, max_samples=args.max_samples)
    logger.info(f"[alias] dataset: {len(loader.dataset)} samples  batch={args.batch}")

    # Load frozen SegGenerator and GMM
    seg_net = SegGenerator(input_nc=41, output_nc=N_SEG).to(DEVICE)
    seg_ckpt = ckpt_dir / "seg_best.pth"
    if seg_ckpt.exists():
        state = torch.load(seg_ckpt, map_location=DEVICE, weights_only=False)
        seg_net.load_state_dict(state["model"])
        logger.info(f"[alias] loaded SegGenerator from {seg_ckpt}")
    else:
        logger.warning("[alias] seg_best.pth not found — random SegGenerator")

    gmm_net = GMM(input_nc_A=22, input_nc_B=3).to(DEVICE)
    gmm_ckpt = ckpt_dir / "gmm_best.pth"
    if gmm_ckpt.exists():
        state = torch.load(gmm_ckpt, map_location=DEVICE, weights_only=False)
        gmm_net.load_state_dict(state["model"])
        logger.info(f"[alias] loaded GMM from {gmm_ckpt}")
    else:
        logger.warning("[alias] gmm_best.pth not found — random GMM")

    for net in [seg_net, gmm_net]:
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    alias  = ALIASGenerator(input_nc=24, ngf=64, seg_nc=N_SEG).to(DEVICE)
    opt    = torch.optim.Adam(alias.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(DEVICE == "cuda"))
    vgg    = VGGLoss().to(DEVICE)
    l1_fn  = nn.L1Loss()

    best, no_imp, best_path = float("inf"), 0, ckpt_dir / "alias_best.pth"

    for epoch in range(1, args.epochs + 1):
        alias.train()
        total, t0 = 0.0, time.time()
        last_ssim = last_psnr = 0.0

        for batch in loader:
            ag, cl, cm, pose, pers, pm = _unpack(batch)

            with torch.no_grad():
                # Segmentation prediction
                seg_inp  = _make_seg_input(cm, cl, pm, pose)
                seg_pred = seg_net(seg_inp)                    # (B,7,H,W) sigmoid
                seg_hard = (seg_pred > 0.5).float()            # binarise
                seg_cloth = seg_hard[:, 2:3]                   # upper-cloth channel

                # GMM warp
                inp_A    = torch.cat([seg_cloth, pose, ag], dim=1)
                theta, grid = gmm_net(inp_A, cl)
                warped_cl = F.grid_sample(cl, grid, padding_mode='border',
                                          align_corners=False)
                warped_cm = F.grid_sample(cm, grid, padding_mode='zeros',
                                          align_corners=False)

                # Misalignment mask: predicted cloth region not covered by warped mask
                misalign = (seg_cloth - warped_cm).clamp(min=0)

                # seg_div = seg + misalign_mask as extra channel
                seg_div = torch.cat([seg_hard, misalign], dim=1)   # (B,8,H,W)

            # ALIAS input: agnostic(3) + pose(18) + warped_cloth(3) = 24ch
            alias_inp = torch.cat([ag, pose, warped_cl], dim=1)

            opt.zero_grad()
            with autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                fake = alias(alias_inp, seg_hard, seg_div, misalign)
                l_l1 = l1_fn(fake, pers) * 1.0
                l_vgg = vgg(fake, pers)  * 2.0
                loss  = l_l1 + l_vgg

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(alias.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            total += loss.item()

        avg = total / len(loader)
        with torch.no_grad():
            last_ssim = ssim_metric(fake, pers).item()
            last_psnr = psnr_metric(fake, pers).item()
        logger.info(f"[alias] E{epoch:03d}  loss={avg:.4f}  SSIM={last_ssim:.4f}  "
                    f"PSNR={last_psnr:.2f}dB  best={best:.4f}  time={time.time()-t0:.1f}s")
        _save({"epoch": epoch, "model": alias.state_dict(),
               "opt": opt.state_dict(), "loss": avg},
              ckpt_dir / f"alias_epoch{epoch:03d}.pth")
        _keep(ckpt_dir, "alias")

        if avg < best - 1e-4:
            best, no_imp = avg, 0
            _save({"epoch": epoch, "model": alias.state_dict(), "loss": avg}, best_path)
            logger.info(f"[alias]   => new best (loss={best:.4f}  SSIM={last_ssim:.4f})")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                logger.info(f"[alias] early stop at epoch {epoch}")
                break

    logger.info(f"[alias] done. best={best:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VITON-HD training")
    p.add_argument("--data",        default=str(ROOT / "dataset" / "train" / "tensors"))
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch",       type=int, default=32)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--patience",    type=int, default=20)
    p.add_argument("--stage",       choices=["seg", "gmm", "alias", "all"], default="all")
    p.add_argument("--max-samples", type=int, default=None, dest="max_samples")
    p.add_argument("--ckpt-dir",    default=str(ROOT / "checkpoints" / "viton_hd"),
                   dest="ckpt_dir")
    p.add_argument("--log-dir",     default=str(ROOT / "logs" / "viton_hd"),
                   dest="log_dir")
    return p.parse_args()


def main():
    args = parse_args()
    if DEVICE == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_dir) / f"train_{ts}.txt"
    logger   = _logger("viton_hd", log_path)

    logger.info(f"VITON-HD training  stage={args.stage}  device={DEVICE}")
    logger.info(f"  data={args.data}  epochs={args.epochs}  batch={args.batch}  lr={args.lr}")

    if args.stage in ("seg",   "all"):  train_seg(args, logger)
    if args.stage in ("gmm",   "all"):  train_gmm(args, logger)
    if args.stage in ("alias", "all"):  train_alias(args, logger)


if __name__ == "__main__":
    main()
