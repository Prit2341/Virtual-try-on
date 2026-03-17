#!/usr/bin/env python3
"""
VITON-HD 2-Stage Training
=========================
Stage 1 (warp):  WarpNet learns cloth deformation flow
Stage 2 (tryon): TryOnNet synthesizes final try-on image

Optimized for GTX 1650 (4 GB VRAM):
  batch=2, AMP=on, InstanceNorm, gradient accumulation

Usage:
  python train.py --stage warp  --epochs 30
  python train.py --stage tryon --epochs 30 --warp-ckpt checkpoints/warp_epoch_030.pth
  python train.py --stage warp  --resume checkpoints/warp_epoch_010.pth
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from pathlib import Path
from tqdm import tqdm

from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth

# ── Defaults (GTX 1650) ────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = "dataset/train/tensors"
CKPT_DIR    = "checkpoints"
LOG_DIR     = "logs"

BATCH_SIZE  = 2
ACCUM_STEPS = 2          # effective batch = 4
LR          = 2e-4
BETAS       = (0.5, 0.999)
NUM_WORKERS = 0           # Windows — no fork
DECAY_START = 20          # epoch where LR decay begins

LAMBDA_L1   = 10.0
LAMBDA_VGG  = 5.0
LAMBDA_TV   = 1.0

CLOTH_LABELS = [4, 7, 17]  # upper-clothes, dress, scarf

torch.backends.cudnn.benchmark = True


# ── Dataset ─────────────────────────────────────────────────────────────────────

class VITONDataset(Dataset):
    def __init__(self, root):
        self.files = sorted(Path(root).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files in {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torch.load(self.files[idx], map_location="cpu", weights_only=False)


# ── VGG Perceptual Loss ────────────────────────────────────────────────────────

class VGGLoss(nn.Module):
    """Multi-scale L1 feature matching on VGG16 relu1_2 … relu4_3."""

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
        return ((x + 1) / 2 - self.mean) / self.std   # [-1,1] → ImageNet

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


# ── Helpers ─────────────────────────────────────────────────────────────────────

def tv_loss(flow):
    """Total variation on flow field — encourages spatial smoothness."""
    return (torch.mean(torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])) +
            torch.mean(torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])))


def person_cloth_mask(parse_map):
    """(B, H, W) int64 → (B, 1, H, W) float32 clothing region mask."""
    mask = torch.zeros_like(parse_map, dtype=torch.float32)
    for lbl in CLOTH_LABELS:
        mask += (parse_map == lbl).float()
    return mask.unsqueeze(1).clamp_(0, 1)


def make_scheduler(optimizer, epochs, decay_start, steps_per_epoch):
    """Constant LR for first decay_start epochs, then linear decay to 0."""
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


def log_images(writer, tag, images, step, n=4):
    writer.add_images(tag, ((images[:n] + 1) / 2).clamp(0, 1), step)


# ── Stage 1: WarpNet ───────────────────────────────────────────────────────────

def train_warp(args):
    dataset = VITONDataset(args.data)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    model = WarpNet().to(DEVICE)
    vgg   = VGGLoss().to(DEVICE).eval()

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    steps_per_epoch = max(len(loader) // args.accum, 1)
    sched = make_scheduler(opt, args.epochs, args.decay_start, steps_per_epoch)
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "warp"))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    step = (start_epoch - 1) * len(loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[warp] E{epoch:02d}", dynamic_ncols=True)

        for i, batch in enumerate(pbar):
            ag   = batch["agnostic"].to(DEVICE)
            cl   = batch["cloth"].to(DEVICE)
            cm   = batch["cloth_mask"].unsqueeze(1).to(DEVICE)
            pose = batch["pose_map"].to(DEVICE)
            per  = batch["person"].to(DEVICE)
            pm   = batch["parse_map"].to(DEVICE)

            inp = torch.cat([ag, pose, cl, cm], 1)                # 25ch

            with autocast("cuda", enabled=args.amp):
                flow   = model(inp)
                warped = warp_cloth(cl, flow)
                pcm    = person_cloth_mask(pm)

                l1 = F.l1_loss(warped * pcm, per * pcm)
                tv = tv_loss(flow)
                vg = vgg(warped * pcm, per * pcm)

                loss = (LAMBDA_L1 * l1 + LAMBDA_TV * tv + LAMBDA_VGG * vg) / args.accum

            scaler.scale(loss).backward()

            if (i + 1) % args.accum == 0 or (i + 1) == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            step += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", TV=f"{tv.item():.4f}",
                             VGG=f"{vg.item():.3f}")

            if step % 50 == 0:
                writer.add_scalar("warp/loss_total", loss.item() * args.accum, step)
                writer.add_scalar("warp/l1", l1.item(), step)
                writer.add_scalar("warp/tv", tv.item(), step)
                writer.add_scalar("warp/vgg", vg.item(), step)
                writer.add_scalar("warp/lr", sched.get_last_lr()[0], step)

        # Checkpoint every 5 epochs + final
        if epoch % 5 == 0 or epoch == args.epochs:
            save_ckpt(f"{CKPT_DIR}/warp_epoch_{epoch:03d}.pth", model, opt, epoch)
            with torch.no_grad():
                log_images(writer, "warp/cloth",  cl, step)
                log_images(writer, "warp/warped", warped, step)
                log_images(writer, "warp/target", per * pcm, step)
            print(f"  -> Checkpoint saved: warp_epoch_{epoch:03d}.pth")

    writer.close()
    print("\nWarpNet training complete!")


# ── Stage 2: TryOnNet ──────────────────────────────────────────────────────────

def train_tryon(args):
    dataset = VITONDataset(args.data)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    # Frozen WarpNet
    warp = WarpNet().to(DEVICE).eval()
    if args.warp_ckpt:
        warp.load_state_dict(
            torch.load(args.warp_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"Loaded WarpNet: {args.warp_ckpt}")
    else:
        print("WARNING: No --warp-ckpt provided — WarpNet is random!")
    for p in warp.parameters():
        p.requires_grad = False

    model = TryOnNet().to(DEVICE)
    vgg   = VGGLoss().to(DEVICE).eval()

    opt   = torch.optim.Adam(model.parameters(), lr=args.lr, betas=BETAS)
    steps_per_epoch = max(len(loader) // args.accum, 1)
    sched = make_scheduler(opt, args.epochs, args.decay_start, steps_per_epoch)
    scaler = GradScaler(enabled=args.amp)
    writer = SummaryWriter(os.path.join(LOG_DIR, "tryon"))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    step = (start_epoch - 1) * len(loader)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[tryon] E{epoch:02d}", dynamic_ncols=True)

        for i, batch in enumerate(pbar):
            ag   = batch["agnostic"].to(DEVICE)
            cl   = batch["cloth"].to(DEVICE)
            cm   = batch["cloth_mask"].unsqueeze(1).to(DEVICE)
            pose = batch["pose_map"].to(DEVICE)
            per  = batch["person"].to(DEVICE)

            with torch.no_grad():
                warp_in = torch.cat([ag, pose, cl, cm], 1)
                flow    = warp(warp_in)
                warped  = warp_cloth(cl, flow)

            inp = torch.cat([ag, warped, pose], 1)                # 24ch

            with autocast("cuda", enabled=args.amp):
                out = model(inp)
                l1  = F.l1_loss(out, per)
                vg  = vgg(out, per)
                loss = (LAMBDA_L1 * l1 + LAMBDA_VGG * vg) / args.accum

            scaler.scale(loss).backward()

            if (i + 1) % args.accum == 0 or (i + 1) == len(loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

            step += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", VGG=f"{vg.item():.3f}")

            if step % 50 == 0:
                writer.add_scalar("tryon/loss_total", loss.item() * args.accum, step)
                writer.add_scalar("tryon/l1", l1.item(), step)
                writer.add_scalar("tryon/vgg", vg.item(), step)
                writer.add_scalar("tryon/lr", sched.get_last_lr()[0], step)

        if epoch % 5 == 0 or epoch == args.epochs:
            save_ckpt(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth", model, opt, epoch)
            with torch.no_grad():
                log_images(writer, "tryon/output",   out, step)
                log_images(writer, "tryon/person",   per, step)
                log_images(writer, "tryon/warped",   warped, step)
                log_images(writer, "tryon/agnostic", ag, step)
            print(f"  -> Checkpoint saved: tryon_epoch_{epoch:03d}.pth")

    writer.close()
    print("\nTryOnNet training complete!")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="VITON-HD Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage",       required=True, choices=["warp", "tryon"])
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch",       type=int, default=BATCH_SIZE)
    p.add_argument("--accum",       type=int, default=ACCUM_STEPS)
    p.add_argument("--lr",          type=float, default=LR)
    p.add_argument("--decay-start", type=int, default=DECAY_START, dest="decay_start")
    p.add_argument("--workers",     type=int, default=NUM_WORKERS)
    p.add_argument("--data",        default=DATA_DIR)
    p.add_argument("--amp",         action="store_true", default=True)
    p.add_argument("--no-amp",      dest="amp", action="store_false")
    p.add_argument("--resume",      default="", help="Resume from checkpoint")
    p.add_argument("--warp-ckpt",   default="", dest="warp_ckpt",
                   help="Pretrained WarpNet checkpoint (required for tryon stage)")
    args = p.parse_args()

    print(f"\n{'='*50}")
    print(f"  VITON-HD Training — {args.stage.upper()}")
    print(f"  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM   : {mem_gb:.1f} GB")
    print(f"  Batch  : {args.batch} x {args.accum} accum = {args.batch * args.accum} effective")
    print(f"  AMP    : {args.amp}")
    print(f"  Epochs : {args.epochs} (decay from epoch {args.decay_start})")
    print(f"{'='*50}\n")

    if args.stage == "warp":
        train_warp(args)
    else:
        train_tryon(args)


if __name__ == "__main__":
    main()
