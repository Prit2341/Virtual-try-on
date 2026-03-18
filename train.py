#!/usr/bin/env python3
"""
VITON-HD 2-Stage Training
=========================
Stage 1 (warp):  WarpNet learns cloth deformation flow
Stage 2 (tryon): TryOnNet synthesizes final try-on image

Optimized for RTX 4070 (12 GB VRAM):
  batch=4 x 2 accum, AMP, channels-last, gradient checkpointing

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
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from pathlib import Path
from tqdm import tqdm

from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.discriminator import PatchDiscriminator
from model.warp_utils import warp_cloth

# ── Defaults (RTX 4070 12 GB) ──────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = "dataset/train/tensors"
CKPT_DIR    = "checkpoints"
LOG_DIR     = "logs"

BATCH_SIZE  = 4
ACCUM_STEPS = 2          # effective batch = 8 (RTX 4070 12 GB, less peak VRAM)
LR          = 2e-4
BETAS       = (0.5, 0.999)
NUM_WORKERS = 0           # Windows — no fork
DECAY_START = 50          # epoch where LR decay begins

LAMBDA_L1   = 10.0
LAMBDA_VGG  = 5.0
LAMBDA_TV   = 1.0
LAMBDA_GAN  = 1.0         # adversarial loss weight (Stage 2 only)
PATIENCE    = 7           # early stopping: stop after N epochs with no improvement
MIN_DELTA   = 1e-4        # minimum loss decrease to count as improvement

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


def save_ckpt_gan(path, gen, disc, opt_G, opt_D, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": gen.state_dict(),
                "disc": disc.state_dict(),
                "optimizer": opt_G.state_dict(),
                "optimizer_D": opt_D.state_dict(),
                "epoch": epoch}, path)


def log_images(writer, tag, images, step, n=4):
    writer.add_images(tag, ((images[:n] + 1) / 2).clamp(0, 1), step)


# ── Stage 1: WarpNet ───────────────────────────────────────────────────────────

def train_warp(args):
    dataset = VITONDataset(args.data)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    model = WarpNet().to(DEVICE).to(memory_format=torch.channels_last)
    model.enable_gradient_checkpointing()
    vgg   = VGGLoss().to(DEVICE).to(memory_format=torch.channels_last).eval()

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
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        pbar = tqdm(loader, desc=f"[warp] E{epoch:02d}", dynamic_ncols=True)
        epoch_loss = 0.0
        n_batches  = 0

        for i, batch in enumerate(pbar):
            ag   = batch["agnostic"].to(DEVICE, memory_format=torch.channels_last)
            cl   = batch["cloth"].to(DEVICE, memory_format=torch.channels_last)
            cm   = batch["cloth_mask"].unsqueeze(1).to(DEVICE, memory_format=torch.channels_last)
            pose = batch["pose_map"].to(DEVICE, memory_format=torch.channels_last)
            per  = batch["person"].to(DEVICE, memory_format=torch.channels_last)
            pm   = batch["parse_map"].to(DEVICE)

            inp = torch.cat([ag, pose, cl, cm], 1)                # 25ch

            with autocast(enabled=args.amp):
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

            del inp, flow, warped, pcm  # free VRAM between batches

            step += 1
            batch_loss = loss.item() * args.accum
            epoch_loss += batch_loss
            n_batches  += 1
            pbar.set_postfix(L1=f"{l1.item():.4f}", TV=f"{tv.item():.4f}",
                             VGG=f"{vg.item():.3f}")

            if step % 50 == 0:
                writer.add_scalar("warp/loss_total", batch_loss, step)
                writer.add_scalar("warp/l1", l1.item(), step)
                writer.add_scalar("warp/tv", tv.item(), step)
                writer.add_scalar("warp/vgg", vg.item(), step)
                writer.add_scalar("warp/lr", sched.get_last_lr()[0], step)

        # ── Epoch stats ───────────────────────────────────────────────────────
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        avg_loss = epoch_loss / max(n_batches, 1)
        writer.add_scalar("warp/epoch_loss", avg_loss, epoch)
        print(f"  Epoch {epoch:02d} avg loss: {avg_loss:.4f}")

        # Checkpoint every 5 epochs + final
        if epoch % 5 == 0 or epoch == args.epochs:
            save_ckpt(f"{CKPT_DIR}/warp_epoch_{epoch:03d}.pth", model, opt, epoch)
            with torch.no_grad():
                log_images(writer, "warp/cloth",  cl, step)
                log_images(writer, "warp/warped", warped, step)
                log_images(writer, "warp/target", per * pcm, step)
            print(f"  -> Checkpoint saved: warp_epoch_{epoch:03d}.pth")

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            save_ckpt(f"{CKPT_DIR}/warp_best.pth", model, opt, epoch)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.patience} epochs")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best loss: {best_loss:.4f})")
            save_ckpt(f"{CKPT_DIR}/warp_epoch_{epoch:03d}.pth", model, opt, epoch)
            break

    writer.close()
    print(f"\nWarpNet training complete! Best loss: {best_loss:.4f}")


# ── Stage 2: TryOnNet + PatchGAN ──────────────────────────────────────────────

def train_tryon(args):
    dataset = VITONDataset(args.data)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                         num_workers=args.workers, pin_memory=True, drop_last=True)

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    # Frozen WarpNet
    warp = WarpNet().to(DEVICE).to(memory_format=torch.channels_last).eval()
    if args.warp_ckpt:
        warp.load_state_dict(
            torch.load(args.warp_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"Loaded WarpNet: {args.warp_ckpt}")
    else:
        print("WARNING: No --warp-ckpt provided — WarpNet is random!")
    for p in warp.parameters():
        p.requires_grad = False

    # Generator + Discriminator
    gen  = TryOnNet().to(DEVICE).to(memory_format=torch.channels_last)
    gen.enable_gradient_checkpointing()
    disc = PatchDiscriminator().to(DEVICE).to(memory_format=torch.channels_last)
    vgg  = VGGLoss().to(DEVICE).to(memory_format=torch.channels_last).eval()

    opt_G = torch.optim.Adam(gen.parameters(),  lr=args.lr, betas=BETAS)
    opt_D = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=BETAS)

    steps_per_epoch = max(len(loader) // args.accum, 1)
    sched_G = make_scheduler(opt_G, args.epochs, args.decay_start, steps_per_epoch)
    sched_D = make_scheduler(opt_D, args.epochs, args.decay_start, steps_per_epoch)
    scaler  = GradScaler(enabled=args.amp)
    writer  = SummaryWriter(os.path.join(LOG_DIR, "tryon"))

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE, weights_only=False)
        gen.load_state_dict(ckpt["model"])
        opt_G.load_state_dict(ckpt["optimizer"])
        if "disc" in ckpt:
            disc.load_state_dict(ckpt["disc"])
        if "optimizer_D" in ckpt:
            opt_D.load_state_dict(ckpt["optimizer_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']}")

    step = (start_epoch - 1) * len(loader)
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        gen.train()
        disc.train()
        pbar = tqdm(loader, desc=f"[tryon] E{epoch:02d}", dynamic_ncols=True)
        epoch_g_loss = 0.0
        n_batches    = 0

        for i, batch in enumerate(pbar):
            ag   = batch["agnostic"].to(DEVICE, memory_format=torch.channels_last)
            cl   = batch["cloth"].to(DEVICE, memory_format=torch.channels_last)
            cm   = batch["cloth_mask"].unsqueeze(1).to(DEVICE, memory_format=torch.channels_last)
            pose = batch["pose_map"].to(DEVICE, memory_format=torch.channels_last)
            per  = batch["person"].to(DEVICE, memory_format=torch.channels_last)
            pm   = batch["parse_map"].to(DEVICE)

            # Warp cloth (frozen)
            with torch.no_grad():
                warp_in     = torch.cat([ag, pose, cl, cm], 1)
                flow        = warp(warp_in)
                warped      = warp_cloth(cl, flow)
                warped_mask = warp_cloth(cm, flow)
                parse_oh    = F.one_hot(pm.long(), 18).permute(0, 3, 1, 2).float()
                del warp_in, flow  # free intermediate warp tensors

            # Assemble 43ch input
            inp = torch.cat([ag, warped, warped_mask, pose, parse_oh], 1)
            del parse_oh  # no longer needed

            # ── Train Discriminator ───────────────────────────────────────────
            with autocast(enabled=args.amp):
                fake = gen(inp)
                d_real = disc(per)
                d_fake = disc(fake.detach())
                d_loss = 0.5 * (
                    F.mse_loss(d_real, torch.ones_like(d_real)) +
                    F.mse_loss(d_fake, torch.zeros_like(d_fake))
                )

            opt_D.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)
            scaler.update()
            sched_D.step()
            del d_real, d_fake  # free D intermediates

            # ── Train Generator ───────────────────────────────────────────────
            with autocast(enabled=args.amp):
                d_fake_for_g = disc(fake)
                g_gan = F.mse_loss(d_fake_for_g, torch.ones_like(d_fake_for_g))
                g_l1  = F.l1_loss(fake, per)
                g_vgg = vgg(fake, per)
                g_loss = LAMBDA_L1 * g_l1 + LAMBDA_VGG * g_vgg + LAMBDA_GAN * g_gan

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()
            sched_G.step()
            del fake, inp, d_fake_for_g  # free G intermediates

            step += 1
            g_total = g_loss.item()
            epoch_g_loss += g_total
            n_batches    += 1
            pbar.set_postfix(
                G=f"{g_total:.3f}", D=f"{d_loss.item():.3f}",
                L1=f"{g_l1.item():.4f}", VGG=f"{g_vgg.item():.3f}",
            )

            if step % 50 == 0:
                writer.add_scalar("tryon/g_total", g_total, step)
                writer.add_scalar("tryon/g_l1", g_l1.item(), step)
                writer.add_scalar("tryon/g_vgg", g_vgg.item(), step)
                writer.add_scalar("tryon/g_gan", g_gan.item(), step)
                writer.add_scalar("tryon/d_loss", d_loss.item(), step)
                writer.add_scalar("tryon/lr", sched_G.get_last_lr()[0], step)

        # ── Epoch stats ───────────────────────────────────────────────────────
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        avg_loss = epoch_g_loss / max(n_batches, 1)
        writer.add_scalar("tryon/epoch_g_loss", avg_loss, epoch)
        print(f"  Epoch {epoch:02d} avg G loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            save_ckpt_gan(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth",
                          gen, disc, opt_G, opt_D, epoch)
            with torch.no_grad():
                log_images(writer, "tryon/output",   fake, step)
                log_images(writer, "tryon/person",   per, step)
                log_images(writer, "tryon/warped",   warped, step)
                log_images(writer, "tryon/agnostic", ag, step)
            print(f"  -> Checkpoint saved: tryon_epoch_{epoch:03d}.pth")

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_loss < best_loss - MIN_DELTA:
            best_loss = avg_loss
            patience_counter = 0
            save_ckpt_gan(f"{CKPT_DIR}/tryon_best.pth",
                          gen, disc, opt_G, opt_D, epoch)
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{args.patience} epochs")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best G loss: {best_loss:.4f})")
            save_ckpt_gan(f"{CKPT_DIR}/tryon_epoch_{epoch:03d}.pth",
                          gen, disc, opt_G, opt_D, epoch)
            break

    writer.close()
    print(f"\nTryOnNet training complete! Best G loss: {best_loss:.4f}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="VITON-HD Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage",       required=True, choices=["warp", "tryon"])
    p.add_argument("--epochs",      type=int, default=200)
    p.add_argument("--patience",    type=int, default=PATIENCE,
                   help="Early stopping patience (epochs with no improvement)")
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
    print(f"  Early stop: patience={args.patience}, min_delta={MIN_DELTA}")
    print(f"{'='*50}\n")

    if args.stage == "warp":
        train_warp(args)
    else:
        train_tryon(args)


if __name__ == "__main__":
    main()
