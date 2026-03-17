#!/usr/bin/env python3
"""
model/train.py — VITON-HD 2-Stage Training Loop
=================================================

Stage 1 — WarpNet  (geometric matching, cloth warping)
Stage 2 — TryOnNet (synthesis with PatchGAN discriminator)

Improvements over v1:
  • Mixed precision (AMP)     — fp16 autocast + GradScaler, ~1.5× faster on RTX 4070
  • Linear LR decay           — LR decays to 0 over the second half of training
  • Single forward pass       — TryOnNet computed once, reused for D and G losses
  • TensorBoard image logging — warped cloth / fake vs real person every N epochs
  • Scheduler state saved     — resume restores exact LR position

Usage:
  python model/train.py --stage warp
  python model/train.py --stage tryon
  python model/train.py --stage warp --limit 10 --epochs 2   # smoke test
  python model/train.py --stage tryon --resume checkpoints/tryon_epoch_05.pth
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.config import Config
from model.dataset import VITONDataset
from model.networks import (
    WarpNet,
    TryOnNet,
    PatchDiscriminator,
    VGGPerceptualLoss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = Config.AMP and DEVICE.type == "cuda"


# ─────────────────────────────── HELPERS ──────────────────────────────────────

def _denorm(t: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1] for TensorBoard image logging."""
    return (t.float().clamp(-1, 1) + 1.0) / 2.0


def _linear_decay(n_epochs: int, decay_start: int):
    """LambdaLR: full LR until decay_start epoch, then linear decay to 0."""
    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        span = max(1, n_epochs - decay_start)
        return max(0.0, 1.0 - (epoch - decay_start) / span)
    return lr_lambda


def _write_summary(path: Path, reason: str, best_loss: float, best_epoch: int,
                   final_epoch: int, final_fmag: float) -> None:
    """Write a human-readable training summary file when training ends."""
    import datetime
    lines = [
        "=== WarpNet Training Summary ===",
        f"Status:          {reason}",
        f"Epochs run:      {final_epoch}",
        f"Best total loss: {best_loss:.4f}  (epoch {best_epoch})",
        f"Final fmag:      {final_fmag:.4f}",
        f"Best checkpoint: warp_best.pth",
        f"Timestamp:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Next step:  python -m model.train --stage tryon",
    ]
    path.write_text("\n".join(lines) + "\n")
    log.info("Summary written → %s", path)


# ─────────────────────────────── LOSSES ───────────────────────────────────────

def flow_loss(flow: torch.Tensor) -> torch.Tensor:
    """TV smoothness + magnitude term.

    TV:        penalises non-smooth flow (neighbour differences).
    Magnitude: 0.1 * mean |flow| prevents zero-flow collapse without
               tricks like exp or relu — simple and stable.
    """
    dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs()
    dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs()
    tv        = dx.mean() + dy.mean()
    magnitude = flow.abs().mean()
    return tv + 0.1 * magnitude


def gan_loss(pred: torch.Tensor, is_real: bool) -> torch.Tensor:
    """LSGAN: MSE against target 1 (real) or 0 (fake)."""
    target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return nn.functional.mse_loss(pred, target)


# ─────────────────────────────── WARP STAGE ───────────────────────────────────

def train_warp(args):
    """Train Stage 1: WarpNet (6-level U-Net, ngf=64)."""
    log.info("=== Stage 1: WarpNet ===")

    dataset = VITONDataset(split="train", limit=args.limit)
    loader  = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )
    log.info("Dataset: %d pairs | %d batches/epoch", len(dataset), len(loader))

    vis_batch = next(iter(loader))
    n = min(Config.VIS_SAMPLES, Config.BATCH_SIZE)

    # ── WarpNet: 6 levels, ngf=64 — full capacity for geometric matching ───────
    warp_net = WarpNet().to(DEVICE)
    n_params = sum(p.numel() for p in warp_net.parameters() if p.requires_grad)
    log.info("WarpNet params: %s", f"{n_params:,}")

    vgg_loss = VGGPerceptualLoss().to(DEVICE)
    l1_loss  = nn.L1Loss()

    optim = torch.optim.Adam(
        warp_net.parameters(), lr=Config.LR_G,
        betas=(Config.BETA1, Config.BETA2),
    )
    scheduler = LambdaLR(optim, lr_lambda=_linear_decay(args.epochs, Config.LR_DECAY_START))
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        warp_net.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer      = SummaryWriter(Config.LOG_DIR / "warp")
    best_loss   = float("inf")
    best_epoch  = start_epoch
    global_step = start_epoch * len(loader)

    # ── Early stopping state ───────────────────────────────────────────────────
    PLATEAU_PATIENCE  = 10     # stop if no improvement for this many epochs
    COLLAPSE_PATIENCE = 3      # stop if fmag < threshold for this many epochs
    FMAG_THRESHOLD    = 0.01   # flow magnitude below this = collapsed
    plateau_counter   = 0
    collapse_counter  = 0
    avg_fmag          = 0.0    # safe default if loop never runs
    stop_reason       = f"COMPLETED (all {args.epochs} epochs)"

    for epoch in range(start_epoch, args.epochs):
        warp_net.train()

        # Running averages reset each epoch
        run_l1 = run_mask = run_vgg = run_tv = run_fmag = run_total = 0.0
        n_batches = len(loader)

        pbar = tqdm(loader, desc=f"Warp E{epoch+1:03d}", unit="batch",
                    dynamic_ncols=True, leave=True)

        for batch_idx, batch in enumerate(pbar):
            cloth      = batch["cloth"].to(DEVICE)
            cloth_mask = batch["cloth_mask"].to(DEVICE)
            agnostic   = batch["agnostic"].to(DEVICE)
            pose_map   = batch["pose_map"].to(DEVICE)
            gt_region  = batch["cloth_region"].to(DEVICE)   # (B,3,H,W) zeros outside upper-body
            parse_map  = batch["parse_map"].to(DEVICE)      # (B,H,W)   int64 labels 0-17
            # Upper-body mask: labels 4=upper-clothes, 7=dress
            gt_mask = ((parse_map == 4) | (parse_map == 7)).float().unsqueeze(1)  # (B,1,H,W)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                warped_cloth, warped_mask, flow = warp_net(cloth, cloth_mask, agnostic, pose_map)

                # Texture: warped cloth pixels vs ground-truth cloth in upper-body region
                loss_l1   = l1_loss(warped_cloth * gt_mask, gt_region)
                # Shape: warped cloth MASK must match the person's upper-body silhouette.
                # This is the primary geometric signal — forces real spatial warping.
                # Zero or constant flow → warped_mask ≠ gt_mask → high loss.
                loss_mask = l1_loss(warped_mask, gt_mask)
                loss_vgg  = vgg_loss(warped_cloth, gt_region)
                loss_tv   = flow_loss(flow)
                loss = (
                    Config.LAMBDA_WARP_L1   * loss_l1
                    + Config.LAMBDA_WARP_MASK * loss_mask
                    + Config.LAMBDA_WARP_VGG  * loss_vgg
                    + Config.LAMBDA_WARP_TV   * loss_tv
                )

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(warp_net.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            # ── Running averages ──────────────────────────────────────────────
            flow_mag   = flow.abs().mean().item()
            run_l1    += loss_l1.item()
            run_mask  += loss_mask.item()
            run_vgg   += loss_vgg.item()
            run_tv    += loss_tv.item()
            run_total += loss.item()
            run_fmag  += flow_mag
            global_step += 1
            i = batch_idx + 1

            pbar.set_postfix(
                L1=f"{run_l1/i:.3f}",
                mask=f"{run_mask/i:.3f}",
                TV=f"{run_tv/i:.4f}",
                fmag=f"{run_fmag/i:.4f}",
                tot=f"{run_total/i:.3f}",
            )

            if global_step % Config.LOG_EVERY == 0:
                writer.add_scalar("warp/loss_total", loss.item(),      global_step)
                writer.add_scalar("warp/loss_l1",    loss_l1.item(),   global_step)
                writer.add_scalar("warp/loss_mask",  loss_mask.item(), global_step)
                writer.add_scalar("warp/loss_vgg",   loss_vgg.item(),  global_step)
                writer.add_scalar("warp/loss_tv",    loss_tv.item(),   global_step)
                writer.add_scalar("warp/lr",         scheduler.get_last_lr()[0], global_step)

        avg_l1   = run_l1   / n_batches
        avg_mask = run_mask / n_batches
        avg_vgg  = run_vgg  / n_batches
        avg_tv   = run_tv   / n_batches
        avg_fmag = run_fmag / n_batches
        avg_loss = run_total / n_batches

        # ── Clear epoch summary ───────────────────────────────────────────────
        log.info(
            "Epoch %d/%d  |  L1=%.4f  mask=%.4f  VGG=%.4f  TV=%.5f  fmag=%.4f  |  total=%.4f  |  lr=%.2e",
            epoch + 1, args.epochs,
            avg_l1, avg_mask, avg_vgg, avg_tv, avg_fmag, avg_loss,
            scheduler.get_last_lr()[0],
        )

        scheduler.step()

        # ── TensorBoard images ────────────────────────────────────────────────
        if (epoch + 1) % Config.LOG_IMAGES_EVERY == 0:
            warp_net.eval()
            with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                vc  = vis_batch["cloth"][:n].to(DEVICE)
                vcm = vis_batch["cloth_mask"][:n].to(DEVICE)
                vag = vis_batch["agnostic"][:n].to(DEVICE)
                vpm = vis_batch["pose_map"][:n].to(DEVICE)
                vp  = vis_batch["person"][:n].to(DEVICE)
                wc, _, _ = warp_net(vc, vcm, vag, vpm)
            writer.add_images("warp/cloth_in",     _denorm(vc),  epoch + 1)
            writer.add_images("warp/warped_cloth", _denorm(wc),  epoch + 1)
            writer.add_images("warp/person_gt",    _denorm(vp),  epoch + 1)
            warp_net.train()

        # ── Checkpoints ───────────────────────────────────────────────────────
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            _save_ckpt(
                Config.CHECKPOINT_DIR / f"warp_epoch_{epoch+1:03d}.pth",
                epoch, warp_net, optim, scheduler,
            )
        improved = avg_loss < best_loss
        if improved:
            best_loss  = avg_loss
            best_epoch = epoch + 1
            _save_ckpt(Config.CHECKPOINT_DIR / "warp_best.pth", epoch, warp_net, optim, scheduler)
            log.info("  ↑ New best: %.4f → warp_best.pth", best_loss)

        # ── Early stopping checks ──────────────────────────────────────────────
        # 1. Flow collapse: fmag too low means the network outputs ~zero flow
        if avg_fmag < FMAG_THRESHOLD:
            collapse_counter += 1
            log.warning("  !! fmag=%.4f < %.2f — collapse count %d/%d",
                        avg_fmag, FMAG_THRESHOLD, collapse_counter, COLLAPSE_PATIENCE)
        else:
            collapse_counter = 0   # reset if flow recovers

        # 2. Loss plateau: no meaningful improvement this epoch
        if improved:
            plateau_counter = 0
        else:
            plateau_counter += 1

        if collapse_counter >= COLLAPSE_PATIENCE:
            stop_reason = (f"EARLY STOP — flow collapsed "
                           f"(fmag={avg_fmag:.4f} for {COLLAPSE_PATIENCE} epochs)")
            log.error("  %s", stop_reason)
            break

        if plateau_counter >= PLATEAU_PATIENCE:
            stop_reason = (f"EARLY STOP — loss plateau "
                           f"(no improvement for {PLATEAU_PATIENCE} epochs)")
            log.warning("  %s", stop_reason)
            break

    writer.close()
    _write_summary(
        Config.CHECKPOINT_DIR / "warp_training_summary.txt",
        stop_reason, best_loss, best_epoch, epoch + 1, avg_fmag,
    )
    log.info("WarpNet done. Best loss: %.4f", best_loss)


# ─────────────────────────────── TRY-ON STAGE ─────────────────────────────────

def train_tryon(args):
    """Train Stage 2: TryOnNet + PatchDiscriminator (frozen WarpNet)."""
    log.info("=== Stage 2: TryOnNet ===")

    dataset = VITONDataset(split="train", limit=args.limit)
    loader  = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )
    n = min(Config.VIS_SAMPLES, Config.BATCH_SIZE)
    vis_batch = next(iter(loader))

    # ── Frozen WarpNet ────────────────────────────────────────────────────────
    warp_ckpt = args.warp_ckpt or str(Config.CHECKPOINT_DIR / "warp_best.pth")
    if not Path(warp_ckpt).exists():
        raise FileNotFoundError(
            f"WarpNet checkpoint not found: {warp_ckpt}\n"
            "Run 'python model/train.py --stage warp' first."
        )
    warp_net = WarpNet().to(DEVICE)
    warp_net.load_state_dict(torch.load(warp_ckpt, map_location=DEVICE)["model"])
    warp_net.eval()
    for p in warp_net.parameters():
        p.requires_grad_(False)
    log.info("Loaded frozen WarpNet from %s", warp_ckpt)

    tryon_net     = TryOnNet().to(DEVICE)
    discriminator = PatchDiscriminator(in_ch=Config.TRYON_IN_CH + 3).to(DEVICE)
    vgg_loss      = VGGPerceptualLoss().to(DEVICE)
    l1_loss       = nn.L1Loss()

    lr_lambda = _linear_decay(args.epochs, Config.LR_DECAY_START)
    optim_G   = torch.optim.Adam(tryon_net.parameters(),
                                  lr=Config.LR_G, betas=(Config.BETA1, Config.BETA2))
    optim_D   = torch.optim.Adam(discriminator.parameters(),
                                  lr=Config.LR_D, betas=(Config.BETA1, Config.BETA2))
    sched_G   = LambdaLR(optim_G, lr_lambda=lr_lambda)
    sched_D   = LambdaLR(optim_D, lr_lambda=lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=DEVICE)
        tryon_net.load_state_dict(ckpt["model"])
        discriminator.load_state_dict(ckpt["disc"])
        optim_G.load_state_dict(ckpt["optim_G"])
        optim_D.load_state_dict(ckpt["optim_D"])
        if "sched_G" in ckpt:
            sched_G.load_state_dict(ckpt["sched_G"])
            sched_D.load_state_dict(ckpt["sched_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        log.info("Resumed from %s (epoch %d)", args.resume, start_epoch)

    Config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    writer      = SummaryWriter(Config.LOG_DIR / "tryon")
    best_loss   = float("inf")
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        tryon_net.train()
        discriminator.train()
        epoch_g_loss = 0.0
        pbar = tqdm(loader, desc=f"TryOn E{epoch+1:03d}", unit="batch", dynamic_ncols=True)

        for batch in pbar:
            cloth       = batch["cloth"].to(DEVICE)
            cloth_mask  = batch["cloth_mask"].to(DEVICE)
            agnostic    = batch["agnostic"].to(DEVICE)
            pose_map    = batch["pose_map"].to(DEVICE)
            parse_oh    = batch["parse_one_hot"].to(DEVICE)
            real_person = batch["person"].to(DEVICE)

            # Warp cloth (no grad through frozen WarpNet)
            with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                warped_cloth, warped_mask, _ = warp_net(cloth, cloth_mask, agnostic, pose_map)

            condition = torch.cat(
                [agnostic, warped_cloth, warped_mask, pose_map, parse_oh], dim=1
            )   # (B, 43, H, W)

            # ── TryOnNet forward: ONE pass, shared by D-step and G-step ───────
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                fake_person = tryon_net(agnostic, warped_cloth, warped_mask, pose_map, parse_oh)

            # ── Discriminator update ──────────────────────────────────────────
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred_real   = discriminator(condition.detach(), real_person)
                pred_fake_D = discriminator(condition.detach(), fake_person.detach())
                loss_D = (gan_loss(pred_real, True) + gan_loss(pred_fake_D, False)) * 0.5

            optim_D.zero_grad()
            scaler.scale(loss_D).backward()
            scaler.step(optim_D)

            # ── Generator update (fake_person graph still intact) ─────────────
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                pred_fake_G = discriminator(condition, fake_person)
                loss_G_gan  = gan_loss(pred_fake_G, is_real=True)
                loss_G_l1   = l1_loss(fake_person.float(), real_person.float())
                loss_G_vgg  = vgg_loss(fake_person, real_person)
                loss_G = (
                    Config.LAMBDA_GAN * loss_G_gan
                    + Config.LAMBDA_L1  * loss_G_l1
                    + Config.LAMBDA_VGG * loss_G_vgg
                )

            optim_G.zero_grad()
            scaler.scale(loss_G).backward()
            scaler.unscale_(optim_G)
            nn.utils.clip_grad_norm_(tryon_net.parameters(), max_norm=1.0)
            scaler.step(optim_G)

            # Update scale once after both backward passes
            scaler.update()

            epoch_g_loss += loss_G.item()
            global_step  += 1

            if global_step % Config.LOG_EVERY == 0:
                writer.add_scalar("tryon/loss_G",     loss_G.item(),     global_step)
                writer.add_scalar("tryon/loss_D",     loss_D.item(),     global_step)
                writer.add_scalar("tryon/loss_G_l1",  loss_G_l1.item(),  global_step)
                writer.add_scalar("tryon/loss_G_vgg", loss_G_vgg.item(), global_step)
                writer.add_scalar("tryon/loss_G_gan", loss_G_gan.item(), global_step)
                writer.add_scalar("tryon/lr_G", sched_G.get_last_lr()[0], global_step)

            pbar.set_postfix(
                G=f"{loss_G.item():.4f}",
                D=f"{loss_D.item():.4f}",
                lr=f"{sched_G.get_last_lr()[0]:.2e}",
            )

        avg_loss = epoch_g_loss / len(loader)
        log.info("Epoch %d/%d | avg G=%.4f | lr=%.2e",
                 epoch + 1, args.epochs, avg_loss, sched_G.get_last_lr()[0])

        sched_G.step()
        sched_D.step()

        # ── TensorBoard images ────────────────────────────────────────────────
        if (epoch + 1) % Config.LOG_IMAGES_EVERY == 0:
            tryon_net.eval()
            with torch.no_grad(), torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                vc   = vis_batch["cloth"][:n].to(DEVICE)
                vcm  = vis_batch["cloth_mask"][:n].to(DEVICE)
                vag  = vis_batch["agnostic"][:n].to(DEVICE)
                vpm  = vis_batch["pose_map"][:n].to(DEVICE)
                vpo  = vis_batch["parse_one_hot"][:n].to(DEVICE)
                vr   = vis_batch["person"][:n].to(DEVICE)
                vwc, vwm, _ = warp_net(vc, vcm, vag, vpm)
                vf   = tryon_net(vag, vwc, vwm, vpm, vpo)
            writer.add_images("tryon/warped_cloth", _denorm(vwc), epoch + 1)
            writer.add_images("tryon/fake_person",  _denorm(vf),  epoch + 1)
            writer.add_images("tryon/real_person",  _denorm(vr),  epoch + 1)
            tryon_net.train()

        # ── Checkpoints ───────────────────────────────────────────────────────
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            _save_tryon_ckpt(
                Config.CHECKPOINT_DIR / f"tryon_epoch_{epoch+1:03d}.pth",
                epoch, tryon_net, discriminator, optim_G, optim_D, sched_G, sched_D,
            )
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_tryon_ckpt(
                Config.CHECKPOINT_DIR / "tryon_best.pth",
                epoch, tryon_net, discriminator, optim_G, optim_D, sched_G, sched_D,
            )
            log.info("  ↑ New best: %.4f → tryon_best.pth", best_loss)

    writer.close()
    log.info("TryOnNet done. Best G-loss: %.4f", best_loss)


# ─────────────────────────────── CHECKPOINT UTILS ─────────────────────────────

def _save_ckpt(path: Path, epoch: int, model: nn.Module, optim, scheduler=None) -> None:
    state = {"epoch": epoch, "model": model.state_dict(), "optim": optim.state_dict()}
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)
    log.info("Saved: %s", path)


def _save_tryon_ckpt(
    path: Path, epoch: int,
    model: nn.Module, disc: nn.Module,
    optim_G, optim_D, sched_G=None, sched_D=None,
) -> None:
    state = {
        "epoch":   epoch,
        "model":   model.state_dict(),
        "disc":    disc.state_dict(),
        "optim_G": optim_G.state_dict(),
        "optim_D": optim_D.state_dict(),
    }
    if sched_G is not None:
        state["sched_G"] = sched_G.state_dict()
        state["sched_D"] = sched_D.state_dict()
    torch.save(state, path)
    log.info("Saved: %s", path)


# ─────────────────────────────── ENTRY POINT ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VITON-HD 2-Stage Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage",    required=True, choices=["warp", "tryon"])
    parser.add_argument("--epochs",   type=int, default=Config.N_EPOCHS)
    parser.add_argument("--warp-ckpt", default=None,
                        help="WarpNet checkpoint path (for --stage tryon).")
    parser.add_argument("--resume",   default=None,
                        help="Checkpoint to resume from.")
    parser.add_argument("--limit",    type=int, default=0,
                        help="Use first N pairs only (0 = all).")
    args = parser.parse_args()

    log.info("Device  : %s", DEVICE)
    log.info("AMP     : %s", USE_AMP)
    log.info("Stage   : %s", args.stage)
    log.info("Epochs  : %d", args.epochs)
    log.info("LR decay starts at epoch %d", Config.LR_DECAY_START)

    if args.stage == "warp":
        train_warp(args)
    else:
        train_tryon(args)


if __name__ == "__main__":
    main()
