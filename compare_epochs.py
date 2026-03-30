#!/usr/bin/env python3
"""
compare_epochs.py — Compare WarpNet checkpoints at epoch 5 vs epoch 25.
Evaluates L1, VGG perceptual, TV, and total loss on a fixed batch.

Usage:
  python compare_epochs.py
  python compare_epochs.py --limit 50   # use only first 50 samples
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.config import Config
from model.dataset import VITONDataset
from model.networks import WarpNet, VGGPerceptualLoss
from model.train import tv_loss

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = Config.AMP and DEVICE.type == "cuda"

CHECKPOINTS = {
    "epoch_05": Config.CHECKPOINT_DIR / "warp_epoch_005.pth",
    "epoch_25": Config.CHECKPOINT_DIR / "warp_epoch_025.pth",
}


def evaluate(ckpt_path, loader, vgg_loss_fn, l1_loss_fn, label):
    warp_net = WarpNet().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    warp_net.load_state_dict(ckpt["model"])
    warp_net.eval()

    total_l1, total_vgg, total_tv, total_loss = 0.0, 0.0, 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            cloth      = batch["cloth"].to(DEVICE)
            cloth_mask = batch["cloth_mask"].to(DEVICE)
            agnostic   = batch["agnostic"].to(DEVICE)
            pose_map   = batch["pose_map"].to(DEVICE)
            gt_region  = batch["cloth_region"].to(DEVICE)
            gt_mask = (
                (batch["parse_map"] == 4) | (batch["parse_map"] == 7)
            ).float().unsqueeze(1).to(DEVICE)

            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                warped_cloth, _, flow = warp_net(cloth, cloth_mask, agnostic, pose_map)
                loss_l1  = l1_loss_fn(warped_cloth * gt_mask, gt_region)
                loss_vgg = vgg_loss_fn(warped_cloth, gt_region)
                loss_tv  = tv_loss(flow)
                loss = (
                    Config.LAMBDA_WARP_L1  * loss_l1
                    + Config.LAMBDA_WARP_VGG * loss_vgg
                    + Config.LAMBDA_WARP_TV  * loss_tv
                )

            total_l1   += loss_l1.item()
            total_vgg  += loss_vgg.item()
            total_tv   += loss_tv.item()
            total_loss += loss.item()
            n_batches  += 1

    avg = lambda x: x / n_batches
    print(f"\n{'='*50}")
    print(f"  {label}  ({n_batches} batches × batch_size={Config.BATCH_SIZE})")
    print(f"{'='*50}")
    print(f"  L1   (×{Config.LAMBDA_WARP_L1:.0f}):  raw={avg(total_l1):.4f}  weighted={Config.LAMBDA_WARP_L1*avg(total_l1):.4f}")
    print(f"  VGG  (×{Config.LAMBDA_WARP_VGG:.0f}):   raw={avg(total_vgg):.4f}  weighted={Config.LAMBDA_WARP_VGG*avg(total_vgg):.4f}")
    print(f"  TV   (×{Config.LAMBDA_WARP_TV:.0f}):   raw={avg(total_tv):.4f}  weighted={Config.LAMBDA_WARP_TV*avg(total_tv):.4f}")
    print(f"  TOTAL:      {avg(total_loss):.4f}")

    return {
        "l1": avg(total_l1), "vgg": avg(total_vgg),
        "tv": avg(total_tv), "total": avg(total_loss),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0,
                        help="Use first N dataset pairs (0 = all).")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"AMP    : {USE_AMP}")
    print(f"Split  : {args.split}")

    dataset = VITONDataset(split=args.split, limit=args.limit)
    loader  = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,          # safe on Windows
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )
    print(f"Dataset: {len(dataset)} pairs | {len(loader)} batches")

    vgg_loss_fn = VGGPerceptualLoss().to(DEVICE)
    l1_loss_fn  = nn.L1Loss()

    results = {}
    for label, ckpt_path in CHECKPOINTS.items():
        results[label] = evaluate(ckpt_path, loader, vgg_loss_fn, l1_loss_fn, label)

    # ── Difference summary ─────────────────────────────────────────────────────
    e5, e25 = results["epoch_05"], results["epoch_25"]
    print(f"\n{'='*50}")
    print("  IMPROVEMENT: epoch_05  →  epoch_25")
    print(f"{'='*50}")
    for k in ("l1", "vgg", "tv", "total"):
        diff = e5[k] - e25[k]
        pct  = (diff / e5[k] * 100) if e5[k] != 0 else 0
        direction = "↓ better" if diff > 0 else "↑ worse"
        print(f"  {k:6s}: {e5[k]:.4f} → {e25[k]:.4f}  ({direction}: {abs(diff):.4f} / {abs(pct):.1f}%)")


if __name__ == "__main__":
    main()
