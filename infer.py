#!/usr/bin/env python3
"""
VITON-HD Inference — Generate Try-On Results
=============================================
Loads trained WarpNet + TryOnNet, runs on test samples,
and saves side-by-side comparison strips.

Usage:
  python infer.py --n 8
  python infer.py --n 16 --save results/
  python infer.py --warp-ckpt checkpoints/warp_epoch_030.pth --tryon-ckpt checkpoints/tryon_epoch_030.pth
"""

import argparse
import os
import random

import torch
import numpy as np
import cv2
from pathlib import Path

from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = "dataset/train/tensors"
CKPT_DIR    = "checkpoints"
RESULTS_DIR = "results"


def tensor_to_rgb(t):
    """(3, H, W) tensor [-1, 1] → uint8 RGB (H, W, 3)."""
    return ((t.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description="VITON-HD Inference")
    p.add_argument("--n",          type=int, default=8, help="Number of samples")
    p.add_argument("--data",       default=DATA_DIR)
    p.add_argument("--warp-ckpt",  default="", dest="warp_ckpt")
    p.add_argument("--tryon-ckpt", default="", dest="tryon_ckpt")
    p.add_argument("--save",       default=RESULTS_DIR)
    args = p.parse_args()

    # Auto-detect latest checkpoints if not specified
    if not args.warp_ckpt:
        ckpts = sorted(Path(CKPT_DIR).glob("warp_*.pth"))
        args.warp_ckpt = str(ckpts[-1]) if ckpts else ""
    if not args.tryon_ckpt:
        ckpts = sorted(Path(CKPT_DIR).glob("tryon_*.pth"))
        args.tryon_ckpt = str(ckpts[-1]) if ckpts else ""

    # Load models
    warp_net  = WarpNet().to(DEVICE).eval()
    tryon_net = TryOnNet().to(DEVICE).eval()

    if args.warp_ckpt:
        warp_net.load_state_dict(
            torch.load(args.warp_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"WarpNet  : {args.warp_ckpt}")
    else:
        print("WARNING: No WarpNet checkpoint found!")

    if args.tryon_ckpt:
        tryon_net.load_state_dict(
            torch.load(args.tryon_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"TryOnNet : {args.tryon_ckpt}")
    else:
        print("WARNING: No TryOnNet checkpoint found!")

    # Pick random samples
    files = sorted(Path(args.data).glob("*.pt"))
    chosen = random.sample(files, min(args.n, len(files)))

    os.makedirs(args.save, exist_ok=True)
    print(f"\nGenerating {len(chosen)} results → {args.save}/\n")

    for f in chosen:
        data = torch.load(f, map_location="cpu", weights_only=False)

        ag   = data["agnostic"].unsqueeze(0).to(DEVICE)
        cl   = data["cloth"].unsqueeze(0).to(DEVICE)
        cm   = data["cloth_mask"].unsqueeze(0).unsqueeze(0).to(DEVICE)
        pose = data["pose_map"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            warp_in = torch.cat([ag, pose, cl, cm], 1)
            flow    = warp_net(warp_in)
            warped  = warp_cloth(cl, flow)

            tryon_in = torch.cat([ag, warped, pose], 1)
            output   = tryon_net(tryon_in)

        # Build comparison strip: person | cloth | agnostic | warped | output
        person_rgb   = tensor_to_rgb(data["person"])
        cloth_rgb    = tensor_to_rgb(data["cloth"])
        agnostic_rgb = tensor_to_rgb(data["agnostic"])
        warped_rgb   = tensor_to_rgb(warped[0])
        output_rgb   = tensor_to_rgb(output[0])

        # Add labels
        strip = np.concatenate(
            [person_rgb, cloth_rgb, agnostic_rgb, warped_rgb, output_rgb], axis=1
        )

        out_path = Path(args.save) / f"{f.stem}.jpg"
        cv2.imwrite(
            str(out_path),
            cv2.cvtColor(strip, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        print(f"  {f.stem}")

    print(f"\nDone! {len(chosen)} results saved to {args.save}/")


if __name__ == "__main__":
    main()
