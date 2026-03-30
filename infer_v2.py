#!/usr/bin/env python3
"""
VITON V2 Inference — GMM (TPS) + Composition TryOnNet
======================================================
Usage:
  python infer_v2.py --n 10
  python infer_v2.py --n 16 --save results/v2/
"""

import argparse
import os
import random

import torch
import numpy as np
import cv2
from pathlib import Path

from model.gmm_model import GMMNet
from model.tryon_model_v2 import TryOnNetV2

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
_BASE       = Path(__file__).resolve().parent
DATA_DIR    = str(_BASE / "dataset" / "train" / "tensors")
CKPT_DIR    = str(_BASE / "checkpoints")
RESULTS_DIR = str(_BASE / "results" / "v2")


def tensor_to_rgb(t):
    return ((t.cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).clip(0, 255).astype(np.uint8)


def mask_to_gray(t):
    """(1, H, W) mask [0,1] -> (H, W, 3) grayscale RGB."""
    m = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return np.stack([m, m, m], axis=2)


def main():
    p = argparse.ArgumentParser(description="VITON V2 Inference")
    p.add_argument("--n",         type=int, default=8)
    p.add_argument("--data",      default=DATA_DIR)
    p.add_argument("--gmm-ckpt",  default="", dest="gmm_ckpt")
    p.add_argument("--tryon-ckpt", default="", dest="tryon_ckpt")
    p.add_argument("--save",      default=RESULTS_DIR)
    args = p.parse_args()

    # Auto-detect checkpoints
    if not args.gmm_ckpt:
        best = Path(CKPT_DIR) / "gmm_best.pth"
        args.gmm_ckpt = str(best) if best.exists() else ""
    if not args.tryon_ckpt:
        best = Path(CKPT_DIR) / "tryon_v2_best.pth"
        args.tryon_ckpt = str(best) if best.exists() else ""

    # Load models
    gmm = GMMNet().to(DEVICE).eval()
    tryon = TryOnNetV2().to(DEVICE).eval()

    if args.gmm_ckpt:
        gmm.load_state_dict(
            torch.load(args.gmm_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"GMM      : {args.gmm_ckpt}")
    else:
        print("WARNING: No GMM checkpoint!")

    if args.tryon_ckpt:
        tryon.load_state_dict(
            torch.load(args.tryon_ckpt, map_location=DEVICE, weights_only=False)["model"]
        )
        print(f"TryOnNet : {args.tryon_ckpt}")
    else:
        print("WARNING: No TryOnNet checkpoint!")

    files = sorted(Path(args.data).glob("*.pt"))
    chosen = random.sample(files, min(args.n, len(files)))

    os.makedirs(args.save, exist_ok=True)
    print(f"\nGenerating {len(chosen)} results -> {args.save}/\n")

    for f in chosen:
        data = torch.load(f, map_location="cpu", weights_only=False)

        ag   = data["agnostic"].unsqueeze(0).to(DEVICE)
        cl   = data["cloth"].unsqueeze(0).to(DEVICE)
        cm   = data["cloth_mask"].unsqueeze(0).unsqueeze(0).to(DEVICE)
        pose = data["pose_map"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            warped_cloth, warped_mask, _ = gmm(cl, cm, ag, pose)

            inp = torch.cat([ag, warped_cloth, warped_mask, pose], dim=1)
            output, rendered, alpha = tryon(inp, warped_cloth=warped_cloth)

        # Build strip: person | cloth | agnostic | warped | alpha | output
        person_rgb   = tensor_to_rgb(data["person"])
        cloth_rgb    = tensor_to_rgb(data["cloth"])
        agnostic_rgb = tensor_to_rgb(data["agnostic"])
        warped_rgb   = tensor_to_rgb(warped_cloth[0])
        alpha_rgb    = mask_to_gray(alpha[0])
        output_rgb   = tensor_to_rgb(output[0])

        strip = np.concatenate(
            [person_rgb, cloth_rgb, agnostic_rgb, warped_rgb, alpha_rgb, output_rgb],
            axis=1,
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
