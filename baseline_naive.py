#!/usr/bin/env python3
"""
Model 1 — Naive Copy-Paste Baseline
=====================================
No learning. Simply pastes the flat cloth image onto the agnostic
(clothing-erased) person body using the cloth mask.

This is the simplest possible "virtual try-on" and serves as the
lower-bound baseline to show why a learned model is needed.

Usage:
  python baseline_naive.py
  python baseline_naive.py --n 6 --save results/naive/
"""

import argparse
import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR    = "dataset/train/tensors"
RESULTS_DIR = "results/naive"


def to_rgb(t):
    """(3,H,W) [-1,1] tensor → uint8 (H,W,3)."""
    return ((t.cpu().float().numpy().transpose(1, 2, 0) + 1) * 127.5
            ).clip(0, 255).astype(np.uint8)


def mask_to_rgb(t):
    m = t.squeeze().cpu().float().numpy()
    m = (m * 255).clip(0, 255).astype(np.uint8)
    return np.stack([m, m, m], axis=-1)


def naive_tryon(person, cloth, cloth_mask, agnostic):
    """
    Paste cloth onto agnostic body using cloth_mask.
    Returns uint8 (H,W,3) composite image.
    """
    person_rgb   = to_rgb(person)
    cloth_rgb    = to_rgb(cloth)
    agnostic_rgb = to_rgb(agnostic)

    mask = cloth_mask.squeeze().cpu().float().numpy()  # (H, W) in [0,1]
    mask3 = np.stack([mask, mask, mask], axis=-1)       # (H, W, 3)

    # Paste cloth on agnostic where mask = 1, keep agnostic elsewhere
    output = (cloth_rgb * mask3 + agnostic_rgb * (1 - mask3)).clip(0, 255).astype(np.uint8)
    return person_rgb, cloth_rgb, agnostic_rgb, mask3, output


def save_figure(name, person_rgb, cloth_rgb, agnostic_rgb, mask3, output_rgb, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    imgs   = [person_rgb, cloth_rgb, agnostic_rgb,
              (mask3 * 255).astype(np.uint8), output_rgb]
    titles = ["Person\n(Ground Truth)", "Cloth\n(Input)",
              "Agnostic\n(Cloth Erased)", "Cloth Mask", "Naive Output\n(Copy-Paste)"]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis("off")

    fig.suptitle(f"Model 1 — Naive Baseline  |  {name}", fontsize=11,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = save_dir / f"{name}_naive.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Also save the output image alone
    cv2.imwrite(
        str(save_dir / f"{name}_output.jpg"),
        cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    return out_path


def main():
    p = argparse.ArgumentParser(description="Naive Copy-Paste Baseline")
    p.add_argument("--n",    type=int, default=6)
    p.add_argument("--data", default=DATA_DIR)
    p.add_argument("--save", default=RESULTS_DIR)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    save_dir = Path(args.save)
    save_dir.mkdir(parents=True, exist_ok=True)

    data_path  = Path(args.data)
    merged     = data_path / "_merged.pt"
    KEYS       = ["person", "cloth", "agnostic", "cloth_mask"]

    if merged.exists():
        print(f"Loading merged file: {merged}")
        md      = torch.load(merged, map_location="cpu", weights_only=False)
        n_total = md["person"].shape[0]
        indices = random.sample(range(n_total), min(args.n, n_total))
        samples = [(f"sample_{i:04d}", {k: md[k][i] for k in KEYS}) for i in indices]
    else:
        files   = sorted(data_path.glob("*.pt"))
        chosen  = random.sample(files, min(args.n, len(files)))
        samples = [(f.stem, torch.load(f, map_location="cpu", weights_only=False))
                   for f in chosen]

    print(f"\nModel 1 — Naive Baseline")
    print(f"Generating {len(samples)} results → {save_dir}/\n")

    for name, data in samples:
        person_rgb, cloth_rgb, agnostic_rgb, mask3, output_rgb = naive_tryon(
            data["person"], data["cloth"], data["cloth_mask"], data["agnostic"]
        )
        out_path = save_figure(name, person_rgb, cloth_rgb, agnostic_rgb,
                               mask3, output_rgb, save_dir)
        print(f"  ✓ {name}  →  {out_path.name}")

    print(f"\nDone! Saved to {save_dir}/")


if __name__ == "__main__":
    main()
