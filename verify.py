#!/usr/bin/env python3
"""
Preprocessing Verification — Visual Spot-Check
================================================
After running preprocess.py, use this to inspect a random sample.

Usage:
  python verify.py                          # random sample from train
  python verify.py --split test --n 5      # 5 random samples from test
  python verify.py --name 00000_00         # specific person stem
"""

import argparse
import random
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

DATASET = Path("d:/Virtual_try_on/dataset")

# SegFormer cloth label names (index → name)
LABEL_NAMES = {
    0: "background", 1: "hat",       2: "hair",      3: "sunglasses",
    4: "upper",      5: "skirt",     6: "pants",     7: "dress",
    8: "belt",       9: "left-shoe", 10: "right-shoe", 11: "face",
    12: "left-leg",  13: "right-leg", 14: "left-arm",  15: "right-arm",
    16: "bag",       17: "scarf",
}

# Color palette for parsing visualisation (one color per label)
PALETTE = np.array([
    [0,   0,   0  ],  # 0  background — black
    [255, 200, 100],  # 1  hat
    [139, 69,  19 ],  # 2  hair — brown
    [128, 128, 128],  # 3  sunglasses
    [255, 80,  80 ],  # 4  upper-clothes — red
    [80,  80,  255],  # 5  skirt
    [0,   128, 0  ],  # 6  pants — green
    [255, 140, 0  ],  # 7  dress — orange
    [160, 82,  45 ],  # 8  belt
    [200, 200, 50 ],  # 9  left-shoe
    [200, 200, 80 ],  # 10 right-shoe
    [255, 224, 189],  # 11 face — skin
    [150, 255, 150],  # 12 left-leg
    [100, 220, 100],  # 13 right-leg
    [100, 180, 255],  # 14 left-arm — blue
    [50,  130, 255],  # 15 right-arm
    [180, 50,  180],  # 16 bag — purple
    [255, 255, 0  ],  # 17 scarf — yellow
], dtype=np.uint8)


def colorise_parse(parse_map: np.ndarray) -> np.ndarray:
    """uint8 (H, W) label map → RGB (H, W, 3)."""
    h, w = parse_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lbl in range(18):
        rgb[parse_map == lbl] = PALETTE[lbl]
    return rgb


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) tensor in [-1, 1] → uint8 RGB (H, W, 3)."""
    arr = ((t.numpy().transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


def visualise_heatmap(pose_map: torch.Tensor) -> np.ndarray:
    """(18, H, W) → max-projection grayscale RGB (H, W, 3)."""
    peak = pose_map.numpy().max(axis=0)        # (H, W)
    peak = (peak / peak.max() * 255).clip(0, 255).astype(np.uint8) if peak.max() > 0 else peak.astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(peak, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def verify_sample(split: str, stem_p: str, stem_c: str):
    base = DATASET / split

    # ── Load all outputs ───────────────────────────────────────────────────────
    tensor_path = base / "tensors" / f"{stem_p}__{stem_c}.pt"
    if not tensor_path.exists():
        print(f"[ERROR] Tensor not found: {tensor_path}")
        print("  → Run preprocess.py first.")
        sys.exit(1)

    bundle = torch.load(tensor_path, map_location="cpu")

    person_rgb   = tensor_to_uint8(bundle["person"])
    cloth_rgb    = tensor_to_uint8(bundle["cloth"])
    agnostic_rgb = tensor_to_uint8(bundle["agnostic"])
    parse_rgb    = colorise_parse(bundle["parse_map"].numpy().astype(np.uint8))
    cloth_mask   = (bundle["cloth_mask"].numpy() * 255).astype(np.uint8)
    pose_vis     = visualise_heatmap(bundle["pose_map"])

    # ── Checks ────────────────────────────────────────────────────────────────
    issues = []
    pm = bundle["parse_map"].numpy()

    if (pm == 4).sum() + (pm == 7).sum() > 0:
        issues.append("⚠  Upper-clothes label still present in parse map!")
    if bundle["cloth_mask"].sum() == 0:
        issues.append("⚠  Cloth mask is all zeros — rembg may have failed.")
    if bundle["pose_map"].sum() == 0:
        issues.append("⚠  Pose heatmap is all zeros — pose estimation failed.")
    if bundle["agnostic"].min() == bundle["agnostic"].max():
        issues.append("⚠  Agnostic image is constant — something is wrong.")

    print(f"\n{'='*60}")
    print(f"  Person : {stem_p}  |  Cloth : {stem_c}  |  Split : {split}")
    print(f"{'='*60}")
    print(f"  Tensor shapes:")
    for k, v in bundle.items():
        print(f"    {k:12s}: {tuple(v.shape)}  dtype={v.dtype}")
    print(f"\n  Unique parse labels: {sorted(np.unique(pm).tolist())}")
    label_coverage = {LABEL_NAMES[l]: int((pm == l).sum()) for l in np.unique(pm)}
    for name, cnt in sorted(label_coverage.items(), key=lambda x: -x[1]):
        pct = cnt / (pm.size) * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:14s}: {bar}  {pct:.1f}%")

    if issues:
        print("\n  ISSUES DETECTED:")
        for iss in issues:
            print(f"    {iss}")
    else:
        print("\n  ✓ All checks passed.")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"[{split}]  person={stem_p}  cloth={stem_c}",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.05)

    panels = [
        (person_rgb,   "Person\n(input)"),
        (cloth_rgb,    "Cloth\n(input)"),
        (parse_rgb,    "Parsing\n(Step 2)"),
        (pose_vis,     "Pose Heatmap\n(Step 3)"),
        (np.stack([cloth_mask]*3, axis=-1), "Cloth Mask\n(Step 4)"),
        (agnostic_rgb, "Agnostic\n(Step 5)"),
    ]

    for i, (img, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def pick_random(split: str, n: int):
    tensor_dir = DATASET / split / "tensors"
    if not tensor_dir.exists():
        print(f"[ERROR] No tensors found at {tensor_dir}")
        print("  → Run preprocess.py first.")
        sys.exit(1)

    files = list(tensor_dir.glob("*.pt"))
    if not files:
        print("[ERROR] Tensor directory is empty.")
        sys.exit(1)

    chosen = random.sample(files, min(n, len(files)))
    for f in chosen:
        parts = f.stem.split("__")
        if len(parts) == 2:
            verify_sample(split, parts[0], parts[1])
        else:
            print(f"[SKIP] Unexpected filename: {f.name}")


def main():
    parser = argparse.ArgumentParser(description="VITON-HD Preprocessing Verifier")
    parser.add_argument("--split",  default="train", choices=["train", "test"])
    parser.add_argument("--n",      type=int, default=1, help="Number of random samples.")
    parser.add_argument("--name",   default="", help="Specific person stem (e.g. 00000_00).")
    args = parser.parse_args()

    if args.name:
        # Find a matching pair
        pairs_file = DATASET / f"{args.split}_pairs.txt"
        with open(pairs_file) as f:
            for line in f:
                p, c = line.strip().split()
                if Path(p).stem == args.name:
                    verify_sample(args.split, Path(p).stem, Path(c).stem)
                    return
        print(f"[ERROR] No pair found for person '{args.name}' in {args.split}_pairs.txt")
    else:
        pick_random(args.split, args.n)


if __name__ == "__main__":
    main()
