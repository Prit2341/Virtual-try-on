#!/usr/bin/env python3
"""
Retrain all 7 virtual try-on models sequentially with all bug fixes applied.

Usage:
    python retrain_all.py                    # train all 7 models
    python retrain_all.py --models baseline v2   # specific models only
    python retrain_all.py --epochs 50        # override epoch count
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    {
        "name": "baseline",
        "label": "Baseline (WarpNet + TryOnNet U-Net)",
        "cmd": [PYTHON, str(ROOT / "train.py"), "--stage", "both", "--batch", "128"],
        "ckpt": ROOT / "checkpoints" / "tryon_best.pth",
    },
    {
        "name": "v2",
        "label": "V2 (GMM TPS warp + Composition TryOnNet) — FIXED cloth labels",
        "cmd": [PYTHON, str(ROOT / "train_v2.py"), "--stage", "both", "--batch", "64"],
        "ckpt": ROOT / "checkpoints" / "v2" / "tryon_best.pth",
    },
    {
        "name": "resnet_gen",
        "label": "ResNet Generator (9 ResBlocks) — FIXED warp loss",
        "cmd": [PYTHON, str(ROOT / "models" / "resnet_gen" / "train.py"), "--stage", "both", "--batch", "64"],
        "ckpt": ROOT / "checkpoints" / "resnet_gen" / "resnet_gen_best.pth",
    },
    {
        "name": "attention_unet",
        "label": "Attention U-Net (self-attention) — FIXED warp loss",
        "cmd": [PYTHON, str(ROOT / "models" / "attention_unet" / "train.py"), "--stage", "both", "--batch", "128"],
        "ckpt": ROOT / "checkpoints" / "attention_unet" / "tryon_best.pth",
    },
    {
        "name": "single_stage",
        "label": "Single Stage (5-level U-Net, no explicit warping)",
        "cmd": [PYTHON, str(ROOT / "models" / "single_stage" / "train.py"), "--batch", "192"],
        "ckpt": ROOT / "checkpoints" / "single_stage" / "model_best.pth",
    },
    {
        "name": "spade",
        "label": "SPADE (spatially adaptive norm) — FIXED warp loss",
        "cmd": [PYTHON, str(ROOT / "models" / "spade" / "train.py"), "--stage", "both", "--batch", "80"],
        "ckpt": ROOT / "checkpoints" / "spade" / "tryon_best.pth",
    },
    {
        "name": "multiscale",
        "label": "Multi-Scale (CoarseNet 128px + RefineNet 256px)",
        "cmd": [PYTHON, str(ROOT / "models" / "multiscale" / "train.py"), "--stage", "both", "--batch", "192"],
        "ckpt": ROOT / "checkpoints" / "multiscale" / "refine_best.pth",
    },
]


def fmt(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def main():
    import argparse
    p = argparse.ArgumentParser(description="Retrain all 7 models")
    p.add_argument("--models", nargs="+", default=None,
                   help="Specific models to train (default: all)")
    p.add_argument("--epochs", type=int, default=100,
                   help="Epoch count for all models")
    args = p.parse_args()

    # Filter models if specified
    if args.models:
        models = [m for m in MODELS if m["name"] in args.models]
    else:
        models = MODELS

    print("=" * 70)
    print(f"  VIRTUAL TRY-ON — FULL RETRAIN ({len(models)} models)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Epochs : {args.epochs}")
    print("=" * 70)

    for m in models:
        print(f"  {m['name']:<18} {m['label']}")
    print()

    total_start = time.perf_counter()
    results = {}

    for i, m in enumerate(models, 1):
        name = m["name"]
        cmd = m["cmd"] + ["--epochs", str(args.epochs)]

        print()
        print("=" * 70)
        print(f"  [{i}/{len(models)}] {m['label']}")
        print(f"  CMD: {' '.join(str(c) for c in cmd)}")
        print("=" * 70)

        t0 = time.perf_counter()
        try:
            result = subprocess.run(cmd, cwd=str(ROOT))
            elapsed = time.perf_counter() - t0

            if result.returncode == 0:
                print(f"\n  {name} DONE in {fmt(elapsed)}")
                results[name] = ("OK", elapsed)
            else:
                print(f"\n  {name} FAILED (exit code {result.returncode}) after {fmt(elapsed)}")
                results[name] = ("FAILED", elapsed)

        except KeyboardInterrupt:
            elapsed = time.perf_counter() - t0
            results[name] = ("INTERRUPTED", elapsed)
            print(f"\n  {name} INTERRUPTED after {fmt(elapsed)}")
            print("\n  Stopping. Completed models are still saved.")
            break

    # ── Summary ──────────────────────────────────────────────────────────
    total = time.perf_counter() - total_start
    print()
    print("=" * 70)
    print(f"  SUMMARY  (total: {fmt(total)})")
    print("=" * 70)
    print(f"  {'Model':<18} {'Status':<12} {'Time':>10}  Label")
    print(f"  {'-'*18} {'-'*12} {'-'*10}  {'-'*30}")

    for m in models:
        name = m["name"]
        if name in results:
            status, elapsed = results[name]
            print(f"  {name:<18} {status:<12} {fmt(elapsed):>10}  {m['label']}")
        else:
            print(f"  {name:<18} {'NOT RUN':<12} {'':>10}  {m['label']}")

    print()
    print("  Next steps:")
    print("    python compare_all.py --n 16")
    print()


if __name__ == "__main__":
    main()
