#!/usr/bin/env python3
"""
Run all virtual try-on models sequentially.

Trains each model one after the other using optimised batch sizes.
Skips models that already have a trained best checkpoint unless --force is used.

Usage:
    python run_all.py                          # train all missing models
    python run_all.py --models baseline v2     # specific models only
    python run_all.py --force                  # retrain everything
    python run_all.py --dry-run                # show what would run, don't execute
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
# Each entry:
#   script   : path to train.py relative to ROOT
#   ckpt     : checkpoint path that signals "already trained"
#   args     : extra CLI args (stage, etc.)
#   label    : display name

MODELS = {
    "baseline": {
        "script": "train.py",
        "ckpt":   "checkpoints/tryon_best.pth",
        "args":   ["--stage", "both"],
        "label":  "Baseline (CNN U-Net, BatchNorm+MaxPool+Bilinear, L1)",
    },
    "v2": {
        "script": "train_v2.py",
        "ckpt":   "checkpoints/v2/tryon_best.pth",
        "args":   ["--stage", "both", "--epochs", "100"],
        "label":  "V2 (GMM TPS warp + Composition TryOnNet)",
    },
    "resnet_gen": {
        "script": "models/resnet_gen/train.py",
        "ckpt":   "checkpoints/resnet_gen/resnet_gen_best.pth",
        "args":   ["--stage", "both"],
        "label":  "ResNet Generator (9 ResBlocks, no skip connections)",
    },
    "attention_unet": {
        "script": "models/attention_unet/train.py",
        "ckpt":   "checkpoints/attention_unet/tryon_best.pth",
        "args":   ["--stage", "both"],
        "label":  "Attention U-Net (self-attention at bottleneck)",
    },
    "single_stage": {
        "script": "models/single_stage/train.py",
        "ckpt":   "checkpoints/single_stage/model_best.pth",
        "args":   [],
        "label":  "Single Stage (5-level U-Net, no explicit warping)",
    },
    "spade": {
        "script": "models/spade/train.py",
        "ckpt":   "checkpoints/spade/tryon_best.pth",
        "args":   ["--stage", "both"],
        "label":  "SPADE (spatially adaptive normalisation)",
    },
    "multiscale": {
        "script": "models/multiscale/train.py",
        "ckpt":   "checkpoints/multiscale/refine_best.pth",
        "args":   ["--stage", "both"],
        "label":  "Multi-Scale (CoarseNet 128px + RefineNet 256px)",
    },
    "viton_hd": {
        "script": "models/viton_hd/train.py",
        "ckpt":   "checkpoints/viton_hd/alias_best.pth",
        "args":   ["--stage", "all"],
        "label":  "VITON-HD (SegGen + GMM/TPS + ALIAS)",
    },
}

# Optimal batch sizes from find_optimal_batch.py (RTX 4000 Ada 20 GB, AMP fp16)
BATCH_SIZES = {
    # Tuned for RTX 4000 Ada 20 GB — GPU at 100% compute, VRAM ~70-80% target.
    "baseline":       96,  # CNN-Based ngf=32, no VGG — light, can push high
    "v2":             48,  # GMM+TPS+VGG — correlation layer adds memory
    "resnet_gen":     48,  # same architecture as v2
    "attention_unet": 96,  # attention overhead but 20 GB handles it
    "single_stage":   64,  # 5-level U-Net + VGG
    "spade":          64,  # SPADE norm + VGG
    "multiscale":     80,  # coarse+refine dual-stage
    "viton_hd":       24,  # ALIAS is heaviest — 3-stage pipeline
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s   = divmod(rem, 60)
    if td.days:
        return f"{td.days}d {h:02d}h {m:02d}m"
    return f"{h:02d}h {m:02d}m {s:02d}s"


def is_trained(model_key: str) -> bool:
    ckpt = ROOT / MODELS[model_key]["ckpt"]
    return ckpt.exists()


def run_model(model_key: str, extra_args: list, dry_run: bool,
              batch_override: int = None) -> bool:
    cfg    = MODELS[model_key]
    script = ROOT / cfg["script"]
    batch  = batch_override if batch_override else BATCH_SIZES[model_key]
    cmd    = [sys.executable, str(script), "--batch", str(batch)] + cfg["args"] + extra_args

    print(f"\n{'='*70}")
    print(f"  MODEL : {cfg['label']}")
    print(f"  BATCH : {batch}" + (" (override)" if batch_override else ""))
    print(f"  CMD   : {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")

    if dry_run:
        print("  [DRY RUN] skipping execution")
        return True

    t0 = time.perf_counter()
    try:
        env = os.environ.copy()
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(cmd, cwd=ROOT, env=env)
        elapsed = time.perf_counter() - t0
        if result.returncode == 0:
            print(f"\n  DONE in {fmt_duration(elapsed)}")
            return True
        else:
            print(f"\n  FAILED (exit code {result.returncode}) after {fmt_duration(elapsed)}")
            return False
    except KeyboardInterrupt:
        print(f"\n  INTERRUPTED by user")
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train all virtual try-on models sequentially")
    p.add_argument("--models",   nargs="+", default=list(MODELS.keys()),
                   choices=list(MODELS.keys()), metavar="MODEL",
                   help="Models to run (default: all)")
    p.add_argument("--force",    action="store_true",
                   help="Retrain even if checkpoint already exists")
    p.add_argument("--dry-run",  action="store_true",
                   help="Print commands without executing")
    p.add_argument("--epochs",      type=int, default=None,
                   help="Override epoch count for all models")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit dataset size (e.g. 3000 for quick runs)")
    p.add_argument("--batch",       type=int, default=None,
                   help="Override batch size for all models")
    args = p.parse_args()

    extra = []
    if args.epochs:
        extra += ["--epochs", str(args.epochs)]
    if args.max_samples:
        extra += ["--max-samples", str(args.max_samples)]

    print(f"\nVirtual Try-On — Training Pipeline")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models  : {args.models}")
    print(f"Force   : {args.force}")

    # Show status
    print(f"\n{'Model':<18} {'Status':<20} {'Checkpoint'}")
    print(f"{'-'*18}  {'-'*20}  {'-'*40}")
    for key in args.models:
        trained = is_trained(key)
        status  = "TRAINED" if trained else "not trained"
        ckpt    = MODELS[key]["ckpt"]
        will_run = not trained or args.force
        action  = "will train" if will_run else "will SKIP"
        print(f"  {key:<16} {status:<20} {ckpt}  [{action}]")

    print()

    total_start = time.perf_counter()
    results = {}

    for key in args.models:
        if is_trained(key) and not args.force:
            print(f"\n  SKIP {key} — checkpoint exists ({MODELS[key]['ckpt']})")
            print(f"       Use --force to retrain.")
            results[key] = "skipped"
            continue

        try:
            ok = run_model(key, extra, args.dry_run, batch_override=args.batch)
            results[key] = "ok" if ok else "failed"
        except KeyboardInterrupt:
            results[key] = "interrupted"
            print(f"\nTraining interrupted. Completed so far:")
            break

    # Final summary
    total_elapsed = time.perf_counter() - total_start
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY  (total: {fmt_duration(total_elapsed)})")
    print(f"{'='*70}")
    for key, status in results.items():
        label = MODELS[key]["label"]
        print(f"  {key:<18} {status.upper():<12}  {label}")

    not_run = [k for k in args.models if k not in results]
    for key in not_run:
        print(f"  {key:<18} {'NOT RUN':<12}  {MODELS[key]['label']}")

    print()


if __name__ == "__main__":
    main()
