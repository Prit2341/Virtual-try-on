#!/usr/bin/env python3
"""
Automated Rotation Training Loop
==================================
Solves HDD storage problem by rotating batches automatically:
  1. Copy batch from HDD → SSD
  2. Train for N epochs (resume from best checkpoint)
  3. Delete batch from SSD
  4. Copy next batch
  5. Repeat until all data seen

Set it running overnight — no manual steps needed.

Usage:
  python train_loop.py --stage warp
  python train_loop.py --stage tryon --warp-ckpt checkpoints/warp_best.pth
  python train_loop.py --stage warp --batch-size 2000 --epochs-per-batch 20
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HDD_TENSORS = Path("D:/viton_tensors")
SSD_TENSORS = Path("C:/Virtual_try_on/dataset/train/tensors")
LOG_DIR     = Path("C:/Virtual_try_on/logs")
PYTHON      = str(Path(__file__).resolve().parent / "venv" / "Scripts" / "python.exe")
TRAIN_V1    = str(Path(__file__).resolve().parent / "train.py")
TRAIN_V2    = str(Path(__file__).resolve().parent / "train_v2.py")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            encoding="utf-8"
        )
    ]
)
log = logging.getLogger("rotation")


def get_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def copy_batch(files: list[Path], dst: Path) -> int:
    """Copy a list of .pt files to dst. Returns count copied."""
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in files:
        target = dst / f.name
        if not target.exists():
            shutil.copy2(f, target)
            copied += 1
    return copied


def clear_ssd(dst: Path):
    """Delete all .pt files from SSD tensor dir."""
    removed = 0
    for f in dst.glob("*.pt"):
        f.unlink()
        removed += 1
    log.info("  Cleared %d files from SSD.", removed)


def run_training(stage: str, epochs: int, warp_ckpt: str, batch_size: int, version: int = 1):
    """Run train.py (v1) or train_v2.py (v2) for N epochs, resuming from best checkpoint."""
    ckpt_dir = Path("C:/Virtual_try_on/checkpoints")
    train_script = TRAIN_V2 if version == 2 else TRAIN_V1

    # Determine checkpoint name based on version + stage
    if version == 2:
        if stage == "gmm":
            best_ckpt = ckpt_dir / "gmm_best.pth"
        else:
            best_ckpt = ckpt_dir / "tryon_v2_best.pth"
    else:
        best_ckpt = ckpt_dir / f"{stage}_best.pth"

    cmd = [
        PYTHON, train_script,
        "--stage", stage,
        "--epochs", str(epochs),
        "--data", str(SSD_TENSORS),
        "--batch", str(batch_size),
    ]
    if best_ckpt.exists():
        cmd += ["--resume", str(best_ckpt)]
        log.info("  Resuming from %s", best_ckpt)
    else:
        log.info("  Starting fresh (no checkpoint found).")

    if stage == "tryon" and warp_ckpt:
        if version == 2:
            cmd += ["--gmm-ckpt", warp_ckpt]
        else:
            cmd += ["--warp-ckpt", warp_ckpt]

    log.info("  Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Automated rotation training — HDD batch copy + train loop"
    )
    parser.add_argument("--stage",           required=True, choices=["warp", "tryon", "gmm"])
    parser.add_argument("--hdd",             default=str(HDD_TENSORS))
    parser.add_argument("--ssd",             default=str(SSD_TENSORS))
    parser.add_argument("--batch-size",      type=int, default=5000,
                        help="Number of tensors to copy per rotation (default 5000)")
    parser.add_argument("--epochs-per-batch",type=int, default=20,
                        help="Training epochs per batch rotation (default 20)")
    parser.add_argument("--train-batch",     type=int, default=16,
                        help="Training batch size (GPU batch, default 16)")
    parser.add_argument("--warp-ckpt",       default="",
                        help="WarpNet checkpoint for tryon stage")
    parser.add_argument("--rounds",          type=int, default=0,
                        help="How many full passes over data (0 = 1 pass)")
    parser.add_argument("--version",        type=int, default=1, choices=[1, 2],
                        help="Model version: 1=WarpNet+TryOn, 2=GMM+TryOnV2")
    args = parser.parse_args()

    hdd = Path(args.hdd)
    ssd = Path(args.ssd)
    LOG_DIR.mkdir(exist_ok=True)

    if not hdd.exists():
        log.error("HDD path not found: %s", hdd)
        sys.exit(1)

    all_files = sorted(hdd.glob("*.pt"))
    if not all_files:
        log.error("No .pt files found in %s", hdd)
        sys.exit(1)

    passes = max(1, args.rounds + 1)
    total_files = len(all_files)
    batch_sz    = args.batch_size
    n_batches   = (total_files + batch_sz - 1) // batch_sz

    log.info("=" * 60)
    log.info("  ROTATION TRAINING LOOP")
    log.info("=" * 60)
    log.info("  Stage          : %s", args.stage)
    log.info("  HDD source     : %s  (%d files)", hdd, total_files)
    log.info("  SSD dest       : %s", ssd)
    log.info("  Batch size     : %d tensors", batch_sz)
    log.info("  Epochs/batch   : %d", args.epochs_per_batch)
    log.info("  Batches total  : %d", n_batches)
    log.info("  Data passes    : %d", passes)
    log.info("  Free SSD now   : %.1f GB", get_free_gb(ssd.parent))
    est_gb = batch_sz * 5.7 / 1024
    free_gb = get_free_gb(ssd.parent)
    if est_gb > free_gb * 0.85:
        log.warning("  ⚠ Batch (~%.1f GB) may exceed safe SSD capacity (%.1f GB free)!", est_gb, free_gb)
        log.warning("  Consider reducing --batch-size to %d", int(free_gb * 0.85 * 1024 / 5.7))
    log.info("  Est. per batch : %.1f GB", est_gb)
    log.info("")

    for p in range(passes):
        log.info("━" * 60)
        log.info("  DATA PASS %d / %d", p + 1, passes)
        log.info("━" * 60)

        for b in range(n_batches):
            batch_files = all_files[b * batch_sz : (b + 1) * batch_sz]
            log.info("")
            log.info("  [Batch %d/%d]  files %d–%d  (free SSD: %.1f GB)",
                     b + 1, n_batches,
                     b * batch_sz, min((b + 1) * batch_sz, total_files) - 1,
                     get_free_gb(ssd.parent))

            # ── 1. Clear old batch ──────────────────────────────────────────
            log.info("  Step 1: Clearing SSD tensors…")
            clear_ssd(ssd)

            # ── 2. Copy new batch ──────────────────────────────────────────
            log.info("  Step 2: Copying %d files from HDD → SSD…", len(batch_files))
            t0 = time.time()
            copied = copy_batch(batch_files, ssd)
            log.info("  Copied %d files in %.1fs  (SSD free: %.1f GB)",
                     copied, time.time() - t0, get_free_gb(ssd.parent))

            # ── 3. Train ───────────────────────────────────────────────────
            log.info("  Step 3: Training for %d epochs…", args.epochs_per_batch)
            ok = run_training(args.stage, args.epochs_per_batch,
                              args.warp_ckpt, args.train_batch, args.version)
            if not ok:
                log.error("  Training failed on batch %d — stopping.", b + 1)
                sys.exit(1)

    log.info("")
    log.info("=" * 60)
    log.info("  ALL BATCHES COMPLETE")
    log.info("  Best checkpoint: checkpoints/%s_best.pth", args.stage)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
