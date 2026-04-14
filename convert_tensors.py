#!/usr/bin/env python3
"""
Convert HDD tensors (512x384) → HDD tensors_256 (256x192)
============================================================
Downsamples all tensor bundles on HDD itself (no SSD needed).
Shrinks each file from ~22 MB to ~5.7 MB.
One-time conversion — then train_loop.py rotates small files to SSD.

Usage:
  python convert_tensors.py
  python convert_tensors.py --limit 5000   # convert first 5000 only
  python convert_tensors.py --workers 4    # parallel workers
"""

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torch.nn.functional as F
from tqdm import tqdm

SRC_DIR = Path("D:/Virtual_try_on/dataset/train/tensors")
DST_DIR = Path("D:/Virtual_try_on/dataset/train/tensors_256")  # same HDD, smaller files
OUT_H, OUT_W = 256, 192


def convert_one(src: Path, dst: Path) -> str:
    """Load one 512x384 bundle, downsample to 256x192, save."""
    if dst.exists():
        return "skip"
    try:
        bundle = torch.load(src, map_location="cpu", weights_only=False)

        def resize_img(t):
            # (3, H, W) → (1, 3, H, W) → resize → (3, H, W)
            return F.interpolate(
                t.unsqueeze(0), size=(OUT_H, OUT_W),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        def resize_map(t, mode="nearest"):
            # (H, W) → (1, 1, H, W) → resize → (H, W)
            return F.interpolate(
                t.float().unsqueeze(0).unsqueeze(0),
                size=(OUT_H, OUT_W), mode=mode
            ).squeeze(0).squeeze(0)

        def resize_pose(t):
            # (18, H, W) → (1, 18, H, W) → resize → (18, H, W)
            return F.interpolate(
                t.unsqueeze(0), size=(OUT_H, OUT_W),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        out = {
            "person":     resize_img(bundle["person"]),
            "cloth":      resize_img(bundle["cloth"]),
            "agnostic":   resize_img(bundle["agnostic"]),
            "parse_map":  resize_map(bundle["parse_map"], mode="nearest").long(),
            "cloth_mask": resize_map(bundle["cloth_mask"], mode="nearest"),
            "pose_map":   resize_pose(bundle["pose_map"]),
        }

        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, dst)
        return "done"

    except Exception as e:
        return f"fail:{e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",     default=str(SRC_DIR))
    parser.add_argument("--dst",     default=str(DST_DIR))
    parser.add_argument("--limit",   type=int, default=0, help="Max files to convert (0=all)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel threads")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if not src.exists():
        print(f"[ERROR] Source not found: {src}")
        sys.exit(1)

    files = sorted(src.glob("*.pt"))
    if not files:
        print(f"[ERROR] No .pt files in {src}")
        sys.exit(1)

    if args.limit > 0:
        files = files[:args.limit]

    dst.mkdir(parents=True, exist_ok=True)

    print(f"Source : {src}  ({len(files)} files)")
    print(f"Dest   : {dst}")
    print(f"Resize : 512x384 → {OUT_H}x{OUT_W}")
    print(f"Workers: {args.workers}")
    print()

    done = skip = fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(convert_one, f, dst / f.name): f
            for f in files
        }
        with tqdm(total=len(files), unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result == "done":
                    done += 1
                elif result == "skip":
                    skip += 1
                else:
                    fail += 1
                pbar.set_postfix(done=done, skip=skip, fail=fail)
                pbar.update(1)

    total = done + skip
    size_gb = total * 5.7 / 1024
    print(f"\nDone: {done}  Skipped(already exist): {skip}  Failed: {fail}")
    print(f"Estimated SSD usage: {size_gb:.1f} GB for {total} files")
    print(f"\nRun training with:")
    print(f"  venv/Scripts/python train.py --stage warp")


if __name__ == "__main__":
    main()
