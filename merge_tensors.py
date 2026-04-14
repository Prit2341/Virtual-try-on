#!/usr/bin/env python3
"""
Merge individual .pt tensor files into a single compact file.
Eliminates per-file torch.load overhead (~10-50x faster data loading).

Usage:
  python merge_tensors.py                          # default: dataset/train/tensors
  python merge_tensors.py --data dataset/test/tensors
"""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm


def merge(data_dir: str):
    root = Path(data_dir)
    files = sorted(root.glob("*.pt"))
    if not files:
        print(f"No .pt files found in {root}")
        return

    print(f"Found {len(files)} tensor files in {root}")

    keys = ["person", "cloth", "agnostic", "parse_map", "cloth_mask", "pose_map"]
    sample = torch.load(files[0], map_location="cpu", weights_only=False)

    # Pre-allocate contiguous tensors
    N = len(files)
    merged = {}
    for k in keys:
        shape = sample[k].shape
        dtype = sample[k].dtype
        merged[k] = torch.empty((N, *shape), dtype=dtype)
        print(f"  {k}: ({N}, {', '.join(str(s) for s in shape)}) {dtype}")

    # Fill in
    for i, f in enumerate(tqdm(files, desc="Merging")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        for k in keys:
            merged[k][i] = d[k]

    out_path = root / "_merged.pt"
    torch.save(merged, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved: {out_path}  ({size_mb:.0f} MB)")
    print(f"Load with: data = torch.load('{out_path}', map_location='cpu')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/train/tensors")
    merge(p.parse_args().data)
