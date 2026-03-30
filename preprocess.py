#!/usr/bin/env python3
"""
VITON-HD Preprocessing  —  Case 1 (Dataset Images)
====================================================
Reads directly from the archive (no model inference needed).
Everything — parsing, agnostic, cloth-mask — is already pre-computed.
Only work done here: resize to target resolution + pose heatmaps from JSON.

Output bundle per pair  (saved to dataset/{split}/tensors/NAME.pt):
  person     : (3, H, W)  float32  [-1, 1]
  cloth      : (3, H, W)  float32  [-1, 1]
  agnostic   : (3, H, W)  float32  [-1, 1]
  cloth_mask : (H, W)     float32  {0.0, 1.0}
  pose_map   : (18, H, W) float32  [0, 1]  Gaussian heatmaps
  parse_map  : (H, W)     int64    CIHP label indices 0-19

Usage:
  python preprocess.py                        # both splits
  python preprocess.py --split train
  python preprocess.py --split test
  python preprocess.py --split train --limit 500   # quick smoke-test
  python preprocess.py --resume                    # skip existing .pt files
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

HEIGHT = 256
WIDTH  = 192

ARCHIVE_DIR = Path(__file__).resolve().parent / "archive"
OUT_DIR     = Path(__file__).resolve().parent / "dataset"

# OpenPose BODY_25 index → COCO-18 index
# COCO-18: 0=nose 1=neck 2=R-sho 3=R-elb 4=R-wri 5=L-sho 6=L-elb 7=L-wri
#          8=R-hip 9=R-kne 10=R-ank 11=L-hip 12=L-kne 13=L-ank
#          14=R-eye 15=L-eye 16=R-ear 17=L-ear
BODY25_TO_COCO18 = {
    0:  0,   # nose
    1:  1,   # neck
    2:  2,   # R-shoulder
    3:  3,   # R-elbow
    4:  4,   # R-wrist
    5:  5,   # L-shoulder
    6:  6,   # L-elbow
    7:  7,   # L-wrist
    9:  8,   # R-hip
    10: 9,   # R-knee
    11: 10,  # R-ankle
    12: 11,  # L-hip
    13: 12,  # L-knee
    14: 13,  # L-ankle
    15: 14,  # R-eye
    16: 15,  # L-eye
    17: 16,  # R-ear
    18: 17,  # L-ear
}

POSE_SIGMA    = 4    # Gaussian radius in pixels at target resolution
CONF_THRESH   = 0.1  # ignore keypoints below this confidence

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_rgb(img: np.ndarray) -> np.ndarray:
    """Resize RGB to (HEIGHT, WIDTH)."""
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)


def resize_mask(img: np.ndarray) -> np.ndarray:
    """Resize grayscale mask to (HEIGHT, WIDTH) with linear interp, then binarize."""
    resized = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return (resized >= 128).astype(np.float32)


def resize_parse(img: np.ndarray) -> np.ndarray:
    """Resize label map to (HEIGHT, WIDTH) with nearest-neighbor (no label mixing)."""
    return cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 (H, W, 3) → float32 tensor (3, H, W) in [-1, 1]."""
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    return t / 127.5 - 1.0


def make_pose_heatmaps(json_path: Path, src_h: int, src_w: int) -> np.ndarray:
    """
    Parse OpenPose BODY_25 JSON and return 18-channel Gaussian heatmaps
    at (HEIGHT, WIDTH) resolution.

    Args:
        json_path : path to *_keypoints.json
        src_h     : original image height (keypoints are in this space)
        src_w     : original image width

    Returns:
        heatmaps : float32 (18, HEIGHT, WIDTH) in [0, 1]
    """
    heatmaps = np.zeros((18, HEIGHT, WIDTH), dtype=np.float32)

    try:
        with open(json_path) as f:
            data = json.load(f)
        if not data.get("people"):
            return heatmaps
        kps = data["people"][0]["pose_keypoints_2d"]  # flat [x, y, conf, ...]
    except Exception:
        return heatmaps

    # Scale factors from original → target resolution
    sx = WIDTH  / src_w
    sy = HEIGHT / src_h

    # Pre-build Gaussian kernel grid
    gy, gx = np.mgrid[0:HEIGHT, 0:WIDTH]

    for body25_idx, coco18_idx in BODY25_TO_COCO18.items():
        base = body25_idx * 3
        if base + 2 >= len(kps):
            continue
        x, y, conf = kps[base], kps[base + 1], kps[base + 2]
        if conf < CONF_THRESH:
            continue

        cx = x * sx
        cy = y * sy
        gauss = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * POSE_SIGMA ** 2))
        heatmaps[coco18_idx] = np.maximum(heatmaps[coco18_idx], gauss)

    return heatmaps


# ── Per-pair processing ───────────────────────────────────────────────────────

def process_pair(
    person_name: str,
    cloth_name:  str,
    split:       str,
    out_dir:     Path,
    resume:      bool,
) -> bool:
    """
    Process one (person, cloth) pair and save a .pt bundle.
    Returns True on success, False if skipped or failed.
    """
    person_stem = Path(person_name).stem
    cloth_stem  = Path(cloth_name).stem
    out_path    = out_dir / f"{person_stem}__{cloth_stem}.pt"

    if resume and out_path.exists():
        return False  # already done

    src = ARCHIVE_DIR / split

    # Paths
    person_path  = src / "image"                    / person_name
    cloth_path   = src / "cloth"                    / cloth_name
    mask_path    = src / "cloth-mask"               / cloth_name
    agnostic_path= src / "agnostic-v3.2"            / person_name
    parse_path   = src / "image-parse-v3"           / (person_stem + ".png")
    pose_path    = src / "openpose_json"             / (person_stem + "_keypoints.json")

    # Skip if any required file is missing (unzip still in progress)
    for p in [person_path, cloth_path, mask_path, agnostic_path, parse_path, pose_path]:
        if not p.exists():
            return False

    try:
        # Load & resize RGB images
        person_img   = resize_rgb(load_rgb(person_path))
        cloth_img    = resize_rgb(load_rgb(cloth_path))
        agnostic_img = resize_rgb(load_rgb(agnostic_path))

        # Cloth mask — grayscale, binarize
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        cloth_mask = resize_mask(mask_raw)

        # Parse map — palette PNG, nearest-neighbor resize
        parse_arr = np.array(Image.open(parse_path))
        parse_map = resize_parse(parse_arr)

        # Pose heatmaps from OpenPose JSON
        pose_map = make_pose_heatmaps(pose_path, src_h=1024, src_w=768)

        # Save bundle
        torch.save(
            {
                "person":     to_tensor(person_img),
                "cloth":      to_tensor(cloth_img),
                "agnostic":   to_tensor(agnostic_img),
                "cloth_mask": torch.from_numpy(cloth_mask),
                "parse_map":  torch.from_numpy(parse_map.astype(np.int64)),
                "pose_map":   torch.from_numpy(pose_map),
            },
            out_path,
        )
        return True

    except Exception as e:
        log.warning("  FAILED %s__%s : %s", person_stem, cloth_stem, e)
        return False


# ── Split runner ──────────────────────────────────────────────────────────────

def run_split(split: str, limit: int, resume: bool) -> None:
    pairs_file = ARCHIVE_DIR / f"{split}_pairs.txt"
    if not pairs_file.exists():
        log.error("Pairs file not found: %s", pairs_file)
        return

    pairs = []
    for line in pairs_file.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))

    if limit:
        pairs = pairs[:limit]

    out_dir = OUT_DIR / split / "tensors"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Split: %s  |  pairs: %d  |  output: %s", split, len(pairs), out_dir)

    done = skipped = failed = 0
    for person_name, cloth_name in tqdm(pairs, desc=split, unit="pair"):
        result = process_pair(person_name, cloth_name, split, out_dir, resume)
        if result is True:
            done += 1
        elif result is False:
            skipped += 1

    log.info(
        "Done: %d  |  Skipped (missing/existing): %d  |  Total attempted: %d",
        done, skipped, len(pairs),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VITON-HD Preprocessing (Case 1)")
    ap.add_argument("--split",  default="both", choices=["train", "test", "both"])
    ap.add_argument("--limit",  type=int, default=0, help="Max pairs per split (0=all)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="Skip pairs whose .pt already exists (default: on)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Reprocess all pairs even if .pt already exists")
    args = ap.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for split in splits:
        run_split(split, args.limit, args.resume)


if __name__ == "__main__":
    main()
