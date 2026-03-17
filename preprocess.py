#!/usr/bin/env python3
"""
VITON-HD Industrial Preprocessing Pipeline
==========================================
Target : 512 (H) x 384 (W)
GPU    : RTX 4070 (CUDA)
Python : 3.10

Steps:
  1. Input validation & resize
  2. Human parsing    → dataset/{split}/parsing/{person}.png
  3. Pose estimation  → dataset/{split}/pose/{person}.pt      (18, H, W)
  4. Cloth mask       → dataset/{split}/cloth_mask/{cloth}.png
  5. Agnostic person  → dataset/{split}/agnostic/{person}.png
  6. Normalise & save → dataset/{split}/tensors/{person}__{cloth}.pt

Usage:
  python preprocess.py --split both
  python preprocess.py --split train --batch 16
  python preprocess.py --split test  --limit 50   # quick test run
"""

import os
import sys
import warnings
import argparse
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────── GLOBAL CONFIG ────────────────────────────────

HEIGHT     = 512
WIDTH      = 384
BATCH_SIZE = 8          # GPU parse batch size; raise to 16 if VRAM allows
POSE_SIGMA = 6          # Gaussian radius (px) for pose heatmap channels
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── SegFormer cloth label indices ──────────────────────────────────────────────
#  0:bg  1:hat  2:hair  3:sunglasses  4:upper-clothes  5:skirt  6:pants
#  7:dress  8:belt  9:left-shoe  10:right-shoe  11:face  12:left-leg
#  13:right-leg  14:left-arm  15:right-arm  16:bag  17:scarf
LABELS_REMOVE = {4, 7, 17}          # upper-clothes, dress, scarf  → erase
LABELS_KEEP   = set(range(18)) - LABELS_REMOVE

# ── MediaPipe landmark index → COCO-18 keypoint index ─────────────────────────
#  COCO-18: 0=nose  1=neck  2=R-shoulder  3=R-elbow  4=R-wrist
#           5=L-shoulder  6=L-elbow  7=L-wrist  8=R-hip  9=R-knee
#           10=R-ankle  11=L-hip  12=L-knee  13=L-ankle
#           14=R-eye  15=L-eye  16=R-ear  17=L-ear
MP_TO_COCO18 = {
    0:  0,   # nose
    12: 2,   # right shoulder
    14: 3,   # right elbow
    16: 4,   # right wrist
    11: 5,   # left shoulder
    13: 6,   # left elbow
    15: 7,   # left wrist
    24: 8,   # right hip
    26: 9,   # right knee
    28: 10,  # right ankle
    23: 11,  # left hip
    25: 12,  # left knee
    27: 13,  # left ankle
    5:  14,  # right eye
    2:  15,  # left eye
    8:  16,  # right ear
    7:  17,  # left ear
}   # NOTE: neck (idx 1) = midpoint(R-shoulder, L-shoulder) → computed below

# ─────────────────────────────── LOGGING ──────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────── MODEL SINGLETONS ─────────────────────────────────
# Loaded once, reused for the entire run.

_seg_processor = None
_seg_model     = None
_pose_model    = None
_rembg_session = None


def get_seg_model():
    """SegFormer-b2 fine-tuned on human clothes (HuggingFace Hub)."""
    global _seg_processor, _seg_model
    if _seg_model is None:
        from transformers import (
            SegformerImageProcessor,
            AutoModelForSemanticSegmentation,
        )
        log.info("Loading SegFormer human-parsing model → %s …", DEVICE)
        _seg_processor = SegformerImageProcessor.from_pretrained(
            "mattmdjaga/segformer_b2_clothes"
        )
        _seg_model = (
            AutoModelForSemanticSegmentation.from_pretrained(
                "mattmdjaga/segformer_b2_clothes"
            )
            .to(DEVICE)
            .eval()
        )
        log.info("  ✓ SegFormer ready.")
    return _seg_processor, _seg_model


def get_pose_model():
    """MediaPipe Pose Landmarker (Tasks API — mediapipe 0.10+)."""
    global _pose_model
    if _pose_model is None:
        import urllib.request
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path("models/pose_landmarker_full.task").resolve()
        model_url  = (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        )
        if not model_path.exists():
            log.info("Downloading pose_landmarker_full.task (~7 MB) …")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(model_url, str(model_path))
            log.info("  ✓ Model saved to %s", model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=1,
        )
        _pose_model = mp_vision.PoseLandmarker.create_from_options(options)
        log.info("  ✓ MediaPipe Pose Landmarker ready (Tasks API).")
    return _pose_model


def get_rembg_session():
    """rembg U2Net cloth-specific segmentation session."""
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        log.info("Loading rembg u2net_cloth_seg model …")
        _rembg_session = new_session("u2net_cloth_seg")
        log.info("  ✓ rembg ready.")
    return _rembg_session


# ──────────────────── STEP 1 — LOAD & VALIDATE ────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """
    Load image, ensure RGB, resize to (WIDTH, HEIGHT).
    Returns float32 array, shape (H, W, 3), range [0, 255].
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Image is not 3-channel: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)
    return img.astype(np.float32)


# ──────────────────── STEP 2 — HUMAN PARSING ──────────────────────────────────

def parse_batch(images: list) -> list:
    """
    GPU-batched human parsing.

    Args:
        images: list of float32 RGB arrays (H, W, 3)

    Returns:
        list of uint8 label maps (H, W), values 0–17
    """
    processor, model = get_seg_model()
    pils = [Image.fromarray(img.astype(np.uint8)) for img in images]
    inputs = processor(images=pils, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits          # (B, 18, h, w)  — low-res

    # Upsample to target resolution
    logits = F.interpolate(
        logits, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False
    )
    preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)
    return [preds[i] for i in range(len(images))]


# ──────────────────── STEP 3 — POSE ESTIMATION ────────────────────────────────

def _gaussian(H: int, W: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """2-D Gaussian blob centred at (cx, cy)."""
    xs = np.arange(W, dtype=np.float32)[None, :]   # (1, W)
    ys = np.arange(H, dtype=np.float32)[:, None]   # (H, 1)
    return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))


def pose_to_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Run MediaPipe Pose Landmarker (Tasks API) and produce 18-channel heatmap.

    Args:
        img: float32 RGB (H, W, 3)

    Returns:
        float32 tensor (18, H, W), each channel ∈ [0, 1]
        Zero tensor if no person detected.
    """
    import mediapipe as mp

    detector = get_pose_model()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img.astype(np.uint8))
    result   = detector.detect(mp_image)

    heatmap = np.zeros((18, HEIGHT, WIDTH), dtype=np.float32)
    if not result.pose_landmarks:
        return heatmap

    landmarks = result.pose_landmarks[0]   # first (only) person
    kp_xy = {}

    for mp_idx, coco_idx in MP_TO_COCO18.items():
        lm = landmarks[mp_idx]
        if lm.visibility >= 0.5:
            cx = lm.x * WIDTH
            cy = lm.y * HEIGHT
            kp_xy[coco_idx] = (cx, cy)
            heatmap[coco_idx] = _gaussian(HEIGHT, WIDTH, cx, cy, POSE_SIGMA)

    # Neck = midpoint(R-shoulder, L-shoulder)
    if 2 in kp_xy and 5 in kp_xy:
        cx = (kp_xy[2][0] + kp_xy[5][0]) / 2.0
        cy = (kp_xy[2][1] + kp_xy[5][1]) / 2.0
        heatmap[1] = _gaussian(HEIGHT, WIDTH, cx, cy, POSE_SIGMA)

    return heatmap


# ──────────────────── STEP 4 — CLOTH MASK ─────────────────────────────────────

def extract_cloth_mask(cloth_img: np.ndarray) -> np.ndarray:
    """
    Use rembg (U2Net cloth model) to produce a binary cloth mask.

    Args:
        cloth_img: float32 RGB (H, W, 3)

    Returns:
        uint8 mask (H, W) — 1 = cloth, 0 = background
    """
    from rembg import remove

    session = get_rembg_session()
    pil = Image.fromarray(cloth_img.astype(np.uint8))
    rgba = remove(pil, session=session)          # → RGBA PIL image

    alpha = np.array(rgba)[:, :, 3]             # alpha channel is the mask
    alpha = cv2.resize(alpha, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    return (alpha > 127).astype(np.uint8)


# ──────────────────── STEP 5 — AGNOSTIC IMAGE ─────────────────────────────────

def create_agnostic(person_img: np.ndarray, parse_map: np.ndarray) -> np.ndarray:
    """
    Erase upper-clothes / dress pixels from person image.
    Fill erased region with mid-gray (128). Keep face, hair, arms, legs.

    Args:
        person_img: float32 RGB (H, W, 3)
        parse_map:  uint8 label map (H, W)

    Returns:
        float32 RGB (H, W, 3)
    """
    remove_mask = np.isin(parse_map, list(LABELS_REMOVE))
    agnostic = person_img.copy()
    agnostic[remove_mask] = 128.0
    return agnostic


# ──────────────────── STEP 6 — NORMALISE ──────────────────────────────────────

def to_tensor_norm(img: np.ndarray) -> torch.Tensor:
    """
    float32 RGB (H, W, 3), range [0, 255]
    → torch.Tensor (3, H, W), range [-1, 1]
    """
    t = torch.from_numpy(img).permute(2, 0, 1).float()
    return (t / 127.5) - 1.0


# ──────────────────── SAVE HELPERS ────────────────────────────────────────────

def save_rgb(img: np.ndarray, path: Path, quality: int = 95):
    """Save float32 RGB (H, W, 3) as JPEG."""
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])


def save_mask(mask: np.ndarray, path: Path):
    """Save uint8 binary mask (H, W) as PNG (0 / 255)."""
    cv2.imwrite(str(path), (mask * 255).astype(np.uint8))


def save_parse(parse_map: np.ndarray, path: Path):
    """Save uint8 label map (H, W) as PNG (raw label values 0–17)."""
    cv2.imwrite(str(path), parse_map)


# ──────────────────── DIRECTORY SETUP ─────────────────────────────────────────

SUBDIRS = ["person", "cloth", "cloth_mask", "parsing", "pose", "agnostic", "tensors"]


def make_output_dirs(dataset_root: Path, split: str) -> dict:
    dirs = {}
    for name in SUBDIRS:
        p = dataset_root / split / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def is_done(dirs: dict, stem_p: str, stem_c: str) -> bool:
    """Return True if the tensor bundle already exists (resume support)."""
    return (dirs["tensors"] / f"{stem_p}__{stem_c}.pt").exists()


# ──────────────────── CORE BATCH FLUSH ────────────────────────────────────────

def flush_batch(batch: list, dirs: dict, stats: dict):
    """
    Process one accumulated batch:
      - GPU parse (all at once)
      - Per-image: pose, cloth-mask, agnostic, save
    """
    if not batch:
        return

    person_imgs = [item["person_img"] for item in batch]

    # ── Step 2: GPU batch parse ────────────────────────────────────────────────
    try:
        parse_maps = parse_batch(person_imgs)
    except Exception as e:
        log.error("Batch parse failed: %s — skipping batch.", e)
        stats["failed"] += len(batch)
        return

    for item, parse_map in zip(batch, parse_maps):
        person_img = item["person_img"]
        cloth_img  = item["cloth_img"]
        stem_p     = item["stem_p"]
        stem_c     = item["stem_c"]

        try:
            # ── Step 2: save parsing map ───────────────────────────────────────
            save_parse(parse_map, dirs["parsing"] / f"{stem_p}.png")

            # ── Step 3: pose heatmap ───────────────────────────────────────────
            pose_map = pose_to_heatmap(person_img)          # (18, H, W)
            torch.save(
                torch.from_numpy(pose_map),
                dirs["pose"] / f"{stem_p}.pt",
            )

            # ── Step 4: cloth mask ─────────────────────────────────────────────
            cloth_mask = extract_cloth_mask(cloth_img)      # (H, W) uint8
            save_mask(cloth_mask, dirs["cloth_mask"] / f"{stem_c}.png")

            # ── Step 5: agnostic image ─────────────────────────────────────────
            agnostic = create_agnostic(person_img, parse_map)
            save_rgb(agnostic, dirs["agnostic"] / f"{stem_p}.png")

            # ── Save resized originals ─────────────────────────────────────────
            save_rgb(person_img, dirs["person"] / f"{stem_p}.jpg")
            save_rgb(cloth_img,  dirs["cloth"]  / f"{stem_c}.jpg")

            # ── Step 6: normalised tensor bundle ──────────────────────────────
            torch.save(
                {
                    "person":     to_tensor_norm(person_img),                 # (3,H,W)
                    "cloth":      to_tensor_norm(cloth_img),                  # (3,H,W)
                    "agnostic":   to_tensor_norm(agnostic),                   # (3,H,W)
                    "parse_map":  torch.from_numpy(parse_map).long(),         # (H,W)
                    "cloth_mask": torch.from_numpy(cloth_mask.astype(np.float32)),  # (H,W)
                    "pose_map":   torch.from_numpy(pose_map),                 # (18,H,W)
                },
                dirs["tensors"] / f"{stem_p}__{stem_c}.pt",
            )

            stats["done"] += 1

        except Exception as e:
            log.warning("  ✗ %s | %s : %s", stem_p, stem_c, e)
            stats["failed"] += 1


# ──────────────────── SPLIT PIPELINE ──────────────────────────────────────────

def process_split(split: str, dataset_root: Path, batch_size: int, limit: int):
    person_dir = dataset_root / "raw" / "image"
    cloth_dir  = dataset_root / "raw" / "cloth"
    pairs_file = dataset_root / f"{split}_pairs.txt"

    if not pairs_file.exists():
        log.error("Pairs file not found: %s", pairs_file)
        return

    # Read pairs
    pairs = []
    with open(pairs_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    if limit > 0:
        pairs = pairs[:limit]
        log.info("[%s] Limited to %d pairs (--limit).", split.upper(), limit)

    log.info(
        "[%s] %d pairs | batch=%d | device=%s",
        split.upper(), len(pairs), batch_size, DEVICE,
    )

    dirs  = make_output_dirs(dataset_root, split)
    stats = {"done": 0, "skipped": 0, "failed": 0}
    batch = []

    with tqdm(total=len(pairs), desc=f"[{split}]", unit="pair", dynamic_ncols=True) as pbar:
        for person_name, cloth_name in pairs:
            stem_p = Path(person_name).stem
            stem_c = Path(cloth_name).stem

            # ── Resume: skip already-done pairs ───────────────────────────────
            if is_done(dirs, stem_p, stem_c):
                stats["skipped"] += 1
                pbar.update(1)
                pbar.set_postfix(**stats)
                continue

            # ── Load & validate ────────────────────────────────────────────────
            try:
                person_img = load_image(person_dir / person_name)
                cloth_img  = load_image(cloth_dir  / cloth_name)
            except Exception as e:
                log.warning("  ✗ Load failed (%s / %s): %s", person_name, cloth_name, e)
                stats["failed"] += 1
                pbar.update(1)
                pbar.set_postfix(**stats)
                continue

            batch.append(
                {"person_img": person_img, "cloth_img": cloth_img,
                 "stem_p": stem_p, "stem_c": stem_c}
            )

            # ── Flush when batch is full ───────────────────────────────────────
            if len(batch) >= batch_size:
                flush_batch(batch, dirs, stats)
                batch.clear()

            pbar.update(1)
            pbar.set_postfix(**stats)

        # ── Flush remaining ────────────────────────────────────────────────────
        if batch:
            flush_batch(batch, dirs, stats)

    log.info(
        "[%s] Finished — done=%d  skipped=%d  failed=%d",
        split.upper(), stats["done"], stats["skipped"], stats["failed"],
    )


# ──────────────────── ENTRY POINT ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VITON-HD Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", default="both", choices=["train", "test", "both"],
        help="Which split to preprocess.",
    )
    parser.add_argument(
        "--dataset", default="d:/Virtul_try_on/dataset",
        help="Path to dataset root (contains train/, test/, *_pairs.txt).",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help="GPU batch size for human parsing. Raise to 16 on 12 GB VRAM.",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only N pairs per split (0 = all). Useful for testing.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("VITON-HD Preprocessing  |  %dH × %dW  |  %s", HEIGHT, WIDTH, DEVICE)
    log.info("=" * 60)

    dataset_root = Path(args.dataset)
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        process_split(split, dataset_root, args.batch, args.limit)

    log.info("All splits done.")


if __name__ == "__main__":
    main()
