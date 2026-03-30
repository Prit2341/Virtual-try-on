#!/usr/bin/env python3
"""
pipeline.py — VITON-HD Full Preprocessing Pipeline
====================================================
Connects all 6 preprocessing steps and batch-processes the dataset.

Step flow per pair:
  step1  →  load & resize person + cloth
  step2  →  GPU-batch human parsing (SegFormer)
  step3  →  pose estimation → 18-ch heatmap (MediaPipe)
  step4  →  cloth mask extraction (rembg U2Net)  [parallelised across batch]
  step5  →  agnostic person image (erase upper-clothes, feathered fill)
  step6  →  normalise + save .pt bundle

Output folder structure (created automatically):
  dataset/{split}/
    ├── person/       resized person images (.jpg)
    ├── cloth/        resized cloth images  (.jpg)
    ├── parsing/      human parsing maps    (.png, raw label values)
    ├── pose/         pose heatmap tensors  (.pt,  shape 18×H×W)
    ├── cloth_mask/   binary cloth masks    (.png, 0 or 255)
    ├── agnostic/     agnostic person imgs  (.png)
    └── tensors/      normalised bundles    (.pt,  all inputs for model)

Resume support: pairs whose .pt bundle already exists are skipped.
Failed pairs are written to dataset/{split}/failed.txt for later retry.

Usage:
  python pipeline.py --split both               # process train + test
  python pipeline.py --split train --batch 16   # larger GPU batch
  python pipeline.py --split train --limit 20   # quick test run (20 pairs)
  python pipeline.py --split train --workers 4  # cloth mask parallelism
"""

import argparse
import logging
import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from steps import (
    step1_validate,
    step2_parsing,
    step3_pose,
    step4_cloth_mask,
    step5_agnostic,
    step6_normalize,
)

# ─────────────────────────────── CONFIG ───────────────────────────────────────

BATCH_SIZE   = 8    # GPU batch size for SegFormer parsing
MASK_WORKERS = 4    # threads for parallel cloth mask extraction
DATASET      = Path(__file__).resolve().parent / "dataset"

SUBDIRS = [
    "person", "cloth", "parsing",
    "pose", "cloth_mask", "agnostic", "tensors",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────── HELPERS ──────────────────────────────────────

def make_output_dirs(dataset_root: Path, split: str) -> dict:
    """Create all output subdirectories under dataset_root/split/."""
    dirs = {}
    for name in SUBDIRS:
        p = dataset_root / split / name
        p.mkdir(parents=True, exist_ok=True)
        dirs[name] = p
    return dirs


def is_done(dirs: dict, stem_p: str, stem_c: str) -> bool:
    """True if the tensor bundle for this pair already exists (resume)."""
    return (dirs["tensors"] / f"{stem_p}__{stem_c}.pt").exists()


def _bgr(img: np.ndarray) -> np.ndarray:
    """float32 RGB → uint8 BGR for cv2.imwrite."""
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_intermediates(
    dirs:       dict,
    person_img: np.ndarray,
    cloth_img:  np.ndarray,
    agnostic:   np.ndarray,
    parse_map:  np.ndarray,
    cloth_mask: np.ndarray,
    stem_p:     str,
    stem_c:     str,
):
    """Save all PNG / JPEG intermediate outputs for visual inspection."""
    cv2.imwrite(
        str(dirs["person"]  / f"{stem_p}.jpg"), _bgr(person_img),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(
        str(dirs["cloth"]   / f"{stem_c}.jpg"), _bgr(cloth_img),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    cv2.imwrite(str(dirs["parsing"]    / f"{stem_p}.png"), parse_map)
    cv2.imwrite(str(dirs["agnostic"]   / f"{stem_p}.png"), _bgr(agnostic))
    cv2.imwrite(
        str(dirs["cloth_mask"] / f"{stem_c}.png"),
        (cloth_mask * 255).astype(np.uint8),
    )


# ─────────────────────────────── BATCH FLUSH ──────────────────────────────────

def _log_fail(fh, stem_p: str, stem_c: str, reason: str) -> None:
    """Append a failed pair to the open failed.txt file handle (may be None)."""
    if fh is not None:
        fh.write(f"{stem_p}\t{stem_c}\t{reason}\n")
        fh.flush()


def flush_batch(
    batch:        list,
    dirs:         dict,
    stats:        dict,
    mask_workers: int,
    failed_fh,
) -> None:
    """
    Process one accumulated batch:
      1. GPU parse all person images         (step 2, batched)
      2. Cloth masks for all cloth images    (step 4, parallelised with threads)
      3. Per-image: pose (3), agnostic (5), save intermediates + bundle (6)
    """
    if not batch:
        return

    # ── Step 2: GPU batch parsing ──────────────────────────────────────────────
    try:
        parse_maps = step2_parsing.run_batch(
            [item["person_img"] for item in batch]
        )
    except Exception as exc:
        log.error("Batch parsing failed (%d images): %s — skipping batch.", len(batch), exc)
        for item in batch:
            _log_fail(failed_fh, item["stem_p"], item["stem_c"], f"parse_batch: {exc}")
        stats["failed"] += len(batch)
        return

    # ── Step 4: Parallel cloth mask extraction ─────────────────────────────────
    # rembg's ONNXRuntime session is thread-safe; running N masks in parallel
    # gives a ~mask_workers× speedup since this step is the CPU bottleneck.
    cloth_masks = [None] * len(batch)
    with ThreadPoolExecutor(max_workers=mask_workers) as pool:
        future_to_idx = {
            pool.submit(step4_cloth_mask.run, item["cloth_img"]): i
            for i, item in enumerate(batch)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                cloth_masks[i] = future.result()
            except Exception as exc:
                log.warning("  ✗ Cloth mask failed (%s): %s", batch[i]["stem_c"], exc)

    # ── Per-image steps 3, 5, 6 ───────────────────────────────────────────────
    for item, parse_map, cloth_mask in zip(batch, parse_maps, cloth_masks):
        person_img = item["person_img"]
        cloth_img  = item["cloth_img"]
        stem_p     = item["stem_p"]
        stem_c     = item["stem_c"]

        if cloth_mask is None:
            _log_fail(failed_fh, stem_p, stem_c, "cloth_mask failed")
            stats["failed"] += 1
            continue

        try:
            # Step 3: pose heatmap (18, H, W)
            pose_map = step3_pose.run(person_img)
            torch.save(
                torch.from_numpy(pose_map),
                dirs["pose"] / f"{stem_p}.pt",
            )

            # Step 5: agnostic image (H, W, 3) float32
            agnostic = step5_agnostic.run(person_img, parse_map)

            # Save PNG/JPEG intermediates for human review
            save_intermediates(
                dirs, person_img, cloth_img,
                agnostic, parse_map, cloth_mask,
                stem_p, stem_c,
            )

            # Step 6: save normalised tensor bundle
            step6_normalize.save_bundle(
                path       = dirs["tensors"] / f"{stem_p}__{stem_c}.pt",
                person_img = person_img,
                cloth_img  = cloth_img,
                agnostic   = agnostic,
                parse_map  = parse_map,
                cloth_mask = cloth_mask,
                pose_map   = pose_map,
            )

            stats["done"] += 1

        except Exception as exc:
            log.warning("  ✗ %s | %s : %s", stem_p, stem_c, exc)
            _log_fail(failed_fh, stem_p, stem_c, str(exc))
            stats["failed"] += 1


# ─────────────────────────────── SPLIT PIPELINE ───────────────────────────────

def process_split(
    split:        str,
    dataset_root: Path,
    batch_size:   int,
    mask_workers: int,
    limit:        int,
) -> None:
    person_dir = dataset_root / split / "image"
    cloth_dir  = dataset_root / split / "cloth"
    pairs_file = dataset_root / f"{split}_pairs.txt"

    if not pairs_file.exists():
        log.error("Pairs file not found: %s", pairs_file)
        return

    # Read all pairs
    pairs = []
    with open(pairs_file) as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))

    if limit > 0:
        pairs = pairs[:limit]
        log.info("[%s] --limit applied: %d pairs.", split.upper(), limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "[%s] %d pairs | batch=%d | mask_workers=%d | device=%s",
        split.upper(), len(pairs), batch_size, mask_workers, device,
    )

    dirs      = make_output_dirs(dataset_root, split)
    stats     = {"done": 0, "skipped": 0, "failed": 0}
    batch     = []
    fail_path = dataset_root / split / "failed.txt"
    failed_fh = open(fail_path, "a", encoding="utf-8")

    try:
        with tqdm(
            total=len(pairs),
            desc=f"[{split}]",
            unit="pair",
            dynamic_ncols=True,
        ) as pbar:

            for person_name, cloth_name in pairs:
                stem_p = Path(person_name).stem
                stem_c = Path(cloth_name).stem

                # ── Resume: skip already-completed pairs ───────────────────────
                if is_done(dirs, stem_p, stem_c):
                    stats["skipped"] += 1
                    pbar.update(1)
                    pbar.set_postfix(**stats)
                    continue

                # ── Step 1: load & validate ────────────────────────────────────
                try:
                    person_img = step1_validate.load_image(person_dir / person_name)
                    cloth_img  = step1_validate.load_image(cloth_dir  / cloth_name)
                except Exception as exc:
                    log.warning("  ✗ Load failed (%s / %s): %s", person_name, cloth_name, exc)
                    _log_fail(failed_fh, stem_p, stem_c, f"load: {exc}")
                    stats["failed"] += 1
                    pbar.update(1)
                    pbar.set_postfix(**stats)
                    continue

                batch.append(
                    {
                        "person_img": person_img,
                        "cloth_img":  cloth_img,
                        "stem_p":     stem_p,
                        "stem_c":     stem_c,
                    }
                )

                # ── Flush when batch is full ───────────────────────────────────
                if len(batch) >= batch_size:
                    flush_batch(batch, dirs, stats, mask_workers, failed_fh)
                    batch.clear()

                pbar.update(1)
                pbar.set_postfix(**stats)

            # ── Flush the final partial batch ──────────────────────────────────
            if batch:
                flush_batch(batch, dirs, stats, mask_workers, failed_fh)

    finally:
        failed_fh.close()

    # Remove empty failed.txt to avoid clutter
    if fail_path.exists() and fail_path.stat().st_size == 0:
        fail_path.unlink()

    log.info(
        "[%s] Finished — done=%d  skipped=%d  failed=%d",
        split.upper(), stats["done"], stats["skipped"], stats["failed"],
    )
    if stats["failed"] > 0:
        log.info("  Failed pairs logged to: %s", fail_path)


# ─────────────────────────────── ENTRY POINT ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VITON-HD Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split", default="both", choices=["train", "test", "both"],
        help="Dataset split to preprocess.",
    )
    parser.add_argument(
        "--dataset", default=str(DATASET),
        help="Dataset root containing train/, test/, *_pairs.txt.",
    )
    parser.add_argument(
        "--batch", type=int, default=BATCH_SIZE,
        help="GPU batch size for SegFormer parsing. Use 16 on ≥12 GB VRAM.",
    )
    parser.add_argument(
        "--workers", type=int, default=MASK_WORKERS,
        help="Thread count for parallel cloth mask extraction (step 4).",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Process only first N pairs per split. 0 = all.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("=" * 60)
    log.info("VITON-HD Pipeline  |  512H × 384W  |  %s", device)
    log.info("=" * 60)

    dataset_root = Path(args.dataset)
    splits = ["train", "test"] if args.split == "both" else [args.split]

    for split in splits:
        process_split(split, dataset_root, args.batch, args.workers, args.limit)

    log.info("All splits complete.")


if __name__ == "__main__":
    main()
