"""
Step 4 — Cloth Mask Extraction
================================
Model : U2Net cloth segmentation (via rembg)
Name  : u2net_cloth_seg

Extracts a binary mask separating the garment from the background,
then cleans it with morphological post-processing:
  1. Close small interior holes  (e.g. buttons, patterns creating gaps)
  2. Keep only the largest connected component  (drop stray noise islands)
  3. Dilate slightly to cover seam edges missed by U2Net

Low-contrast fix:
  Light-coloured garments on white/light backgrounds confuse U2Net.
  When the cloth image has a near-white background (mean corner brightness
  > BG_BRIGHT_THRESH), a neutral gray background is composited before
  inference to restore contrast, then removed after masking.

Output: uint8 binary mask (H, W) — 1 = cloth, 0 = background
"""

import threading

import cv2
import numpy as np
from PIL import Image

HEIGHT           = 512
WIDTH            = 384
MODEL_NAME       = "u2net_cloth_seg"
BG_BRIGHT_THRESH = 220   # corner mean brightness above this → swap to gray bg
BG_GRAY          = 127   # gray value used as replacement background

# ── Lazy-loaded singleton (thread-safe) ───────────────────────────────────────
_session      = None
_session_lock = threading.Lock()


def _load():
    global _session
    with _session_lock:
        if _session is None:
            from rembg import new_session
            import logging
            logging.getLogger(__name__).info("Loading rembg (%s) …", MODEL_NAME)
            _session = new_session(MODEL_NAME)
    return _session


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological cleanup pipeline for a binary uint8 mask.

    Steps:
      1. Close (dilate → erode) with a 15×15 ellipse kernel to fill small
         holes created by buttons, zips, and fabric patterns.
      2. Largest-connected-component filter: keep the biggest foreground
         blob and discard stray noise islands.
      3. Dilate by 3 px to cover seam edges that U2Net slightly under-segments.

    Args:
        mask: uint8 (H, W), values 0 or 1.

    Returns:
        Cleaned uint8 (H, W), values 0 or 1.
    """
    # 1. Close interior holes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 2. Keep largest connected component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        # stats row 0 = background; find the largest foreground label
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    # 3. Slight dilation to recover under-segmented seam edges
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.dilate(mask, k_dilate, iterations=1)

    return mask


def _add_gray_background(img: np.ndarray) -> np.ndarray:
    """
    Replace near-white background with neutral gray to improve U2Net contrast.

    Detects the background colour from the four image corners (each 10×10 px).
    If the mean brightness exceeds BG_BRIGHT_THRESH the entire near-white region
    (pixels brighter than BG_BRIGHT_THRESH in all channels) is filled with gray.

    Args:
        img: uint8 RGB (H, W, 3)

    Returns:
        uint8 RGB (H, W, 3) with background replaced, or original if not bright.
    """
    corners = np.concatenate([
        img[:10,  :10,  :].reshape(-1, 3),
        img[:10,  -10:, :].reshape(-1, 3),
        img[-10:, :10,  :].reshape(-1, 3),
        img[-10:, -10:, :].reshape(-1, 3),
    ])
    if corners.mean() < BG_BRIGHT_THRESH:
        return img   # dark or coloured background — leave as-is

    # Flood-fill from corners is fragile; use a simple threshold approach:
    # pixels where ALL channels are bright are treated as background
    bg_mask = np.all(img >= BG_BRIGHT_THRESH, axis=2)     # (H, W) bool
    result  = img.copy()
    result[bg_mask] = BG_GRAY
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def run(cloth_img: np.ndarray) -> np.ndarray:
    """
    Extract and clean a binary cloth mask using U2Net cloth segmentation.

    Args:
        cloth_img: float32 RGB (H, W, 3), range [0, 255].

    Returns:
        uint8 binary mask (H, W) — 1 = cloth, 0 = background.
    """
    from rembg import remove

    session  = _load()
    uint8    = cloth_img.astype(np.uint8)
    enhanced = _add_gray_background(uint8)              # fix low-contrast light garments
    pil      = Image.fromarray(enhanced)
    rgba     = remove(pil, session=session)             # → RGBA PIL image

    alpha = np.array(rgba)[:, :, 3]                    # raw alpha channel (H', W')
    alpha = cv2.resize(alpha, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

    mask  = (alpha > 127).astype(np.uint8)              # binarize
    mask  = _clean_mask(mask)                           # morphological cleanup

    return mask
