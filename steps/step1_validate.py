"""
Step 1 — Input Validation & Resize
===================================
Load person/cloth image, ensure RGB, resize to target resolution.

Handles:
  • BGR images      (OpenCV default)
  • RGBA PNG        → composite on white background (common for product photos)
  • Grayscale       → broadcast to 3-channel RGB

Output: float32 numpy array (H, W, 3), range [0, 255]
"""

import cv2
import numpy as np
from pathlib import Path

HEIGHT = 512
WIDTH  = 384


def load_image(path: Path) -> np.ndarray:
    """
    Load image from disk, normalise channels, resize to (HEIGHT × WIDTH).

    Args:
        path: Path to .jpg / .png image file.

    Returns:
        float32 RGB array, shape (512, 384, 3), range [0, 255].

    Raises:
        FileNotFoundError: if the file cannot be read.
        ValueError: if the image has an unsupported channel layout.
    """
    # IMREAD_UNCHANGED preserves alpha channel in RGBA PNGs
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    # ── Channel normalisation ─────────────────────────────────────────────────
    if img.ndim == 2:
        # Grayscale → RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif img.shape[2] == 4:
        # RGBA → composite on white background
        bgr   = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0   # (H, W, 1) in [0,1]
        white = np.full_like(bgr, 255.0)
        img   = (alpha * bgr + (1.0 - alpha) * white).astype(np.uint8)
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        raise ValueError(f"Unsupported channel count {img.shape[2]}: {path}")

    # ── Resize ───────────────────────────────────────────────────────────────
    img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LANCZOS4)

    return img.astype(np.float32)
