"""
Step 5 — Agnostic Person Image
================================
Remove upper-body clothing from the person image.

The erase mask is dilated slightly before filling so that clothing
boundary seams (often 1–3 px outside the segmentation label) are also
removed.  The boundary between erased and kept regions is then feathered
with a Gaussian blur to avoid a sharp hard edge, which reduces grid-like
artefacts in the model's synthesised output.

Labels erased (SegFormer cloth classes):
  4  upper-clothes
  7  dress
  17 scarf

Constants:
  DILATE_PX     — kernel half-size for mask dilation (covers seam bleed)
  FEATHER_SIGMA — Gaussian sigma for soft edge blend (pixels)

Output: float32 RGB (H, W, 3), same range as input [0, 255]
"""

import cv2
import numpy as np

LABELS_REMOVE  = {4, 7, 17}
FILL_VALUE     = 128.0   # mid-gray fill for the erased region
DILATE_PX      = 5       # dilation radius in pixels to cover boundary seams
FEATHER_SIGMA  = 3       # Gaussian sigma for soft edge blending


def run(person_img: np.ndarray, parse_map: np.ndarray) -> np.ndarray:
    """
    Create the agnostic person image by removing clothing regions.

    Produces a soft-edge fill:
      1. Build binary erase mask from clothing labels.
      2. Dilate mask by DILATE_PX to cover label boundary seams.
      3. Gaussian-blur the dilated mask to get a weight map in [0, 1].
      4. Blend: out = person × (1 - weight) + gray × weight.

    Args:
        person_img: float32 RGB (H, W, 3), range [0, 255].
        parse_map:  uint8 label map (H, W), values 0–17.

    Returns:
        float32 RGB (H, W, 3) — clothing region replaced with feathered gray.
    """
    # Binary mask for clothing labels
    erase_mask = np.isin(parse_map, list(LABELS_REMOVE)).astype(np.uint8)

    # Dilate to fully cover seam edges missed by the segmentation boundary
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (DILATE_PX * 2 + 1, DILATE_PX * 2 + 1)
    )
    dilated = cv2.dilate(erase_mask, k, iterations=1)          # (H, W) uint8

    # Gaussian feathering → smooth blend weight in [0, 1]
    weight = cv2.GaussianBlur(
        dilated.astype(np.float32),
        ksize=(0, 0),
        sigmaX=FEATHER_SIGMA,
    )                                                           # (H, W) float32
    weight = weight[:, :, None]                                 # (H, W, 1) for broadcast

    fill     = np.full_like(person_img, FILL_VALUE)
    agnostic = person_img * (1.0 - weight) + fill * weight

    return agnostic.astype(np.float32)
