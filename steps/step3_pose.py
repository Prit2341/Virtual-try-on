"""
Step 3 — Pose Estimation → 18-channel Gaussian Heatmap
=======================================================
Model : MediaPipe Pose (complexity=2, static mode)

COCO-18 keypoints produced:
  0:nose  1:neck  2:R-shoulder  3:R-elbow  4:R-wrist
  5:L-shoulder  6:L-elbow  7:L-wrist  8:R-hip  9:R-knee  10:R-ankle
  11:L-hip  12:L-knee  13:L-ankle  14:R-eye  15:L-eye  16:R-ear  17:L-ear

Neck (idx 1) = midpoint of L/R shoulders (both must pass visibility threshold).

Improvements:
  • Visibility threshold — keypoints with visibility < MIN_VISIBILITY are
    skipped; their heatmap channels stay zero instead of placing a blob on
    an unreliable estimate (e.g. an occluded wrist behind the body).
  • Vectorised Gaussian — all 18 channels computed in one NumPy broadcast
    instead of 18 separate calls → ~10× faster heatmap generation.

Output: float32 array (18, H, W), each channel ∈ [0, 1]
"""

import numpy as np

HEIGHT         = 512
WIDTH          = 384
SIGMA          = 6          # Gaussian blob radius (pixels)
MIN_VISIBILITY = 0.5        # skip keypoints below this MediaPipe confidence

# MediaPipe landmark index → COCO-18 keypoint index
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
}   # neck (idx 1) = midpoint(right-shoulder, left-shoulder) — computed below

# ── Lazy-loaded singleton ──────────────────────────────────────────────────────
_pose = None


def _load():
    global _pose
    if _pose is None:
        import mediapipe as mp
        import logging
        logging.getLogger(__name__).info(
            "Loading MediaPipe Pose (complexity=2) …"
        )
        _pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
    return _pose


def _build_heatmap(kp_xy: dict) -> np.ndarray:
    """
    Vectorised Gaussian heatmap for all 18 channels at once.

    Builds all channels via a single (18, H, W) NumPy broadcast:
      dx² = (xs[None,None,:] - cx[:,None,None])²
      dy² = (ys[None,:,None] - cy[:,None,None])²
      heat = exp(-(dx²+dy²) / 2σ²)

    Channels without a visible keypoint (NaN centre) evaluate to NaN via
    the subtraction and are then zeroed out with np.nan_to_num.

    Args:
        kp_xy: dict  coco_idx → (cx_pixels, cy_pixels)

    Returns:
        float32 (18, H, W)
    """
    # Centre coords — NaN marks missing / low-visibility keypoints
    cxs = np.full(18, np.nan, dtype=np.float32)
    cys = np.full(18, np.nan, dtype=np.float32)
    for idx, (cx, cy) in kp_xy.items():
        cxs[idx] = cx
        cys[idx] = cy

    xs = np.arange(WIDTH,  dtype=np.float32)  # (W,)
    ys = np.arange(HEIGHT, dtype=np.float32)  # (H,)

    # Broadcast: (18,1,1) - (1,1,W) → (18,1,W);  (18,1,1) - (1,H,1) → (18,H,1)
    dx2 = (xs[None, None, :] - cxs[:, None, None]) ** 2   # (18, 1, W)
    dy2 = (ys[None, :, None] - cys[:, None, None]) ** 2   # (18, H, 1)
    heatmap = np.exp(-(dx2 + dy2) / (2.0 * SIGMA ** 2))   # (18, H, W)

    # NaN centres → NaN heatmap entries → replace with 0
    return np.nan_to_num(heatmap, nan=0.0).astype(np.float32)


# ── Public API ─────────────────────────────────────────────────────────────────

def run(img: np.ndarray) -> np.ndarray:
    """
    Detect body keypoints and produce 18-channel Gaussian heatmap.

    Keypoints with MediaPipe visibility < MIN_VISIBILITY are excluded so
    that occluded joints (e.g. back-facing wrist) do not corrupt the map.

    Args:
        img: float32 RGB (H, W, 3), range [0, 255].

    Returns:
        float32 array (18, H, W).
        Returns all-zeros heatmap if no person is detected.
    """
    pose   = _load()
    result = pose.process(img.astype(np.uint8))

    if result.pose_landmarks is None:
        return np.zeros((18, HEIGHT, WIDTH), dtype=np.float32)

    lm    = result.pose_landmarks.landmark
    kp_xy = {}   # coco_idx → (cx_pixels, cy_pixels)

    for mp_idx, coco_idx in MP_TO_COCO18.items():
        lmk = lm[mp_idx]
        if lmk.visibility >= MIN_VISIBILITY:
            kp_xy[coco_idx] = (lmk.x * WIDTH, lmk.y * HEIGHT)

    # Neck = midpoint of R-shoulder (2) and L-shoulder (5) — only if both visible
    if 2 in kp_xy and 5 in kp_xy:
        kp_xy[1] = (
            (kp_xy[2][0] + kp_xy[5][0]) / 2.0,
            (kp_xy[2][1] + kp_xy[5][1]) / 2.0,
        )

    return _build_heatmap(kp_xy)
