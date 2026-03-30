"""
Step 6 — Normalize & Save Tensor Bundle
=========================================
Convert float32 [0, 255] images → torch tensors in [-1, 1].
Save all preprocessed data for one pair as a single .pt file.

Bundle contents saved per pair:
  person     : (3, 512, 384)  float32  [-1, 1]
  cloth      : (3, 512, 384)  float32  [-1, 1]
  agnostic   : (3, 512, 384)  float32  [-1, 1]
  parse_map  : (512, 384)     int64    label indices 0–17
  cloth_mask : (512, 384)     float32  0.0 / 1.0
  pose_map   : (18, 512, 384) float32  Gaussian heatmaps [0, 1]
"""

import torch
import numpy as np
from pathlib import Path


def image_to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert float32 RGB array to normalised tensor.

    Args:
        img: float32 (H, W, 3), range [0, 255].

    Returns:
        torch.Tensor (3, H, W), range [-1, 1].
    """
    t = torch.from_numpy(img).permute(2, 0, 1).float()   # (3, H, W)
    return (t / 127.5) - 1.0


def save_bundle(
    path: Path,
    person_img:  np.ndarray,
    cloth_img:   np.ndarray,
    agnostic:    np.ndarray,
    parse_map:   np.ndarray,
    cloth_mask:  np.ndarray,
    pose_map:    np.ndarray,
) -> None:
    """
    Save all preprocessed tensors for one (person, cloth) pair.

    Args:
        path:        Output .pt file path.
        person_img:  float32 (H, W, 3)  [0, 255]
        cloth_img:   float32 (H, W, 3)  [0, 255]
        agnostic:    float32 (H, W, 3)  [0, 255]
        parse_map:   uint8   (H, W)     label indices 0–17
        cloth_mask:  uint8   (H, W)     0 / 1
        pose_map:    float32 (18, H, W) [0, 1]
    """
    torch.save(
        {
            "person":     image_to_tensor(person_img),
            "cloth":      image_to_tensor(cloth_img),
            "agnostic":   image_to_tensor(agnostic),
            "parse_map":  torch.from_numpy(parse_map.astype(np.int64)),
            "cloth_mask": torch.from_numpy(cloth_mask.astype(np.float32)),
            "pose_map":   torch.from_numpy(pose_map),
        },
        path,
    )
