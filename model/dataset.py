"""
model/dataset.py — PyTorch Dataset for VITON-HD
=================================================
Reads preprocessed .pt bundles produced by pipeline.py.

Each item returned by __getitem__:
  person          (3,  512, 384)  float32  [-1, 1]
  cloth           (3,  512, 384)  float32  [-1, 1]
  agnostic        (3,  512, 384)  float32  [-1, 1]
  pose_map        (18, 512, 384)  float32  [0, 1]
  cloth_mask      (1,  512, 384)  float32  {0, 1}
  parse_map       (512, 384)      int64    labels 0–17
  parse_one_hot   (18, 512, 384)  float32  one-hot encoding of parse_map
  cloth_region    (3,  512, 384)  float32  [-1,1]  ground-truth cloth on person
                                           (used as warp supervision target)
  stem_p          str             person image stem (for logging)
  stem_c          str             cloth image stem  (for logging)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from model.config import Config


class VITONDataset(Dataset):
    """
    Dataset that reads pair .pt bundles saved by pipeline.py.

    Args:
        split:        "train" or "test"
        dataset_root: path to dataset root (contains train/, test/)
        limit:        if > 0, only load the first N bundles (for quick tests)
    """

    def __init__(self, split: str = "train", dataset_root: Path = None, limit: int = 0):
        if dataset_root is None:
            dataset_root = Config.DATASET_ROOT

        self.tensor_dir = Path(dataset_root) / split / "tensors"
        if not self.tensor_dir.exists():
            raise FileNotFoundError(
                f"Tensor directory not found: {self.tensor_dir}\n"
                "Run pipeline.py first."
            )

        # Discover all .pt bundles
        self.files = sorted(self.tensor_dir.glob("*.pt"))
        if limit > 0:
            self.files = self.files[:limit]

        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {self.tensor_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path   = self.files[idx]
        bundle = torch.load(path, map_location="cpu", weights_only=True)

        # ── Unpack bundle ──────────────────────────────────────────────────────
        person    = bundle["person"]      # (3, H, W)
        cloth     = bundle["cloth"]       # (3, H, W)
        agnostic  = bundle["agnostic"]    # (3, H, W)
        parse_map = bundle["parse_map"]   # (H, W) int64
        cloth_mask = bundle["cloth_mask"].unsqueeze(0)  # (1, H, W)
        pose_map  = bundle["pose_map"]    # (18, H, W)

        # ── One-hot parse map → (18, H, W) float32 ────────────────────────────
        parse_one_hot = F.one_hot(parse_map, num_classes=Config.N_PARSE_CLASSES)
        parse_one_hot = parse_one_hot.permute(2, 0, 1).float()   # (18, H, W)

        # ── Ground-truth cloth region on person ────────────────────────────────
        # Extract the upper-clothes region from the person image using parse map.
        # This serves as the supervision target for WarpNet.
        # Labels 4 (upper-clothes) + 7 (dress) → cloth region mask
        upper_mask = (
            (parse_map == 4) | (parse_map == 7)
        ).float().unsqueeze(0)                                    # (1, H, W)
        cloth_region = person * upper_mask                        # (3, H, W)

        # ── Stem names ────────────────────────────────────────────────────────
        stem        = path.stem              # "{stem_p}__{stem_c}"
        parts       = stem.split("__")
        stem_p      = parts[0] if len(parts) >= 2 else stem
        stem_c      = parts[1] if len(parts) >= 2 else ""

        return {
            "person":        person,
            "cloth":         cloth,
            "agnostic":      agnostic,
            "pose_map":      pose_map,
            "cloth_mask":    cloth_mask,
            "parse_map":     parse_map,
            "parse_one_hot": parse_one_hot,
            "cloth_region":  cloth_region,
            "stem_p":        stem_p,
            "stem_c":        stem_c,
        }
