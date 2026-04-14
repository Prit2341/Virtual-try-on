"""
VITON V2 Pipeline — GMM (TPS Warp) + Composition TryOnNet
===========================================================
Key improvements over V1:
  1. GMM with TPS warp instead of dense optical flow
     - 25 control points → smooth, physically plausible deformation
     - Feature correlation for explicit cloth-person matching
     - No TV/flow regularization needed
  2. Composition-based TryOnNet
     - Predicts alpha mask + rendered person
     - Final = alpha * warped_cloth + (1-alpha) * rendered
     - Preserves warped cloth texture exactly in clothing region

Stage 1 (GMM):    Learns TPS warp — trained with L1 + VGG + mask loss
Stage 2 (TryOn):  Learns composition — trained with L1 + VGG + alpha regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.gmm_model import GMMNet
from model.tryon_model_v2 import TryOnNetV2


class VITONV2(nn.Module):
    """
    Full VITON V2 pipeline.

    Inputs:
      cloth      (B, 3, H, W)   — flat cloth image
      cloth_mask (B, 1, H, W)   — binary cloth mask
      agnostic   (B, 3, H, W)   — person with cloth region blanked
      pose       (B, 18, H, W)  — pose heatmaps

    Outputs:
      output        (B, 3, H, W)  — final try-on image
      warped_cloth  (B, 3, H, W)
      warped_mask   (B, 1, H, W)
      alpha         (B, 1, H, W)  — composition mask
      theta         (B, 2, G, G)  — TPS control-point offsets
    """

    def __init__(self, in_h=256, in_w=192, grid_size=5, ngf=64):
        super().__init__()
        self.gmm   = GMMNet(in_h=in_h, in_w=in_w, grid_size=grid_size, ngf=ngf)
        self.tryon = TryOnNetV2(in_channels=25, ngf=ngf)

    def forward(self, cloth, cloth_mask, agnostic, pose):
        # Stage 1: GMM — TPS warp
        warped_cloth, warped_mask, theta = self.gmm(cloth, cloth_mask, agnostic, pose)

        # Stage 2: TryOnNet — composition
        tryon_in = torch.cat([agnostic, warped_cloth, warped_mask, pose], dim=1)  # 25ch
        output, rendered, alpha = self.tryon(tryon_in, warped_cloth=warped_cloth)

        return output, warped_cloth, warped_mask, alpha, theta
