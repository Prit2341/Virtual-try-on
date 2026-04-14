import torch
import torch.nn as nn
from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth


class VITON(nn.Module):
    """
    Full VITON pipeline: WarpNet → warp cloth → TryOnNet → try-on image.

    Inputs:
      agnostic   (B, 3, H, W)  — person with cloth region masked out
      cloth      (B, 3, H, W)  — flat cloth image
      cloth_mask (B, 1, H, W)  — binary cloth mask
      pose       (B, 18, H, W) — pose heatmaps

    Outputs:
      output       (B, 3, H, W) — synthesized try-on image
      warped_cloth (B, 3, H, W)
      warped_mask  (B, 1, H, W)
      flow         (B, 2, H/2, W/2)
    """

    def __init__(self):
        super().__init__()
        self.warp  = WarpNet(in_channels=25)   # agnostic(3)+pose(18)+cloth(3)+mask(1)
        self.tryon = TryOnNet(in_channels=25)  # agnostic(3)+warped(3)+wmask(1)+pose(18)

    def forward(self, agnostic, cloth, cloth_mask, pose, parse_map=None):
        # Stage 1: predict flow and warp cloth
        warp_input  = torch.cat([agnostic, pose, cloth, cloth_mask], 1)  # 25ch
        flow        = self.warp(warp_input)
        warped_cloth = warp_cloth(cloth, flow)
        warped_mask  = warp_cloth(cloth_mask, flow)

        # Stage 2: synthesize try-on image (25ch — no parse onehot needed)
        tryon_input = torch.cat([agnostic, warped_cloth, warped_mask, pose], 1)
        output = self.tryon(tryon_input)

        return output, warped_cloth, warped_mask, flow
