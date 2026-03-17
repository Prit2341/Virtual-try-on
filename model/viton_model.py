import torch
import torch.nn as nn
from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth


class VITON(nn.Module):
    """
    Full VITON-HD pipeline: WarpNet → warp cloth → TryOnNet → try-on image.
    """

    def __init__(self):
        super().__init__()
        self.warp = WarpNet(in_channels=25)
        self.tryon = TryOnNet(in_channels=24)

    def forward(self, agnostic, cloth, cloth_mask, pose):
        # Stage 1: predict flow and warp cloth
        warp_input = torch.cat([agnostic, pose, cloth, cloth_mask], 1)  # 25ch
        flow = self.warp(warp_input)
        warped_cloth = warp_cloth(cloth, flow)

        # Stage 2: synthesize try-on image
        tryon_input = torch.cat([agnostic, warped_cloth, pose], 1)      # 24ch
        output = self.tryon(tryon_input)

        return output, warped_cloth, flow
