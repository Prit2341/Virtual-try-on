import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.tryon = TryOnNet(in_channels=43)

    def forward(self, agnostic, cloth, cloth_mask, pose, parse_map):
        # Stage 1: predict flow and warp cloth
        warp_input = torch.cat([agnostic, pose, cloth, cloth_mask], 1)  # 25ch
        flow = self.warp(warp_input)
        warped_cloth = warp_cloth(cloth, flow)
        warped_mask = warp_cloth(cloth_mask, flow)

        # Build one-hot parse map (18 classes)
        parse_onehot = F.one_hot(parse_map.long(), 18).permute(0, 3, 1, 2).float()

        # Stage 2: synthesize try-on image (43ch)
        tryon_input = torch.cat([
            agnostic, warped_cloth, warped_mask, pose, parse_onehot
        ], 1)
        output = self.tryon(tryon_input)

        return output, warped_cloth, warped_mask, flow
