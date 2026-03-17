import torch
import torch.nn as nn
from warp_model import WarpNet
from tryon_model import TryOnNet
from warp_utils import warp_cloth


class VITON(nn.Module):

    def __init__(self):
        super().__init__()

        self.warp = WarpNet()
        self.tryon = TryOnNet()

    def forward(self, person_inputs, cloth):

        flow = self.warp(person_inputs)

        warped_cloth = warp_cloth(cloth, flow)

        tryon_input = torch.cat([person_inputs, warped_cloth], 1)

        output = self.tryon(tryon_input)

        return output, warped_cloth