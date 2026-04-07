"""
Coarse-to-fine multiscale virtual try-on network.

CoarseNet  — runs WarpNet + TryOnNet at half resolution (128×96).
RefineNet  — 3-level U-Net that refines the coarse output at full resolution.

The coarse stage gives a rough spatial layout; the refine stage adds detail.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.warp_model import WarpNet
from model.tryon_model import TryOnNet
from model.warp_utils import warp_cloth


# ---------------------------------------------------------------------------
# Building blocks for RefineNet
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → conv."""

    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_c, in_c, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# CoarseNet
# ---------------------------------------------------------------------------

class CoarseNet(nn.Module):
    """
    Coarse try-on network at half resolution (128×96).

    Contains a WarpNet and TryOnNet, both with reduced ngf for efficiency.
    Inputs are expected at HALF resolution (128×96); caller must downsample.

    Args:
        ngf : base feature channels (default 32 — lightweight for coarse stage)
    """

    def __init__(self, ngf: int = 32):
        super().__init__()
        self.warp  = WarpNet(in_channels=25, ngf=ngf, flow_scale=0.25)
        self.tryon = TryOnNet(in_channels=25, ngf=ngf)

    def forward(
        self,
        ag:   torch.Tensor,   # (B, 3,  H/2, W/2)
        cl:   torch.Tensor,   # (B, 3,  H/2, W/2)
        cm:   torch.Tensor,   # (B, 1,  H/2, W/2)
        pose: torch.Tensor,   # (B, 18, H/2, W/2)
    ) -> tuple:
        """
        Returns:
            out     : (B, 3,  H/2, W/2)  coarse try-on output
            warped  : (B, 3,  H/2, W/2)  warped cloth at half resolution
            wm      : (B, 1,  H/2, W/2)  warped cloth mask at half resolution
        """
        warp_inp = torch.cat([ag, pose, cl, cm], dim=1)  # 25ch
        flow     = self.warp(warp_inp)
        warped   = warp_cloth(cl, flow)
        wm       = warp_cloth(cm, flow)
        warped   = warped * wm                            # mask-composite

        tryon_inp = torch.cat([ag, warped, wm, pose], dim=1)  # 25ch
        out       = self.tryon(tryon_inp)

        return out, warped, wm


# ---------------------------------------------------------------------------
# RefineNet
# ---------------------------------------------------------------------------

class RefineNet(nn.Module):
    """
    Refinement network — takes coarse output and full-resolution warped cloth,
    outputs a sharper full-resolution try-on image.

    Input channels (28):
        agnostic(3) + warped_full(3) + warped_mask_full(1) + coarse_up(3) + pose(18) = 28

    Architecture: 3-level U-Net encoder-decoder.
    """

    def __init__(self, in_channels: int = 28, ngf: int = 64):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(in_channels, ngf)        # H/2
        self.e2 = ConvBlock(ngf,         ngf * 2)    # H/4
        self.e3 = ConvBlock(ngf * 2,     ngf * 4)    # H/8

        # Decoder
        self.d1 = UpBlock(ngf * 4, ngf * 2, ngf * 2)  # H/4 — skip=e2
        self.d2 = UpBlock(ngf * 2, ngf,     ngf)       # H/2 — skip=e1

        # Final upsample H/2 → H
        self.up_final = nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1)
        self.out      = nn.Conv2d(ngf // 2, 3, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)          # H/2
        e2 = self.e2(e1)         # H/4
        e3 = self.e3(e2)         # H/8

        d1 = self.d1(e3, e2)     # H/4
        d2 = self.d2(d1, e1)     # H/2

        out = self.up_final(d2)  # H
        return torch.tanh(self.out(out))
