"""
SPADE (Spatially Adaptive Denormalization) synthesis network for virtual try-on.

SPADE conditions each decoder layer on a spatially-varying pose/segmentation map,
enabling fine-grained spatial control over the generated image.

Reference: Park et al., "Semantic Image Synthesis with Spatially-Adaptive
           Normalization" (CVPR 2019).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SPADE normalization layer
# ---------------------------------------------------------------------------

class SPADE(nn.Module):
    """
    Spatially Adaptive Denormalization.

    Learns per-pixel affine parameters (gamma, beta) from a conditioning
    segmentation/pose map, applied after InstanceNorm.

    Args:
        norm_nc  : number of channels of the feature map to normalise
        label_nc : number of channels of the conditioning map (default 18 for pose)
        nhidden  : hidden channels in the shared MLP (default 128)
    """

    def __init__(self, norm_nc: int, label_nc: int = 18, nhidden: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma_conv = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.beta_conv  = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Resize seg to match x's spatial dimensions
        seg_resized = F.interpolate(seg, size=x.shape[2:], mode="bilinear",
                                    align_corners=True)
        normalized = self.norm(x)
        h     = self.shared_mlp(seg_resized)
        gamma = self.gamma_conv(h)
        beta  = self.beta_conv(h)
        return normalized * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# SPADE Residual Block
# ---------------------------------------------------------------------------

class SPADEResBlock(nn.Module):
    """
    Residual block using SPADE normalisation before each conv.

    Pattern per conv: SPADE → LeakyReLU(0.2) → Conv2d

    Args:
        fin      : input channels
        fout     : output channels
        label_nc : channels of conditioning map passed to SPADE
    """

    def __init__(self, fin: int, fout: int, label_nc: int = 18):
        super().__init__()
        fmid = min(fin, fout)

        # Main path
        self.spade1 = SPADE(fin,  label_nc)
        self.conv1  = nn.Conv2d(fin,  fmid, 3, 1, 1)

        self.spade2 = SPADE(fmid, label_nc)
        self.conv2  = nn.Conv2d(fmid, fout, 3, 1, 1)

        # Shortcut (only needed when fin != fout)
        self.learned_shortcut = (fin != fout)
        if self.learned_shortcut:
            self.spade_s = SPADE(fin, label_nc)
            self.conv_s  = nn.Conv2d(fin, fout, 1, bias=False)

        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def _shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.spade_s(x, seg))
        return x

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        x_s = self._shortcut(x, seg)
        dx  = self.conv1(self.lrelu(self.spade1(x, seg)))
        dx  = self.conv2(self.lrelu(self.spade2(dx, seg)))
        return x_s + dx


# ---------------------------------------------------------------------------
# SPADETryOnNet
# ---------------------------------------------------------------------------

class SPADETryOnNet(nn.Module):
    """
    SPADE-based synthesis network for virtual try-on.

    Encoder: standard Conv + InstanceNorm + LeakyReLU (no SPADE).
    Decoder: SPADEResBlocks conditioned on pose map.

    Args:
        in_channels : input channels (default 25 for tryon_inp = ag+warped+wm+pose)
        ngf         : base feature channels (default 64)
        label_nc    : conditioning channels passed to SPADE (pose = 18)

    Inputs to forward():
        x   : (B, in_channels, H, W)  concatenated tryon input
        seg : (B, label_nc, H, W)     pose map (conditioning signal)

    Output: (B, 3, H, W) RGB in [-1, 1]
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64, label_nc: int = 18):
        super().__init__()
        lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Encoder — 4 stride-2 conv blocks
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, ngf,     4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf),
            lrelu,
        )  # H/2
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf,     ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  # H/4
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # H/8
        self.enc4 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # H/16

        # Decoder — SPADE ResBlocks + nearest-neighbour upsample
        self.d4 = SPADEResBlock(ngf * 8, ngf * 8, label_nc)
        self.up4 = nn.Upsample(scale_factor=2, mode="nearest")   # H/8

        self.d3 = SPADEResBlock(ngf * 8, ngf * 4, label_nc)
        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")   # H/4

        self.d2 = SPADEResBlock(ngf * 4, ngf * 2, label_nc)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")   # H/2

        self.d1 = SPADEResBlock(ngf * 2, ngf, label_nc)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")   # H

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)   # H/2
        e2 = self.enc2(e1)  # H/4
        e3 = self.enc3(e2)  # H/8
        e4 = self.enc4(e3)  # H/16

        # Decode with SPADE conditioning
        out = self.d4(e4, seg)
        out = self.up4(out)    # H/8

        out = self.d3(out, seg)
        out = self.up3(out)    # H/4

        out = self.d2(out, seg)
        out = self.up2(out)    # H/2

        out = self.d1(out, seg)
        out = self.up1(out)    # H

        return self.out_conv(out)
