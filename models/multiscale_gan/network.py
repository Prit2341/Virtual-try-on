"""
Multiscale GAN — Best-Performing Multiscale Model + PatchGAN Adversarial Training
==================================================================================
Takes the existing CoarseNet + RefineNet (the top-performing model at SSIM 0.9179)
and adds:
  1. PatchGAN discriminator at the refine stage
  2. Feature matching loss for GAN stability
  3. Perceptual loss with higher VGG weight

Expected improvement: push SSIM from 0.9179 → 0.93+ by enforcing high-frequency
texture realism through adversarial supervision.

Architecture
------------
  CoarseNet (unchanged)  — WarpNet + TryOnNet at 128×96
  RefineNet  (unchanged) — 3-level U-Net at 256×192
  PatchGAN               — 70×70 discriminator on RefineNet output
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse proven CoarseNet and RefineNet unchanged
from models.multiscale.network import CoarseNet, RefineNet  # noqa: F401


# ---------------------------------------------------------------------------
# PatchGAN Discriminator (same as cp_viton but standalone for clarity)
# ---------------------------------------------------------------------------

class MultiscalePatchGAN(nn.Module):
    """
    Multi-scale PatchGAN discriminator.

    Evaluates realism at two scales:
      - Full resolution  (256×192)
      - Half resolution  (128×96)

    Taking the average of both scale outputs gives more robust gradient signal
    at fine details (full) and global structure (half).

    Input: refined image (3ch) + condition (agnostic + coarse_upsampled = 6ch) = 9ch
    """

    def __init__(self, in_channels: int = 9, ndf: int = 64):
        super().__init__()
        self.disc_full = self._make_disc(in_channels, ndf)
        self.disc_half = self._make_disc(in_channels, ndf)

    @staticmethod
    def _make_disc(in_channels: int, ndf: int) -> nn.Sequential:
        """Build a single 70×70 PatchGAN discriminator."""
        return nn.Sequential(
            # No norm on first layer
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(ndf,       ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(ndf * 2,   ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Stride-1 conv + output
            nn.Conv2d(ndf * 4,   ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8,   1,        4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns predictions at both scales.
        x: (B, 9, H, W)
        """
        pred_full = self.disc_full(x)
        x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
        pred_half = self.disc_half(x_half)
        return pred_full, pred_half


class MultiScaleDiscriminatorWithFeatures(nn.Module):
    """
    Same PatchGAN but returns intermediate feature maps for feature matching loss.

    Feature matching: match discriminator intermediate activations between
    real and fake samples — prevents mode collapse and stabilises training.
    """

    def __init__(self, in_channels: int = 9, ndf: int = 64):
        super().__init__()
        # Split into 4 feature levels for feature matching
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf,     ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.output = nn.Conv2d(ndf * 8, 1, 4, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns (prediction, [feat1, feat2, feat3, feat4]) for feature matching.
        """
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        pred = self.output(f4)
        return pred, [f1, f2, f3, f4]
