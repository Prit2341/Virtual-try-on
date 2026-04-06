"""
Self-Attention U-Net for virtual try-on.

Inserts a non-local self-attention block at the bottleneck (after e4)
to capture long-range spatial dependencies in both the warp and synthesis nets.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (same as WarpNet / TryOnNet)
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
# Self-Attention (non-local block)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Non-local self-attention block.

    Q/K/V projections → scaled dot-product attention → residual with
    learnable gamma parameter (initialised to 0 so it starts as identity).

    Args:
        channels  : number of input/output channels
        reduction : Q/K channel reduction factor (default 8)
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.query = nn.Conv2d(channels, mid, 1, bias=False)
        self.key   = nn.Conv2d(channels, mid, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = mid ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W

        Q = self.query(x).view(B, -1, N).permute(0, 2, 1)  # (B, N, mid)
        K = self.key(x).view(B, -1, N)                      # (B, mid, N)
        V = self.value(x).view(B, C, N)                     # (B, C, N)

        attn = torch.bmm(Q, K) * self.scale                 # (B, N, N)
        attn = torch.softmax(attn, dim=-1)

        out = torch.bmm(V, attn.permute(0, 2, 1))           # (B, C, N)
        out = out.view(B, C, H, W)

        return self.gamma * out + x


# ---------------------------------------------------------------------------
# AttentionWarpNet
# ---------------------------------------------------------------------------

class AttentionWarpNet(nn.Module):
    """
    WarpNet with self-attention at the bottleneck.

    Input  (25ch): agnostic(3) + pose(18) + cloth(3) + cloth_mask(1)
    Output (2ch):  flow field at H/2 resolution
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64, flow_scale: float = 0.8):
        super().__init__()
        self.flow_scale = flow_scale

        # Encoder
        self.e1 = ConvBlock(in_channels, ngf)
        self.e2 = ConvBlock(ngf,         ngf * 2)
        self.e3 = ConvBlock(ngf * 2,     ngf * 4)
        self.e4 = ConvBlock(ngf * 4,     ngf * 8)

        # Bottleneck self-attention
        self.attn = SelfAttention(ngf * 8)

        # Decoder
        self.d1 = UpBlock(ngf * 8, ngf * 4, ngf * 4)
        self.d2 = UpBlock(ngf * 4, ngf * 2, ngf * 2)
        self.d3 = UpBlock(ngf * 2, ngf,     ngf)

        # Flow head
        self.flow_head = nn.Conv2d(ngf, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)          # H/2
        e2 = self.e2(e1)         # H/4
        e3 = self.e3(e2)         # H/8
        e4 = self.e4(e3)         # H/16

        e4 = self.attn(e4)       # self-attention at bottleneck

        d1 = self.d1(e4, e3)     # H/8
        d2 = self.d2(d1, e2)     # H/4
        d3 = self.d3(d2, e1)     # H/2

        return torch.tanh(self.flow_head(d3)) * self.flow_scale


# ---------------------------------------------------------------------------
# AttentionTryOnNet
# ---------------------------------------------------------------------------

class AttentionTryOnNet(nn.Module):
    """
    TryOnNet with self-attention at the bottleneck.

    Input  (25ch): agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18)
    Output (3ch):  RGB image in [-1, 1]
    """

    def __init__(self, in_channels: int = 25, ngf: int = 64):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(in_channels, ngf)
        self.e2 = ConvBlock(ngf,         ngf * 2)
        self.e3 = ConvBlock(ngf * 2,     ngf * 4)
        self.e4 = ConvBlock(ngf * 4,     ngf * 8)

        # Bottleneck self-attention
        self.attn = SelfAttention(ngf * 8)

        # Decoder
        self.d1 = UpBlock(ngf * 8, ngf * 4, ngf * 4)
        self.d2 = UpBlock(ngf * 4, ngf * 2, ngf * 2)
        self.d3 = UpBlock(ngf * 2, ngf,     ngf)

        # Final upsample to full resolution
        self.up_final = nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1)
        self.out      = nn.Conv2d(ngf // 2, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)          # H/2
        e2 = self.e2(e1)         # H/4
        e3 = self.e3(e2)         # H/8
        e4 = self.e4(e3)         # H/16

        e4 = self.attn(e4)       # self-attention at bottleneck

        d1 = self.d1(e4, e3)     # H/8
        d2 = self.d2(d1, e2)     # H/4
        d3 = self.d3(d2, e1)     # H/2

        out = self.up_final(d3)  # H
        return torch.tanh(self.out(out))
