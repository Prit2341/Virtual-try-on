import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → conv to reduce channels."""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c, 4, 2, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c + skip_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], 1)
        return self.conv(x)


class SelfAttention(nn.Module):
    """Lightweight self-attention for bottleneck feature maps."""

    def __init__(self, channels):
        super().__init__()
        mid = max(channels // 8, 1)
        self.query = nn.Conv2d(channels, mid, 1)
        self.key = nn.Conv2d(channels, mid, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, mid)
        k = self.key(x).view(B, -1, H * W)                       # (B, mid, HW)
        attn = F.softmax(torch.bmm(q, k), dim=-1)                # (B, HW, HW)
        v = self.value(x).view(B, -1, H * W)                     # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


class WarpNet(nn.Module):
    """
    Predicts a 2-channel flow field to warp flat cloth onto the person.

    Input (25ch): agnostic(3) + pose(18) + cloth(3) + cloth_mask(1)
    Output:       flow (2, H/2, W/2) — upsampled in warp_cloth()

    Improvements over v1:
      - 5 encoder blocks (deeper)
      - Concat skip connections (preserves detail)
      - Self-attention at bottleneck
    """

    def __init__(self, in_channels=25, ngf=64):
        super().__init__()
        self.use_checkpointing = False

        # Encoder  (H → H/2 → H/4 → H/8 → H/16 → H/32)
        self.e1 = ConvBlock(in_channels, ngf)       # 64
        self.e2 = ConvBlock(ngf, ngf * 2)            # 128
        self.e3 = ConvBlock(ngf * 2, ngf * 4)        # 256
        self.e4 = ConvBlock(ngf * 4, ngf * 8)        # 512
        self.e5 = ConvBlock(ngf * 8, ngf * 8)        # 512  (bottleneck)

        # Self-attention at bottleneck (small spatial size → cheap)
        self.attn = SelfAttention(ngf * 8)

        # Decoder with concat skip connections
        self.d1 = UpBlock(ngf * 8, ngf * 8, ngf * 8)   # + e4 → 512
        self.d2 = UpBlock(ngf * 8, ngf * 4, ngf * 4)   # + e3 → 256
        self.d3 = UpBlock(ngf * 4, ngf * 2, ngf * 2)   # + e2 → 128
        self.d4 = UpBlock(ngf * 2, ngf, ngf)            # + e1 → 64

        # Flow head — output at H/2 resolution
        self.flow = nn.Conv2d(ngf, 2, 3, padding=1)

    def enable_gradient_checkpointing(self):
        self.use_checkpointing = True

    def forward(self, x):
        if self.use_checkpointing and self.training:
            e1 = checkpoint(self.e1, x, use_reentrant=False)
            e2 = checkpoint(self.e2, e1, use_reentrant=False)
            e3 = checkpoint(self.e3, e2, use_reentrant=False)
            e4 = checkpoint(self.e4, e3, use_reentrant=False)
            e5 = checkpoint(self.e5, e4, use_reentrant=False)
        else:
            e1 = self.e1(x)     # H/2
            e2 = self.e2(e1)    # H/4
            e3 = self.e3(e2)    # H/8
            e4 = self.e4(e3)    # H/16
            e5 = self.e5(e4)    # H/32

        e5 = self.attn(e5)

        d1 = self.d1(e5, e4)  # H/16
        d2 = self.d2(d1, e3)  # H/8
        d3 = self.d3(d2, e2)  # H/4
        d4 = self.d4(d3, e1)  # H/2

        return self.flow(d4)
