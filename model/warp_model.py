import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class WarpNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(21, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)

        # Decoder
        self.d1 = DeconvBlock(512, 256)
        self.d2 = DeconvBlock(256, 128)
        self.d3 = DeconvBlock(128, 64)

        # flow prediction
        self.flow = nn.Conv2d(64, 2, 3, padding=1)

    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        d1 = self.d1(e4)
        d2 = self.d2(d1)
        d3 = self.d3(d2)

        flow = self.flow(d3)

        return flow