import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator (LSGAN).

    Input: RGB image (3, H, W)
    Output: patch prediction map — each value ∈ R classifies a ~70×70 receptive field.

    Used in Stage 2 (TryOnNet) to push generated images toward realism.
    """

    def __init__(self, in_channels=3, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            # Layer 1 — no InstanceNorm on first layer
            nn.Conv2d(in_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, True),

            # Layer 2
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            # Layer 4 — stride 1
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),

            # Output — 1-channel patch map
            nn.Conv2d(ndf * 8, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.model(x)
