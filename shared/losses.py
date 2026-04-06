"""
Shared loss functions for VITON training.

VGGLoss  — perceptual loss using VGG16 feature slices.
smooth_loss — second-order flow regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGGLoss(nn.Module):
    """
    Perceptual loss using VGG16 feature maps.

    Uses 4 slice points:
      relu1_2  (block 1, after 2nd conv)
      relu2_2  (block 2, after 2nd conv)
      relu3_3  (block 3, after 3rd conv)
      relu4_3  (block 4, after 3rd conv)

    Inputs are expected in [-1, 1]; they are re-normalised to ImageNet stats
    internally before being passed through VGG.
    """

    # VGG16 feature layer indices for each slice (0-indexed in features list)
    _SLICE_ENDS = [4, 9, 16, 23]   # relu1_2, relu2_2, relu3_3, relu4_3

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), device=None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = vgg.features

        # Build separate sequential blocks for each slice
        self.slices = nn.ModuleList()
        prev = 0
        for end in self._SLICE_ENDS:
            self.slices.append(nn.Sequential(*list(features.children())[prev:end]))
            prev = end

        # Freeze weights — we use VGG only for feature extraction
        for p in self.parameters():
            p.requires_grad_(False)

        self.weights = weights

        # ImageNet normalisation constants (mean and std for [0, 1] inputs)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def _normalise(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to ImageNet-normalised [0, 1]."""
        x = (x + 1.0) * 0.5          # [-1,1] → [0,1]
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 3, H, W) predicted image in [-1, 1]
            target: (B, 3, H, W) ground-truth image in [-1, 1]

        Returns:
            Scalar perceptual loss.
        """
        pred   = self._normalise(pred)
        target = self._normalise(target)

        loss = torch.tensor(0.0, device=pred.device)
        p_feat = pred
        t_feat = target
        for i, (slice_net, w) in enumerate(zip(self.slices, self.weights)):
            p_feat = slice_net(p_feat)
            t_feat = slice_net(t_feat)
            loss = loss + w * F.l1_loss(p_feat, t_feat.detach())

        return loss


def smooth_loss(flow: torch.Tensor) -> torch.Tensor:
    """
    Second-order flow regularization.

    Penalizes the Laplacian of the flow field (second-order derivatives),
    discouraging abrupt changes in flow direction without biasing toward zero.

    Args:
        flow: (B, 2, H, W) optical flow field

    Returns:
        Scalar regularization loss.
    """
    # Second-order finite differences via convolution with Laplacian kernel
    laplacian_kernel = torch.tensor(
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]],
        dtype=flow.dtype, device=flow.device
    ).view(1, 1, 3, 3)

    # Apply to each flow channel independently
    loss = torch.tensor(0.0, device=flow.device, dtype=flow.dtype)
    for c in range(flow.shape[1]):
        ch = flow[:, c:c+1, :, :]
        lap = F.conv2d(ch, laplacian_kernel, padding=1)
        loss = loss + lap.abs().mean()

    return loss / flow.shape[1]


# CIHP/parse-v3 labels for clothing region
CLOTH_LABELS = [5, 6, 7]   # upper-clothes=5, skirt=6, dress=7


def person_cloth_mask(parse_map: torch.Tensor) -> torch.Tensor:
    """Extract clothing region mask from parse map.

    Args:
        parse_map: (B, H, W) int64 parse label map

    Returns:
        (B, 1, H, W) float32 binary mask of clothing region.
    """
    mask = torch.zeros_like(parse_map, dtype=torch.float32)
    for lbl in CLOTH_LABELS:
        mask = mask + (parse_map == lbl).float()
    return mask.unsqueeze(1).clamp_(0, 1)
