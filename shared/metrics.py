"""
Evaluation metrics for virtual try-on.

All functions expect tensors in [-1, 1] with shape (B, 3, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM) approximation.

    Uses avg_pool2d with kernel=11 and padding=5 to estimate local statistics,
    following the standard SSIM formulation.

    Args:
        pred:   (B, 3, H, W) in [-1, 1]
        target: (B, 3, H, W) in [-1, 1]

    Returns:
        Scalar mean SSIM across batch and channels.
    """
    C1 = 0.01 ** 2  # stability constant for luminance
    C2 = 0.03 ** 2  # stability constant for contrast

    kernel_size = 11
    padding     = 5

    # Compute local means using average pooling
    mu_p  = F.avg_pool2d(pred,   kernel_size, stride=1, padding=padding)
    mu_t  = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)

    mu_p2  = mu_p * mu_p
    mu_t2  = mu_t * mu_t
    mu_pt  = mu_p * mu_t

    # Local variances and covariance
    sigma_p2  = F.avg_pool2d(pred   * pred,   kernel_size, stride=1, padding=padding) - mu_p2
    sigma_t2  = F.avg_pool2d(target * target, kernel_size, stride=1, padding=padding) - mu_t2
    sigma_pt  = F.avg_pool2d(pred   * target, kernel_size, stride=1, padding=padding) - mu_pt

    numerator   = (2 * mu_pt  + C1) * (2 * sigma_pt  + C2)
    denominator = (mu_p2 + mu_t2 + C1) * (sigma_p2 + sigma_t2 + C2)

    ssim_map = numerator / (denominator + 1e-8)
    return ssim_map.mean()


def psnr_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio (PSNR).

    Inputs are in [-1, 1], so the dynamic range is 2.0 and max_range^2 = 4.0.

    Args:
        pred:   (B, 3, H, W) in [-1, 1]
        target: (B, 3, H, W) in [-1, 1]

    Returns:
        Scalar mean PSNR (dB) across batch.
    """
    max_range_sq = 4.0  # (1 - (-1))^2 = 4

    mse_per_image = ((pred - target) ** 2).mean(dim=(1, 2, 3))  # (B,)
    # Clamp to avoid log(0) when pred == target exactly
    mse_per_image = mse_per_image.clamp(min=1e-10)

    psnr_per_image = 10.0 * torch.log10(
        torch.tensor(max_range_sq, device=pred.device, dtype=pred.dtype) / mse_per_image
    )
    return psnr_per_image.mean()


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute all metrics at once. Returns dict with L1, SSIM, PSNR."""
    l1   = F.l1_loss(pred, target).item()
    ssim = ssim_metric(pred, target).item()
    psnr = psnr_metric(pred, target).item()
    return {"L1": l1, "SSIM": ssim, "PSNR": psnr}


def metrics_header() -> str:
    """Column header for epoch metrics table."""
    return (f"  {'Epoch':>5}  {'L1':>7}  {'VGG':>7}  {'SSIM':>6}  {'PSNR':>7}  "
            f"{'LR':>10}  {'Time':>8}  {'Best L1':>7}  Status")


def metrics_separator() -> str:
    return (f"  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*6}  {'─'*7}  "
            f"{'─'*10}  {'─'*8}  {'─'*7}  {'─'*20}")


def metrics_row(epoch, avg_l1, avg_vgg, ssim, psnr, lr, time_str, best_l1, status) -> str:
    return (f"  {epoch:>5}  {avg_l1:>7.4f}  {avg_vgg:>7.4f}  {ssim:>6.4f}  {psnr:>7.2f}  "
            f"  {lr:>8.2e}  {time_str:>8}  {best_l1:>7.4f}  {status}")
