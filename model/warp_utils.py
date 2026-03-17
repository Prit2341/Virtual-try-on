import torch
import torch.nn.functional as F


def warp_cloth(cloth, flow):
    """
    Warp cloth image using a predicted flow field.

    Args:
        cloth: (B, C, H, W) cloth image
        flow:  (B, 2, h, w) flow field (any resolution — upsampled to match cloth)

    Returns:
        (B, C, H, W) warped cloth
    """
    B, C, H, W = cloth.size()

    # Upsample flow to cloth resolution
    flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=cloth.device),
        torch.linspace(-1, 1, W, device=cloth.device),
        indexing="ij",
    )

    grid = torch.stack((grid_x, grid_y), 2)         # (H, W, 2)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)

    flow = flow.permute(0, 2, 3, 1)                 # (B, H, W, 2)
    new_grid = grid + flow

    warped = F.grid_sample(cloth, new_grid, padding_mode="border", align_corners=True)
    return warped
