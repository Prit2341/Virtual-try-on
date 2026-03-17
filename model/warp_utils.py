import torch
import torch.nn.functional as F


def warp_cloth(cloth, flow):

    B, C, H, W = cloth.size()

    # resize flow to cloth resolution
    flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H),
        torch.linspace(-1, 1, W),
        indexing="ij"
    )

    grid = torch.stack((grid_x, grid_y), 2).to(cloth.device)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)

    flow = flow.permute(0, 2, 3, 1)

    new_grid = grid + flow

    warped = F.grid_sample(cloth, new_grid, padding_mode="border")

    return warped