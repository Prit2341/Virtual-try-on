"""
CP-VTON Geometric Matching Module (GMM)
========================================
Replaces the optical-flow WarpNet with a Thin Plate Spline (TPS) warp.

Instead of predicting a dense per-pixel flow field (unstable, chaotic),
GMM predicts displacements for a sparse grid of control points and lets
TPS interpolate a smooth, coherent warp for the entire cloth.

Architecture
------------
  ClothEncoder   (3+1 ch)  → cloth feature map at H/16
  PersonEncoder  (3+18 ch) → person feature map at H/16
  Correlation    → cross-correlation volume (grid_size^4 channels at H/16)
  RegressionHead → 2 * grid_size^2 TPS control-point offsets
  TPSGridGen     → dense (H, W, 2) sampling grid from control points
  grid_sample    → warped cloth / warped mask

Inputs
------
  cloth      (B, 3, H, W)   flat cloth image
  cloth_mask (B, 1, H, W)   binary cloth mask
  agnostic   (B, 3, H, W)   person with cloth region blanked
  pose       (B, 18, H, W)  OpenPose heatmaps

Outputs
-------
  warped_cloth  (B, 3, H, W)
  warped_mask   (B, 1, H, W)
  theta         (B, 2, grid_size, grid_size)  predicted control-point offsets

Reference: Han et al., "VITON: An Image-based Virtual Try-on Network" (CVPR 2018)
           Wang et al., "Toward Characteristic-Preserving Image-based Virtual Try-On
                         Network" (ECCV 2018)  [CP-VTON]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm → ReLU (stride-2 downsampling)."""

    def __init__(self, in_c: int, out_c: int, stride: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureExtractor(nn.Module):
    """
    Lightweight feature extractor — produces feature maps at H/16 resolution.

    Architecture: 4 × stride-2 ConvBnRelu blocks
      in_c → 64 → 128 → 256 → 512

    At 256×192 input, output is 16×12 with 512 channels.
    """

    def __init__(self, in_channels: int, ngf: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(in_channels, ngf),           # H/2
            ConvBnRelu(ngf,         ngf * 2),       # H/4
            ConvBnRelu(ngf * 2,     ngf * 4),       # H/8
            ConvBnRelu(ngf * 4,     ngf * 8),       # H/16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Correlation layer
# ---------------------------------------------------------------------------

class FeatureCorrelation(nn.Module):
    """
    Computes cross-correlation between cloth and person feature maps.

    For each spatial location (i, j) in the person feature map, computes
    the dot product with every location (k, l) in the cloth feature map.
    This gives an explicit matching cost volume.

    Output shape: (B, H*W, H, W)  — H, W are the feature map spatial dims.
    """

    def __init__(self):
        super().__init__()

    def forward(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """
        fa: person features  (B, C, H, W)
        fb: cloth  features  (B, C, H, W)
        returns correlation  (B, H*W, H, W)
        """
        B, C, H, W = fa.size()

        # Normalize for stable dot products
        fa = F.normalize(fa.view(B, C, -1), dim=1)  # (B, C, H*W)
        fb = F.normalize(fb.view(B, C, -1), dim=1)  # (B, C, H*W)

        # (B, H*W_person, H*W_cloth)
        corr = torch.bmm(fa.permute(0, 2, 1), fb)   # (B, H*W, H*W)
        corr = corr.view(B, H * W, H, W)            # (B, H*W, H, W)
        return corr


# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

class ThetaRegressor(nn.Module):
    """
    Regresses TPS control-point offsets from the correlation volume.

    Input:  (B, H*W, H_feat, W_feat) correlation volume
    Output: (B, 2 * grid_size^2) flattened control-point offsets
    """

    def __init__(self, feat_h: int, feat_w: int, grid_size: int = 5, ngf: int = 64):
        super().__init__()
        in_channels = feat_h * feat_w   # correlation volume channels

        self.conv = nn.Sequential(
            ConvBnRelu(in_channels, ngf * 4, stride=2),   # /2
            ConvBnRelu(ngf * 4,    ngf * 2, stride=2),   # /4
        )

        # Compute flattened size after the two downsampling convs
        dummy_h = feat_h // 4
        dummy_w = feat_w // 4
        flat = ngf * 2 * dummy_h * dummy_w

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2 * grid_size * grid_size),
        )
        self.grid_size = grid_size

    def forward(self, corr: torch.Tensor) -> torch.Tensor:
        x = self.conv(corr)
        theta = self.fc(x)
        return theta.view(-1, 2, self.grid_size, self.grid_size)


# ---------------------------------------------------------------------------
# TPS grid generator
# ---------------------------------------------------------------------------

class TPSGridGenerator(nn.Module):
    """
    Thin Plate Spline grid generator.

    Given predicted offsets `theta` for a regular grid of control points,
    computes a dense (H, W, 2) sampling grid for F.grid_sample.

    Control points are arranged uniformly in [-1, 1] × [-1, 1].
    The predicted theta shifts these points; TPS solves for the global warp.

    Args:
        out_h, out_w : output image resolution
        grid_size    : number of control points per dimension (default 5 → 25 pts)
    """

    def __init__(self, out_h: int, out_w: int, grid_size: int = 5):
        super().__init__()
        self.out_h     = out_h
        self.out_w     = out_w
        self.grid_size = grid_size
        self.num_ctrl  = grid_size * grid_size

        # Source control-point positions (regular grid in [-1, 1])
        xs = torch.linspace(-1, 1, grid_size)
        ys = torch.linspace(-1, 1, grid_size)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ctrl_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)  # (N, 2)
        self.register_buffer("ctrl_pts", ctrl_pts)   # (N, 2)

        # Target output grid (all pixel locations in [-1, 1])
        ty = torch.linspace(-1, 1, out_h)
        tx = torch.linspace(-1, 1, out_w)
        gy, gx = torch.meshgrid(ty, tx, indexing="ij")
        target_pts = torch.stack([gx.flatten(), gy.flatten()], dim=1)  # (H*W, 2)
        self.register_buffer("target_pts", target_pts)  # (H*W, 2)

        # Precompute the TPS basis for the output grid
        # (stays constant — only control-point offsets change per sample)
        P = self._build_tps_basis(target_pts, ctrl_pts)   # (H*W, N+3)
        self.register_buffer("P", P)

    # ------------------------------------------------------------------
    # TPS math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rbf(r2: torch.Tensor) -> torch.Tensor:
        """TPS radial basis function: r^2 * log(r^2), safe at r=0."""
        # Add small epsilon inside log to avoid -inf at r=0
        return r2 * torch.log(r2 + 1e-10)

    def _build_tps_basis(self, query_pts: torch.Tensor,
                         ctrl_pts: torch.Tensor) -> torch.Tensor:
        """
        Build the TPS evaluation matrix P for a set of query points.

        P_i = [1, x_i, y_i, phi(||x_i - c_1||^2), ..., phi(||x_i - c_N||^2)]
        Shape: (Q, N+3)
        """
        Q = query_pts.shape[0]
        N = ctrl_pts.shape[0]

        # ||query - ctrl||^2  (Q, N)
        diff = query_pts.unsqueeze(1) - ctrl_pts.unsqueeze(0)   # (Q, N, 2)
        r2   = (diff ** 2).sum(dim=2)                           # (Q, N)
        phi  = self._rbf(r2)                                    # (Q, N)

        # Affine part: [1, x, y]
        ones = torch.ones(Q, 1, device=query_pts.device, dtype=query_pts.dtype)
        affine = torch.cat([ones, query_pts], dim=1)            # (Q, 3)

        return torch.cat([affine, phi], dim=1)                  # (Q, N+3)

    def _solve_tps_weights(self, src: torch.Tensor,
                           dst: torch.Tensor) -> torch.Tensor:
        """
        Solve for TPS weights W given source and target control points.

        Solves: K * W = dst  where K is the (N+3) × (N+3) TPS kernel matrix.

        Returns W: (B, N+3, 2)
        """
        B, N, _ = src.shape
        device, dtype = src.device, src.dtype

        # Build kernel K (N, N)
        diff = src.unsqueeze(2) - src.unsqueeze(1)   # (B, N, N, 2)
        r2   = (diff ** 2).sum(dim=3)                # (B, N, N)
        K    = self._rbf(r2)                         # (B, N, N)

        # Affine constraints block: P = [1, x, y] (B, N, 3)
        ones = torch.ones(B, N, 1, device=device, dtype=dtype)
        P_blk = torch.cat([ones, src], dim=2)        # (B, N, 3)

        # Assemble full system matrix (B, N+3, N+3)
        top    = torch.cat([K,                    P_blk],          dim=2)  # (B, N, N+3)
        zeros3 = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        bottom = torch.cat([P_blk.permute(0,2,1), zeros3],         dim=2)  # (B, 3, N+3)
        system = torch.cat([top, bottom], dim=1)     # (B, N+3, N+3)

        # RHS: dst coordinates padded with zeros for the 3 constraint rows
        zeros_pad = torch.zeros(B, 3, 2, device=device, dtype=dtype)
        rhs = torch.cat([dst, zeros_pad], dim=1)     # (B, N+3, 2)

        # Solve with least-squares (more stable than torch.linalg.solve when
        # the matrix is near-singular for small deformations)
        W = torch.linalg.lstsq(system, rhs).solution  # (B, N+3, 2)
        return W

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, 2, grid_size, grid_size) — predicted control-point offsets

        Returns sampling grid (B, out_h, out_w, 2) for F.grid_sample.
        """
        B = theta.shape[0]
        device, dtype = theta.device, theta.dtype

        # Source control points (same for all batch items)
        src = self.ctrl_pts.unsqueeze(0).expand(B, -1, -1).to(dtype)  # (B, N, 2)

        # Target control points = source + predicted offset
        offsets = theta.view(B, 2, self.num_ctrl).permute(0, 2, 1)    # (B, N, 2)
        dst = src + offsets                                             # (B, N, 2)

        # Solve TPS weights
        W = self._solve_tps_weights(src, dst)   # (B, N+3, 2)

        # Evaluate TPS at all output pixel locations
        # P: (H*W, N+3)  →  warp: (H*W, 2)
        P = self.P.to(dtype)                    # (H*W, N+3)
        warp = P.matmul(W[0])                  # start with batch item 0

        # Batch over all items
        warp_list = [P.matmul(W[b]) for b in range(B)]
        warp_all  = torch.stack(warp_list, dim=0)           # (B, H*W, 2)

        grid = warp_all.view(B, self.out_h, self.out_w, 2)  # (B, H, W, 2)
        return grid


# ---------------------------------------------------------------------------
# Full GMM module
# ---------------------------------------------------------------------------

class GMMNet(nn.Module):
    """
    Geometric Matching Module — predicts TPS warp from cloth + person inputs.

    Replaces the optical-flow WarpNet. Key advantages:
    - Predicts sparse control points (25 pts) instead of dense per-pixel flow
    - TPS guarantees smooth, physically plausible deformations
    - No need for TV loss or flow regularization
    - Much more stable training

    Inputs
    ------
      cloth      (B, 3, H, W)
      cloth_mask (B, 1, H, W)
      agnostic   (B, 3, H, W)   person representation (cloth region blanked)
      pose       (B, 18, H, W)  OpenPose heatmaps

    Outputs
    -------
      warped_cloth (B, 3, H, W)
      warped_mask  (B, 1, H, W)
      theta        (B, 2, grid_size, grid_size)   control-point offsets (for loss)
    """

    def __init__(
        self,
        in_h: int      = 256,
        in_w: int      = 192,
        grid_size: int = 5,
        ngf: int       = 64,
    ):
        super().__init__()
        self.grid_size = grid_size

        # Separate feature extractors for cloth and person
        self.cloth_encoder  = FeatureExtractor(in_channels=4,  ngf=ngf)   # cloth(3)+mask(1)
        self.person_encoder = FeatureExtractor(in_channels=21, ngf=ngf)   # agnostic(3)+pose(18)

        # Feature map size at H/16
        feat_h = in_h // 16
        feat_w = in_w // 16

        self.correlation   = FeatureCorrelation()
        self.regressor     = ThetaRegressor(feat_h, feat_w, grid_size=grid_size, ngf=ngf)
        self.tps_generator = TPSGridGenerator(in_h, in_w, grid_size=grid_size)

    def forward(
        self,
        cloth:      torch.Tensor,   # (B, 3, H, W)
        cloth_mask: torch.Tensor,   # (B, 1, H, W)
        agnostic:   torch.Tensor,   # (B, 3, H, W)
        pose:       torch.Tensor,   # (B, 18, H, W)
    ):
        # 1. Extract features
        cloth_feat  = self.cloth_encoder(torch.cat([cloth, cloth_mask], dim=1))
        person_feat = self.person_encoder(torch.cat([agnostic, pose],   dim=1))

        # 2. Compute correlation
        corr = self.correlation(person_feat, cloth_feat)   # (B, H*W, H, W)

        # 3. Regress TPS control-point offsets
        theta = self.regressor(corr)                       # (B, 2, grid_size, grid_size)

        # 4. Generate dense warp grid via TPS
        grid = self.tps_generator(theta)                   # (B, H, W, 2)

        # 5. Warp cloth and mask
        warped_cloth = F.grid_sample(cloth,      grid, padding_mode="border", align_corners=True)
        warped_mask  = F.grid_sample(cloth_mask, grid, padding_mode="zeros",  align_corners=True)

        return warped_cloth, warped_mask, theta
