"""
VITON-HD network architectures adapted for 256×192.

Three networks following the VITON-HD paper (CVPR 2021):
  SegGenerator  — predicts target segmentation (7-class) from agnostic parse + pose + cloth
  GMM           — Geometric Matching Module: TPS warp via feature correlation
  ALIASGenerator — ALIAS (Alignment-Aware) synthesis with misalignment mask conditioning

Parse map label convention (18-class input from SegFormer):
  0=bg, 1=hat, 2=hair, 3=glasses, 4=upper-clothes, 5=skirt,
  6=pants, 7=dress, 8=belt, 9=left-shoe, 10=right-shoe,
  11=face, 12=left-leg, 13=right-leg, 14=left-arm, 15=right-arm,
  16=bag, 17=scarf

7-class merged output (used by ALIASGenerator):
  0=background, 1=paste(keep body/accessories), 2=upper-clothing,
  3=hair, 4=left-arm, 5=right-arm, 6=noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm
import numpy as np

# ---------------------------------------------------------------------------
# Parse label helpers
# ---------------------------------------------------------------------------

# Mapping: 18-class parse label → 7-class merged label
_LABEL_MAP_18_TO_7 = {
    0:  0,   # background  → background
    1:  1,   # hat         → paste (keep)
    2:  3,   # hair        → hair
    3:  1,   # glasses     → paste
    4:  2,   # upper-clothes → upper
    5:  1,   # skirt       → paste (lower body)
    6:  1,   # pants       → paste
    7:  2,   # dress       → upper
    8:  1,   # belt        → paste
    9:  1,   # left-shoe   → paste
    10: 1,   # right-shoe  → paste
    11: 1,   # face        → paste
    12: 1,   # left-leg    → paste
    13: 1,   # right-leg   → paste
    14: 4,   # left-arm    → left-arm
    15: 5,   # right-arm   → right-arm
    16: 1,   # bag         → paste
    17: 6,   # scarf       → noise
}

# Precompute as a lookup tensor (18,) — register as buffer where needed
_MAP_TENSOR = torch.tensor([_LABEL_MAP_18_TO_7[i] for i in range(18)], dtype=torch.long)

# Labels removed to make agnostic parse (upper-clothes, dress, scarf)
AGNOSTIC_REMOVE = [4, 7, 17]

N_PARSE   = 18   # raw parse classes
N_SEG     = 7    # merged classes for ALIAS conditioning
GRID_SIZE = 5    # TPS control-point grid


def remap_parse_18_to_7(parse_map: torch.Tensor) -> torch.Tensor:
    """
    Map integer parse map (B,H,W) or (H,W) with values 0-17 → 0-6.
    Values outside [0, 17] are clamped to background (0) before mapping.
    Works on CPU or CUDA.
    """
    lut = _MAP_TENSOR.to(parse_map.device)
    return lut[parse_map.long().clamp(0, N_PARSE - 1)]


def make_parse_agnostic_onehot(parse_map: torch.Tensor) -> torch.Tensor:
    """
    From integer parse map (B,H,W) produce one-hot (B, N_PARSE, H, W)
    with clothing labels (4, 7, 17) zeroed out.
    Values outside [0, 17] are clamped to 0 (background).
    """
    p = parse_map.long().clamp(0, N_PARSE - 1).clone()
    for lbl in AGNOSTIC_REMOVE:
        p[p == lbl] = 0
    # one-hot: (B,H,W) → (B,N_PARSE,H,W)
    B, H, W = p.shape
    oh = F.one_hot(p, N_PARSE).permute(0, 3, 1, 2).float()
    return oh


def parse_7_onehot(parse_map: torch.Tensor) -> torch.Tensor:
    """
    From integer parse map (B,H,W) produce one-hot (B, N_SEG, H, W)
    using the 7-class merged mapping.
    """
    p7 = remap_parse_18_to_7(parse_map)
    B, H, W = p7.shape
    return F.one_hot(p7, N_SEG).permute(0, 3, 1, 2).float()


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

def _norm(nc):
    return nn.InstanceNorm2d(nc)


class ConvNormRelu(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p),
            _norm(out_c),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# SegGenerator  (U-Net, 256×192)
# ---------------------------------------------------------------------------

class SegGenerator(nn.Module):
    """
    Predicts 7-class target segmentation.

    Input (41 ch):
        cloth_mask (1) + cloth_masked (3) + parse_agnostic_onehot (18)
        + pose_map (18) + noise (1)

    Output (7 ch): logits → sigmoid  (multi-label, not softmax)
    """

    def __init__(self, input_nc: int = 41, output_nc: int = N_SEG):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_nc, 64, 3, 1, 1), _norm(64), nn.ReLU(True),
            nn.Conv2d(64,       64, 3, 1, 1), _norm(64), nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,  128, 3, 1, 1), _norm(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1), _norm(128), nn.ReLU(True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), _norm(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), _norm(256), nn.ReLU(True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), _norm(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), _norm(512), nn.ReLU(True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1), _norm(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), _norm(512), nn.ReLU(True),
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(512, 256, 3, 1, 1), _norm(256), nn.ReLU(True))
        self.dec4 = nn.Sequential(
            nn.Conv2d(512+256, 256, 3, 1, 1), _norm(256), nn.ReLU(True),
            nn.Conv2d(256,     256, 3, 1, 1), _norm(256), nn.ReLU(True),
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(256, 128, 3, 1, 1), _norm(128), nn.ReLU(True))
        self.dec3 = nn.Sequential(
            nn.Conv2d(256+128, 128, 3, 1, 1), _norm(128), nn.ReLU(True),
            nn.Conv2d(128,     128, 3, 1, 1), _norm(128), nn.ReLU(True),
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(128, 64, 3, 1, 1), _norm(64), nn.ReLU(True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(128+64, 64, 3, 1, 1), _norm(64), nn.ReLU(True),
            nn.Conv2d(64,     64, 3, 1, 1), _norm(64), nn.ReLU(True),
        )

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                 nn.Conv2d(64, 64, 3, 1, 1), _norm(64), nn.ReLU(True))
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, 1, 1), _norm(64), nn.ReLU(True),
            nn.Conv2d(64,    output_nc, 3, 1, 1),
        )

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.conv1(x)                          # 256×192
        c2 = self.conv2(self.pool(c1))              # 128× 96
        c3 = self.conv3(self.pool(c2))              #  64× 48
        c4 = self.drop(self.conv4(self.pool(c3)))   #  32× 24
        c5 = self.drop(self.conv5(self.pool(c4)))   #  16× 12

        d4 = self.dec4(torch.cat([c4, self.up4(c5)], 1))  #  32× 24
        d3 = self.dec3(torch.cat([c3, self.up3(d4)], 1))  #  64× 48
        d2 = self.dec2(torch.cat([c2, self.up2(d3)], 1))  # 128× 96
        d1 = self.dec1(torch.cat([c1, self.up1(d2)], 1))  # 256×192
        return d1   # raw logits — use BCEWithLogitsLoss; apply sigmoid externally when needed


# ---------------------------------------------------------------------------
# GMM — Geometric Matching Module (TPS)
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """4 stride-2 conv layers: (H,W) → (H/16, W/16) with 512 channels."""

    def __init__(self, input_nc: int, ngf: int = 64):
        super().__init__()
        nf = ngf
        layers = [nn.Conv2d(input_nc, nf, 4, 2, 1), nn.ReLU(True), _norm(nf)]
        for _ in range(3):
            nf_prev, nf = nf, min(nf * 2, 512)
            layers += [nn.Conv2d(nf_prev, nf, 4, 2, 1), nn.ReLU(True), _norm(nf)]
        layers += [nn.Conv2d(nf, 512, 3, 1, 1), nn.ReLU(True), _norm(512)]
        layers += [nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        b, c, h, w = A.shape
        A = A.permute(0, 3, 2, 1).reshape(b, w * h, c)
        B = B.reshape(b, c, h * w)
        corr = torch.bmm(A, B).reshape(b, w * h, h, w)
        return corr


class FeatureRegression(nn.Module):
    """
    Regresses 2*GRID_SIZE^2 TPS control-point offsets.
    input_nc: channels of correlation map  = (H/16)*(W/16)
             for 256×192 → 16×12 → 192
    """

    def __init__(self, input_nc: int = 192, output_size: int = 2 * GRID_SIZE ** 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, 4, 2, 1), _norm(512), nn.ReLU(True),
            nn.Conv2d(512,      256, 4, 2, 1), _norm(256), nn.ReLU(True),
            nn.Conv2d(256,      128, 3, 1, 1), _norm(128), nn.ReLU(True),
            nn.Conv2d(128,       64, 3, 1, 1), _norm(64),  nn.ReLU(True),
        )
        # After 2 stride-2 on 16×12 → 4×3; 64 ch → 768 features
        self.linear = nn.Linear(64 * 4 * 3, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.linear(x.reshape(x.size(0), -1))
        return self.tanh(x)


class TpsGridGen(nn.Module):
    """
    Thin-Plate-Spline grid generator for a GRID_SIZE×GRID_SIZE control grid.
    Produces a sampling grid for F.grid_sample.
    """

    def __init__(self, out_h: int = 256, out_w: int = 192, dtype=torch.float):
        super().__init__()
        self.N = GRID_SIZE * GRID_SIZE

        gx, gy = np.meshgrid(np.linspace(-0.9, 0.9, out_w),
                              np.linspace(-0.9, 0.9, out_h))
        grid_X = torch.tensor(gx, dtype=dtype).unsqueeze(0).unsqueeze(3)
        grid_Y = torch.tensor(gy, dtype=dtype).unsqueeze(0).unsqueeze(3)

        coords = np.linspace(-0.9, 0.9, GRID_SIZE)
        P_Y, P_X = np.meshgrid(coords, coords)
        P_X = torch.tensor(P_X, dtype=dtype).reshape(self.N, 1)
        P_Y = torch.tensor(P_Y, dtype=dtype).reshape(self.N, 1)
        P_X_base, P_Y_base = P_X.clone(), P_Y.clone()

        Li = self._compute_L_inv(P_X, P_Y).unsqueeze(0)
        P_X = P_X.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)
        P_Y = P_Y.unsqueeze(2).unsqueeze(3).unsqueeze(4).transpose(0, 4)

        self.register_buffer('grid_X',   grid_X,   False)
        self.register_buffer('grid_Y',   grid_Y,   False)
        self.register_buffer('P_X_base', P_X_base, False)
        self.register_buffer('P_Y_base', P_Y_base, False)
        self.register_buffer('Li',       Li,        False)
        self.register_buffer('P_X',      P_X,       False)
        self.register_buffer('P_Y',      P_Y,       False)

    def _compute_L_inv(self, X, Y):
        N = X.size(0)
        Xm, Ym = X.expand(N, N), Y.expand(N, N)
        d2 = (Xm - Xm.t()).pow(2) + (Ym - Ym.t()).pow(2)
        d2[d2 == 0] = 1
        K = d2 * torch.log(d2)
        O = torch.ones(N, 1)
        Z = torch.zeros(3, 3)
        P = torch.cat([O, X, Y], 1)
        L = torch.cat([torch.cat([K, P], 1), torch.cat([P.t(), Z], 1)], 0)
        return torch.inverse(L)

    def _apply_tps(self, theta, points):
        # theta: (B, 2*N)
        B = theta.size(0)
        # Q_X/Q_Y: (B, N, 1) — control-point positions = base + offset
        pb_x = self.P_X_base.unsqueeze(0).expand(B, self.N, 1)  # (B,N,1)
        pb_y = self.P_Y_base.unsqueeze(0).expand(B, self.N, 1)  # (B,N,1)
        Q_X = theta[:, :self.N].unsqueeze(2) + pb_x             # (B,N,1)
        Q_Y = theta[:, self.N:].unsqueeze(2) + pb_y             # (B,N,1)

        ph, pw = points.size(1), points.size(2)
        P_X = self.P_X.expand(1, ph, pw, 1, self.N)
        P_Y = self.P_Y.expand(1, ph, pw, 1, self.N)

        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand(B, self.N, self.N), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand(B, self.N, self.N), Q_Y)
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)

        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand(B, 3, self.N), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand(B, 3, self.N), Q_Y)
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(1, 4).repeat(1, ph, pw, 1, 1)

        px = points[:, :, :, 0].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        py = points[:, :, :, 1].unsqueeze(3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if B == 1:
            dx, dy = px - P_X, py - P_Y
        else:
            dx = px - P_X.expand_as(px)
            dy = py - P_Y.expand_as(py)

        d2 = dx.pow(2) + dy.pow(2)
        d2[d2 == 0] = 1
        U = d2 * torch.log(d2)

        pxb = points[:, :, :, 0].unsqueeze(3)
        pyb = points[:, :, :, 1].unsqueeze(3)
        if B == 1:
            pxb = pxb.expand((B,) + pxb.size()[1:])
            pyb = pyb.expand((B,) + pyb.size()[1:])

        Xp = (A_X[:, :, :, :, 0] + A_X[:, :, :, :, 1] * pxb + A_X[:, :, :, :, 2] * pyb
              + (W_X * U.expand_as(W_X)).sum(4))
        Yp = (A_Y[:, :, :, :, 0] + A_Y[:, :, :, :, 1] * pxb + A_Y[:, :, :, :, 2] * pyb
              + (W_Y * U.expand_as(W_Y)).sum(4))
        return torch.cat([Xp, Yp], 3)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self._apply_tps(theta,
                               torch.cat([self.grid_X, self.grid_Y], 3))


class GMM(nn.Module):
    """
    Geometric Matching Module.

    inputA (22ch): seg_cloth (1) + pose (18) + agnostic (3)
    inputB (3ch) : cloth image
    Returns: theta (TPS params), warped_grid
    """

    def __init__(self, input_nc_A: int = 22, input_nc_B: int = 3):
        super().__init__()
        feat_h = 256 // 16   # = 16
        feat_w = 192 // 16   # = 12
        corr_nc = feat_h * feat_w  # = 192

        self.extractA = FeatureExtractor(input_nc_A)
        self.extractB = FeatureExtractor(input_nc_B)
        self.corr     = FeatureCorrelation()
        self.regress  = FeatureRegression(input_nc=corr_nc,
                                          output_size=2 * GRID_SIZE ** 2)
        self.grid_gen = TpsGridGen(out_h=256, out_w=192)

    def forward(self, inputA: torch.Tensor, inputB: torch.Tensor):
        fA = F.normalize(self.extractA(inputA), dim=1)
        fB = F.normalize(self.extractB(inputB), dim=1)
        corr = self.corr(fA, fB)
        theta = self.regress(corr)
        grid = self.grid_gen(theta)
        return theta, grid


# ---------------------------------------------------------------------------
# ALIASGenerator — ALIAS normalization + synthesis
# ---------------------------------------------------------------------------

class MaskNorm(nn.Module):
    def __init__(self, norm_nc: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_nc, affine=False)

    def _norm_region(self, region, mask):
        b, c, h, w = region.shape
        n = mask.sum((2, 3), keepdim=True).clamp(min=1)
        mu = region.sum((2, 3), keepdim=True) / n
        return self.norm(region + (1 - mask) * mu) * (n / (h * w)).sqrt()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.detach()
        return self._norm_region(x * mask, mask) + self._norm_region(x * (1 - mask), 1 - mask)


class ALIASNorm(nn.Module):
    """
    ALIAS normalization: InstanceNorm or MaskNorm + learned affine from seg map.
    norm_type: 'aliasinstance' or 'aliasmask'
    """

    def __init__(self, norm_type: str, norm_nc: int, label_nc: int):
        super().__init__()
        self.noise_scale = nn.Parameter(torch.zeros(norm_nc))

        assert norm_type.startswith('alias')
        base = norm_type[len('alias'):]
        if base == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif base == 'mask':
            self.param_free_norm = MaskNorm(norm_nc)
        else:
            raise ValueError(f"Unknown alias norm type: {base}")
        self._use_mask = (base == 'mask')

        nhidden = 128
        self.shared   = nn.Sequential(nn.Conv2d(label_nc, nhidden, 3, 1, 1), nn.ReLU(True))
        self.gamma_c  = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)
        self.beta_c   = nn.Conv2d(nhidden, norm_nc, 3, 1, 1)

    def forward(self, x, seg, misalign_mask=None):
        b, c, h, w = x.shape
        noise = (torch.randn(b, w, h, 1, device=x.device) * self.noise_scale).transpose(1, 3)
        if self._use_mask and misalign_mask is not None:
            normed = self.param_free_norm(x + noise, misalign_mask)
        else:
            normed = self.param_free_norm(x + noise)
        h_seg = self.shared(seg)
        return normed * (1 + self.gamma_c(h_seg)) + self.beta_c(h_seg)


class ALIASResBlock(nn.Module):
    def __init__(self, input_nc: int, output_nc: int,
                 label_nc: int, use_mask_norm: bool = True):
        super().__init__()
        self.learned_shortcut = (input_nc != output_nc)
        mid = min(input_nc, output_nc)

        self.conv0 = spectral_norm(nn.Conv2d(input_nc, mid,    3, 1, 1))
        self.conv1 = spectral_norm(nn.Conv2d(mid,      output_nc, 3, 1, 1))
        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(input_nc, output_nc, 1, bias=False))

        snorm = 'aliasmask' if use_mask_norm else 'aliasinstance'
        snc   = label_nc  # label_nc already includes misalign channel for mask blocks
        self.norm0 = ALIASNorm(snorm, input_nc, snc)
        self.norm1 = ALIASNorm(snorm, mid,   snc)
        if self.learned_shortcut:
            self.norm_s = ALIASNorm(snorm, input_nc, snc)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _shortcut(self, x, seg, mm):
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg, mm))
        return x

    def forward(self, x, seg, misalign_mask=None):
        seg = F.interpolate(seg, size=x.shape[2:], mode='nearest')
        if misalign_mask is not None:
            misalign_mask = F.interpolate(misalign_mask, size=x.shape[2:], mode='nearest')
        xs = self._shortcut(x, seg, misalign_mask)
        dx = self.conv0(self.relu(self.norm0(x,  seg, misalign_mask)))
        dx = self.conv1(self.relu(self.norm1(dx, seg, misalign_mask)))
        return xs + dx


class ALIASGenerator(nn.Module):
    """
    ALIAS generator for 256×192 (5 upsampling layers = 'normal' mode).

    Input (24ch): img_agnostic(3) + pose(18) + warped_cloth(3)
    seg      (7ch)  — full 7-class predicted segmentation
    seg_div  (8ch)  — seg (7) + misalign_mask (1)
    misalign_mask (1ch)

    Latent at 8×6 → upsample ×32 → 256×192
    """

    def __init__(self, input_nc: int = 24, ngf: int = 64,
                 seg_nc: int = N_SEG):
        super().__init__()
        nf = ngf
        self.sh, self.sw = 256 // 32, 192 // 32   # = 8, 6

        # Multi-scale input projections (6 scales: 8×6 → 256×192)
        self.conv_0 = nn.Conv2d(input_nc, nf * 16, 3, 1, 1)
        for i in range(1, 6):
            self.add_module(f'conv_{i}', nn.Conv2d(input_nc, 16, 3, 1, 1))

        # Decoder blocks — use seg_div (seg_nc+1) for first blocks, seg_nc for last
        self.head    = ALIASResBlock(nf * 16,      nf * 16, seg_nc + 1)
        self.mid0    = ALIASResBlock(nf * 16 + 16, nf * 16, seg_nc + 1)
        self.mid1    = ALIASResBlock(nf * 16 + 16, nf * 16, seg_nc + 1)
        self.up0     = ALIASResBlock(nf * 16 + 16, nf * 8,  seg_nc + 1)
        self.up1     = ALIASResBlock(nf * 8  + 16, nf * 4,  seg_nc + 1)
        self.up2     = ALIASResBlock(nf * 4  + 16, nf * 2,  seg_nc,     use_mask_norm=False)
        self.up3     = ALIASResBlock(nf * 2  + 16, nf * 1,  seg_nc,     use_mask_norm=False)

        self.out_conv = nn.Conv2d(nf, 3, 3, 1, 1)
        self.up  = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor,
                seg: torch.Tensor,
                seg_div: torch.Tensor,
                misalign_mask: torch.Tensor) -> torch.Tensor:
        # 6 scales: 8×6 → 16×12 → 32×24 → 64×48 → 128×96 → 256×192
        scales = [F.interpolate(x, size=(self.sh * 2**i, self.sw * 2**i), mode='nearest')
                  for i in range(6)]
        feats  = [self._modules[f'conv_{i}'](scales[i]) for i in range(6)]

        # latent = 8×6
        h = self.head(feats[0], seg_div, misalign_mask)

        h = self.up(h)                                                          # 16×12
        h = self.mid0(torch.cat([h, feats[1]], 1), seg_div, misalign_mask)

        h = self.up(h)                                                          # 32×24
        h = self.mid1(torch.cat([h, feats[2]], 1), seg_div, misalign_mask)

        h = self.up(h)                                                          # 64×48
        h = self.up0(torch.cat([h, feats[3]], 1), seg_div, misalign_mask)

        h = self.up(h)                                                          # 128×96
        h = self.up1(torch.cat([h, feats[4]], 1), seg_div, misalign_mask)

        h = self.up(h)                                                          # 256×192
        h = self.up2(torch.cat([h, feats[5]], 1), seg)

        # final refinement at 256×192 (reuse feats[5] — same spatial scale)
        h = self.up3(torch.cat([h, feats[5]], 1), seg)

        return torch.tanh(self.out_conv(self.relu(h)))
