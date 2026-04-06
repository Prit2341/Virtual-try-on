# VITON Virtual Try-On — Multi-Architecture Benchmark

A PyTorch implementation comparing **seven** virtual try-on architectures on the VITON dataset. Given a person image and a target clothing item, each model generates a photorealistic try-on result at 256×192 resolution.

---

## 1. Project Overview

Virtual try-on requires two key capabilities:
1. **Geometric alignment** — warping the flat clothing image to fit the person's pose and body shape.
2. **Appearance synthesis** — compositing the warped cloth onto the person while preserving skin, hair, and background.

Most models in this project use a **2-stage pipeline**:
```
Stage 1 (Warp):  [agnostic | pose | cloth | cloth_mask] (25ch)  →  warped cloth
Stage 2 (Try-on): [agnostic | warped | warped_mask | pose] (25ch) →  RGB output
```

The `single_stage` model skips warping entirely and learns implicit alignment in one pass.

---

## 2. Models

| # | Model | Key Architectural Difference |
|---|-------|------------------------------|
| 1 | **baseline** | WarpNet (optical flow) + TryOnNet (4-level U-Net) |
| 2 | **v2** | GMMNet (TPS warp with 25 control points) + TryOnNetV2 (5-level U-Net + alpha blending) |
| 3 | **resnet_gen** | WarpNet + ResNet9 generator (no skip connections, 9 residual blocks) |
| 4 | **attention_unet** | Self-attention non-local block at bottleneck of both warp and synthesis nets |
| 5 | **single_stage** | Deep 5-level U-Net — no explicit warp stage, learns alignment implicitly |
| 6 | **spade** | WarpNet + SPADE decoder conditioned on pose map (spatially-adaptive normalisation) |
| 7 | **multiscale** | Coarse-to-fine: CoarseNet at 128×96, RefineNet upsamples to 256×192 |

### Architecture Details

**baseline** (`model/warp_model.py`, `model/tryon_model.py`)
- WarpNet: 4-block U-Net → 2ch optical flow field → `grid_sample` warp
- TryOnNet: 4-block U-Net encoder-decoder with skip connections

**v2** (`model/gmm_model.py`, `model/tryon_model_v2.py`)
- GMMNet: TPS warp via learned control-point offsets; smoothly-varying deformation
- TryOnNetV2: predicts rendered region + alpha mask; output = α·warped + (1−α)·rendered

**resnet_gen** (`models/resnet_gen/network.py`)
- ResNetGenerator: reflection-padded 7×7 head → 2× stride-2 downsampling → 9 ResBlocks → 2× stride-2 upsampling → Tanh

**attention_unet** (`models/attention_unet/network.py`)
- AttentionWarpNet / AttentionTryOnNet: identical to baseline but with a non-local self-attention block (gamma=0 init) inserted at the bottleneck (H/16)

**single_stage** (`models/single_stage/network.py`)
- SingleStageTryOn: 5-level U-Net (H → H/32 encoder, H/32 → H decoder), input is raw cloth — no pre-warping

**spade** (`models/spade/network.py`)
- SPADETryOnNet: Conv encoder + SPADEResBlock decoder; each normalisation layer receives the pose map to condition gamma/beta spatially

**multiscale** (`models/multiscale/network.py`)
- CoarseNet: WarpNet (ngf=32) + TryOnNet (ngf=32) at 128×96
- RefineNet: 3-level U-Net that takes coarse output + full-res warped cloth and upsamples to 256×192

---

## 3. Setup

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `torchvision`, `Pillow`, `tqdm`, `opencv-python`.

### Data Preparation

1. Download the VITON dataset and place images under `dataset/train/` and `dataset/test/`.
2. Run preprocessing to generate `.pt` tensor files:

```bash
python preprocess.py --split train
python preprocess.py --split test
```

Each `.pt` file contains:
| Key | Shape | Range |
|-----|-------|-------|
| `agnostic` | (3, 256, 192) | [-1, 1] |
| `cloth` | (3, 256, 192) | [-1, 1] |
| `cloth_mask` | (256, 192) | [0, 1] |
| `pose_map` | (18, 256, 192) | [-1, 1] |
| `person` | (3, 256, 192) | [-1, 1] |

Expected directory layout after preprocessing:
```
dataset/
  train/
    tensors/   ← .pt files
  test/
    tensors/
```

---

## 4. Training

All training scripts accept `--data`, `--epochs`, `--batch`, `--lr`, `--patience` flags.
Default: epochs=100, batch=8, lr=2e-4, patience=20.

### Baseline (V1)

```bash
python train.py --stage warp
python train.py --stage tryon
```

### V2 (GMM + composition)

```bash
python train_v2.py --stage gmm
python train_v2.py --stage tryon
```

### ResNet Generator

```bash
python models/resnet_gen/train.py --stage both
# or separately:
python models/resnet_gen/train.py --stage warp
python models/resnet_gen/train.py --stage tryon
```

### Attention U-Net

```bash
python models/attention_unet/train.py --stage both
```

### Single Stage

```bash
python models/single_stage/train.py
```

### SPADE

```bash
python models/spade/train.py --stage both
```

### Multiscale (Coarse-to-Fine)

```bash
python models/multiscale/train.py --stage both
# or separately:
python models/multiscale/train.py --stage coarse
python models/multiscale/train.py --stage refine
```

### Resuming Training

Each script saves the last 3 epoch checkpoints alongside `*_best.pth`.
Pass `--ckpt-dir` to specify a custom checkpoint directory.

---

## 5. Inference / Visual Results

Each model's `infer.py` saves a side-by-side image strip.

### Baseline

```bash
python infer.py --n 8
```

### V2

```bash
python infer_v2.py --n 8
```

### ResNet Generator

```bash
python models/resnet_gen/infer.py --n 8 --data dataset/test/tensors
# Saves: results/resnet_gen/results_strip.jpg
# Columns: person | cloth | agnostic | warped | output
```

### Attention U-Net

```bash
python models/attention_unet/infer.py --n 8
# Saves: results/attention_unet/results_strip.jpg
```

### Single Stage

```bash
python models/single_stage/infer.py --n 8
# Saves: results/single_stage/results_strip.jpg
# Columns: person | cloth | agnostic | output  (no warp column)
```

### SPADE

```bash
python models/spade/infer.py --n 8
# Saves: results/spade/results_strip.jpg
```

### Multiscale

```bash
python models/multiscale/infer.py --n 8
# Saves: results/multiscale/results_strip.jpg
# Columns: person | cloth | agnostic | coarse | refined
```

All scripts accept:
- `--n N` — number of test samples (default 8)
- `--data PATH` — path to `.pt` tensor directory
- `--save PATH` — output directory
- `--ckpt-dir PATH` — checkpoint directory

---

## 6. Comparison Across All Models

`compare_all.py` evaluates every model that has trained checkpoints, then produces:
- A terminal table of L1 / SSIM / PSNR metrics
- `results/comparison_summary.csv`
- `results/comparison_grid.jpg` — side-by-side grid for N test samples

```bash
# Evaluate all models
python compare_all.py

# Evaluate specific models only
python compare_all.py --models baseline resnet_gen spade

# Custom options
python compare_all.py --n 16 --split test --batch 8
```

### Metrics

| Metric | Description |
|--------|-------------|
| **L1** | Mean absolute pixel error (lower is better) |
| **SSIM** | Structural Similarity Index (higher is better, max 1.0) |
| **PSNR** | Peak Signal-to-Noise Ratio in dB (higher is better) |

All metrics are computed in the [-1, 1] range (PSNR uses max_range²=4.0).

---

## 7. Checkpoints Directory Structure

```
checkpoints/
  warp_best.pth          ← baseline WarpNet
  tryon_best.pth         ← baseline TryOnNet
  v2/
    gmm_best.pth
    tryon_best.pth
  resnet_gen/
    warp_best.pth
    resnet_gen_best.pth
    warp_epoch*.pth      ← last 3 epoch checkpoints
    resnet_gen_epoch*.pth
  attention_unet/
    warp_best.pth
    tryon_best.pth
    warp_epoch*.pth
    tryon_epoch*.pth
  single_stage/
    model_best.pth
    model_epoch*.pth
  spade/
    warp_best.pth
    tryon_best.pth
  multiscale/
    coarse_best.pth
    refine_best.pth
    coarse_epoch*.pth
    refine_epoch*.pth
```

### Log Files

```
logs/
  resnet_gen/    train_YYYYMMDD_HHMMSS.txt
  attention_unet/
  single_stage/
  spade/
  multiscale/
```

---

## Citation

If you use this codebase, please cite the relevant papers:

- Han et al., *"VITON: An Image-based Virtual Try-on Network"*, CVPR 2018
- Wang et al., *"Toward Characteristic-Preserving Image-based Virtual Try-On Network"*, ECCV 2018 (CP-VTON)
- Park et al., *"Semantic Image Synthesis with Spatially-Adaptive Normalization"*, CVPR 2019 (SPADE)
