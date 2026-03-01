# VITON-HD — Virtual Try-On

A PyTorch implementation of a 2-stage virtual try-on system based on VITON-HD. Given a person image and a clothing item, the model generates a realistic try-on result.

---

## Architecture

Two sequential networks trained independently:

```
Stage 1 — WarpNet
  Input : cloth(3) + cloth_mask(1) + agnostic(3) + pose(18) = 25 channels
  Output: flow field (2, H, W) → warped cloth (3, H, W) + warped mask (1, H, W)
  Loss  : L1 + VGG perceptual + Total Variation

Stage 2 — TryOnNet
  Input : agnostic(3) + warped_cloth(3) + warped_mask(1) + pose(18) + parse_onehot(18) = 43 channels
  Output: try-on RGB image (3, 512, 384)
  Loss  : L1 + VGG perceptual + PatchGAN adversarial (LSGAN)
```

Both networks use a **U-Net** encoder-decoder with skip connections and **InstanceNorm**.

---

## Project Structure

```
Virtul_try_on/
├── dataset/
│   ├── raw/
│   │   ├── image/          # person images (11,647)
│   │   └── cloth/          # garment images (11,647)
│   ├── train_pairs.txt     # 11,647 training pairs
│   ├── test_pairs.txt      # 2,032 test pairs
│   └── train/              # preprocessed outputs
│       ├── person/
│       ├── cloth/
│       ├── parsing/        # SegFormer segmentation maps
│       ├── pose/           # MediaPipe 18-keypoint heatmaps
│       ├── cloth_mask/     # U2Net cloth masks
│       ├── agnostic/       # clothing-removed person images
│       └── tensors/        # .pt bundles for fast DataLoader
├── model/
│   ├── config.py           # all hyperparameters
│   ├── dataset.py          # VITONDataset (reads .pt bundles)
│   ├── networks.py         # WarpNet, TryOnNet, PatchDiscriminator, VGGLoss
│   └── train.py            # 2-stage training loop with AMP + TensorBoard
├── steps/                  # modular preprocessing pipeline
│   ├── step1_validate.py
│   ├── step2_parsing.py
│   ├── step3_pose.py
│   ├── step4_cloth_mask.py
│   ├── step5_agnostic.py
│   └── step6_normalize.py
├── preprocess.py           # main all-in-one preprocessing script
├── pipeline.py             # alternative: preprocessing via steps/ modules
├── verify.py               # visual spot-check of preprocessed outputs
└── requirements.txt
```

---

## Requirements

- Python 3.12
- CUDA-capable GPU (tested on GTX 1650 4 GB and RTX 4070 12 GB)

Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Training and inference |
| `transformers` | SegFormer human parsing |
| `mediapipe` | Pose estimation (18 keypoints) |
| `rembg`, `onnxruntime` | U2Net cloth mask extraction |
| `tensorboard` | Training visualization |
| `opencv-python`, `Pillow` | Image I/O |

---

## Quick Start

### 1 — Preprocess

```bash
# Preprocess training data (limit for a quick smoke test)
python preprocess.py --split train --limit 1000

# Full dataset
python preprocess.py --split both
```

Outputs go to `dataset/train/tensors/*.pt` (one bundle per image pair).

### 2 — Train Stage 1: WarpNet

```bash
python model/train.py --stage warp --epochs 30
```

### 3 — Train Stage 2: TryOnNet

```bash
python model/train.py --stage tryon --epochs 30
```

### 4 — Resume from checkpoint

```bash
python model/train.py --stage warp --resume checkpoints/warp_epoch_10.pth
```

### 5 — Verify preprocessing

```bash
python verify.py               # random sample from train split
python verify.py --split test --n 5
python verify.py --name 00000_00
```

---

## GPU Configuration

### GTX 1650 (4 GB VRAM) — recommended settings in `model/config.py`

```python
BATCH_SIZE  = 4
NGF         = 48
NDF         = 48
NUM_WORKERS = 0     # required on Windows
AMP         = True
N_EPOCHS    = 30
LR_DECAY_START = 20
```

### RTX 4070 (12 GB VRAM) — default config settings

```python
BATCH_SIZE  = 12
NGF         = 64
NDF         = 64
NUM_WORKERS = 4
AMP         = True
N_EPOCHS    = 100
LR_DECAY_START = 50
```

> **Windows users**: Always set `NUM_WORKERS = 0` — Python multiprocessing fork is not supported on Windows.

---

## Training Details

| Setting | Value |
|---|---|
| Image resolution | 512 × 384 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Learning rate (G & D) | 2e-4 |
| LR schedule | Linear decay to 0 over second half of training |
| Mixed precision | fp16 autocast + GradScaler |
| TensorBoard logging | every 50 iterations (scalars), every 10 epochs (images) |
| Checkpoint interval | every 5 epochs |

### Loss Weights

| Loss | WarpNet | TryOnNet |
|---|---|---|
| L1 | 10.0 | 10.0 |
| VGG perceptual | 5.0 | 5.0 |
| Total variation (flow) | 1.0 | — |
| PatchGAN (LSGAN) | — | 1.0 |

---

## Preprocessing Pipeline

Six steps run sequentially per image pair:

1. **Validate & resize** — checks images, resizes to 512×384
2. **Human parsing** — SegFormer (`mattmdjaga/segformer_b2_clothes`) → 18-class label map
3. **Pose estimation** — MediaPipe → 18-keypoint Gaussian heatmaps saved as `.pt`
4. **Cloth mask** — U2Net via `rembg` → binary cloth mask
5. **Agnostic** — erases clothing region (labels 4, 7, 17) from person image
6. **Normalize & bundle** — tensors normalized to [-1, 1], saved as a single `.pt` file

---

## TensorBoard

```bash
tensorboard --logdir logs
```

Logs scalars (loss curves) and image grids (warped cloth, fake vs. real person) during training.

---

## Common Issues

| Problem | Fix |
|---|---|
| `RuntimeError` on DataLoader | Set `NUM_WORKERS = 0` on Windows |
| CUDA OOM | Reduce `BATCH_SIZE`, or set `NGF = NDF = 48` |
| NaN losses | Set `AMP = False` to debug; check input normalization |
| Slow preprocessing | Increase `--batch` arg (default 8); reduce `--limit` for testing |
