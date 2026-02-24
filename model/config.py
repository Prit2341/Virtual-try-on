"""
model/config.py — All training hyperparameters in one place.
Edit this file to tune the model. Do not scatter constants across files.
"""

from pathlib import Path


class Config:
    # ── Image ──────────────────────────────────────────────────────────────────
    HEIGHT = 512
    WIDTH  = 384
    N_PARSE_CLASSES = 18      # SegFormer output classes (0–17)

    # ── Dataset ────────────────────────────────────────────────────────────────
    DATASET_ROOT   = Path("d:/Virtual_try_on/dataset")
    CHECKPOINT_DIR = Path("d:/Virtual_try_on/checkpoints")
    LOG_DIR        = Path("d:/Virtual_try_on/logs")

    # ── DataLoader ─────────────────────────────────────────────────────────────
    BATCH_SIZE  = 12     # RTX 4070 12 GB at 512×384 with AMP (~10 GB used)
    NUM_WORKERS = 4      # Windows: keep 0 (no multiprocessing fork support)
    PIN_MEMORY  = True

    # ── Mixed Precision ────────────────────────────────────────────────────────
    AMP = True           # fp16 autocast + GradScaler; ~1.5× faster on RTX 4070
                         # set False to debug NaN losses

    # ── Optimiser ──────────────────────────────────────────────────────────────
    LR_G   = 2e-4        # generator / warp net learning rate
    LR_D   = 2e-4        # discriminator learning rate
    BETA1  = 0.5
    BETA2  = 0.999

    # ── Training ───────────────────────────────────────────────────────────────
    N_EPOCHS      = 100
    SAVE_EVERY    = 5    # save checkpoint every N epochs
    LOG_EVERY     = 50   # log scalar losses every N iterations

    # ── LR Schedule ────────────────────────────────────────────────────────────
    # Linear decay from full LR → 0 over the second half of training.
    # Set LR_DECAY_START = N_EPOCHS to disable decay.
    LR_DECAY_START = 50  # epoch to begin decay (keep full LR for first 50 epochs)

    # ── Visualisation ──────────────────────────────────────────────────────────
    LOG_IMAGES_EVERY = 10   # write sample images to TensorBoard every N epochs
    VIS_SAMPLES      = 4    # number of samples shown in TensorBoard image grids

    # ── Loss weights ───────────────────────────────────────────────────────────
    #   WarpNet
    LAMBDA_WARP_L1  = 10.0   # L1 on warped cloth vs. ground-truth cloth region
    LAMBDA_WARP_VGG = 5.0    # VGG perceptual on warped cloth
    LAMBDA_WARP_TV  = 1.0    # total-variation smoothness on flow field

    #   TryOnNet
    LAMBDA_L1   = 10.0   # L1 reconstruction
    LAMBDA_VGG  = 5.0    # VGG perceptual
    LAMBDA_GAN  = 1.0    # PatchGAN adversarial

    # ── Network capacity ───────────────────────────────────────────────────────
    NGF = 64     # base feature channels for generators / WarpNet
    NDF = 64     # base feature channels for discriminator

    # ── WarpNet input channels ─────────────────────────────────────────────────
    #   cloth(3) + cloth_mask(1) + agnostic(3) + pose_map(18) = 25
    WARP_IN_CH = 25

    # ── TryOnNet input channels ────────────────────────────────────────────────
    #   agnostic(3) + warped_cloth(3) + warped_mask(1)
    #   + pose_map(18) + parse_one_hot(18) = 43
    TRYON_IN_CH  = 43
    TRYON_OUT_CH = 3     # RGB output
