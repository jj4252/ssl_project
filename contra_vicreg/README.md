# VICReg Self-Supervised Learning Pipeline

This folder contains a VICReg (Bardes et al., 2022) implementation for self-supervised learning with ViT-S/16 backbone.

## Overview

- **VICReg**: Variance-Invariance-Covariance Regularization for SSL
- **Backbone**: ViT-S/16 (same architecture as MoCo/KD setups)
- **Training**: Two augmented views per image, no negatives, no queue
- **Evaluation**: Linear probing on CIFAR-10/100

## Key Features

- **VICReg Loss**: Implements invariance, variance, and covariance terms
- **ViT-S/16 Encoder**: Reuses existing ViT-S/16 architecture
- **3-Layer Projection Head**: 384 → 2048 → 2048 → 2048 with BatchNorm
- **Strong Augmentations**: RandomResizedCrop, HorizontalFlip, ColorJitter, Grayscale, GaussianBlur
- **Comprehensive Diagnostics**: Per-dim std, pairwise similarity, covariance magnitude
- **Compatible Evaluation**: Works with existing `evaluate_checkpoints.py`

## Architecture

### Model Components

1. **Encoder**: ViT-S/16 backbone
   - Patch size: 16
   - Embedding dim: 384
   - Depth: 12
   - Heads: 6
   - Output: CLS token [B, 384]

2. **Projection Head**: 3-layer MLP
   - Linear(384 → 2048) + BatchNorm + ReLU
   - Linear(2048 → 2048) + BatchNorm + ReLU
   - Linear(2048 → 2048)
   - Output: [B, 2048]

### VICReg Loss

The total loss consists of three terms:

1. **Invariance**: MSE between two views of the same image
   - `L_inv = mean((z1 - z2)^2)`
   - Weight: `lambda_invariance = 25.0`

2. **Variance**: Penalize dimensions with std < gamma
   - `L_var = mean(relu(gamma - std(z)))` (averaged over both views)
   - Weight: `mu_variance = 25.0`
   - Target std: `gamma = 1.0`

3. **Covariance**: Penalize off-diagonal covariance elements
   - `L_cov = mean(off_diag(cov(z))^2)` (averaged over both views)
   - Weight: `nu_covariance = 1.0`

**Total Loss**: `L = lambda_inv * L_inv + mu_var * L_var + nu_cov * L_cov`

## Configuration

### Training Config (`train_config_vicreg.yaml`)

```yaml
training_mode: "vicreg"

batch_size: 128
num_epochs: 200
learning_rate: 0.001
weight_decay: 0.0001
warmup_epochs: 10

vicreg:
  proj_dim: 2048
  proj_hidden_dim: 2048
  lambda_invariance: 25.0
  mu_variance: 25.0
  nu_covariance: 1.0
  gamma: 1.0
```

### Model Config (`model_config_vicreg.yaml`)

```yaml
backbone_name: "vit_small_patch16_224"
image_size: 96
```

### Data Config (`data_config.yaml`)

Same as MoCo/KD - supports CIFAR-10/100 and HuggingFace dataset with cached tensors.

## Usage

### Training

```bash
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_vicreg.yaml \
    --model_config model_config_vicreg.yaml \
    --device cuda
```

**Note:** Make sure you're in the `contra_vicreg` directory or provide full paths to the config files.

### Evaluation

```bash
python evaluate_checkpoints.py \
    --model_config model_config_vicreg.yaml \
    --checkpoint_dir /path/to/checkpoints \
    --dataset cifar10 \
    --mode vicreg \
    --epochs 1 50 100 150 200
```

**Important:** The model config file is named `model_config_vicreg.yaml` (not `model_config_vit_vicreg.yaml`).

## Diagnostics

During training, the script prints periodic diagnostics:

- **Total loss** and individual components (invariance, variance, covariance)
- **Per-dimension std**: mean, min, max (min should be >0.1 to avoid collapse)
- **Pairwise cosine similarity**: Should be 0.1-0.5 (not drift to 1.0)
- **Covariance magnitude**: Mean squared off-diagonal elements

### Warning Signs

- **Collapse risk**: `min_std < 0.01` → Very low variance in some dimensions
- **Feature collapse**: `pairwise_sim > 0.9` → Features are too similar
- **Low variance**: `min_std < 0.1` → Monitor for collapse

## Checkpoints

VICReg checkpoints contain:
- `model`: Full model state dict (encoder + projection head)
- `optimizer`, `scheduler`, `scaler`: Training state
- `epoch`, `global_step`: Training progress

For evaluation, only the encoder is used (projection head is discarded).

## Differences from MoCo/KD

- **No Teacher**: VICReg is fully self-supervised, no DINOv2 teacher
- **No Queue**: No negative queue, uses only two views per image
- **No Momentum Encoder**: Single encoder (no EMA update)
- **Different Loss**: VICReg loss (invariance + variance + covariance) vs InfoNCE
- **Projection Head**: 3-layer MLP with BatchNorm (vs 2-layer for MoCo)

## Hyperparameters

Default VICReg hyperparameters (good starting point):
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 128 (can increase to 256 if memory allows)
- Loss weights: λ=25.0, μ=25.0, ν=1.0
- Target std (γ): 1.0
- Gradient clipping: 1.0

## References

- **VICReg Paper**: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning", ICLR 2022
- **Original Implementation**: https://github.com/facebookresearch/vicreg

