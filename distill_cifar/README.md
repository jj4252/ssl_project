# Knowledge Distillation on CIFAR10 - Dataset Distribution Test

This folder contains an experiment to test whether the poor accuracy in `distill_cached_random` is due to a dataset distribution mismatch.

## Experiment Hypothesis

**Question**: Is the poor accuracy (11-13% on CIFAR10) due to training on a different dataset distribution than CIFAR10?

**Test**: Train KD on CIFAR10, evaluate on CIFAR10 (same distribution)

**Expected Result**: If distribution mismatch is the issue, accuracy should improve significantly when training and evaluation use the same dataset.

## Methodology

Same as `distill_cached_random`:
- **Frozen projection layers** (prevents collapse)
- **Cosine similarity loss** (per-sample matching)
- **Same model architecture** (ViT-S/16 student, DINOv2 ViT-B/14 teacher)
- **Same training hyperparameters** (LR=5e-4, batch=64, etc.)

**Only difference**: Dataset
- Training: CIFAR10 (50K images)
- Evaluation: CIFAR10 (10K test images)
- Same distribution = no mismatch

## Files

- `distill_trainer.py` - Main training script (same as distill_cached_random, with frozen projections and cosine loss)
- `data_loader.py` - CIFAR10 dataset loader
- `data_config.yaml` - CIFAR10 data configuration
- `train_config_kd.yaml` - Training configuration (same as distill_cached_random)
- `model_config_kd.yaml` - Model configuration (same as distill_cached_random)
- `evaluate_checkpoints.py` - Evaluation script for linear probing on CIFAR10
- `transforms.py` - Image transforms (same as distill_cached_random)
- `optimizer.py` - Optimizer and scheduler (same as distill_cached_random)
- `dinov2_patcher.py` - DINOv2 compatibility patcher (same as distill_cached_random)

## Usage

### Training

```bash
cd /scratch/$USER/Nov_14_distill/ssl_project/distill_cifar

python distill_trainer.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --model_config model_config_kd.yaml \
  --no_resume  # Start fresh
```

### Evaluation

```bash
python evaluate_checkpoints.py \
  --model_config model_config_kd.yaml \
  --checkpoint_dir /scratch/$USER/Nov_14_distill/checkpoints_cifar10 \
  --dataset cifar10 \
  --epochs 1 10 20 50 \
  --batch_size 128 \
  --device cuda
```

## Expected Results

If distribution mismatch is the issue:
- **Accuracy should improve significantly** (from 11-13% to 50-70%+)
- **Features should remain diverse** (pairwise similarity < 0.7)
- **Loss should decrease properly** (from ~0.97 to ~0.1-0.2)

If distribution mismatch is NOT the issue:
- **Accuracy may still be low** (indicating other problems)
- **Need to investigate further** (feature quality, loss computation, etc.)

## Key Differences from distill_cached_random

| Aspect | distill_cached_random | distill_cifar |
|--------|----------------------|---------------|
| **Training Dataset** | Custom dataset (250K images, 96x96) | CIFAR10 (50K images, 32x32) |
| **Evaluation Dataset** | CIFAR10 (10K images, 32x32) | CIFAR10 (10K images, 32x32) |
| **Distribution Match** | ❌ Different | ✅ Same |
| **Image Size** | 96x96 (native) | 32x32 → 96x96 (upscaled) |
| **Caching** | Yes (cached tensors) | No (direct loading) |
| **Methodology** | Same | Same |

## Notes

- CIFAR10 images are 32x32, will be upscaled to 96x96 for student
- Teacher receives 224x224 (upscaled from 96x96)
- Full CIFAR10 training set (50K images) will be used
- Checkpoints saved to `/scratch/$USER/Nov_14_distill/checkpoints_cifar10`
