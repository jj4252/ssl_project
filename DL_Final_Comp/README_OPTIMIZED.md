# Optimized Training Configuration

This document describes the optimized training setup for significantly faster training with minimal performance loss.

## Key Optimizations

### 1. Reduced Multi-Crop
- **Before**: 2 global + 8 local = 10 crops per image
- **After**: 2 global + 2 local = 4 crops per image
- **Speedup**: ~2.5x reduction in forward passes

### 2. Restricted Loss Pairings
- Student matches only against teacher global + local views
- Avoids expensive local-to-local comparisons
- **Speedup**: ~20-30% reduction in loss computation

### 3. Reduced Projection Head
- **out_dim**: 65536 → 32768 (50% reduction)
- **hidden_dim**: 2048 → 1536 (25% reduction)
- **Speedup**: ~30% reduction in projection head compute

### 4. Shorter Training
- **Epochs**: 200 → 75 (62% reduction)
- **Warmup**: 10 → 5 epochs
- **Speedup**: 2.67x reduction in total training time

### 5. Performance Optimizations
- **torch.compile**: Model compilation for faster execution
- **channels_last**: Better memory layout for convolutions
- **BF16**: Mixed precision with bfloat16
- **TF32**: TensorFloat-32 for faster matmuls (Ampere+)
- **Fused AdamW**: Optimized optimizer implementation
- **24 data workers**: Faster data loading with prefetching

## Expected Speedup

Combined optimizations should provide **~5-8x speedup**:
- Reduced crops: ~2.5x
- Restricted loss: ~1.2x
- Smaller head: ~1.3x
- Fewer epochs: ~2.7x
- System optimizations: ~1.2-1.5x

**Total**: Training time reduced from days to **a few hours**

## Usage

### Quick Start

```bash
bash pretrain_optimized.sh
```

Or manually:

```bash
python train_dino_main.py \
    --data_config data_config_optimized.yaml \
    --model_config model_config.yaml \
    --train_config train_config_optimized.yaml \
    --device cuda
```

## Configuration Files

- `data_config_optimized.yaml`: Optimized data loading (24 workers, 2 local crops)
- `train_config_optimized.yaml`: Optimized training (75 epochs, 5 warmup, smaller head)

## Performance vs Accuracy Trade-off

These optimizations are designed to maintain competitive accuracy while dramatically reducing training time:

- **Multi-crop reduction**: Minimal impact (2 local crops still provide good diversity)
- **Restricted loss**: Slight impact, but teacher averaging compensates
- **Smaller head**: Small impact, 32k dims still sufficient
- **Fewer epochs**: DINO converges early, 75 epochs often sufficient

The model should still achieve strong k-NN performance while training in hours instead of days.

