# Optimized Knowledge Distillation Training

This folder contains an **optimized version** of the knowledge distillation trainer designed to train in **20-40 minutes per epoch** on A100/L4 GPUs (down from multiple hours).

## Key Optimizations

1. ✅ **Step-capped epochs**: Only `max_steps_per_epoch` batches per epoch (default: 500)
2. ✅ **No local crops**: Disabled for ~70% speedup
3. ✅ **Simplified augmentations**: Only essential transforms (5-10× faster CPU)
4. ✅ **Teacher feature caching**: Optional (10× speedup when enabled)
5. ✅ **Only compile student**: Teacher is frozen, no benefit from compilation
6. ✅ **Optimized DataLoader**: 4 workers, `persistent_workers=False`
7. ✅ **Step-based checkpointing**: Save every N steps
8. ✅ **Detailed logging**: GPU/data/batch timing

## Files

- `distill_trainer.py` - Optimized trainer with all optimizations
- `transforms.py` - Simplified transforms (SimpleTransform, FastMultiCropTransform)
- `train_config_kd.yaml` - Training config with optimization settings
- `model_config_kd.yaml` - Model config
- `data_config.yaml` - Data loading config
- `train_kd_optimized.ipynb` - Complete notebook from data loading to training

## Usage

### Command Line

```bash
cd distill_optimized
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_kd.yaml \
    --model_config model_config_kd.yaml \
    --device cuda \
    --max_steps_per_epoch 500  # Optional override
```

### Notebook

Open `train_kd_optimized.ipynb` and run all cells.

## Configuration

### `train_config_kd.yaml`

Key settings:
- `max_steps_per_epoch: 500` - Process only 500 batches per epoch
- `use_local_crops: false` - Disable local crops (major speedup)
- `cache_teacher_features: false` - Enable for 10× speedup (optional)
- `compile_student: true` - Compile only student (never teacher)
- `num_workers: 4` - Optimized for Slurm
- `persistent_workers: false` - Must be False to avoid KeyError

## Expected Performance

With default settings (`max_steps_per_epoch: 500`):
- **Per epoch**: ~20-40 minutes (500 steps × 2.4-4.8s/step)
- **10 epochs**: ~3-7 hours
- **50 epochs**: ~17-33 hours
- **100 epochs**: ~33-67 hours

## Teacher Feature Caching

To enable teacher feature caching (10× speedup after first epoch):

1. Set `cache_teacher_features: true` in `train_config_kd.yaml`
2. Set `teacher_feature_dir: "/scratch/$USER/ssl_project/cache/features"`
3. First epoch will be slower (computing and caching features)
4. Subsequent epochs will be much faster (loading from cache)

**Note**: Cache directory must have sufficient disk space (~500GB for full dataset).

## Differences from Original

| Feature | Original | Optimized |
|---------|----------|-----------|
| Local crops | Enabled (8 crops) | Disabled (0 crops) |
| Augmentations | Full DINO-style | Simplified (4 transforms) |
| Teacher compile | Yes | No (frozen, no benefit) |
| Student compile | Yes | Yes |
| DataLoader workers | Variable | 4 (optimized) |
| Persistent workers | True | False (avoids KeyError) |
| Step capping | No | Yes (500 steps/epoch) |
| Teacher caching | No | Optional |
| Checkpointing | Epoch-based | Step + epoch-based |

## Troubleshooting

### KeyError with DataLoader

If you see `KeyError: _ResumeIteration`, ensure:
- `persistent_workers: false` in config
- Using `itertools.islice` to limit iterations (already implemented)

### Slow Training

1. Verify GPU usage: `nvidia-smi`
2. Check DataLoader bottleneck: Look at `data` time in progress bar
3. Enable teacher caching if running multiple epochs
4. Reduce `max_steps_per_epoch` for shorter jobs

### Out of Memory

1. Reduce `batch_size` in `train_config_kd.yaml`
2. Reduce `num_workers` (try 2 instead of 4)
3. Disable teacher caching if enabled

