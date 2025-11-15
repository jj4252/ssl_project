# Knowledge Distillation Training - Cached Tensor Support (2-Stage Pipeline)

This folder contains a **2-stage training pipeline** that uses cached preprocessed tensors to dramatically speed up data loading. This solves data-loading bottlenecks by preprocessing images once and storing them as tensor shards.

## Key Feature: 2-Stage Pipeline

**Stage 1 (Precompute)**: Preprocess all 500K images once and save as tensor shards
- Deterministic preprocessing: resize, normalize
- Saved as shard files (e.g., `images_shard_00000.pt`)
- Creates `index.json` for fast lookup

**Stage 2 (Training)**: Load from cached tensors instead of HuggingFace/raw images
- Fast tensor loading (no JPEG decoding, no HF API calls)
- Minimal augmentations applied on cached tensors
- Expected 10-20× speedup in data loading

## Files

- `precompute_cache.py` - Stage 1: Precompute and cache image tensors
- `distill_trainer.py` - Stage 2: Training with cached tensor support
- `data_loader.py` - Contains `CachedTensorDataset` and factory function
- `data_config.yaml` - Config with cache settings
- `train_config_kd.yaml` - Training config
- `model_config_kd.yaml` - Model config

## Usage

### Stage 1: Precompute Cache (Resumable)

First, run the precomputation script to create cached tensors:

```bash
cd /scratch/jj4252/Nov_14_distill/ssl_project/distill_cached
python precompute_cache.py \
    --data_config data_config.yaml \
    --train_config train_config_kd.yaml \
    --batch_size 256
```

This will:
- Load all 500K images from HuggingFace
- Apply deterministic preprocessing (resize to 224×224, normalize)
- Save as shard files in `cache_dir` (default: `/scratch/jj4252/Nov_14_distill/cache_images`)
- Create `index.json` for fast lookup

**Resumable**: If the job is interrupted, simply run the same command again. The script will:
- Detect existing shards from `index.json`
- Skip already processed samples
- Continue from the last completed shard
- Update `index.json` incrementally after each shard

**Expected time**: ~2-4 hours for 500K images (one-time cost, can be split across multiple runs)

**Disk space**: ~300 GB for float32 tensors (3×224×224 per image)

**Example resume scenario**:
```bash
# First run: processes shards 0-20, then job is killed
python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml

# Second run: automatically detects shards 0-20, skips them, continues from shard 21
python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml
```

### Stage 2: Train with Cached Data

After precomputation, enable cached mode in `data_config.yaml`:

```yaml
use_cached: true  # Enable cached mode
cache_dir: "/scratch/$USER/Nov_14_distill/cache_images"
```

Then run training:

```bash
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_kd.yaml \
    --model_config model_config_kd.yaml \
    --device cuda
```

Training will now load from cached tensors instead of HuggingFace, resulting in much faster data loading.

## Configuration

### `data_config.yaml`

```yaml
# Cache configuration
use_cached: false  # Set to true after running precompute_cache.py
cache_dir: "/scratch/$USER/Nov_14_distill/cache_images"
cache_shard_size: 10000  # Samples per shard file
```

### Cache Directory Structure

After precomputation, the cache directory will contain:

```
cache_images/
├── index.json                    # Index file mapping global_idx → shard
├── images_shard_00000.pt        # Shard 0: 10k samples
├── images_shard_00001.pt        # Shard 1: 10k samples
├── ...
└── images_shard_00049.pt        # Final shard: remaining samples
```

Each shard file contains:
```python
{
    "images": torch.Tensor  # Shape [N, 3, 224, 224], float32, normalized
}
```

## Performance

### Data Loading Speed

- **Original mode** (HuggingFace): ~20-25 seconds per batch (data-bound)
- **Cached mode**: ~0.1-0.5 seconds per batch (10-20× faster)

### Expected Training Time

With cached mode + step-capped epochs (`max_steps_per_epoch: 500`):
- **Per epoch**: ~5-10 minutes (down from 20-40 minutes)
- **10 epochs**: ~1-2 hours
- **100 epochs**: ~8-17 hours

## How It Works

1. **CachedTensorDataset**:
   - Loads `index.json` to map global index → (shard_path, local_index)
   - Lazily loads shard files into memory (cached per shard)
   - Returns preprocessed tensors [3, H, W]

2. **build_pretraining_dataloader**:
   - If `use_cached: true` → uses `CachedTensorDataset`
   - If `use_cached: false` → uses original `PretrainDataset` (HuggingFace)

3. **Training loop**:
   - Receives batches of tensors [B, 3, H, W] (same as before)
   - No changes to KD loss, optimizer, or training logic

## Memory Considerations

- **Shard cache**: Each shard is ~1.2 GB (10k samples × 3×224×224 × 4 bytes)
- **Total cache**: ~300 GB for 500K images (float32)
- **Memory usage**: Only one shard loaded at a time per worker

To reduce disk space:
- Use float16 instead of float32 (half the size)
- Reduce image size (e.g., 192 instead of 224)
- Use compression (optional, not implemented)

## Troubleshooting

### Cache not found

If you see `FileNotFoundError: Index file not found`:
1. Run `precompute_cache.py` first
2. Check `cache_dir` path in `data_config.yaml` (should be `/scratch/jj4252/Nov_14_distill/cache_images`)
3. Verify `index.json` exists in cache directory

### Resume not working

If resume doesn't work correctly:
1. Check that `index.json` exists and is valid JSON
2. Verify shard files exist on disk (check a few manually)
3. If `index.json` is corrupted, you can delete it and restart (will regenerate all shards)
4. Make sure `shuffle=False` in DataLoader (required for deterministic order)

### Out of memory during precomputation

- Reduce `batch_size` in `precompute_cache.py` (default: 256)
- Reduce `num_workers` in data_config

### Slow data loading even with cache

- Check disk I/O speed (should be on `/scratch/` or fast SSD)
- Reduce `num_workers` if too many workers cause contention
- Verify shard files are on local filesystem (not network mount)

## Example Workflow

```bash
# Navigate to project directory
cd /scratch/jj4252/Nov_14_distill/ssl_project/distill_cached

# 1. Precompute cache (one-time, ~2-4 hours, resumable)
python precompute_cache.py \
    --data_config data_config.yaml \
    --train_config train_config_kd.yaml \
    --batch_size 256

# If interrupted, just run again - it will resume automatically:
# python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml

# 2. Update data_config.yaml: set use_cached: true

# 3. Train with cached data (much faster)
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_kd.yaml \
    --model_config model_config_kd.yaml \
    --device cuda
```

## Differences from Original

| Feature | Original | Cached |
|---------|----------|--------|
| Data source | HuggingFace API | Cached tensor shards |
| Preprocessing | Every epoch | Once (precomputed) |
| Data loading | ~20-25s/batch | ~0.1-0.5s/batch |
| Disk space | Minimal | ~300 GB |
| Setup time | None | 2-4 hours (one-time) |

