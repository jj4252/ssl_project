# Knowledge Distillation Training - Fixed Shard Subset

This folder contains a training regime that uses **fixed shard-level subset** to train on large cached datasets. This is an experiment to test whether random sampling degrades performance compared to using a fixed set of shards.

## Key Feature: Fixed Shard Subset

Instead of randomly sampling shards each epoch (like `distill_cached_random`), this implementation:

1. **Uses cached tensors only** (no direct HuggingFace image loading during training)
2. **Uses a fixed set of shard files** (e.g., first 25 shards) for all epochs
3. **Same shards every epoch** (no resampling)
4. **Sequential access within shards** for better cache locality and I/O performance
5. **Works with `max_steps_per_epoch`** to cap the number of batches
6. **Deterministic coverage**: Model sees the same subset of images every epoch

## Experiment Purpose

This setup is designed to test whether **random sampling degrades performance** compared to fixed shard selection. By comparing results from:
- `distill_cached_random`: Uses random shard sampling (15 shards per epoch, different each epoch)
- `distill_cached_fixed`: Uses fixed shard subset (25 shards, same every epoch)

We can determine if the randomness in shard selection affects model performance.

## How It Works

### Architecture

- **`CachedTensorDataset`**: Loads the full cached dataset (~250K images)
- **`FixedShardSubsetCachedDataset`**: Wraps the full dataset and uses a fixed set of shard files
- **`set_epoch()`**: No-op (kept for compatibility, but doesn't change shards)
- **Sequential access**: All images from selected shards are accessed sequentially for better performance

### Configuration

In `data_config.yaml`:

```yaml
# Fixed shard-level subset (for testing if random sampling degrades performance)
fixed_shards: 25  # Number of fixed shard files to use (~50K images with 2048 per shard)
```

### Training Flow

1. **Initialization**: First 25 shard files are selected from all available shards
2. **Index Mapping**: All images from selected shards are mapped sequentially
3. **Training**: Only images from selected shards are used (accessed sequentially within each shard)
4. **All Epochs**: Same shards are used across all epochs (no resampling)

### Example

- **Full cached dataset**: 250,000 images in ~122 shards (2048 images per shard)
- **Fixed shards**: 25 shards (first 25 shards)
- **Samples per epoch**: ~51,200 images (25 × 2048)
- **Coverage per epoch**: ~20% of cached data
- **All epochs**: Model sees the same 51,200 images every epoch
- **Performance**: Sequential access within shards provides excellent cache locality

## Files

- `distill_trainer.py` - Main training script (same as `distill_cached_random`)
- `data_loader.py` - Contains `FixedShardSubsetCachedDataset` and factory function
- `data_config.yaml` - Config with `fixed_shards: 25` setting
- `train_config_kd.yaml` - Training config (same as `distill_cached_random`)
- `model_config_kd.yaml` - Model config (same as `distill_cached_random`)
- `evaluate_checkpoints.py` - Evaluation script for linear probing on CIFAR10/100

## Usage

### Training with Fixed Shard Subset

```bash
cd /scratch/jj4252/Nov_14_distill/ssl_project/distill_cached_fixed

python distill_trainer.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --model_config model_config_kd.yaml
```

### Evaluation on CIFAR10/100

After training, evaluate checkpoints using linear probing:

```bash
bash evaluate_checkpoints.sh
```

Or manually:

```bash
python evaluate_checkpoints.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --model_config model_config_kd.yaml \
  --checkpoint_dir /scratch/$USER/Nov_14_distill/checkpoints \
  --dataset cifar10
```

### Configuration Options

**`data_config.yaml`**:
- `fixed_shards`: Number of fixed shard files to use (default: 25)
- `cache_root`: Path to cached tensor directory
- `cache_shard_size`: Number of images per shard (default: 2048)

**`train_config_kd.yaml`**:
- `max_steps_per_epoch`: Maximum batches per epoch (default: 500)
- `batch_size`: Batch size (default: 64)
- Other training hyperparameters

### Example: 25 Fixed Shards

With `fixed_shards: 25`, `cache_shard_size: 2048`, and `batch_size: 64`:
- **Samples per epoch**: 25 × 2048 = 51,200 images
- **Batches per epoch**: 51,200 / 64 = ~800 batches
- **With `max_steps_per_epoch: 500`**: Will process only 500 batches (capped)
- **All epochs**: Same 51,200 images are used every epoch

## Benefits

1. **Deterministic**: Same subset of images every epoch (reproducible)
2. **Better cache locality**: Sequential access within fixed shards (excellent performance)
3. **Fewer shard files loaded**: Only 25 shard files per epoch
4. **Memory efficient**: Smaller dataset per epoch reduces memory pressure
5. **Compatible with step-capping**: Works seamlessly with `max_steps_per_epoch`
6. **Partial cache support**: Works with incomplete cache (e.g., 250K out of 500K)

## Differences from `distill_cached_random`

| Feature | `distill_cached_random` | `distill_cached_fixed` |
|---------|------------------------|------------------------|
| Dataset size per epoch | Random shard subset (~30K) | Fixed shard subset (~50K) |
| Sampling method | Random shard files per epoch | Fixed shard files (first N) |
| Shards per epoch | 15 (random) | 25 (fixed) |
| Access pattern | Sequential within selected shards | Sequential within fixed shards |
| Epoch variation | Different shards each epoch | Same shards every epoch |
| Coverage | Stochastic over epochs | Deterministic (same subset) |
| Use case | Large dataset with fast epochs | Testing if random sampling degrades performance |

## Experiment Design

This setup is designed to test the hypothesis: **Does random sampling degrade performance?**

### Comparison Setup

1. **`distill_cached_random`**: 
   - 15 random shards per epoch (~30K images)
   - Different shards each epoch
   - Stochastic coverage over many epochs

2. **`distill_cached_fixed`** (this folder):
   - 25 fixed shards (~50K images)
   - Same shards every epoch
   - Deterministic coverage

### Evaluation

Both setups should be evaluated on CIFAR10/100 using linear probing to compare:
- Final accuracy
- Training dynamics (loss curves)
- Feature quality (pairwise similarity, variance)

If fixed shards perform better, it suggests that random sampling may be degrading performance. If random shards perform better, it suggests that diversity in training data (even with same total coverage) helps.

## Notes

- The cached dataset is assumed to be **partial** (~250K images currently)
- Fixed shards ensure **deterministic coverage** (same subset every epoch)
- Each epoch uses the **same fixed subset** (no resampling)
- The implementation is **config-driven** and compatible with existing training code
- Evaluation should be done on CIFAR10/100 to test generalization
