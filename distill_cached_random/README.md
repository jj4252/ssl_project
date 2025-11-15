# Knowledge Distillation Training - Random Shard-Level Sampling Per Epoch

This folder contains a training regime that uses **random shard-level sampling per epoch** to train on large cached datasets efficiently. Instead of randomly sampling individual indices, it randomly selects shard files and accesses all images within those shards sequentially for better performance.

## Key Feature: Random Shard-Level Sampling

Instead of iterating over all cached images each epoch, this implementation:

1. **Uses cached tensors only** (no direct HuggingFace image loading during training)
2. **Assumes partial cache** (~250K images currently cached, out of 500K total)
3. **Samples random shard files per epoch** (e.g., 15 shards = ~30K images per epoch)
4. **Sequential access within shards** for better cache locality and I/O performance
5. **Works with `max_steps_per_epoch`** to cap the number of batches
6. **Stochastic coverage**: Over many epochs, the model sees most/all of the 250K images multiple times

## How It Works

### Architecture

- **`CachedTensorDataset`**: Loads the full cached dataset (~250K images)
- **`RandomShardSubsetCachedDataset`**: Wraps the full dataset and samples random shard files each epoch
- **`set_epoch()`**: Called at the start of each epoch to resample random shards
- **Sequential access**: All images from selected shards are accessed sequentially for better performance

### Configuration

In `data_config.yaml`:

```yaml
# Random shard-level sampling per epoch
shards_per_epoch: 15  # Number of random shard files per epoch (~30K images)
random_subset_seed: 42   # Random seed for reproducibility
```

### Training Flow

1. **Epoch Start**: `RandomShardSubsetCachedDataset.set_epoch(epoch)` is called
2. **Resampling**: Random shard files are selected from all available shards
3. **Index Mapping**: All images from selected shards are mapped sequentially
4. **Training**: Only images from selected shards are used (accessed sequentially within each shard)
5. **Next Epoch**: Process repeats with a new random set of shards

### Example

- **Full cached dataset**: 250,000 images in ~122 shards (2048 images per shard)
- **Shards per epoch**: 15 shards
- **Samples per epoch**: ~30,720 images (15 × 2048)
- **Coverage per epoch**: ~12% of cached data
- **Over 10 epochs**: Model sees ~120% of cached data (with overlap)
- **Over 100 epochs**: Model sees most/all cached images multiple times
- **Performance**: Sequential access within shards provides much better cache locality than random index sampling

## Files

- `distill_trainer.py` - Main training script (modified to call `set_epoch`)
- `data_loader.py` - Contains `RandomShardSubsetCachedDataset` and factory function
- `data_config.yaml` - Config with `shards_per_epoch` setting
- `train_config_kd.yaml` - Training config (same as `distill_cached`)
- `model_config_kd.yaml` - Model config (same as `distill_cached`)

## Usage

### Training with Random Subset Sampling

```bash
cd /scratch/jj4252/Nov_14_distill/ssl_project/distill_cached_random

python distill_trainer.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --model_config model_config_kd.yaml
```

### Configuration Options

**`data_config.yaml`**:
- `shards_per_epoch`: Number of random shard files per epoch (default: 15)
- `random_subset_seed`: Random seed for shard sampling (default: 42)
- `cache_root`: Path to cached tensor directory
- `cache_shard_size`: Number of images per shard (default: 2048)

**`train_config_kd.yaml`**:
- `max_steps_per_epoch`: Maximum batches per epoch (default: 500)
- `batch_size`: Batch size (default: 64)
- Other training hyperparameters

### Example: 15 Shards Per Epoch

With `shards_per_epoch: 15`, `cache_shard_size: 2048`, and `batch_size: 64`:
- **Samples per epoch**: 15 × 2048 = 30,720 images
- **Batches per epoch**: 30,720 / 64 = ~480 batches
- **With `max_steps_per_epoch: 500`**: Will process all 480 batches
- **With `max_steps_per_epoch: 250`**: Will process only 250 batches (capped)

## Benefits

1. **Fast epochs**: Only process ~30K images per epoch instead of 250K
2. **Better cache locality**: Sequential access within shards (much faster than random index sampling)
3. **Fewer shard files loaded**: Only 15 shard files per epoch vs. many shards with random indices
4. **Stochastic coverage**: Model sees different shards (and thus different images) each epoch
5. **Memory efficient**: Smaller dataset per epoch reduces memory pressure
6. **Compatible with step-capping**: Works seamlessly with `max_steps_per_epoch`
7. **Partial cache support**: Works with incomplete cache (e.g., 250K out of 500K)

## Differences from `distill_cached`

| Feature | `distill_cached` | `distill_cached_random` |
|---------|------------------|------------------------|
| Dataset size per epoch | Full cached dataset | Random shard subset (~30K) |
| Sampling method | All shards | Random shard files per epoch |
| Access pattern | Sequential | Sequential within selected shards |
| Epoch duration | Longer (processes all data) | Shorter (processes subset) |
| Coverage | All data each epoch | Stochastic over epochs |
| Use case | Full dataset training | Large dataset with fast epochs |

## Notes

- The cached dataset is assumed to be **partial** (~250K images currently)
- Random sampling ensures **stochastic coverage** over many epochs
- Each epoch uses a **different random subset** (resampled at epoch start)
- The implementation is **config-driven** and compatible with existing training code

