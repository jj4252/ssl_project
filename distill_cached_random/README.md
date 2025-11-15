# Knowledge Distillation Training - Random Subset Sampling Per Epoch

This folder contains a training regime that uses **random subset sampling per epoch** to train on large cached datasets efficiently.

## Key Feature: Random Subset Sampling

Instead of iterating over all cached images each epoch, this implementation:

1. **Uses cached tensors only** (no direct HuggingFace image loading during training)
2. **Assumes partial cache** (~250K images currently cached, out of 500K total)
3. **Samples random subset per epoch** (e.g., ~30K images per epoch)
4. **Works with `max_steps_per_epoch`** to cap the number of batches
5. **Stochastic coverage**: Over many epochs, the model sees most/all of the 250K images multiple times

## How It Works

### Architecture

- **`CachedTensorDataset`**: Loads the full cached dataset (~250K images)
- **`RandomSubsetCachedDataset`**: Wraps the full dataset and samples a random subset each epoch
- **`set_epoch()`**: Called at the start of each epoch to resample random indices

### Configuration

In `data_config.yaml`:

```yaml
# Random subset sampling per epoch
samples_per_epoch: 30000  # Number of random samples per epoch
random_subset_seed: 42   # Random seed for reproducibility
```

### Training Flow

1. **Epoch Start**: `RandomSubsetCachedDataset.set_epoch(epoch)` is called
2. **Resampling**: Random indices are sampled from the full cached dataset
3. **Training**: Only the sampled subset is used for that epoch
4. **Next Epoch**: Process repeats with a new random subset

### Example

- **Full cached dataset**: 250,000 images
- **Samples per epoch**: 30,000 images
- **Coverage per epoch**: 12% of cached data
- **Over 10 epochs**: Model sees ~120% of cached data (with overlap)
- **Over 100 epochs**: Model sees most/all cached images multiple times

## Files

- `distill_trainer.py` - Main training script (modified to call `set_epoch`)
- `data_loader.py` - Contains `RandomSubsetCachedDataset` and factory function
- `data_config.yaml` - Config with `samples_per_epoch` setting
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
- `samples_per_epoch`: Number of random samples per epoch (default: 30000)
- `random_subset_seed`: Random seed for subset sampling (default: 42)
- `cache_root`: Path to cached tensor directory

**`train_config_kd.yaml`**:
- `max_steps_per_epoch`: Maximum batches per epoch (default: 500)
- `batch_size`: Batch size (default: 64)
- Other training hyperparameters

### Example: 30K Samples Per Epoch

With `samples_per_epoch: 30000` and `batch_size: 64`:
- **Batches per epoch**: 30,000 / 64 = ~469 batches
- **With `max_steps_per_epoch: 500`**: Will process all 469 batches
- **With `max_steps_per_epoch: 250`**: Will process only 250 batches (capped)

## Benefits

1. **Fast epochs**: Only process ~30K images per epoch instead of 250K
2. **Stochastic coverage**: Model sees different images each epoch
3. **Memory efficient**: Smaller dataset per epoch reduces memory pressure
4. **Compatible with step-capping**: Works seamlessly with `max_steps_per_epoch`
5. **Partial cache support**: Works with incomplete cache (e.g., 250K out of 500K)

## Differences from `distill_cached`

| Feature | `distill_cached` | `distill_cached_random` |
|---------|------------------|------------------------|
| Dataset size per epoch | Full cached dataset | Random subset (~30K) |
| Epoch duration | Longer (processes all data) | Shorter (processes subset) |
| Coverage | All data each epoch | Stochastic over epochs |
| Use case | Full dataset training | Large dataset with fast epochs |

## Notes

- The cached dataset is assumed to be **partial** (~250K images currently)
- Random sampling ensures **stochastic coverage** over many epochs
- Each epoch uses a **different random subset** (resampled at epoch start)
- The implementation is **config-driven** and compatible with existing training code

