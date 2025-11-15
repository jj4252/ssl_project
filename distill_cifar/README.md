# Knowledge Distillation Training on CIFAR-10/100

This folder contains a simplified setup for testing knowledge distillation training on CIFAR-10/100 datasets.

## Key Features

- **CIFAR-10/100 Support**: Easy switching between CIFAR-10 and CIFAR-100
- **20K Image Limit**: Uses only 20,000 images for quick testing
- **Full Epoch Training**: No step-capping - uses full dataset each epoch
- **Same Training Procedure**: Identical KD training as `distill_cached`

## Files

- `data_loader.py` - CIFAR dataset loading (20K limit)
- `distill_trainer.py` - Training script (same as distill_cached)
- `transforms.py` - Data augmentation transforms
- `optimizer.py` - Optimizer and scheduler
- `dinov2_patcher.py` - DINOv2 Python 3.9 compatibility
- `data_config.yaml` - Dataset configuration
- `train_config_kd.yaml` - Training hyperparameters
- `model_config_kd.yaml` - Model configuration

## Configuration

### Dataset Selection

Edit `data_config.yaml`:
```yaml
dataset_name: "cifar10"  # or "cifar100"
max_samples: 20000  # Limit to 20K images
```

### Training Settings

Edit `train_config_kd.yaml`:
- `num_epochs: 10` - Reduced for quick testing
- No `max_steps_per_epoch` - uses full dataset
- `batch_size: 64`

## Usage

### 1. Train on CIFAR-10

```bash
cd DL_Final_Comp/distill_cifar

python distill_trainer.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --model_config model_config_kd.yaml \
  --device cuda
```

### 2. Train on CIFAR-100

Edit `data_config.yaml`:
```yaml
dataset_name: "cifar100"
```

Then run the same command.

## Expected Behavior

- **Dataset**: Downloads CIFAR automatically on first run
- **Images**: Uses first 20,000 images from training split
- **Training**: Full epochs (no step-capping)
- **Checkpoints**: Saves after each epoch
- **Resume**: Auto-resumes from latest checkpoint if interrupted

## Differences from distill_cached

1. **No caching**: Direct CIFAR loading (no precomputation needed)
2. **20K limit**: Smaller dataset for quick testing
3. **No step-capping**: Uses full dataset each epoch
4. **Simpler setup**: No cache directory configuration needed

## Notes

- CIFAR images (32x32) are upscaled to 224x224 for ViT training
- First run will download CIFAR dataset (~170MB for CIFAR-10, ~170MB for CIFAR-100)
- Training should be much faster than the 500K image setup
- Good for validating that the training code works correctly

