# MoCo-v3 Multi-Backbone Support

This folder now supports multiple backbone architectures for MoCo-v3 training:
- **Vision Transformer (ViT)**: Original configuration
- **ResNet-50**: Standard ResNet backbone with MoCo-v3 hyperparameters

## Quick Start

### Training with ViT (existing config)
```bash
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_moco.yaml \
    --model_config model_config_moco_vit.yaml
```

### Training with ResNet-50 (new)
```bash
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_moco_resnet.yaml \
    --model_config model_config_moco_resnet.yaml
```

## Configuration Files

### Model Configs
- **`model_config_moco_vit.yaml`**: ViT-S/16 configuration (96x96 images)
- **`model_config_moco_resnet.yaml`**: ResNet-50 configuration (224x224 images)

### Training Configs
- **`train_config_moco.yaml`**: ViT training config (current settings)
- **`train_config_moco_resnet.yaml`**: ResNet-50 training config with standard MoCo-v3 hyperparameters:
  - Temperature: 0.2 (standard for ResNet)
  - Queue: enabled
  - LR: 0.001 (AdamW) or 0.03 (SGD)
  - Strong augmentations

## Key Differences

### ViT Configuration
- Image size: 96x96 (for CIFAR-10 upscaled)
- Feature extraction: CLS token
- Projection: 384 → 1024 → 256
- Temperature: 0.8 (higher to prevent collapse)
- Queue: Currently disabled (batch-only mode)

### ResNet-50 Configuration
- Image size: 224x224 (standard ImageNet size)
- Feature extraction: Global average pooling
- Projection: 2048 → 2048 → 128
- Temperature: 0.2 (standard MoCo-v3)
- Queue: Enabled (65536 negatives)
- Momentum: 0.999 (standard for ResNet)

## Model Architecture

The `MoCoModel` class (renamed from `MoCoViT`) automatically detects backbone type:
- **Auto-detection**: Checks if "vit" or "resnet" is in the backbone name
- **Manual override**: Set `backbone_type: "vit"` or `"resnet"` in model config

### Feature Extraction
- **ViT**: Extracts CLS token (first token from sequence)
- **ResNet**: Extracts global average pooled features

## Switching Backbones

To switch between backbones, simply change the config files:

1. **For ViT**: Use `model_config_moco_vit.yaml` and `train_config_moco.yaml`
2. **For ResNet-50**: Use `model_config_moco_resnet.yaml` and `train_config_moco_resnet.yaml`

The training script automatically handles the differences in feature extraction and model initialization.

## Notes

- ResNet-50 requires 224x224 images (standard ImageNet size)
- ViT-S/16 uses 96x96 images (optimized for CIFAR-10)
- Both backbones use the same MoCo-v3 contrastive learning framework
- Projection head dimensions are adjusted per backbone (ResNet has 2048-dim features, ViT has 384-dim)

