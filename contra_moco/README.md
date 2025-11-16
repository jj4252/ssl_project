# MoCo-v3 Contrastive Learning Pipeline

This folder contains a refactored training pipeline that supports both Knowledge Distillation (KD) and MoCo-v3 contrastive learning.

## Overview

- **MoCo-v3**: Self-supervised contrastive learning with momentum encoder and negative queue
- **KD**: Original knowledge distillation from DINOv2 teacher (legacy mode)

## Key Features

- **Training Mode Switch**: Use `training_mode: "moco_v3"` or `training_mode: "kd"` in config
- **MoCo-v3 Implementation**: Full MoCo-v3 with momentum encoder, projection head, and negative queue
- **96×96 Images**: Optimized for 96×96 input size (HF 500K dataset + CIFAR10)
- **ViT-Small/16**: Uses ViT-Small/16 as the backbone encoder
- **Two Views**: MoCo-v3 uses strong augmentations to create two views per image

## Configuration

### Training Config (`train_config_moco.yaml`)

```yaml
training_mode: "moco_v3"

batch_size: 256
num_epochs: 100
learning_rate: 0.0005
weight_decay: 0.05
warmup_epochs: 10

moco:
  proj_dim: 256           # Projection head output dimension
  queue_size: 65536       # Number of negative keys
  momentum: 0.99          # EMA momentum for key encoder
  temperature: 0.2        # InfoNCE temperature
```

### Model Config (`model_config_kd.yaml`)

Same as KD mode - uses `student_name` and `student_img_size`:

```yaml
student_name: "vit_small_patch16_224"
student_img_size: 96
```

### Data Config (`data_config.yaml`)

```yaml
dataset_name: "cifar10"  # or use cached HF dataset
image_size: 96
```

## Usage

### Training

```bash
python distill_trainer.py \
    --data_config data_config.yaml \
    --train_config train_config_moco.yaml \
    --model_config model_config_kd.yaml \
    --device cuda
```

### Evaluation

```bash
python evaluate_checkpoints.py \
    --model_config model_config_kd.yaml \
    --checkpoint_dir /path/to/checkpoints \
    --dataset cifar10 \
    --mode moco_v3 \
    --epochs 1 10 50 100
```

## Architecture

### MoCo-v3 Components

1. **Encoder_q (Query Encoder)**: Trainable ViT-Small/16 backbone
2. **Encoder_k (Key Encoder)**: Momentum-updated copy of encoder_q (frozen)
3. **Projection Heads**: MLP (embed_dim → embed_dim → proj_dim) for both encoders
4. **Negative Queue**: FIFO queue of 65,536 negative samples

### Training Process

1. Forward two augmented views through encoder_q and encoder_k
2. Project to lower dimension (256) and L2-normalize
3. Compute InfoNCE loss: positive = same image, negatives = queue
4. Update queue with new keys
5. Momentum update: encoder_k = 0.99 * encoder_k + 0.01 * encoder_q

## Checkpoints

MoCo-v3 checkpoints contain:
- `encoder_q`: Query encoder state dict
- `proj_q`: Query projection head state dict
- `optimizer`, `scheduler`, `scaler`: Training state
- `epoch`, `global_step`: Training progress

For evaluation, only `encoder_q` is needed (projection head is discarded).

## Differences from KD

- **No Teacher**: MoCo-v3 is self-supervised, no DINOv2 teacher needed
- **No Distillation Loss**: Uses InfoNCE contrastive loss instead
- **Two Views**: Always uses two augmented views per image
- **Queue**: Maintains a queue of negative samples for contrastive learning
- **Momentum Encoder**: Key encoder updated via EMA, not gradients

## Hyperparameters

Default MoCo-v3 hyperparameters (good starting point):
- Learning rate: 0.0005
- Weight decay: 0.05
- Batch size: 256
- Queue size: 65,536
- Momentum: 0.99
- Temperature: 0.2
- Projection dim: 256

For CIFAR10-only: 200 epochs recommended for better results.
