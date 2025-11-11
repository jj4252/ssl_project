# Self-Supervised Visual Encoder with DINO

This project implements DINO-style self-supervised learning for visual representation learning on 96×96 images. The model is pretrained on ~500k unlabeled images and evaluated using k-NN classification on a downstream task.

## Project Structure

All files are in the root directory:

```
DL_Final_Comp/
├── datasets.py           # HuggingFace dataset loaders
├── transforms.py         # Multi-crop augmentations
├── vit_model.py          # ViT-S/16 or ViT-B/16 backbone
├── dino_wrapper.py       # DINO teacher-student wrapper
├── train_dino.py         # Training loop
├── optimizer.py          # Optimizer and scheduler
├── train_dino_main.py    # Main training script
├── extract_features.py   # Feature extraction
├── extract_features_main.py
├── knn_eval.py           # k-NN evaluation
├── knn_eval_main.py
├── linear_probe.py       # Linear probe (optional)
├── data_config.yaml      # Data configuration
├── model_config.yaml     # Model configuration
├── train_config.yaml     # Training configuration
├── eval_config.yaml      # Evaluation configuration
├── pretrain.sh           # Pretraining script
├── extract_features.sh   # Feature extraction script
├── evaluate.sh           # Evaluation script
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The dataset will be automatically downloaded from HuggingFace when you first run the code.

## Usage

### 1. Pretraining

Train the DINO model on unlabeled data:

```bash
bash pretrain.sh
```

Or directly:
```bash
python train_dino_main.py \
    --data_config data_config.yaml \
    --model_config model_config.yaml \
    --train_config train_config.yaml \
    --device cuda
```

Checkpoints will be saved to `./checkpoints/`.

### 2. Feature Extraction

Extract features from the frozen encoder:

```bash
bash extract_features.sh checkpoints/checkpoint_latest.pth
```

Or directly:
```bash
python extract_features_main.py \
    --checkpoint checkpoints/checkpoint_latest.pth \
    --data_config data_config.yaml \
    --model_config model_config.yaml \
    --eval_config eval_config.yaml \
    --output_dir ./features \
    --device cuda
```

Features will be saved to `./features/features.pt`.

### 3. Evaluation

Run k-NN evaluation:

```bash
bash evaluate.sh features/features.pt
```

Or directly:
```bash
python knn_eval_main.py \
    --features features/features.pt \
    --eval_config eval_config.yaml
```

## Configuration

### Model Configuration (`model_config.yaml`)

- `model_name`: Choose `"vit_small_patch16_224"` (ViT-S/16, ~22M params) or `"vit_base_patch16_224"` (ViT-B/16, ~86M params)
- `use_cls_token`: `true` to use CLS token, `false` for mean-pooled patches

### Training Configuration (`train_config.yaml`)

- `batch_size`: Batch size (default: 64)
- `num_epochs`: Number of training epochs (default: 200)
- `learning_rate`: Initial learning rate (default: 0.0005)
- `local_crops_number`: Number of local crops (default: 8)

### Evaluation Configuration (`eval_config.yaml`)

- `k_values`: List of k values for k-NN (default: [10, 20, 50])
- `linear_probe`: Enable linear probe evaluation (default: false)

## Key Features

- **DINO-style SSL**: Teacher-student framework with momentum updates
- **Multi-crop augmentation**: 2 global views + 8 local views
- **ViT backbone**: ViT-S/16 or ViT-B/16 (<100M params)
- **k-NN evaluation**: Cosine similarity-based k-NN classification
- **Modular design**: Easy to swap components and run ablations

## Notes

- The encoder remains frozen during evaluation
- Only the training split is used for building the k-NN feature bank
- No adaptation is performed on test data
- Mixed precision training is used for efficiency

