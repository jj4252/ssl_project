# Checkpoint Evaluation on CIFAR10/100

This script evaluates multiple checkpoints on CIFAR10/100 using linear probing.

## Overview

The `evaluate_checkpoints.py` script:
1. Loads checkpoints from different epochs (1, 100, 200, 250, and best)
2. Extracts features from CIFAR10/100 datasets using each checkpoint
3. Trains linear probes (logistic regression) for each checkpoint
4. Compares and reports performance across all checkpoints

## Usage

### Basic Usage

```bash
python evaluate_checkpoints.py \
  --model_config model_config_kd.yaml \
  --checkpoint_dir /scratch/$USER/Nov_14_distill/checkpoints \
  --dataset cifar10 \
  --epochs 1 100 200 250 \
  --output_file evaluation_results.json
```

### Using the Shell Script

```bash
./evaluate_checkpoints.sh
```

This will evaluate on both CIFAR10 and CIFAR100.

## Arguments

- `--model_config`: Path to model config YAML (required)
- `--checkpoint_dir`: Directory containing checkpoints (required)
- `--dataset`: Dataset to evaluate on (`cifar10` or `cifar100`, default: `cifar10`)
- `--epochs`: List of epoch numbers to evaluate (default: `1 100 200 250`)
- `--batch_size`: Batch size for feature extraction (default: `128`)
- `--num_workers`: Number of data loading workers (default: `4`)
- `--linear_probe_C`: Regularization strength for linear probe (default: `1.0`)
- `--output_file`: Output file for results in JSON format (default: `evaluation_results.json`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

## Checkpoint Detection

The script automatically finds:
- **Epoch-specific checkpoints**: `checkpoint_epoch_{N}.pth` for epochs 1, 100, 200, 250
- **Best checkpoint**: `checkpoint_latest.pth` (if available)

If a checkpoint for a specified epoch is not found, a warning is printed and that epoch is skipped.

## Output

### Console Output

The script prints:
- Progress for each checkpoint
- Feature extraction progress
- Linear probe training progress
- Final accuracy for each checkpoint
- Summary table comparing all checkpoints
- Best checkpoint identification

Example output:
```
📊 EVALUATION SUMMARY
============================================================
Dataset: CIFAR10
Linear Probe C: 1.0

Results:
Checkpoint      Accuracy     Accuracy %   
----------------------------------------
best            0.8234       82.34%      
250             0.8156       81.56%      
200             0.8023       80.23%      
100             0.7891       78.91%      
1               0.6543       65.43%      

🏆 Best checkpoint: best (82.34%)
```

### JSON Output

Results are saved to a JSON file with the following structure:

```json
{
  "dataset": "cifar10",
  "linear_probe_C": 1.0,
  "use_cls_token": true,
  "results": {
    "1": {
      "checkpoint_path": "/path/to/checkpoint_epoch_1.pth",
      "accuracy": 0.6543,
      "accuracy_percent": 65.43,
      "train_samples": 50000,
      "test_samples": 10000,
      "feature_dim": 384
    },
    "100": { ... },
    "200": { ... },
    "250": { ... },
    "best": { ... }
  },
  "best_checkpoint": "best"
}
```

## How It Works

1. **Model Loading**: Builds the student model architecture from config
2. **Checkpoint Loading**: Loads each checkpoint's state dict into the model
3. **Feature Extraction**: 
   - Extracts features from CIFAR train set (for each checkpoint)
   - Extracts features from CIFAR test set (once, reused for all checkpoints)
   - Uses CLS token or mean-pooled patches based on config
   - Normalizes features to unit length
4. **Linear Probing**: 
   - Trains logistic regression on training features
   - Evaluates on test features
   - Reports accuracy
5. **Comparison**: Compares all checkpoints and identifies the best

## Notes

- **CIFAR Images**: CIFAR images are 32×32, but will be resized to match your model's `student_img_size` (96×96 in your config)
- **Feature Extraction**: Test features are extracted once and reused for all checkpoints (more efficient)
- **Linear Probe**: Uses scikit-learn's LogisticRegression with L-BFGS solver
- **Normalization**: Features are L2-normalized before linear probing (standard practice)

## Example: Evaluate Specific Epochs

```bash
python evaluate_checkpoints.py \
  --model_config model_config_kd.yaml \
  --checkpoint_dir /scratch/$USER/Nov_14_distill/checkpoints \
  --dataset cifar100 \
  --epochs 50 100 150 200 \
  --linear_probe_C 0.1 \
  --output_file cifar100_custom.json
```

## Troubleshooting

**Checkpoint not found**: Make sure checkpoint files exist in the specified directory. The script will skip missing checkpoints with a warning.

**Out of memory**: Reduce `--batch_size` or use `--device cpu` (slower but uses less GPU memory).

**Low accuracy**: Try adjusting `--linear_probe_C` (try values like 0.1, 1.0, 10.0).

