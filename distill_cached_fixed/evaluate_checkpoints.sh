#!/bin/bash
# Script to evaluate multiple checkpoints on CIFAR10/100

# Configuration
CHECKPOINT_DIR="/scratch/$USER/Nov_14_distill/checkpoints"
MODEL_CONFIG="model_config_kd.yaml"
OUTPUT_DIR="./evaluation_results"

# Create output directory
mkdir -p $OUTPUT_DIR

# Evaluate on CIFAR10
echo "Evaluating on CIFAR10..."
python evaluate_checkpoints.py \
  --model_config $MODEL_CONFIG \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset cifar10 \
  --epochs 1 100 200 250 \
  --batch_size 128 \
  --num_workers 4 \
  --linear_probe_C 1.0 \
  --output_file $OUTPUT_DIR/cifar10_results.json \
  --device cuda

# Evaluate on CIFAR100
echo ""
echo "Evaluating on CIFAR100..."
python evaluate_checkpoints.py \
  --model_config $MODEL_CONFIG \
  --checkpoint_dir $CHECKPOINT_DIR \
  --dataset cifar100 \
  --epochs 1 100 200 250 \
  --batch_size 128 \
  --num_workers 4 \
  --linear_probe_C 1.0 \
  --output_file $OUTPUT_DIR/cifar100_results.json \
  --device cuda

echo ""
echo "✓ Evaluation complete! Results saved to $OUTPUT_DIR/"

