#!/bin/bash

# Feature Extraction Script
# Usage: bash extract_features.sh <checkpoint_path>

CHECKPOINT=$1
if [ -z "$CHECKPOINT" ]; then
    echo "Usage: bash extract_features.sh <checkpoint_path>"
    exit 1
fi

python extract_features_main.py \
    --checkpoint $CHECKPOINT \
    --data_config data_config.yaml \
    --model_config model_config.yaml \
    --eval_config eval_config.yaml \
    --output_dir ./features \
    --device cuda

