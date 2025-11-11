#!/bin/bash

# Evaluation Script
# Usage: bash evaluate.sh <features_path>

FEATURES=$1

if [ -z "$FEATURES" ]; then
    echo "Usage: bash evaluate.sh <features_path>"
    echo "Example: bash evaluate.sh features/features.pt"
    exit 1
fi

python knn_eval_main.py \
    --features $FEATURES \
    --eval_config eval_config.yaml

