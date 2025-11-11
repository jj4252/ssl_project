#!/bin/bash

# Optimized DINO Pretraining Script
# Uses optimized configs for fast training

python train_dino_main.py \
    --data_config data_config_optimized.yaml \
    --model_config model_config.yaml \
    --train_config train_config_optimized.yaml \
    --device cuda \
    --resume_from ""

