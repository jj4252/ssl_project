#!/bin/bash
# Knowledge Distillation Training Script
# Trains a lightweight ViT student to mimic DINOv2 teacher

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training with nohup
nohup python distill_trainer.py \
    --data_config data_config_optimized.yaml \
    --train_config train_config_kd.yaml \
    --model_config model_config_kd.yaml \
    > logs/distill_train_$(date +%F_%H-%M-%S).log 2>&1 &

echo "Training started in background. PID: $!"
echo "Log file: logs/distill_train_$(date +%F_%H-%M-%S).log"
echo "Monitor with: tail -f logs/distill_train_*.log"

