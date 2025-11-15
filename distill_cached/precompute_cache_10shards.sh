#!/bin/bash
# Slurm script to precompute cache with 10 parallel shards
# Original images are 96x96, so cache at 96x96 (no upscaling needed)

#SBATCH --job-name=precompute_cache
#SBATCH --array=0-9
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

SHARD_ID=$SLURM_ARRAY_TASK_ID
NUM_SHARDS=10

python precompute_cache.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --batch_size 256 \
  --shard_id $SHARD_ID \
  --num_shards $NUM_SHARDS

