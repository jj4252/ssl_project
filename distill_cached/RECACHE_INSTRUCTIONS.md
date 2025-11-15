# Instructions for Recaching with 96x96 and 10 Shards

## Changes Made

1. **Cache image size**: Changed from 256x256 to **96x96** (matches original image size)
2. **Number of shards**: Changed from 4 to **10** parallel shards

## Why Recache?

- Original images are 96x96, so caching at 256x256 was wasteful
- 96x96 cache will be ~13.8× smaller in disk space
- 10 shards will process faster (more parallelization)

## Steps to Recache

### 1. Delete Old Cache Files

**Option A: Delete entire cache directory** (if you want a clean start):
```bash
rm -rf /scratch/jj4252/Nov_14_distill/cache_images/*
```

**Option B: Delete only old cache files** (keep directory structure):
```bash
cd /scratch/jj4252/Nov_14_distill/cache_images
rm -f index_shard_*.json images_s*.pt
```

### 2. Submit 10 Parallel Jobs

Use the provided Slurm script:
```bash
cd DL_Final_Comp/distill_cached
sbatch precompute_cache_10shards.sh
```

Or run manually (for testing):
```bash
# Test with shard 0
python precompute_cache.py \
  --data_config data_config.yaml \
  --train_config train_config_kd.yaml \
  --batch_size 256 \
  --shard_id 0 \
  --num_shards 10
```

### 3. Expected Results

With 10 shards and 500K images:
- Each shard handles: 500K ÷ 10 = 50,000 images
- Cache files per shard: 50,000 ÷ 2048 = ~24 files
- Total cache files: 10 × 24 = ~240 files
- Index files: `index_shard_000.json` through `index_shard_009.json`

### 4. Disk Space Savings

**Old (256x256, float32):**
- Per image: 3 × 256 × 256 × 4 bytes = 786,432 bytes
- 500K images: ~375 GB

**New (96x96, float32):**
- Per image: 3 × 96 × 96 × 4 bytes = 110,592 bytes
- 500K images: ~53 GB
- **Savings: ~322 GB (86% reduction!)**

## Verification

After caching completes, verify:
```bash
cd /scratch/jj4252/Nov_14_distill/cache_images
ls index_shard_*.json | wc -l  # Should show 10
ls images_s*.pt | wc -l        # Should show ~240 files
```

## Training Configuration

The setup is now configured for:
- **Cache**: 96x96 (matches original image size)
- **Student**: Trains at 96x96 (native resolution)
- **Teacher**: Automatically upscales 96x96 → 224x224 before DINOv2 forward pass

### Key Changes Made:
1. `data_config.yaml`: `image_size: 96` (student training size)
2. `model_config_kd.yaml`: `student_img_size: 96` (student model size)
3. `distill_trainer.py`: Added automatic upscaling in `extract_teacher_features()`

### Training

Once recaching is complete, training will automatically:
- Load 96x96 images from cache
- Feed 96x96 images to student (native resolution)
- Upscale to 224x224 for teacher (DINOv2 expects 224x224)
- Compute distillation loss between student (96x96) and teacher (224x224) features

No additional changes needed - just run training as usual!

