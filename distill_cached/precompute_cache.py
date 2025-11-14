"""
Precompute cache script for 2-stage training pipeline with parallel sharded processing.

This script preprocesses all images from the pretraining dataset and saves them
as cached tensor shards for fast loading during training. Supports parallel processing
across multiple shards (e.g., 4 Slurm array jobs).

Usage:
    # Single shard (for testing)
    python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml --shard_id 0 --num_shards 4
    
    # Parallel execution (4 Slurm array jobs)
    #SBATCH --array=0-3
    python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml --shard_id $SLURM_ARRAY_TASK_ID --num_shards 4
"""

import argparse
import yaml
import os
import json
import math
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset

from data_loader import build_precompute_dataset


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def precompute_cache(
    data_config: dict,
    train_config: dict = None,
    batch_size: int = 256,
    shard_id: int = 0,
    num_shards: int = 1
):
    """
    Precompute and cache preprocessed image tensors with resumable, sharded support.
    
    This function processes a single shard (slice) of the dataset and can safely resume
    if interrupted. Multiple instances can run in parallel, each handling a different shard.
    
    Args:
        data_config: Data configuration dictionary
        train_config: Optional training config (for compatibility)
        batch_size: Batch size for processing (default: 256)
        shard_id: 0-based index of this shard [0, num_shards-1]
        num_shards: Total number of shards (default: 1 for single-process mode)
    """
    # Validate shard_id
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards-1}], got {shard_id}")
    
    # Get cache configuration (new settings take precedence)
    cache_root = data_config.get('cache_root', data_config.get('cache_dir', './cache_images'))
    cache_root = os.path.expandvars(cache_root)
    cache_image_size = data_config.get('cache_image_size', 256)
    cache_dtype = data_config.get('cache_dtype', 'float32')
    cache_shard_size = data_config.get('cache_shard_size', 2048)  # Default to 2048
    
    # Convert dtype string to torch dtype
    if cache_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    
    # Create cache directory
    cache_path = Path(cache_root)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache root: {cache_root}")
    print(f"Cache image size: {cache_image_size}x{cache_image_size}")
    print(f"Cache dtype: {cache_dtype}")
    print(f"Cache shard size: {cache_shard_size} samples per cache file")
    print(f"Processing shard {shard_id} of {num_shards}")
    
    # Build full dataset for precomputing
    print("\nBuilding precompute dataset...")
    dataset = build_precompute_dataset(data_config)
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples:,}")
    
    # Split dataset into num_shards contiguous slices
    per_shard = math.ceil(total_samples / num_shards)
    global_start = shard_id * per_shard
    global_end = min(total_samples, (shard_id + 1) * per_shard)
    num_samples_in_slice = global_end - global_start
    
    print(f"\nShard {shard_id} assignment:")
    print(f"  Global indices: [{global_start:,}, {global_end:,})")
    print(f"  Samples in this shard: {num_samples_in_slice:,}")
    
    # Per-shard index file
    index_file = cache_path / f"index_shard_{shard_id:03d}.json"
    
    # Initialize or load existing index
    if index_file.exists():
        print(f"\n✓ Found existing index: {index_file}")
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        # Validate index matches current configuration
        if index.get('shard_id') != shard_id:
            raise ValueError(f"Index file shard_id mismatch: expected {shard_id}, got {index.get('shard_id')}")
        if index.get('num_shards') != num_shards:
            raise ValueError(f"Index file num_shards mismatch: expected {num_shards}, got {index.get('num_shards')}")
        if index.get('global_start') != global_start:
            raise ValueError(f"Index file global_start mismatch: expected {global_start}, got {index.get('global_start')}")
        
        processed_samples = index.get('processed_samples', 0)
        index_entries = index.get('shards', [])
        local_shard_idx = len(index_entries)
        
        print(f"  Found {len(index_entries)} existing cache files")
        print(f"  Processed samples: {processed_samples:,} / {num_samples_in_slice:,}")
        print(f"  Resuming from local_shard_idx: {local_shard_idx}")
        
        # Verify existing cache files exist on disk
        for entry in index_entries:
            shard_file = cache_path / entry['shard_path']
            if not shard_file.exists():
                print(f"  ⚠️  Warning: Cache file missing: {entry['shard_path']}")
                print(f"     Will regenerate this file")
    else:
        print(f"\n✓ No existing index found, starting fresh")
        processed_samples = 0
        local_shard_idx = 0
        index_entries = []
        
        # Initialize index structure
        index = {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "global_start": global_start,
            "global_end": global_end,
            "num_samples_in_slice": num_samples_in_slice,
            "processed_samples": 0,
            "cache_image_size": cache_image_size,
            "cache_dtype": cache_dtype,
            "cache_shard_size": cache_shard_size,
            "dataset_name": data_config.get('dataset_name', 'unknown'),
            "shards": []
        }
    
    # Create subset dataset for this shard's slice
    # If resuming, skip already processed samples
    slice_start = global_start + processed_samples
    slice_end = global_end
    
    if processed_samples > 0:
        print(f"\n⏭️  Skipping {processed_samples:,} already processed samples...")
        remaining_indices = list(range(slice_start, slice_end))
        shard_dataset = Subset(dataset, remaining_indices)
        print(f"  Created subset with {len(remaining_indices):,} remaining samples")
    else:
        shard_indices = list(range(global_start, global_end))
        shard_dataset = Subset(dataset, shard_indices)
        print(f"  Created subset with {num_samples_in_slice:,} samples")
    
    # Build DataLoader for this shard's slice
    num_workers = data_config.get('num_workers', 4)
    loader = DataLoader(
        shard_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False  # Deterministic order (required for resumability)
    )
    
    # Process and write cache files
    print(f"\nProcessing images and writing cache files...")
    if processed_samples > 0:
        print(f"  Resuming from local_shard_idx {local_shard_idx}, processed_samples {processed_samples:,}")
    
    shard_buffer = []
    remaining_samples = num_samples_in_slice - processed_samples
    progress_bar = tqdm(total=remaining_samples, desc=f"Shard {shard_id} progress", initial=0)
    
    for batch in loader:
        # Handle batch format
        if isinstance(batch, list):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
            print(f"⚠️  Warning: Unexpected batch type {type(batch)}, skipping...")
            continue
        
        # Add each image in batch to buffer
        for img in batch:
            # Move to CPU and convert to target dtype
            img_cpu = img.cpu()
            if img_cpu.dtype != torch_dtype:
                if torch_dtype == torch.float16:
                    img_cpu = img_cpu.half()
                else:
                    img_cpu = img_cpu.float()
            
            shard_buffer.append(img_cpu)
            processed_samples += 1
            
            # When buffer reaches cache_shard_size, write cache file
            if len(shard_buffer) >= cache_shard_size:
                # Stack and save
                shard_tensor = torch.stack(shard_buffer, dim=0)  # [N, 3, H, W]
                
                # Generate cache file name with shard prefix
                shard_filename = f"images_s{shard_id:03d}_{local_shard_idx:05d}.pt"
                shard_path = cache_path / shard_filename
                
                # Check if cache file already exists (shouldn't happen, but safety check)
                if shard_path.exists():
                    print(f"  ⚠️  Cache file {shard_filename} already exists, skipping write")
                else:
                    torch.save(
                        {"images": shard_tensor},
                        shard_path
                    )
                
                # Calculate global_start for this cache file
                cache_global_start = global_start + (processed_samples - len(shard_buffer))
                
                # Record in index
                index_entry = {
                    "shard_path": shard_filename,
                    "num_samples": len(shard_buffer),
                    "global_start": cache_global_start
                }
                index_entries.append(index_entry)
                
                # Update index metadata
                index["processed_samples"] = processed_samples
                index["shards"] = index_entries
                
                # Write index file incrementally (for safety)
                with open(index_file, 'w') as f:
                    json.dump(index, f, indent=2)
                
                print(f"  ✓ Wrote {shard_filename}: {len(shard_buffer)} samples "
                      f"(global idx {cache_global_start} to {cache_global_start + len(shard_buffer) - 1})")
                
                # Clear buffer and increment
                shard_buffer = []
                local_shard_idx += 1
        
        progress_bar.update(len(batch))
    
    progress_bar.close()
    
    # Write final cache file if there are leftover samples
    if len(shard_buffer) > 0:
        shard_tensor = torch.stack(shard_buffer, dim=0)
        
        # Generate cache file name
        shard_filename = f"images_s{shard_id:03d}_{local_shard_idx:05d}.pt"
        shard_path = cache_path / shard_filename
        
        # Check if final cache file already exists
        if shard_path.exists():
            print(f"  ⚠️  Final cache file {shard_filename} already exists, skipping write")
        else:
            torch.save(
                {"images": shard_tensor},
                shard_path
            )
        
        # Calculate global_start for final cache file
        cache_global_start = global_start + (processed_samples - len(shard_buffer))
        
        index_entry = {
            "shard_path": shard_filename,
            "num_samples": len(shard_buffer),
            "global_start": cache_global_start
        }
        index_entries.append(index_entry)
        
        print(f"  ✓ Wrote final cache file {shard_filename}: {len(shard_buffer)} samples")
        local_shard_idx += 1
    
    # Final index update with all metadata
    index["processed_samples"] = processed_samples
    index["shards"] = index_entries
    
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    # Verify completion for this shard
    total_cached = sum(entry['num_samples'] for entry in index_entries)
    if total_cached == num_samples_in_slice:
        print(f"\n✓ Shard {shard_id} precomputation complete!")
        print(f"  Samples in slice: {num_samples_in_slice:,}")
        print(f"  Cache files written: {len(index_entries)}")
        print(f"  Index file: {index_file}")
    else:
        print(f"\n⚠️  Shard {shard_id} precomputation incomplete!")
        print(f"  Expected: {num_samples_in_slice:,} samples")
        print(f"  Cached: {total_cached:,} samples")
        print(f"  Missing: {num_samples_in_slice - total_cached:,} samples")
        print(f"  Run this script again to resume and complete this shard.")
    
    print(f"\n  To use cached data, set use_cached_tensors: true in data_config.yaml")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute image cache for fast training (supports parallel sharded processing)"
    )
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data config YAML')
    parser.add_argument('--train_config', type=str, default=None,
                       help='Path to training config YAML (optional, for compatibility)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for processing (default: 256)')
    parser.add_argument('--shard_id', type=int, required=True,
                       help='0-based shard index [0, num_shards-1]')
    parser.add_argument('--num_shards', type=int, required=True,
                       help='Total number of shards (e.g., 4 for parallel processing)')
    args = parser.parse_args()
    
    # Validate arguments
    if args.shard_id < 0:
        raise ValueError(f"--shard_id must be >= 0, got {args.shard_id}")
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if args.shard_id >= args.num_shards:
        raise ValueError(f"--shard_id must be < --num_shards ({args.num_shards}), got {args.shard_id}")
    
    # Load configs
    data_cfg = load_config(args.data_config)
    train_cfg = None
    if args.train_config:
        train_cfg = load_config(args.train_config)
    
    # Run precomputation for this shard
    precompute_cache(
        data_cfg,
        train_cfg,
        batch_size=args.batch_size,
        shard_id=args.shard_id,
        num_shards=args.num_shards
    )


if __name__ == '__main__':
    main()
