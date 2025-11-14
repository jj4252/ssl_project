"""
Precompute cache script for 2-stage training pipeline.

This script preprocesses all images from the pretraining dataset and saves them
as cached tensor shards for fast loading during training.

Usage:
    python precompute_cache.py --data_config data_config.yaml --train_config train_config_kd.yaml
"""

import argparse
import yaml
import os
import json
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from data_loader import build_precompute_dataset


def load_config(config_path: str) -> dict:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def precompute_cache(data_config: dict, train_config: dict = None, batch_size: int = 256):
    """
    Precompute and cache preprocessed image tensors with resumable support.
    
    This function can safely resume if interrupted:
    - Checks for existing shards and skips them
    - Continues from the last completed shard
    - Updates index.json incrementally
    
    Args:
        data_config: Data configuration dictionary
        train_config: Optional training config (for compatibility)
        batch_size: Batch size for processing (default: 256)
    """
    # Get cache configuration (new settings take precedence)
    cache_root = data_config.get('cache_root', data_config.get('cache_dir', './cache_images'))
    cache_root = os.path.expandvars(cache_root)
    cache_image_size = data_config.get('cache_image_size', 256)
    cache_dtype = data_config.get('cache_dtype', 'float32')
    cache_shard_size = data_config.get('cache_shard_size', 10000)
    
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
    print(f"Shard size: {cache_shard_size} samples per shard")
    
    # Build dataset for precomputing
    print("\nBuilding precompute dataset...")
    dataset = build_precompute_dataset(data_config)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples:,}")
    
    # Check for existing index and shards (resume support)
    index_file = cache_path / "index.json"
    existing_shards = set()
    index_entries = []
    start_shard_idx = 0
    start_global_idx = 0
    
    if index_file.exists():
        print(f"\n✓ Found existing index: {index_file}")
        with open(index_file, 'r') as f:
            existing_index = json.load(f)
            index_entries = existing_index.get('shards', [])
            start_shard_idx = len(index_entries)
            
            # Calculate how many samples were already processed
            if index_entries:
                last_entry = index_entries[-1]
                start_global_idx = last_entry['global_start'] + last_entry['num_samples']
                print(f"  Found {len(index_entries)} existing shards")
                print(f"  Last shard: {last_entry['shard_path']} ({last_entry['num_samples']} samples)")
                print(f"  Resuming from global_idx: {start_global_idx:,}")
                
                # Verify existing shards exist on disk
                for entry in index_entries:
                    shard_file = cache_path / entry['shard_path']
                    if shard_file.exists():
                        existing_shards.add(entry['shard_path'])
                    else:
                        print(f"  ⚠️  Warning: Shard file missing: {entry['shard_path']}")
                        print(f"     Will regenerate this shard")
            else:
                print(f"  Index exists but empty, starting fresh")
    else:
        print(f"\n✓ No existing index found, starting fresh")
    
    # Create a subset dataset if resuming (to skip already processed samples)
    if start_global_idx > 0:
        print(f"\n⏭️  Skipping {start_global_idx:,} already processed samples...")
        from torch.utils.data import Subset
        # Create subset starting from start_global_idx
        remaining_indices = list(range(start_global_idx, total_samples))
        dataset = Subset(dataset, remaining_indices)
        print(f"  Created subset with {len(remaining_indices):,} remaining samples")
    
    # Build DataLoader
    num_workers = data_config.get('num_workers', 4)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False  # Deterministic order (required for resumability)
    )
    
    # Process and write shards
    print(f"\nProcessing images and writing shards...")
    if start_global_idx > 0:
        print(f"  Resuming from shard {start_shard_idx}, global_idx {start_global_idx:,}")
    
    shard_buffer = []
    shard_idx = start_shard_idx
    global_idx = start_global_idx
    
    # Progress bar: show remaining samples
    remaining_samples = total_samples - start_global_idx
    progress_bar = tqdm(total=remaining_samples, desc="Processing images", initial=0)
    
    # Track if we need to break after processing remaining samples
    samples_processed = 0
    
    for batch in loader:
        # Handle batch format
        if isinstance(batch, list):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
            print(f"⚠️  Warning: Unexpected batch type {type(batch)}, skipping...")
            continue
        
        # Add each image in batch to buffer
        for img in batch:
            shard_buffer.append(img.cpu().clone())
            global_idx += 1
            samples_processed += 1
            
            if len(shard_buffer) >= cache_shard_size:
                # Write shard
                shard_tensor = torch.stack(shard_buffer, dim=0)  # [N, 3, H, W]
                
                # Convert to specified dtype
                if torch_dtype == torch.float16:
                    shard_tensor = shard_tensor.half()
                else:
                    shard_tensor = shard_tensor.float()
                
                shard_path = f"images_shard_{shard_idx:05d}.pt"
                shard_file = cache_path / shard_path
                
                # Check if shard already exists (shouldn't happen, but safety check)
                if shard_file.exists():
                    print(f"  ⚠️  Shard {shard_idx} already exists, skipping write")
                else:
                    torch.save(
                        {"images": shard_tensor, "indices": list(range(global_idx - len(shard_buffer), global_idx))},
                        shard_file
                    )
                
                # Record in index
                index_entry = {
                    "shard_path": shard_path,
                    "num_samples": len(shard_buffer),
                    "global_start": global_idx - len(shard_buffer)
                }
                index_entries.append(index_entry)
                
                # Update index.json incrementally (for safety)
                index = {
                    "num_samples": total_samples,
                    "shards": index_entries
                }
                with open(index_file, 'w') as f:
                    json.dump(index, f, indent=2)
                
                print(f"  ✓ Wrote shard {shard_idx}: {len(shard_buffer)} samples "
                      f"(global idx {global_idx - len(shard_buffer)} to {global_idx - 1})")
                
                # Clear buffer and increment
                shard_buffer = []
                shard_idx += 1
        
        progress_bar.update(len(batch))
    
    progress_bar.close()
    
    # Write final shard if there are leftover samples
    if len(shard_buffer) > 0:
        shard_tensor = torch.stack(shard_buffer, dim=0)
        
        # Convert to specified dtype
        if torch_dtype == torch.float16:
            shard_tensor = shard_tensor.half()
        else:
            shard_tensor = shard_tensor.float()
        
        shard_path = f"images_shard_{shard_idx:05d}.pt"
        shard_file = cache_path / shard_path
        
        # Check if final shard already exists
        if shard_file.exists():
            print(f"  ⚠️  Final shard {shard_idx} already exists, skipping write")
        else:
            torch.save(
                {"images": shard_tensor, "indices": list(range(global_idx - len(shard_buffer), global_idx))},
                shard_file
            )
        
        index_entry = {
            "shard_path": shard_path,
            "num_samples": len(shard_buffer),
            "global_start": global_idx - len(shard_buffer)
        }
        index_entries.append(index_entry)
        
        print(f"  ✓ Wrote final shard {shard_idx}: {len(shard_buffer)} samples")
        shard_idx += 1
    
    # Final index update with metadata
    index = {
        "num_samples": total_samples,
        "shards": index_entries,
        "cache_image_size": cache_image_size,
        "cache_dtype": cache_dtype,
        "cache_shard_size": cache_shard_size,
        "config_snapshot": {
            "dataset_name": data_config.get('dataset_name', 'unknown'),
            "image_size": data_config.get('image_size', 224),
        }
    }
    
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    # Verify completion
    total_cached = sum(entry['num_samples'] for entry in index_entries)
    if total_cached == total_samples:
        print(f"\n✓ Cache precomputation complete!")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Total shards: {len(index_entries)}")
        print(f"  Index file: {index_file}")
        print(f"\n  To use cached data, set use_cached_tensors: true in data_config.yaml")
    else:
        print(f"\n⚠️  Cache precomputation incomplete!")
        print(f"  Expected: {total_samples:,} samples")
        print(f"  Cached: {total_cached:,} samples")
        print(f"  Missing: {total_samples - total_cached:,} samples")
        print(f"  Run this script again to resume and complete the cache.")


def main():
    parser = argparse.ArgumentParser(description="Precompute image cache for fast training")
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data config YAML')
    parser.add_argument('--train_config', type=str, default=None,
                       help='Path to training config YAML (optional, for compatibility)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for processing (default: 256)')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    train_cfg = None
    if args.train_config:
        train_cfg = load_config(args.train_config)
    
    # Run precomputation
    precompute_cache(data_cfg, train_cfg, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

