"""
Data loading module for DINO training with optional cached tensor support.
Modified for random subset sampling per epoch.

IMPORTANT: If you have a file named 'datasets.py' in your project directory,
it will conflict with the HuggingFace 'datasets' package. 
Please rename or delete any local 'datasets.py' file.
"""

import sys
import json
import os
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Check if there's a conflicting local datasets.py file
current_dir = Path(__file__).parent.absolute()
local_datasets_file = current_dir / 'datasets.py'

if local_datasets_file.exists():
    raise ImportError(
        f"ERROR: Found a local file 'datasets.py' at {local_datasets_file}\n"
        f"This conflicts with the HuggingFace 'datasets' package.\n"
        f"Please rename or delete this file. Suggested name: 'datasets_old.py' or 'datasets_backup.py'"
    )

# Import HuggingFace datasets package
try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError(
        f"Failed to import 'load_dataset' from HuggingFace datasets package.\n"
        f"Make sure you have installed it: pip install datasets\n"
        f"Original error: {e}"
    )

from torch.utils.data import Dataset, Subset
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _inspect_dataset_splits(dataset_name):
    """Helper function to inspect available splits in a dataset"""
    try:
        dataset_dict = load_dataset(dataset_name, split=None)
        if isinstance(dataset_dict, dict):
            return list(dataset_dict.keys())
        else:
            # If it's a DatasetDict, get the keys
            return list(dataset_dict.keys()) if hasattr(dataset_dict, 'keys') else ["train"]
    except:
        return ["train"]  # Default fallback


class PretrainDataset(Dataset):
    """Unlabeled pretraining dataset (~500k images)"""
    def __init__(self, transform=None):
        print("Loading pretraining dataset...")
        # Try different possible split names
        try:
            # First try eval_public/train
            self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                       split="eval_public/train")
        except ValueError:
            try:
                # Try train split
                self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                           split="train")
            except ValueError:
                # Try to inspect available splits
                splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                if splits:
                    self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                               split=splits[0])
                else:
                    raise ValueError("Could not find a valid split for the dataset")
        
        # Try to enable streaming if available
        try:
            if hasattr(self.dataset, 'set_format'):
                self.dataset.set_format('pil')
        except:
            pass  # If not supported, continue without
        
        self.transform = transform
        print(f"Loaded {len(self.dataset)} pretraining images")
    
    def __getitem__(self, idx):
        # Optimized access - try direct indexing first
        try:
            item = self.dataset[idx]
            # Handle both dict and direct image access
            if isinstance(item, dict):
                image = item["image"]
            else:
                image = item
        except:
            # Fallback to slower method if needed
            image = self.dataset[idx]["image"]
        
        if self.transform:
            views = self.transform(image)  # Multi-crop returns list
        else:
            views = image
        return views
    
    def __len__(self):
        return len(self.dataset)


class CachedTensorDataset(Dataset):
    """
    Dataset that loads preprocessed tensors from cached shard files.
    
    This dataset reads from shard files created by precompute_cache.py.
    Supports both legacy single index.json format and new sharded format (index_shard_*.json).
    Each shard contains unnormalized tensors [N, 3, H, W] in range [0, 1].
    The transform should normalize and apply augmentations.
    """
    def __init__(self, cache_dir: str, transform=None, max_shards: int = None, max_samples: int = None):
        """
        Args:
            cache_dir: Directory containing cached shard files and index file(s)
            transform: Transform to apply to each image tensor (should normalize + augment)
            max_shards: Optional limit on number of shard files to use (for testing)
            max_samples: Optional limit on total number of samples to use (for testing)
        """
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        
        # Try to load sharded index files first (new format), fallback to legacy index.json
        shard_index_files = sorted(self.cache_dir.glob("index_shard_*.json"))
        legacy_index_path = self.cache_dir / "index.json"
        
        if shard_index_files:
            # New sharded format: load and merge all shard index files
            print(f"✓ Found {len(shard_index_files)} shard index files (parallel processing format)")
            all_shards = []
            cache_image_size = None
            cache_dtype = None
            total_samples = 0
            
            for shard_index_path in shard_index_files:
                with open(shard_index_path, 'r') as f:
                    shard_index = json.load(f)
                
                # Extract metadata from first shard (should be consistent across all)
                if cache_image_size is None:
                    cache_image_size = shard_index.get("cache_image_size", 256)
                    cache_dtype = shard_index.get("cache_dtype", "float32")
                
                # Add all cache files from this shard index
                for cache_entry in shard_index.get("shards", []):
                    all_shards.append(cache_entry)
                
                # Accumulate total samples
                total_samples += shard_index.get("num_samples_in_slice", 0)
            
            # Sort all cache files by global_start to maintain order
            all_shards.sort(key=lambda x: x["global_start"])
            
            # Apply subset limits if specified (for testing)
            if max_shards is not None and max_shards > 0:
                all_shards = all_shards[:max_shards]
                print(f"  ⚠️  Limited to first {max_shards} shard files (for testing)")
            
            # Build mapping from dataset index to global_start for partial cache support
            # When only a subset of shards exists, we need to map dataset indices (0-N) 
            # to the actual global_start values in cache files
            self._index_to_global = []
            for shard_entry in all_shards:
                global_start = shard_entry["global_start"]
                num_samples = shard_entry["num_samples"]
                # Add all indices covered by this cache file
                for i in range(num_samples):
                    self._index_to_global.append(global_start + i)
                    # Apply max_samples limit if specified
                    if max_samples is not None and len(self._index_to_global) >= max_samples:
                        break
                if max_samples is not None and len(self._index_to_global) >= max_samples:
                    break
            
            # Apply max_samples limit if specified
            if max_samples is not None and max_samples > 0:
                self._index_to_global = self._index_to_global[:max_samples]
                # Also limit shards to only those that are actually used
                used_global_indices = set(self._index_to_global)
                all_shards = [s for s in all_shards 
                             if any(used_global_indices & set(range(s["global_start"], s["global_start"] + s["num_samples"])))]
                print(f"  ⚠️  Limited to first {max_samples:,} samples (for testing)")
            
            # Set num_samples to actual available samples (may be less than sum if gaps exist)
            self.num_samples = len(self._index_to_global)
            self.shards = all_shards
            self.cache_image_size = cache_image_size
            self.cache_dtype = cache_dtype
            
            print(f"  Merged {len(shard_index_files)} shard indices into {len(all_shards)} cache files")
            if self.num_samples != total_samples:
                print(f"  ⚠️  Note: {total_samples:,} samples in shard indices, but {self.num_samples:,} actually available (gaps may exist)")
        elif legacy_index_path.exists():
            # Legacy format: single index.json
            print(f"✓ Found legacy index.json format")
            with open(legacy_index_path, 'r') as f:
                index = json.load(f)
            
            self.shards = index["shards"]
            self.cache_image_size = index.get("cache_image_size", 256)
            self.cache_dtype = index.get("cache_dtype", "float32")
            
            # Apply subset limits if specified (for testing)
            if max_shards is not None and max_shards > 0:
                self.shards = self.shards[:max_shards]
                print(f"  ⚠️  Limited to first {max_shards} shard files (for testing)")
            
            # Build index mapping for legacy format (sequential, no global_start)
            # Legacy format assumes sequential shards starting from 0
            total_samples_available = sum(s["num_samples"] for s in self.shards)
            self._index_to_global = list(range(total_samples_available))
            
            if max_samples is not None and max_samples > 0:
                self._index_to_global = self._index_to_global[:max_samples]
                # Limit shards to only those needed
                samples_per_shard = self.shards[0]["num_samples"] if self.shards else 0
                if samples_per_shard > 0:
                    num_shards_needed = (max_samples + samples_per_shard - 1) // samples_per_shard
                    self.shards = self.shards[:num_shards_needed]
                print(f"  ⚠️  Limited to first {max_samples:,} samples (for testing)")
            
            self.num_samples = len(self._index_to_global)
        else:
            raise FileNotFoundError(
                f"No index files found in {self.cache_dir}\n"
                f"Expected either:\n"
                f"  - New format: index_shard_*.json files\n"
                f"  - Legacy format: index.json\n"
                f"Please run precompute_cache.py first to create the cache."
            )
        
        # Build mapping for efficient idx -> shard lookup
        # For parallel sharded format, we need to map global dataset indices to cache files
        # Each cache file has a global_start that indicates its position in the original dataset
        # We'll use binary search on global_start values to find the right cache file
        
        # Sort shards by global_start for binary search
        self.shards.sort(key=lambda x: x["global_start"])
        
        # Build prefix sum for sequential access (for legacy format compatibility)
        # But for sharded format, we'll use global_start-based lookup
        self.shard_prefix_sum = [0]
        for shard in self.shards:
            self.shard_prefix_sum.append(self.shard_prefix_sum[-1] + shard["num_samples"])
        
        # In-memory cache for loaded shards (per-epoch caching)
        self._shard_cache: Dict[str, torch.Tensor] = {}
        
        print(f"✓ Loaded cached dataset: {self.num_samples:,} samples in {len(self.shards)} cache files")
        print(f"  Cache image size: {self.cache_image_size}x{self.cache_image_size}")
        print(f"  Cache dtype: {self.cache_dtype}")
    
    def _load_shard(self, shard_path: str) -> torch.Tensor:
        """
        Load a shard file into memory (with caching).
        
        Args:
            shard_path: Relative path to shard file (e.g., "images_shard_00000.pt")
        
        Returns:
            Tensor of shape [N, 3, H, W]
        """
        if shard_path not in self._shard_cache:
            full_path = self.cache_dir / shard_path
            if not full_path.exists():
                raise FileNotFoundError(f"Shard file not found: {full_path}")
            
            shard_data = torch.load(full_path, map_location='cpu')
            self._shard_cache[shard_path] = shard_data["images"]
        
        return self._shard_cache[shard_path]
    
    def _find_shard(self, idx: int) -> Tuple[int, int]:
        """
        Find which shard contains the given global index.
        
        Args:
            idx: Global sample index (0-based, corresponds to original dataset position)
        
        Returns:
            Tuple of (shard_index, local_index_within_shard)
        """
        # Check if shards have global_start (parallel sharded format)
        # If they do, use global_start-based lookup
        if self.shards and "global_start" in self.shards[0]:
            # Binary search based on global_start to find the cache file containing this index
            # Cache files are sorted by global_start, and each covers [global_start, global_start + num_samples)
            left, right = 0, len(self.shards) - 1
            best_shard_idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                shard_info = self.shards[mid]
                shard_global_start = shard_info["global_start"]
                shard_num_samples = shard_info["num_samples"]
                shard_global_end = shard_global_start + shard_num_samples
                
                if shard_global_start <= idx < shard_global_end:
                    # Found the shard containing this index
                    best_shard_idx = mid
                    break
                elif idx < shard_global_start:
                    # Index is before this shard, search left
                    right = mid - 1
                else:
                    # Index is after this shard, search right
                    left = mid + 1
            
            if best_shard_idx == -1:
                # Fallback: use the last shard if index is at the boundary
                if idx >= self.shards[-1]["global_start"]:
                    best_shard_idx = len(self.shards) - 1
                else:
                    raise IndexError(
                        f"Index {idx} not found in any cache file. "
                        f"Valid range: [0, {self.shards[-1]['global_start'] + self.shards[-1]['num_samples']})"
                    )
            
            shard_info = self.shards[best_shard_idx]
            local_idx = idx - shard_info["global_start"]
            
            # Safety check
            if local_idx < 0 or local_idx >= shard_info["num_samples"]:
                raise IndexError(
                    f"Local index {local_idx} out of range for shard {best_shard_idx} "
                    f"(global_start={shard_info['global_start']}, num_samples={shard_info['num_samples']})"
                )
            
            return best_shard_idx, local_idx
        else:
            # Legacy format: use prefix sum (sequential)
            left, right = 0, len(self.shard_prefix_sum) - 1
            while left < right:
                mid = (left + right + 1) // 2
                if self.shard_prefix_sum[mid] <= idx:
                    left = mid
                else:
                    right = mid - 1
            
            shard_idx = left
            local_idx = idx - self.shard_prefix_sum[shard_idx]
            return shard_idx, local_idx
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a single image tensor with augmentations applied.
        
        Args:
            idx: Dataset sample index (0-based, relative to available cache)
        
        Returns:
            Tensor of shape [3, H, W] (normalized and augmented)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Map dataset index to global_start if using partial cache
        if hasattr(self, '_index_to_global') and self._index_to_global:
            # Map dataset index to actual global index in cache files
            global_idx = self._index_to_global[idx]
        else:
            # Legacy format or full cache: use index directly
            global_idx = idx
        
        # Find which shard contains this global index
        shard_idx, local_idx = self._find_shard(global_idx)
        shard_info = self.shards[shard_idx]
        
        # Load shard (cached in memory)
        shard_tensor = self._load_shard(shard_info["shard_path"])
        
        # Extract the specific image (unnormalized, [0, 1])
        img = shard_tensor[local_idx].clone()  # [3, H, W], range [0, 1]
        
        # Convert to float32 if needed
        if img.dtype != torch.float32:
            img = img.float()
        
        # Apply transform (should normalize + apply augmentations)
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def clear_shard_cache(self):
        """Clear shard cache (useful for memory management)"""
        self._shard_cache.clear()
    
    def __len__(self) -> int:
        return self.num_samples


class RandomShardSubsetCachedDataset(Dataset):
    """
    Wrapper around CachedTensorDataset that samples random shards each epoch.
    
    Instead of randomly sampling individual indices, this samples entire shard files.
    For each epoch, it randomly picks N shards and uses all images from those shards
    sequentially. This provides much better cache locality and I/O performance.
    
    Key features:
    - Samples random shard files (not individual indices)
    - Uses all images from selected shards sequentially
    - Better cache locality and I/O patterns
    - Resamples shards at the start of each epoch (via set_epoch)
    """
    def __init__(self, base_dataset: CachedTensorDataset, shards_per_epoch: int, seed: int = None):
        """
        Args:
            base_dataset: The full CachedTensorDataset to sample from
            shards_per_epoch: Number of random shard files to use per epoch
            seed: Optional random seed (for reproducibility)
        """
        self.base_dataset = base_dataset
        self.shards_per_epoch = shards_per_epoch
        self.seed = seed
        self.current_epoch = 0
        
        # Get all available shards from base dataset
        self.all_shards = base_dataset.shards.copy()
        self.total_shards = len(self.all_shards)
        
        # Validate shards_per_epoch
        if shards_per_epoch > self.total_shards:
            print(f"⚠️  Warning: shards_per_epoch ({shards_per_epoch}) > total_shards ({self.total_shards}), using all shards")
            self.shards_per_epoch = self.total_shards
        
        # Generate initial random shard selection
        self._resample_shards()
        
        # Build index mapping for current shard selection
        self._build_index_mapping()
        
        print(f"✓ RandomShardSubsetCachedDataset initialized:")
        print(f"  Total shards available: {self.total_shards}")
        print(f"  Shards per epoch: {self.shards_per_epoch}")
        print(f"  Samples per epoch: {self.num_samples:,}")
        print(f"  Coverage: {100 * self.shards_per_epoch / self.total_shards:.1f}% of shards per epoch")
    
    def _resample_shards(self):
        """Resample random shards for the current epoch"""
        if self.seed is not None:
            # Use epoch-specific seed for reproducibility
            random.seed(self.seed + self.current_epoch)
            torch.manual_seed(self.seed + self.current_epoch)
        
        # Sample random shard indices without replacement
        shard_indices = random.sample(range(self.total_shards), self.shards_per_epoch)
        shard_indices.sort()  # Sort for better sequential access
        
        # Get the actual shard entries
        self.current_shards = [self.all_shards[i] for i in shard_indices]
    
    def _build_index_mapping(self):
        """
        Build mapping from dataset index to (shard_idx, local_idx) within selected shards.
        This allows sequential access to images within the selected shards.
        """
        self.index_to_shard_local = []
        self.shard_start_indices = [0]  # Prefix sum for shard boundaries
        
        for shard in self.current_shards:
            num_samples = shard["num_samples"]
            global_start = shard["global_start"]
            
            # Add all indices from this shard
            for local_idx in range(num_samples):
                self.index_to_shard_local.append((len(self.shard_start_indices) - 1, local_idx, global_start + local_idx))
            
            # Update prefix sum
            self.shard_start_indices.append(self.shard_start_indices[-1] + num_samples)
        
        self.num_samples = len(self.index_to_shard_local)
    
    def set_epoch(self, epoch: int):
        """
        Set the current epoch and resample shards.
        Call this at the start of each training epoch.
        """
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self._resample_shards()
            self._build_index_mapping()
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get item from the current epoch's selected shards.
        
        Args:
            idx: Index in the current epoch's subset (0 to num_samples-1)
        
        Returns:
            Tensor from base_dataset at the appropriate shard and local index
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Get shard index in selection, local index within shard, and global index
        shard_idx_in_selection, local_idx, global_idx = self.index_to_shard_local[idx]
        
        # Get the shard info from our selected shards
        shard_info = self.current_shards[shard_idx_in_selection]
        
        # Use base_dataset's shard loading method (reuses shard cache)
        shard_tensor = self.base_dataset._load_shard(shard_info["shard_path"])
        
        # Extract the specific image (unnormalized, [0, 1])
        img = shard_tensor[local_idx].clone()  # [3, H, W], range [0, 1]
        
        # Convert to float32 if needed
        if img.dtype != torch.float32:
            img = img.float()
        
        # Apply transform (should normalize + apply augmentations)
        if self.base_dataset.transform is not None:
            img = self.base_dataset.transform(img)
        
        return img
    
    def __len__(self) -> int:
        return self.num_samples


def build_precompute_dataset(data_config: dict) -> Dataset:
    """
    Build a dataset for precomputing cache (deterministic preprocessing).
    
    This returns a dataset that applies deterministic preprocessing to images
    for caching purposes. It does not include stochastic augmentations.
    
    Stage 1: Decode + resize only (no normalization, no augmentations)
    Normalization will be applied during Stage 2 (training) along with augmentations.
    """
    from transforms import SimpleTransform
    
    image_size = data_config.get('image_size', 224)
    
    # For precomputation, use deterministic transform (no augmentations)
    transform = SimpleTransform(
        image_size=image_size,
        scale=(1.0, 1.0)  # No random scaling, just resize
    )
    
    dataset = PretrainDataset(transform=transform)
    return dataset


def build_pretraining_dataloader(data_config: dict, train_config: dict) -> torch.utils.data.DataLoader:
    """
    Build DataLoader for pretraining.
    
    Supports two modes:
    1. Cached tensor mode: Load from preprocessed tensor shards (fast)
    2. Original mode: Load from HuggingFace/raw images (slower)
    
    Args:
        data_config: Data configuration dictionary
        train_config: Training configuration dictionary
    
    Returns:
        DataLoader for pretraining
    """
    from transforms import SimpleTransform, FastMultiCropTransform
    from tensor_augmentations import TensorSimpleTransform, TensorMultiCropTransform
    
    # Check for cached mode (new setting takes precedence)
    use_cached_tensors = data_config.get('use_cached_tensors', data_config.get('use_cached', None))
    
    # Auto-detect cache if not explicitly set
    if use_cached_tensors is None:
        cache_root = data_config.get('cache_root', data_config.get('cache_dir', './cache_images'))
        cache_root = os.path.expandvars(cache_root)
        cache_path = Path(cache_root)
        
        # Check for new sharded format first, then legacy format
        shard_index_files = list(cache_path.glob("index_shard_*.json"))
        legacy_index_path = cache_path / "index.json"
        
        if shard_index_files:
            print(f"✓ Auto-detected cache at {cache_root} ({len(shard_index_files)} shard index files), enabling cached tensor mode")
            use_cached_tensors = True
        elif legacy_index_path.exists():
            print(f"✓ Auto-detected cache at {cache_root} (legacy format), enabling cached tensor mode")
            use_cached_tensors = True
        else:
            use_cached_tensors = False
    elif use_cached_tensors is False:
        # Explicitly disabled
        use_cached_tensors = False
    else:
        # Explicitly enabled
        use_cached_tensors = True
    
    if use_cached_tensors:
        # Cached mode: load from preprocessed tensor shards and apply full augmentations
        cache_root = data_config.get('cache_root', data_config.get('cache_dir', './cache_images'))
        cache_root = os.path.expandvars(cache_root)
        
        # Get training image size
        image_size = data_config.get('image_size', 224)
        use_multi_crop = train_config.get('use_multi_crop', False)
        use_local_crops = train_config.get('use_local_crops', False)
        
        # Create tensor-based augmentations (work on unnormalized tensors [0, 1])
        if use_multi_crop:
            transform = TensorMultiCropTransform(
                global_crops_scale=tuple(data_config.get('global_crops_scale', [0.4, 1.0])),
                local_crops_scale=tuple(data_config.get('local_crops_scale', [0.05, 0.4])),
                local_crops_number=data_config.get('local_crops_number', 0),
                image_size=image_size,
                use_local_crops=use_local_crops
            )
        else:
            transform = TensorSimpleTransform(
                image_size=image_size,
                scale=(0.2, 1.0)
            )
        
        # Get subset limits for testing (if specified)
        max_shards = data_config.get('max_shards', None)
        max_samples = data_config.get('max_samples', None)
        
        # Build base cached dataset (full dataset)
        base_dataset = CachedTensorDataset(
            cache_dir=cache_root, 
            transform=transform,
            max_shards=max_shards,
            max_samples=max_samples
        )
        
        # Check if random shard sampling is enabled (preferred for performance)
        shards_per_epoch = data_config.get('shards_per_epoch', None)
        # Fallback to old samples_per_epoch for backward compatibility
        if shards_per_epoch is None:
            samples_per_epoch = data_config.get('samples_per_epoch', None)
            if samples_per_epoch is not None:
                # Estimate shards needed (approximate)
                cache_shard_size = data_config.get('cache_shard_size', 2048)
                shards_per_epoch = max(1, (samples_per_epoch + cache_shard_size - 1) // cache_shard_size)
                print(f"⚠️  Note: samples_per_epoch is deprecated, using shards_per_epoch={shards_per_epoch} (estimated)")
        
        random_seed = data_config.get('random_subset_seed', None)
        
        if shards_per_epoch is not None and shards_per_epoch > 0:
            # Use random shard-level sampling (better performance)
            dataset = RandomShardSubsetCachedDataset(
                base_dataset=base_dataset,
                shards_per_epoch=shards_per_epoch,
                seed=random_seed
            )
            print(f"✓ Using random shard subset cached dataset:")
            print(f"  Full cache: {len(base_dataset):,} samples in {len(base_dataset.shards)} shards")
        else:
            # Use full cached dataset
            dataset = base_dataset
            print(f"✓ Using full cached dataset from: {cache_root}")
        
        print(f"  Applying full DINO-style augmentations on cached tensors")
    else:
        # Original mode: load from HuggingFace/raw images
        use_multi_crop = train_config.get('use_multi_crop', False)
        use_local_crops = train_config.get('use_local_crops', False)
        image_size = data_config.get('image_size', 224)
        
        if use_multi_crop:
            transform = FastMultiCropTransform(
                global_crops_scale=tuple(data_config.get('global_crops_scale', [0.4, 1.0])),
                local_crops_scale=tuple(data_config.get('local_crops_scale', [0.05, 0.4])),
                local_crops_number=data_config.get('local_crops_number', 0),
                image_size=image_size,
                use_local_crops=use_local_crops
            )
        else:
            transform = SimpleTransform(
                image_size=image_size,
                scale=(0.2, 1.0)
            )
        
        dataset = PretrainDataset(transform=transform)
        print(f"✓ Using original dataset (HuggingFace)")
    
    # Build DataLoader
    num_workers = train_config.get('num_workers', data_config.get('num_workers', 4))
    persistent_workers = train_config.get('persistent_workers', data_config.get('persistent_workers', False))
    prefetch_factor = train_config.get('prefetch_factor', data_config.get('prefetch_factor', 2)) if num_workers > 0 else None
    pin_memory = train_config.get('pin_memory', data_config.get('pin_memory', True))
    
    # Determine if we should shuffle
    # If using RandomShardSubsetCachedDataset, we already randomize shards per epoch,
    # so we can disable shuffle for better sequential access within shards
    should_shuffle = not isinstance(dataset, RandomShardSubsetCachedDataset)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=should_shuffle,  # Disable shuffle for shard-level sampling (better cache locality)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    return loader


class EvalDataset(Dataset):
    """Labeled evaluation dataset"""
    def __init__(self, split="train", transform=None):
        print(f"Loading evaluation dataset ({split})...")
        # Try different possible split names
        try:
            # First try eval_public/{split}
            self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                       split=f"eval_public/{split}")
        except ValueError:
            try:
                # Try {split} directly
                self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                           split=split)
            except ValueError:
                # Try to inspect available splits
                splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                if splits:
                    self.dataset = load_dataset("tsbpp/fall2025_deeplearning",
                                               split=splits[0])
                else:
                    raise ValueError(f"Could not find a valid split '{split}' for the dataset")
        
        self.transform = transform
        print(f"Loaded {len(self.dataset)} evaluation images")
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item.get("label", -1)  # Default to -1 if no label
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)
