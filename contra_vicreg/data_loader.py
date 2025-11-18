"""
Data loading module for MoCo-v3 training.
Supports both CIFAR-10/100 and HuggingFace dataset with cached tensors.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from torch.utils.data import Dataset, Subset
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import InterpolationMode


class CIFARDataset(Dataset):
    """CIFAR dataset for pretraining (unlabeled, using only images)"""
    def __init__(self, dataset_name="cifar10", root="./data", train=True, transform=None, max_samples=None, return_two_views=False):
        """
        Args:
            dataset_name: "cifar10" or "cifar100"
            root: Root directory for CIFAR data
            train: Use training split (True) or test split (False)
            transform: Transform to apply to images
            max_samples: Maximum number of samples to use (None = use all)
            return_two_views: If True, return two augmented views for SSL (Barlow Twins)
        """
        print(f"Loading {dataset_name.upper()} dataset...")
        
        # Load CIFAR dataset
        if dataset_name.lower() == "cifar10":
            self.dataset = CIFAR10(root=root, train=train, download=True, transform=None)
        elif dataset_name.lower() == "cifar100":
            self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'")
        
        # Limit to max_samples if specified
        if max_samples is not None and len(self.dataset) > max_samples:
            print(f"  Limiting dataset to {max_samples:,} samples (from {len(self.dataset):,})")
            indices = list(range(max_samples))
            self.dataset = Subset(self.dataset, indices)
        
        self.transform = transform
        self.return_two_views = return_two_views
        print(f"  Loaded {len(self.dataset):,} images")
        if return_two_views:
            print(f"  ✓ Returning two views per image for SSL (Barlow Twins)")
    
    def __getitem__(self, idx):
        # Get image (ignore label for pretraining)
        if isinstance(self.dataset, Subset):
            image, _ = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            image, _ = self.dataset[idx]
        
        if self.transform:
            if self.return_two_views:
                # Check if transform already returns two views (e.g., MoCoTransform)
                result = self.transform(image)
                if isinstance(result, tuple) and len(result) == 2:
                    # Transform already returns (view1, view2)
                    return result
                else:
                    # Transform returns single view, apply twice for two views
                    view1 = self.transform(image)
                    view2 = self.transform(image)  # Apply transform again (stochastic)
                    return view1, view2
            else:
                views = self.transform(image)  # Multi-crop returns list
                return views
        else:
            if self.return_two_views:
                return image, image  # Return same image twice if no transform
            return image
    
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
    def __init__(self, cache_dir: str, transform=None, max_shards: int = None, max_samples: int = None, return_two_views: bool = False):
        """
        Args:
            cache_dir: Directory containing cached shard files and index file(s)
            transform: Transform to apply to each image tensor (should normalize + augment)
            max_shards: Optional limit on number of shard files to use (for testing)
            max_samples: Optional limit on total number of samples to use (for testing)
            return_two_views: If True, return two augmented views for SSL (Barlow Twins)
        """
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        self.return_two_views = return_two_views
        
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
            self._index_to_global = []
            for shard_entry in all_shards:
                global_start = shard_entry["global_start"]
                num_samples = shard_entry["num_samples"]
                for i in range(num_samples):
                    self._index_to_global.append(global_start + i)
                    if max_samples is not None and len(self._index_to_global) >= max_samples:
                        break
                if max_samples is not None and len(self._index_to_global) >= max_samples:
                    break
            
            if max_samples is not None and max_samples > 0:
                self._index_to_global = self._index_to_global[:max_samples]
                used_global_indices = set(self._index_to_global)
                all_shards = [s for s in all_shards 
                             if any(used_global_indices & set(range(s["global_start"], s["global_start"] + s["num_samples"])))]
                print(f"  ⚠️  Limited to first {max_samples:,} samples (for testing)")
            
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
            
            if max_shards is not None and max_shards > 0:
                self.shards = self.shards[:max_shards]
                print(f"  ⚠️  Limited to first {max_shards} shard files (for testing)")
            
            total_samples_available = sum(s["num_samples"] for s in self.shards)
            self._index_to_global = list(range(total_samples_available))
            
            if max_samples is not None and max_samples > 0:
                self._index_to_global = self._index_to_global[:max_samples]
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
        
        # Sort shards by global_start for binary search
        self.shards.sort(key=lambda x: x.get("global_start", 0))
        
        # Build prefix sum for sequential access (for legacy format compatibility)
        self.shard_prefix_sum = [0]
        for shard in self.shards:
            self.shard_prefix_sum.append(self.shard_prefix_sum[-1] + shard["num_samples"])
        
        # In-memory cache for loaded shards
        self._shard_cache: Dict[str, torch.Tensor] = {}
        
        print(f"✓ Loaded cached dataset: {self.num_samples:,} samples in {len(self.shards)} cache files")
        print(f"  Cache image size: {self.cache_image_size}x{self.cache_image_size}")
        print(f"  Cache dtype: {self.cache_dtype}")
    
    def _load_shard(self, shard_path: str) -> torch.Tensor:
        """Load a shard file into memory (with caching)."""
        if shard_path not in self._shard_cache:
            full_path = self.cache_dir / shard_path
            if not full_path.exists():
                raise FileNotFoundError(f"Shard file not found: {full_path}")
            shard_data = torch.load(full_path, map_location='cpu')
            self._shard_cache[shard_path] = shard_data["images"]
        return self._shard_cache[shard_path]
    
    def _find_shard(self, idx: int) -> Tuple[int, int]:
        """Find which shard contains the given global index."""
        if self.shards and "global_start" in self.shards[0]:
            # Binary search based on global_start
            left, right = 0, len(self.shards) - 1
            best_shard_idx = -1
            
            while left <= right:
                mid = (left + right) // 2
                shard_info = self.shards[mid]
                shard_global_start = shard_info["global_start"]
                shard_num_samples = shard_info["num_samples"]
                shard_global_end = shard_global_start + shard_num_samples
                
                if shard_global_start <= idx < shard_global_end:
                    best_shard_idx = mid
                    break
                elif idx < shard_global_start:
                    right = mid - 1
                else:
                    left = mid + 1
            
            if best_shard_idx == -1:
                if idx >= self.shards[-1]["global_start"]:
                    best_shard_idx = len(self.shards) - 1
                else:
                    raise IndexError(f"Index {idx} not found in any cache file.")
            
            shard_info = self.shards[best_shard_idx]
            local_idx = idx - shard_info["global_start"]
            return best_shard_idx, local_idx
        else:
            # Legacy format: use prefix sum
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
    
    def __getitem__(self, idx: int):
        """Get a single image tensor with augmentations applied."""
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Map dataset index to global_start if using partial cache
        if hasattr(self, '_index_to_global') and self._index_to_global:
            global_idx = self._index_to_global[idx]
        else:
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
            if self.return_two_views:
                # Check if transform already returns two views (e.g., MoCoTransform)
                result = self.transform(img)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                else:
                    # Transform returns single view, apply twice for two views
                    view1 = self.transform(img)
                    view2 = self.transform(img)  # Apply transform again (stochastic)
                    return view1, view2
            else:
                img = self.transform(img)
                return img
        else:
            # No transform: return two views if requested, otherwise single view
            if self.return_two_views:
                return img, img
            else:
                return img
    
    def __len__(self) -> int:
        return self.num_samples


class RandomShardSubsetCachedDataset(Dataset):
    """
    Wrapper around CachedTensorDataset that samples random shards each epoch.
    
    Instead of randomly sampling individual indices, this samples entire shard files.
    For each epoch, it randomly picks N shards and uses all images from those shards
    sequentially. This provides much better cache locality and I/O performance.
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
        self._build_index_mapping()
        
        print(f"✓ RandomShardSubsetCachedDataset initialized:")
        print(f"  Total shards available: {self.total_shards}")
        print(f"  Shards per epoch: {self.shards_per_epoch}")
        print(f"  Samples per epoch: {self.num_samples:,}")
    
    def _resample_shards(self):
        """Resample random shards for the current epoch"""
        if self.seed is not None:
            random.seed(self.seed + self.current_epoch)
            torch.manual_seed(self.seed + self.current_epoch)
        
        # Sample random shard indices without replacement
        shard_indices = random.sample(range(self.total_shards), self.shards_per_epoch)
        shard_indices.sort()  # Sort for better sequential access
        
        # Get the actual shard entries
        self.current_shards = [self.all_shards[i] for i in shard_indices]
    
    def _build_index_mapping(self):
        """Build mapping from dataset index to (shard_idx, local_idx) within selected shards."""
        self.index_to_shard_local = []
        self.shard_start_indices = [0]
        
        for shard in self.current_shards:
            num_samples = shard["num_samples"]
            global_start = shard["global_start"]
            
            for local_idx in range(num_samples):
                self.index_to_shard_local.append((len(self.shard_start_indices) - 1, local_idx, global_start + local_idx))
            
            self.shard_start_indices.append(self.shard_start_indices[-1] + num_samples)
        
        self.num_samples = len(self.index_to_shard_local)
    
    def set_epoch(self, epoch: int):
        """Set the current epoch and resample shards."""
        if epoch != self.current_epoch:
            self.current_epoch = epoch
            self._resample_shards()
            self._build_index_mapping()
    
    def __getitem__(self, idx: int):
        """Get item from the current epoch's selected shards."""
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
            if self.base_dataset.return_two_views:
                # Check if transform already returns two views (e.g., MoCoTransform)
                result = self.base_dataset.transform(img)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                else:
                    # Transform returns single view, apply twice for two views
                    view1 = self.base_dataset.transform(img)
                    view2 = self.base_dataset.transform(img)  # Apply transform again (stochastic)
                    return view1, view2
            else:
                img = self.base_dataset.transform(img)
                return img
        else:
            # No transform: return two views if requested, otherwise single view
            if self.base_dataset.return_two_views:
                return img, img
            else:
                return img
    
    def __len__(self) -> int:
        return self.num_samples


def build_pretraining_dataloader(data_config: dict, train_config: dict) -> torch.utils.data.DataLoader:
    """
    Build pretraining DataLoader.
    
    Supports both CIFAR-10/100 and HuggingFace dataset with cached tensors.
    
    Args:
        data_config: Data configuration dictionary
        train_config: Training configuration dictionary
    
    Returns:
        DataLoader for pretraining
    """
    from transforms import SimpleTransform, FastMultiCropTransform, MoCoTransform, VICRegTransform
    from tensor_augmentations import TensorSimpleTransform, TensorMultiCropTransform
    
    # Check training mode and determine if we need two views
    training_mode = train_config.get('training_mode', 'kd')
    use_moco_aug = train_config.get('use_moco_aug', False)
    use_vicreg_aug = train_config.get('use_vicreg_aug', False)
    
    # For MoCo-v3, always use MoCo transform (returns two views)
    if training_mode == "moco_v3" or use_moco_aug:
        return_two_views = True
        print("✓ MoCo-v3 mode: will return two views per image")
    elif training_mode == "vicreg" or use_vicreg_aug:
        return_two_views = True
        print("✓ VICReg mode: will return two views per image")
    else:
        # Check if SSL is enabled (for Barlow Twins, need two views)
        ssl_config = train_config.get('ssl', {})
        use_ssl = ssl_config.get('enabled', False)
        return_two_views = use_ssl
    
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
        use_cached_tensors = False
    else:
        use_cached_tensors = True
    
    if use_cached_tensors:
        # Cached mode: load from preprocessed tensor shards and apply full augmentations
        cache_root = data_config.get('cache_root', data_config.get('cache_dir', './cache_images'))
        cache_root = os.path.expandvars(cache_root)
        
        # Get training image size
        image_size = data_config.get('image_size', 96)
        use_multi_crop = train_config.get('use_multi_crop', False)
        
        # For MoCo-v3 or VICReg, always use appropriate transform (works on tensors too)
        if training_mode == "moco_v3" or use_moco_aug:
            # MoCoTransform works on PIL images, but we have tensors
            # We'll use TensorSimpleTransform and apply it twice for two views
            transform = TensorSimpleTransform(
                image_size=image_size,
                scale=(0.2, 1.0)
            )
        elif training_mode == "vicreg" or use_vicreg_aug:
            # VICRegTransform works on PIL images, but we have tensors
            # We'll use TensorSimpleTransform and apply it twice for two views
            transform = TensorSimpleTransform(
                image_size=image_size,
                scale=(0.2, 1.0)
            )
        elif use_multi_crop:
            transform = TensorMultiCropTransform(
                global_crops_scale=tuple(data_config.get('global_crops_scale', [0.4, 1.0])),
                local_crops_scale=tuple(data_config.get('local_crops_scale', [0.05, 0.4])),
                local_crops_number=data_config.get('local_crops_number', 0),
                image_size=image_size,
                use_local_crops=train_config.get('use_local_crops', False)
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
        print(f"  Creating CachedTensorDataset with return_two_views={return_two_views}")
        base_dataset = CachedTensorDataset(
            cache_dir=cache_root, 
            transform=transform,
            max_shards=max_shards,
            max_samples=max_samples,
            return_two_views=return_two_views
        )
        print(f"  ✓ CachedTensorDataset created, return_two_views={base_dataset.return_two_views}")
        
        # Check if random shard sampling is enabled (preferred for performance)
        shards_per_epoch = data_config.get('shards_per_epoch', None)
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
        
        print(f"  Applying augmentations on cached tensors")
    else:
        # Original mode: load from CIFAR or HuggingFace/raw images
        dataset_name = data_config.get('dataset_name', 'cifar10')
        dataset_root = data_config.get('dataset_root', './data')
        max_samples = data_config.get('max_samples', None)
        image_size = data_config.get('image_size', 96)
        use_multi_crop = train_config.get('use_multi_crop', False)
        use_local_crops = train_config.get('use_local_crops', False)
        
        # For MoCo-v3 or VICReg, always use appropriate transform (returns two views)
        if training_mode == "moco_v3" or use_moco_aug:
            transform = MoCoTransform(image_size=image_size)
            print("✓ Using MoCo-v3 style augmentations (two views per image)")
        elif training_mode == "vicreg" or use_vicreg_aug:
            transform = VICRegTransform(image_size=image_size)
            print("✓ Using VICReg style augmentations (two views per image)")
        elif use_multi_crop:
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
        
        # Check if dataset_name is CIFAR or HuggingFace
        if dataset_name.lower() in ['cifar10', 'cifar100']:
            # Create CIFAR dataset
            dataset = CIFARDataset(
                dataset_name=dataset_name,
                root=dataset_root,
                train=True,
                transform=transform,
                max_samples=max_samples,
                return_two_views=return_two_views
            )
            print(f"✓ Using {dataset_name.upper()} dataset: {len(dataset):,} images")
        else:
            # HuggingFace dataset (not cached)
            raise NotImplementedError(
                f"HuggingFace dataset '{dataset_name}' requires cached tensors. "
                f"Please run precompute_cache.py first or set use_cached_tensors=false to use CIFAR."
            )
    
    # Build DataLoader
    num_workers = train_config.get('num_workers', data_config.get('num_workers', 4))
    persistent_workers = train_config.get('persistent_workers', data_config.get('persistent_workers', False))
    prefetch_factor = train_config.get('prefetch_factor', data_config.get('prefetch_factor', 2)) if num_workers > 0 else None
    pin_memory = train_config.get('pin_memory', data_config.get('pin_memory', True))
    
    # For MoCo-v3, drop_last=True is important for queue management
    # For VICReg, we can also drop last for consistent batch sizes
    drop_last = (training_mode == "moco_v3" or use_moco_aug or training_mode == "vicreg" or use_vicreg_aug)
    
    # Determine if we should shuffle
    # If using RandomShardSubsetCachedDataset, we already randomize shards per epoch,
    # so we can disable shuffle for better sequential access within shards
    should_shuffle = not isinstance(dataset, RandomShardSubsetCachedDataset)
    
    # Custom collate function for two-view batches
    def collate_two_views(batch):
        """Collate function for batches that return (view1, view2) tuples"""
        if len(batch) == 0:
            return batch
        
        # Check if first item is a tuple of two views
        first_item = batch[0]
        if isinstance(first_item, tuple) and len(first_item) == 2:
            # Batch is list of (view1, view2) tuples
            # Collate into (batch_view1, batch_view2) tuple of tensors
            view1_list = [item[0] for item in batch]
            view2_list = [item[1] for item in batch]
            view1_batch = torch.stack(view1_list)
            view2_batch = torch.stack(view2_list)
            return (view1_batch, view2_batch)
        elif isinstance(first_item, torch.Tensor):
            # Single tensor - this shouldn't happen if return_two_views=True
            # But handle it gracefully by duplicating
            print(f"⚠️ Warning: collate_two_views received single tensor instead of tuple. "
                  f"First item type: {type(first_item)}, shape: {first_item.shape if hasattr(first_item, 'shape') else 'N/A'}")
            # Fallback: treat as single view and duplicate
            batch_tensor = torch.stack(batch)
            return (batch_tensor, batch_tensor)
        else:
            # Fallback to default collate
            print(f"⚠️ Warning: collate_two_views received unexpected type: {type(first_item)}")
            return torch.utils.data.default_collate(batch)
    
    # Use custom collate for MoCo-v3 (two views) or when return_two_views is True
    collate_fn = collate_two_views if return_two_views else None
    if return_two_views:
        print(f"  ✓ Using custom collate function for two-view batches")
    else:
        print(f"  Using default collate function (return_two_views=False)")
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=should_shuffle,  # Disable shuffle for shard-level sampling (better cache locality)
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn
    )
    
    return loader


class CIFAREvalDataset(Dataset):
    """CIFAR dataset for evaluation (labeled)"""
    def __init__(self, dataset_name="cifar10", root="./data", train=False, transform=None):
        """
        Args:
            dataset_name: "cifar10" or "cifar100"
            root: Root directory for CIFAR data
            train: Use training split (True) or test split (False)
            transform: Transform to apply to images
        """
        print(f"Loading {dataset_name.upper()} evaluation dataset...")
        
        if dataset_name.lower() == "cifar10":
            self.dataset = CIFAR10(root=root, train=train, download=True, transform=None)
        elif dataset_name.lower() == "cifar100":
            self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'")
        
        self.transform = transform
        print(f"  Loaded {len(self.dataset):,} images")
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)

