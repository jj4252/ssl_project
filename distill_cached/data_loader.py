"""
Data loading module for DINO training with optional cached tensor support.

IMPORTANT: If you have a file named 'datasets.py' in your project directory,
it will conflict with the HuggingFace 'datasets' package. 
Please rename or delete any local 'datasets.py' file.
"""

import sys
import json
import os
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

from torch.utils.data import Dataset
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
            # First try 'pretrain' split
            self.dataset = load_dataset("tsbpp/fall2025_deeplearning", split="pretrain")
        except ValueError:
            try:
                # If that fails, try 'train' split (which might be the unlabeled data)
                self.dataset = load_dataset("tsbpp/fall2025_deeplearning", split="train")
                print("Note: Using 'train' split for pretraining (assuming it's unlabeled)")
            except ValueError as e:
                # If both fail, show available splits
                available_splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                raise ValueError(
                    f"Could not find pretraining split. Available splits: {available_splits}\n"
                    f"Please check the dataset structure. Original error: {e}"
                )
        
        # Set format for faster access (if supported)
        try:
            self.dataset = self.dataset.with_format("pil")  # Keep as PIL for transforms
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
    Each shard contains a tensor of shape [N, 3, H, W] and is loaded lazily.
    """
    def __init__(self, cache_dir: str, transform=None):
        """
        Args:
            cache_dir: Directory containing cached shard files and index.json
            transform: Optional transform to apply to each image tensor
        """
        self.cache_dir = Path(cache_dir)
        self.transform = transform
        
        # Load index.json
        index_path = self.cache_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_path}\n"
                f"Please run precompute_cache.py first to create the cache."
            )
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        self.num_samples = index["num_samples"]
        self.shards = index["shards"]
        
        # Build prefix sum for efficient idx -> shard mapping
        self.shard_prefix_sum = [0]
        for shard in self.shards:
            self.shard_prefix_sum.append(self.shard_prefix_sum[-1] + shard["num_samples"])
        
        # In-memory cache for loaded shards
        self._shard_cache: Dict[str, torch.Tensor] = {}
        
        print(f"✓ Loaded cached dataset: {self.num_samples} samples in {len(self.shards)} shards")
    
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
            idx: Global sample index
        
        Returns:
            Tuple of (shard_index, local_index_within_shard)
        """
        # Binary search for the shard containing this index
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
        Get a single image tensor.
        
        Args:
            idx: Global sample index
        
        Returns:
            Tensor of shape [3, H, W] (or transformed version)
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")
        
        # Find which shard contains this index
        shard_idx, local_idx = self._find_shard(idx)
        shard_info = self.shards[shard_idx]
        
        # Load shard (cached in memory)
        shard_tensor = self._load_shard(shard_info["shard_path"])
        
        # Extract the specific image
        img = shard_tensor[local_idx]  # [3, H, W]
        
        # Apply transform if provided
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __len__(self) -> int:
        return self.num_samples


def build_precompute_dataset(data_config: dict) -> Dataset:
    """
    Build a dataset for precomputing cache (deterministic preprocessing).
    
    This returns a dataset that applies deterministic preprocessing to images
    for caching purposes. It does not include stochastic augmentations.
    
    Args:
        data_config: Data configuration dictionary
    
    Returns:
        Dataset that returns preprocessed tensors [3, H, W]
    """
    image_size = data_config.get('image_size', 224)
    
    # Deterministic preprocessing: resize, center crop, normalize
    # No stochastic augmentations (those will be applied during training)
    precompute_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PretrainDataset(transform=precompute_transform)
    return dataset


def build_pretraining_dataloader(data_config: dict, train_config: dict) -> torch.utils.data.DataLoader:
    """
    Build pretraining DataLoader with support for cached mode.
    
    Args:
        data_config: Data configuration dictionary
        train_config: Training configuration dictionary
    
    Returns:
        DataLoader for pretraining
    """
    from transforms import SimpleTransform, FastMultiCropTransform
    
    use_cached = data_config.get('use_cached', False)
    
    if use_cached:
        # Cached mode: load from preprocessed tensor shards
        cache_dir = data_config.get('cache_dir', './cache_images')
        cache_dir = os.path.expandvars(cache_dir)
        
        # For cached mode, cached tensors are already normalized [3, H, W]
        # We apply minimal augmentations that work on tensors
        # Note: Most augmentations (crop, resize) are already done during caching
        # We only apply simple augmentations like random flip
        import random
        from torchvision.transforms import functional as F_torch
        
        class TensorAugmentation:
            """Light augmentation for preprocessed normalized tensors"""
            def __init__(self):
                pass
            
            def __call__(self, img):
                # Random horizontal flip (works on tensors)
                if random.random() < 0.5:
                    img = torch.flip(img, dims=[2])  # Flip width dimension
                return img
        
        transform = TensorAugmentation()
        dataset = CachedTensorDataset(cache_dir=cache_dir, transform=transform)
        print(f"✓ Using cached dataset from: {cache_dir}")
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
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
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
                # If that fails, try loading the full dataset and accessing a subset
                full_dataset = load_dataset("tsbpp/fall2025_deeplearning")
                if isinstance(full_dataset, dict):
                    # If it's a dict, try to find eval_public key
                    if "eval_public" in full_dataset:
                        self.dataset = full_dataset["eval_public"][split]
                    else:
                        # Fallback: use the split directly if it exists
                        self.dataset = full_dataset[split]
                else:
                    raise ValueError("Unexpected dataset structure")
            except (ValueError, KeyError) as e:
                # If all fails, show available splits
                available_splits = _inspect_dataset_splits("tsbpp/fall2025_deeplearning")
                raise ValueError(
                    f"Could not find evaluation split 'eval_public/{split}'. "
                    f"Available splits: {available_splits}\n"
                    f"Please check the dataset structure. Original error: {e}"
                )
        self.transform = transform
        print(f"Loaded {len(self.dataset)} {split} images")
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return len(self.dataset)
