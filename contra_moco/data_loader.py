"""
Data loading module for DINO training with CIFAR-10/100 dataset.

This module provides datasets for knowledge distillation training using CIFAR-10/100.
Limited to 20K images for quick testing.
"""

import os
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


def build_pretraining_dataloader(data_config: dict, train_config: dict) -> torch.utils.data.DataLoader:
    """
    Build pretraining DataLoader for CIFAR dataset.
    
    Args:
        data_config: Data configuration dictionary
        train_config: Training configuration dictionary
    
    Returns:
        DataLoader for pretraining
    """
    from transforms import SimpleTransform, FastMultiCropTransform
    
    # Get dataset configuration
    dataset_name = data_config.get('dataset_name', 'cifar10')
    dataset_root = data_config.get('dataset_root', './data')
    max_samples = data_config.get('max_samples', None)  # None = use full dataset
    
    # Get training image size (CIFAR10 is 32x32, will be upscaled to image_size)
    image_size = data_config.get('image_size', 96)  # Match student model size
    use_multi_crop = train_config.get('use_multi_crop', False)
    use_local_crops = train_config.get('use_local_crops', False)
    use_minimal_aug = train_config.get('use_minimal_aug', False)  # Step 3 diagnostic: minimal augmentation
    
    # Create transforms
    if use_minimal_aug:
        # Step 3 diagnostic: Minimal augmentation (only resize + flip, no crop/blur)
        from transforms import MinimalTransform
        transform = MinimalTransform(image_size=image_size)
        print("✓ Using minimal augmentation (diagnostic mode: resize + flip only)")
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
    
    # Check training mode and determine if we need two views
    training_mode = train_config.get('training_mode', 'kd')
    use_moco_aug = train_config.get('use_moco_aug', False)
    
    # For MoCo-v3, always use MoCo transform (returns two views)
    if training_mode == "moco_v3" or use_moco_aug:
        from transforms import MoCoTransform
        transform = MoCoTransform(image_size=image_size)
        return_two_views = True
        print("✓ Using MoCo-v3 style augmentations (two views per image)")
    else:
        # Check if SSL is enabled (for Barlow Twins, need two views)
        ssl_config = train_config.get('ssl', {})
        use_ssl = ssl_config.get('enabled', False)
        return_two_views = use_ssl  # Return two views if SSL is enabled
    
    # Create dataset
    dataset = CIFARDataset(
        dataset_name=dataset_name,
        root=dataset_root,
        train=True,
        transform=transform,
        max_samples=max_samples,
        return_two_views=return_two_views
    )
    
    print(f"✓ Using {dataset_name.upper()} dataset: {len(dataset):,} images")
    
    # Build DataLoader
    num_workers = train_config.get('num_workers', data_config.get('num_workers', 4))
    persistent_workers = train_config.get('persistent_workers', data_config.get('persistent_workers', False))
    prefetch_factor = train_config.get('prefetch_factor', data_config.get('prefetch_factor', 2)) if num_workers > 0 else None
    pin_memory = train_config.get('pin_memory', data_config.get('pin_memory', True))
    
    # For MoCo-v3, drop_last=True is important for queue management
    drop_last = (training_mode == "moco_v3" or use_moco_aug)
    
    # Custom collate function for two-view batches
    def collate_two_views(batch):
        """Collate function for batches that return (view1, view2) tuples"""
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # Batch is list of (view1, view2) tuples
            # Collate into (batch_view1, batch_view2) tuple of tensors
            view1_list = [item[0] for item in batch]
            view2_list = [item[1] for item in batch]
            view1_batch = torch.stack(view1_list)
            view2_batch = torch.stack(view2_list)
            return (view1_batch, view2_batch)
        else:
            # Fallback to default collate
            return torch.utils.data.default_collate(batch)
    
    # Use custom collate for MoCo-v3 (two views)
    collate_fn = collate_two_views if (training_mode == "moco_v3" or use_moco_aug) else None
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
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

