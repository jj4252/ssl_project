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
    def __init__(self, dataset_name="cifar10", root="./data", train=True, transform=None, max_samples=20000):
        """
        Args:
            dataset_name: "cifar10" or "cifar100"
            root: Root directory for CIFAR data
            train: Use training split (True) or test split (False)
            transform: Transform to apply to images
            max_samples: Maximum number of samples to use (default: 20000)
        """
        print(f"Loading {dataset_name.upper()} dataset...")
        
        # Load CIFAR dataset
        if dataset_name.lower() == "cifar10":
            self.dataset = CIFAR10(root=root, train=train, download=True, transform=None)
        elif dataset_name.lower() == "cifar100":
            self.dataset = CIFAR100(root=root, train=train, download=True, transform=None)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar10' or 'cifar100'")
        
        # Limit to max_samples
        if max_samples is not None and len(self.dataset) > max_samples:
            print(f"  Limiting dataset to {max_samples:,} samples (from {len(self.dataset):,})")
            indices = list(range(max_samples))
            self.dataset = Subset(self.dataset, indices)
        
        self.transform = transform
        print(f"  Loaded {len(self.dataset):,} images")
    
    def __getitem__(self, idx):
        # Get image (ignore label for pretraining)
        if isinstance(self.dataset, Subset):
            image, _ = self.dataset.dataset[self.dataset.indices[idx]]
        else:
            image, _ = self.dataset[idx]
        
        if self.transform:
            views = self.transform(image)  # Multi-crop returns list
        else:
            views = image
        return views
    
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
    max_samples = data_config.get('max_samples', 20000)
    
    # Get training image size
    image_size = data_config.get('image_size', 224)
    use_multi_crop = train_config.get('use_multi_crop', False)
    use_local_crops = train_config.get('use_local_crops', False)
    
    # Create transforms
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
    
    # Create dataset
    dataset = CIFARDataset(
        dataset_name=dataset_name,
        root=dataset_root,
        train=True,
        transform=transform,
        max_samples=max_samples
    )
    
    print(f"✓ Using {dataset_name.upper()} dataset: {len(dataset):,} images")
    
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

