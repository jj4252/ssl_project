"""
Optimized transforms for fast DINO training
- Simplified augmentations (no solarization, reduced color jitter)
- Optional local crops (can be disabled for speed)
"""

import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from PIL import Image


class FastMultiCropTransform:
    """
    Optimized multi-crop augmentation for fast training:
    - 2 global views (large crops)
    - Optional N local views (can be disabled)
    - Simplified augmentations (no solarization, reduced intensity)
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), 
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=0,  # Default to 0 (disabled) for speed
                 image_size=96,
                 use_local_crops=False):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number if use_local_crops else 0
        self.image_size = image_size
        self.use_local_crops = use_local_crops
        
        # Simplified augmentation for global views (fast)
        # Only: random resized crop, horizontal flip, single gaussian blur, normalize
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, 
                                       scale=global_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, 
                                       scale=global_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Simplified local transform (only if local crops enabled)
        if self.use_local_crops:
            self.local_transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size,
                                           scale=local_crops_scale,
                                           interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image):
        crops = []
        # 2 global views
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        # N local views (only if enabled)
        if self.use_local_crops:
            for _ in range(self.local_crops_number):
                crops.append(self.local_transform(image))
        return crops


class SimpleTransform:
    """
    Simple single-crop transform for knowledge distillation
    Minimal augmentation: random resized crop, horizontal flip, single blur, normalize
    """
    def __init__(self, image_size=224, scale=(0.2, 1.0)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, 
                                        scale=scale,
                                        interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        return self.transform(image)


class MinimalTransform:
    """
    Minimal augmentation for debugging (Step 3 diagnostic)
    Only resize, horizontal flip, and normalize - no random crop or blur
    """
    def __init__(self, image_size=96):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),  # Only flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        return self.transform(image)


class EvalTransform:
    """Simple transform for evaluation (no augmentation)"""
    def __init__(self, image_size=96):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        return self.transform(image)


class MoCoTransform:
    """
    MoCo-v3 style augmentation for contrastive learning.
    Returns two views per image with strong augmentations.
    """
    def __init__(self, image_size=96):
        # View 1: RandomResizedCrop + RandomHorizontalFlip + ColorJitter + RandomGrayscale + GaussianBlur
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # View 2: Same augmentations (applied independently)
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.2, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        # Return two independently augmented views
        view1 = self.transform1(image)
        view2 = self.transform2(image)
        return view1, view2

