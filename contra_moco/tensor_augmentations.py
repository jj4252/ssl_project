"""
Tensor-based augmentations for cached preprocessed tensors.

These augmentations work on normalized tensors [C, H, W] that have already been
resized and normalized. They replicate DINO-style augmentations but operate
directly on tensors instead of PIL images.
"""

import torch
import torch.nn.functional as F
import random
import math
from typing import Tuple, List


def random_horizontal_flip(img: torch.Tensor) -> torch.Tensor:
    """Random horizontal flip on tensor [C, H, W]"""
    if random.random() < 0.5:
        img = torch.flip(img, dims=[2])  # Flip width dimension
    return img


def random_resized_crop_tensor(
    img: torch.Tensor,
    size: Tuple[int, int],
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    interpolation: str = 'bilinear'
) -> torch.Tensor:
    """
    Random resized crop on tensor [C, H, W].
    
    Args:
        img: Input tensor [C, H, W]
        size: Output size (height, width)
        scale: Scale range for random crop
        ratio: Aspect ratio range
        interpolation: Interpolation mode ('bilinear' or 'nearest')
    
    Returns:
        Cropped and resized tensor [C, H, W]
    """
    C, H, W = img.shape
    
    # Sample scale and aspect ratio
    scale_val = random.uniform(scale[0], scale[1])
    ratio_val = random.uniform(ratio[0], ratio[1])
    
    # Calculate crop size
    crop_h = int(H * scale_val * math.sqrt(ratio_val))
    crop_w = int(W * scale_val / math.sqrt(ratio_val))
    
    # Clamp to image size
    crop_h = min(crop_h, H)
    crop_w = min(crop_w, W)
    
    # Sample crop position
    top = random.randint(0, H - crop_h) if H > crop_h else 0
    left = random.randint(0, W - crop_w) if W > crop_w else 0
    
    # Crop
    img_cropped = img[:, top:top+crop_h, left:left+crop_w]
    
    # Resize
    img_resized = F.interpolate(
        img_cropped.unsqueeze(0),
        size=size,
        mode=interpolation,
        align_corners=False if interpolation == 'bilinear' else None
    ).squeeze(0)
    
    return img_resized


def color_jitter_tensor(
    img: torch.Tensor,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.2,
    hue: float = 0.1
) -> torch.Tensor:
    """
    Color jitter on unnormalized tensor [C, H, W] in range [0, 1].
    
    Args:
        img: Unnormalized tensor [C, H, W], range [0, 1]
        brightness: Brightness jitter range
        contrast: Contrast jitter range
        saturation: Saturation jitter range
        hue: Hue jitter range
    
    Returns:
        Jittered tensor [C, H, W], range [0, 1]
    """
    # Ensure tensor is in [0, 1] range
    img = torch.clamp(img, 0, 1)
    
    # Apply jitter
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        img = img * brightness_factor
        img = torch.clamp(img, 0, 1)
    
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        img = (img - 0.5) * contrast_factor + 0.5
        img = torch.clamp(img, 0, 1)
    
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        # Convert to grayscale and blend
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        for c in range(3):
            img[c] = gray + (img[c] - gray) * saturation_factor
        img = torch.clamp(img, 0, 1)
    
    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
        # Simple hue shift (approximation)
        # For simplicity, we'll do a basic rotation in RGB space
        if abs(hue_factor) > 1e-6:
            # Rotate RGB channels (scaled to [0, 1])
            shift = int(hue_factor * 3)
            img = torch.roll(img, shift, dims=0)
    
    return img


def random_grayscale_tensor(img: torch.Tensor, p: float = 0.2) -> torch.Tensor:
    """Random grayscale conversion on tensor [C, H, W]"""
    if random.random() < p:
        # Convert to grayscale
        gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        img = gray.unsqueeze(0).repeat(3, 1, 1)
    return img


def gaussian_blur_tensor(
    img: torch.Tensor,
    kernel_size: int = 9,
    sigma: Tuple[float, float] = (0.1, 2.0)
) -> torch.Tensor:
    """
    Gaussian blur on tensor [C, H, W].
    
    Args:
        img: Input tensor [C, H, W]
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Range for random sigma
    
    Returns:
        Blurred tensor [C, H, W]
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    sigma_val = random.uniform(sigma[0], sigma[1])
    
    # Create 2D Gaussian kernel from 1D kernel (outer product)
    kernel_1d = _get_gaussian_kernel(kernel_size, sigma_val, img.dtype, img.device)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]  # Outer product: [kernel_size, kernel_size]
    kernel_2d = kernel_2d / kernel_2d.sum()  # Normalize
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
    
    # Apply blur (conv2d expects [B, C, H, W])
    img_blurred = F.conv2d(
        img.unsqueeze(0),
        kernel,
        padding=kernel_size // 2,
        groups=3
    ).squeeze(0)
    
    return img_blurred


def _get_gaussian_kernel(kernel_size: int, sigma: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Generate 1D Gaussian kernel"""
    coords = torch.arange(kernel_size, dtype=dtype, device=device).float() - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


class TensorSimpleTransform:
    """
    Simple single-crop transform for cached tensors.
    Replicates SimpleTransform but works on unnormalized tensors [0, 1].
    """
    def __init__(self, image_size: int = 96, scale: Tuple[float, float] = (0.2, 1.0)):
        self.image_size = image_size
        self.scale = scale
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to cached tensor [C, H, W].
        
        Args:
            img: Unnormalized tensor [C, H, W] from cache, range [0, 1]
        
        Returns:
            Augmented and normalized tensor [C, H, W]
        """
        # Random resized crop
        img = random_resized_crop_tensor(img, (self.image_size, self.image_size), scale=self.scale)
        
        # Random horizontal flip
        img = random_horizontal_flip(img)
        
        # Random color jitter (80% chance) - works on [0, 1] range
        if random.random() < 0.8:
            img = color_jitter_tensor(img, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        
        # Random grayscale (20% chance)
        img = random_grayscale_tensor(img, p=0.2)
        
        # Random Gaussian blur (50% chance)
        if random.random() < 0.5:
            img = gaussian_blur_tensor(img, kernel_size=9, sigma=(0.1, 2.0))
        
        # Normalize (ImageNet stats)
        if img.device != self.mean.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        img = (img - self.mean) / self.std
        
        return img


class TensorMultiCropTransform:
    """
    Multi-crop transform for cached tensors.
    Replicates FastMultiCropTransform but works on unnormalized tensors [0, 1].
    """
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 0,
        image_size: int = 96,
        use_local_crops: bool = False
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number if use_local_crops else 0
        self.image_size = image_size
        self.use_local_crops = use_local_crops
        # ImageNet normalization stats
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def _apply_global_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply global crop augmentation"""
        # Random resized crop
        img = random_resized_crop_tensor(
            img,
            (self.image_size, self.image_size),
            scale=self.global_crops_scale
        )
        
        # Random horizontal flip
        img = random_horizontal_flip(img)
        
        # Random Gaussian blur (50% chance)
        if random.random() < 0.5:
            img = gaussian_blur_tensor(img, kernel_size=9, sigma=(0.1, 2.0))
        
        # Normalize
        if img.device != self.mean.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        img = (img - self.mean) / self.std
        
        return img
    
    def _apply_local_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply local crop augmentation"""
        # Random resized crop
        img = random_resized_crop_tensor(
            img,
            (self.image_size, self.image_size),
            scale=self.local_crops_scale
        )
        
        # Random horizontal flip
        img = random_horizontal_flip(img)
        
        # Random Gaussian blur (50% chance)
        if random.random() < 0.5:
            img = gaussian_blur_tensor(img, kernel_size=9, sigma=(0.1, 2.0))
        
        # Normalize
        if img.device != self.mean.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        img = (img - self.mean) / self.std
        
        return img
    
    def __call__(self, img: torch.Tensor) -> List[torch.Tensor]:
        """
        Apply multi-crop augmentations to cached tensor.
        
        Args:
            img: Normalized tensor [C, H, W] from cache
        
        Returns:
            List of augmented tensors [C, H, W]
        """
        crops = []
        
        # 2 global views
        crops.append(self._apply_global_transform(img.clone()))
        crops.append(self._apply_global_transform(img.clone()))
        
        # N local views (only if enabled)
        if self.use_local_crops:
            for _ in range(self.local_crops_number):
                crops.append(self._apply_local_transform(img.clone()))
        
        return crops

