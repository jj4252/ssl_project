import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
from PIL import Image

# Check if Solarize is available (added in torchvision 0.8.0)
# and create a fallback if not
if hasattr(transforms, 'Solarize'):
    SolarizeTransform = transforms.Solarize
else:
    # Custom Solarize implementation if not available
    class SolarizeTransform:
        """Solarize transform - inverts pixels above threshold"""
        def __init__(self, threshold=128):
            self.threshold = threshold
        
        def __call__(self, img):
            if isinstance(img, Image.Image):
                import numpy as np
                img_array = np.array(img)
                img_array = np.where(img_array > self.threshold, 255 - img_array, img_array)
                return Image.fromarray(img_array.astype(np.uint8))
            return img

class MultiCropTransform:
    """
    DINO-style multi-crop augmentation:
    - 2 global views (large crops)
    - N local views (small crops)
    """
    def __init__(self, global_crops_scale=(0.4, 1.0), 
                 local_crops_scale=(0.05, 0.4),
                 local_crops_number=8,
                 image_size=96):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.image_size = image_size
        
        # Strong augmentation for global views
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, 
                                       scale=global_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))  # Reduced from 23 for speed
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
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))  # Reduced from 23 for speed
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Stronger augmentation for local views
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size,
                                       scale=local_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))  # Reduced from 23 for speed
            ], p=0.5),
            transforms.RandomApply([
                SolarizeTransform(threshold=128)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image):
        crops = []
        # 2 global views
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        # N local views
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops


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

