"""
Knowledge Distillation Trainer with Cached Tensor Support (2-Stage Pipeline)

This trainer supports both original (HF/raw images) and cached tensor modes.
In cached mode, images are preprocessed once and stored as tensor shards for fast loading.

Key features:
- Step-capped epochs (max_steps_per_epoch)
- Cached tensor support (2-stage pipeline)
- Simplified augmentations
- Teacher feature caching (optional)
- Only compile student (never teacher)
- Optimized DataLoader settings
- Step-based checkpointing
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
# Try to import new GradScaler API (PyTorch 2.0+), fall back to old API
try:
    from torch.amp import GradScaler
    GRADSCALER_NEW_API = True
except ImportError:
    from torch.cuda.amp import GradScaler
    GRADSCALER_NEW_API = False

# Try to import new autocast API (PyTorch 2.0+), fall back to old API
try:
    from torch.amp import autocast  # PyTorch 2.0+
    AUTOCAST_NEW_API = True
except ImportError:
    from torch.cuda.amp import autocast  # PyTorch < 2.0
    AUTOCAST_NEW_API = False
from tqdm import tqdm
import os
import time
from pathlib import Path
import hashlib
import itertools

import timm
from data_loader import build_pretraining_dataloader
from optimizer import build_optimizer, build_scheduler


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_teacher_model(teacher_name="dinov2_vitb14", device="cuda"):
    """
    Load pretrained DINOv2 teacher model from torch.hub
    Automatically patches DINOv2 for Python 3.9 compatibility if needed
    
    Args:
        teacher_name: Model name (e.g., "dinov2_vitb14")
        device: Device to load model on
    
    Returns:
        Frozen teacher model
    """
    import sys
    import warnings
    
    print(f"Loading teacher model: {teacher_name}")
    
    try:
        # For Python < 3.10, use patcher for compatibility
        if sys.version_info < (3, 10):
            try:
                from dinov2_patcher import load_dinov2_with_patch
                teacher = load_dinov2_with_patch(teacher_name, verbose=False)
            except ImportError:
                print("⚠️  Warning: dinov2_patcher not available. Trying direct load...")
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers.*")
                    teacher = torch.hub.load("facebookresearch/dinov2", teacher_name, verbose=False)
        else:
            # Python 3.10+: direct load (no patching needed)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*xFormers.*")
                teacher = torch.hub.load("facebookresearch/dinov2", teacher_name, verbose=False)
        
        teacher = teacher.to(device)
        teacher.eval()
        
        # Freeze all parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        print(f"✓ Teacher loaded: {teacher_name}")
        print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M")
        print(f"  Frozen: True")
        print(f"  ⚠️  Teacher will NOT be compiled (frozen, no benefit)")
        
        return teacher
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher model: {e}")


def build_student_model(model_name="vit_small_patch16_224", 
                       img_size=224, 
                       device="cuda"):
    """
    Build student ViT model from timm (random initialization)
    
    Args:
        model_name: timm model name (e.g., "vit_small_patch16_224" or "vit_small_patch16")
        img_size: Input image size (will override model default if different)
        device: Device to load model on
    
    Returns:
        Student model (trainable)
    """
    print(f"Building student model: {model_name} with img_size={img_size}")
    
    # Try to create model with custom img_size
    try:
        student = timm.create_model(
            model_name,
            pretrained=False,  # Random initialization
            img_size=img_size,
            num_classes=0,  # No classification head
        )
    except Exception as e:
        # If model creation fails (e.g., model name doesn't support custom size),
        # try creating with default size and then patch it
        print(f"  Warning: Could not create model with img_size={img_size}, trying default size first...")
        print(f"  Error: {e}")
        student = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
        )
    
    # Always patch patch_embed to ensure it matches desired img_size
    # (Some models may ignore img_size parameter or have hardcoded checks)
    if hasattr(student, 'patch_embed'):
        # Get current patch size
        if hasattr(student.patch_embed, 'patch_size'):
            patch_size = student.patch_embed.patch_size
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
        else:
            # Default patch size for ViT-S/16
            patch_size = 16
        
        # Check current img_size
        current_img_size = None
        if hasattr(student.patch_embed, 'img_size'):
            current_img_size = student.patch_embed.img_size
            if isinstance(current_img_size, (list, tuple)):
                current_img_size = current_img_size[0]
        
        # Calculate new grid size
        grid_size = img_size // patch_size
        
        # Patch if needed
        if current_img_size != img_size:
            print(f"  Patching patch_embed from {current_img_size}x{current_img_size} to {img_size}x{img_size}")
            if hasattr(student.patch_embed, 'img_size'):
                student.patch_embed.img_size = (img_size, img_size)
            
            # Update grid_size and num_patches
            if hasattr(student.patch_embed, 'grid_size'):
                student.patch_embed.grid_size = (grid_size, grid_size)
            if hasattr(student.patch_embed, 'num_patches'):
                student.patch_embed.num_patches = grid_size * grid_size
            
            print(f"  ✓ Patched: patch_size={patch_size}, grid_size={grid_size}x{grid_size}, num_patches={grid_size * grid_size}")
    
    student = student.to(device)
    student.train()
    
    num_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    
    # Verify the model accepts the desired img_size
    if hasattr(student, 'patch_embed') and hasattr(student.patch_embed, 'img_size'):
        actual_img_size = student.patch_embed.img_size
        if isinstance(actual_img_size, (list, tuple)):
            actual_img_size = actual_img_size[0]
        print(f"✓ Student created: {model_name}")
        print(f"  Parameters: {num_params / 1e6:.2f}M")
        print(f"  Trainable: {trainable_params / 1e6:.2f}M")
        print(f"  Image size: {actual_img_size}x{actual_img_size}")
        if actual_img_size != img_size:
            print(f"  ⚠️  Warning: Model img_size ({actual_img_size}) != requested ({img_size})")
    else:
        print(f"✓ Student created: {model_name}")
        print(f"  Parameters: {num_params / 1e6:.2f}M")
        print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    return student


def extract_teacher_features(teacher, images, use_cls_token=True, 
                            cache_dir=None, cache_key=None, teacher_img_size=224):
    """
    Extract features from frozen teacher model with optional caching
    
    Args:
        teacher: DINOv2 teacher model (expects 224x224 input)
        images: Input images [B, 3, H, W] (may be 96x96)
        use_cls_token: Whether to use CLS token or mean-pool patches
        cache_dir: Optional directory to cache features
        cache_key: Optional key for caching (e.g., batch index)
        teacher_img_size: Target size for teacher (default 224x224 for DINOv2)
    
    Returns:
        cls_embedding: [B, D] CLS token embedding
        patch_embeddings: [B, N, D] patch token embeddings
    """
    # Check cache if enabled
    if cache_dir is not None and cache_key is not None:
        cache_path = Path(cache_dir) / f"{cache_key}.pt"
        if cache_path.exists():
            cached = torch.load(cache_path, map_location=images.device)
            return cached['cls'], cached['patches']
    
    with torch.no_grad():
        # Upscale images to teacher's expected size (96x96 -> 224x224)
        if images.shape[-1] != teacher_img_size or images.shape[-2] != teacher_img_size:
            images = F.interpolate(
                images, 
                size=(teacher_img_size, teacher_img_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # DINOv2 forward_features may return dict or tensor
        features = teacher.forward_features(images)
        
        # Handle DINOv2 output format (dict with keys)
        if isinstance(features, dict):
            if 'x_norm_clstoken' in features:
                cls_embedding = features['x_norm_clstoken']  # [B, D]
            elif 'cls_token' in features:
                cls_embedding = features['cls_token']
            else:
                cls_embedding = features.get('x', features.get('tokens', None))[:, 0]
            
            if 'x_norm_patchtokens' in features:
                patch_embeddings = features['x_norm_patchtokens']  # [B, N, D]
            elif 'patch_tokens' in features:
                patch_embeddings = features['patch_tokens']
            else:
                patch_embeddings = features.get('x', features.get('tokens', None))[:, 1:]
        else:
            # Assume tensor format [B, N+1, D] (CLS + patches)
            if use_cls_token:
                cls_embedding = features[:, 0]  # CLS token [B, D]
            else:
                cls_embedding = features[:, 1:].mean(dim=1)  # Mean-pool patches [B, D]
            
            patch_embeddings = features[:, 1:]  # Patch tokens [B, N, D]
        
        # Normalize embeddings
        cls_embedding = F.normalize(cls_embedding, dim=-1, p=2)
        patch_embeddings = F.normalize(patch_embeddings, dim=-1, p=2)
    
    # Save to cache if enabled
    if cache_dir is not None and cache_key is not None:
        cache_path = Path(cache_dir) / f"{cache_key}.pt"
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({'cls': cls_embedding, 'patches': patch_embeddings}, cache_path)
    
    return cls_embedding, patch_embeddings


def extract_student_features(student, images, use_cls_token=True):
    """
    Extract features from student model
    
    Args:
        student: Student ViT model
        images: Input images [B, 3, H, W]
        use_cls_token: Whether to use CLS token or mean-pool patches
    
    Returns:
        cls_embedding: [B, D] CLS token embedding
        patch_embeddings: [B, N, D] patch token embeddings
    """
    # Student forward_features returns [B, N+1, D] (CLS + patches)
    features = student.forward_features(images)
    
    if use_cls_token:
        cls_embedding = features[:, 0]  # CLS token [B, D]
    else:
        cls_embedding = features[:, 1:].mean(dim=1)  # Mean-pool patches [B, D]
    
    patch_embeddings = features[:, 1:]  # Patch tokens [B, N, D]
    
    # Normalize embeddings
    cls_embedding = F.normalize(cls_embedding, dim=-1, p=2)
    patch_embeddings = F.normalize(patch_embeddings, dim=-1, p=2)
    
    return cls_embedding, patch_embeddings


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss with learnable projections for dimension mismatch.
    
    Handles cases where teacher and student have different embedding dimensions
    by using learnable linear projections.
    """
    def __init__(self, teacher_dim=768, student_dim=384, loss_weights=None):
        """
        Args:
            teacher_dim: Teacher embedding dimension (default: 768 for DINOv2 ViT-B)
            student_dim: Student embedding dimension (default: 384 for ViT-S)
            loss_weights: Dict with 'cls' and 'patch' weights
        """
        super().__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.loss_weights = loss_weights if loss_weights else {'cls': 1.0, 'patch': 0.5}
        
        # Learnable projection layers (only needed if dimensions differ)
        if teacher_dim != student_dim:
            self.teacher_proj_cls = nn.Linear(teacher_dim, student_dim, bias=False)
            self.teacher_proj_patch = nn.Linear(teacher_dim, student_dim, bias=False)
            # Initialize projections with Xavier uniform (standard for linear layers)
            # This allows the projection to learn the optimal mapping
            nn.init.xavier_uniform_(self.teacher_proj_cls.weight)
            nn.init.xavier_uniform_(self.teacher_proj_patch.weight)
        else:
            self.teacher_proj_cls = None
            self.teacher_proj_patch = None
    
    def forward(self, student_cls, student_patches, teacher_cls, teacher_patches):
        """
        Compute distillation loss.
        
        Args:
            student_cls: Student CLS embedding [B, D_s]
            student_patches: Student patch embeddings [B, N_s, D_s]
            teacher_cls: Teacher CLS embedding [B, D_t]
            teacher_patches: Teacher patch embeddings [B, N_t, D_t]
        
        Returns:
            Total loss and component losses dict
        """
        # CLS token loss
        if self.teacher_proj_cls is not None:
            # Project teacher to student dimension
            teacher_cls_proj = self.teacher_proj_cls(teacher_cls)  # [B, D_s]
            teacher_cls_proj = F.normalize(teacher_cls_proj, dim=-1)
            student_cls_norm = F.normalize(student_cls, dim=-1)
            loss_cls = F.mse_loss(student_cls_norm, teacher_cls_proj)
        else:
            # Same dimensions: direct MSE
            student_cls_norm = F.normalize(student_cls, dim=-1)
            teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
            loss_cls = F.mse_loss(student_cls_norm, teacher_cls_norm)
        
        # Patch embeddings loss
        B_s, N_s, D_s = student_patches.shape
        B_t, N_t, D_t = teacher_patches.shape
        
        # Handle patch number mismatch first
        if N_s != N_t:
            if N_s < N_t:
                teacher_patches = teacher_patches[:, :N_s, :]  # Truncate teacher
            else:
                student_patches = student_patches[:, :N_t, :]  # Truncate student
        
        # Handle dimension mismatch
        if self.teacher_proj_patch is not None:
            # Project teacher patches to student dimension
            teacher_patches_proj = self.teacher_proj_patch(teacher_patches)  # [B, N, D_s]
            teacher_patches_proj = F.normalize(teacher_patches_proj, dim=-1)
            student_patches_norm = F.normalize(student_patches, dim=-1)
            loss_patch = F.mse_loss(student_patches_norm, teacher_patches_proj)
        else:
            # Same dimensions: direct MSE
            student_patches_norm = F.normalize(student_patches, dim=-1)
            teacher_patches_norm = F.normalize(teacher_patches, dim=-1)
            loss_patch = F.mse_loss(student_patches_norm, teacher_patches_norm)
        
        # Weighted combination
        total_loss = self.loss_weights['cls'] * loss_cls + self.loss_weights['patch'] * loss_patch
        
        return total_loss, {
            'total': total_loss.item(),
            'cls': loss_cls.item(),
            'patch': loss_patch.item()
        }


def compute_distillation_loss(student_cls, student_patches, 
                             teacher_cls, teacher_patches,
                             loss_weights=None,
                             distillation_loss_module=None):
    """
    Compute distillation loss between student and teacher embeddings.
    
    This is a wrapper function that uses DistillationLoss module if provided,
    otherwise falls back to simple computation (for backward compatibility).
    
    Args:
        student_cls: Student CLS embedding [B, D_s]
        student_patches: Student patch embeddings [B, N_s, D_s]
        teacher_cls: Teacher CLS embedding [B, D_t]
        teacher_patches: Teacher patch embeddings [B, N_t, D_t]
        loss_weights: Dict with 'cls' and 'patch' weights
        distillation_loss_module: DistillationLoss module (if None, uses simple computation)
    
    Returns:
        Total loss and component losses
    """
    if distillation_loss_module is not None:
        return distillation_loss_module(student_cls, student_patches, teacher_cls, teacher_patches)
    
    # Fallback: simple computation (for backward compatibility)
    if loss_weights is None:
        loss_weights = {'cls': 1.0, 'patch': 0.5}
    
    # CLS token loss
    if student_cls.shape[-1] == teacher_cls.shape[-1]:
        student_cls_norm = F.normalize(student_cls, dim=-1)
        teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
        loss_cls = F.mse_loss(student_cls_norm, teacher_cls_norm)
    else:
        # Different dimensions: use simple projection (mean pooling)
        D_s = student_cls.shape[-1]
        D_t = teacher_cls.shape[-1]
        if D_s < D_t:
            ratio = D_t // D_s
            if D_t % D_s == 0:
                teacher_proj = teacher_cls.view(teacher_cls.shape[0], D_s, ratio).mean(dim=-1)
            else:
                teacher_proj = teacher_cls[:, :D_s]
                if D_t > D_s:
                    remaining = teacher_cls[:, D_s:].view(teacher_cls.shape[0], -1, D_s).mean(dim=1)
                    teacher_proj = (teacher_proj + remaining) / 2
            student_cls_norm = F.normalize(student_cls, dim=-1)
            teacher_proj_norm = F.normalize(teacher_proj, dim=-1)
            loss_cls = F.mse_loss(student_cls_norm, teacher_proj_norm)
        else:
            ratio = D_s // D_t
            if D_s % D_t == 0:
                student_proj = student_cls.view(student_cls.shape[0], D_t, ratio).mean(dim=-1)
            else:
                student_proj = student_cls[:, :D_t]
                if D_s > D_t:
                    remaining = student_cls[:, D_t:].view(student_cls.shape[0], -1, D_t).mean(dim=1)
                    student_proj = (student_proj + remaining) / 2
            student_proj_norm = F.normalize(student_proj, dim=-1)
            teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
            loss_cls = F.mse_loss(student_proj_norm, teacher_cls_norm)
    
    # Patch embeddings loss
    B_s, N_s, D_s = student_patches.shape
    B_t, N_t, D_t = teacher_patches.shape
    
    if N_s != N_t:
        if N_s < N_t:
            teacher_patches = teacher_patches[:, :N_s, :]
        else:
            student_patches = student_patches[:, :N_t, :]
    
    if D_s == D_t:
        student_patches_norm = F.normalize(student_patches, dim=-1)
        teacher_patches_norm = F.normalize(teacher_patches, dim=-1)
        loss_patch = F.mse_loss(student_patches_norm, teacher_patches_norm)
    else:
        # Different dimensions: use simple projection
        if D_s < D_t:
            ratio = D_t // D_s
            if D_t % D_s == 0:
                teacher_proj = teacher_patches.view(teacher_patches.shape[0], teacher_patches.shape[1], D_s, ratio).mean(dim=-1)
            else:
                teacher_proj = teacher_patches[:, :, :D_s]
                if D_t > D_s:
                    remaining = teacher_patches[:, :, D_s:].view(teacher_patches.shape[0], teacher_patches.shape[1], -1, D_s).mean(dim=2)
                    teacher_proj = (teacher_proj + remaining) / 2
            student_patches_norm = F.normalize(student_patches, dim=-1)
            teacher_proj_norm = F.normalize(teacher_proj, dim=-1)
            loss_patch = F.mse_loss(student_patches_norm, teacher_proj_norm)
        else:
            ratio = D_s // D_t
            if D_s % D_t == 0:
                student_proj = student_patches.view(student_patches.shape[0], student_patches.shape[1], D_t, ratio).mean(dim=-1)
            else:
                student_proj = student_patches[:, :, :D_t]
                if D_s > D_t:
                    remaining = student_patches[:, :, D_t:].view(student_patches.shape[0], student_patches.shape[1], -1, D_t).mean(dim=2)
                    student_proj = (student_proj + remaining) / 2
            student_proj_norm = F.normalize(student_proj, dim=-1)
            teacher_patches_norm = F.normalize(teacher_patches, dim=-1)
            loss_patch = F.mse_loss(student_proj_norm, teacher_patches_norm)
    
    # Weighted combination
    total_loss = loss_weights['cls'] * loss_cls + loss_weights['patch'] * loss_patch
    
    return total_loss, {
        'total': total_loss.item(),
        'cls': loss_cls.item(),
        'patch': loss_patch.item()
    }


def train_epoch(teacher, student, dataloader, optimizer, scheduler, 
                device, scaler, epoch, num_epochs, loss_weights, 
                use_cls_token=True, use_multi_crop=False,
                max_steps_per_epoch=None, cache_teacher_features=False,
                teacher_feature_dir=None, save_every=0, global_step=0,
                checkpoint_dir=None, distillation_loss_module=None):
    """
    Train one epoch with optimizations
    
    Args:
        teacher: Frozen teacher model
        student: Trainable student model
        dataloader: Data loader
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        scaler: GradScaler for mixed precision
        epoch: Current epoch
        num_epochs: Total epochs
        loss_weights: Loss weights dict
        use_cls_token: Whether to use CLS token
        use_multi_crop: Whether to use multi-crop augmentation
        max_steps_per_epoch: Maximum number of steps per epoch
        cache_teacher_features: Whether to cache teacher features
        teacher_feature_dir: Directory for teacher feature cache
        save_every: Save checkpoint every N steps (0 = only at end)
        global_step: Global step counter
        checkpoint_dir: Checkpoint directory
    
    Returns:
        Average loss, metrics, and updated global_step
    """
    student.train()
    total_loss = 0
    total_metrics = {'cls': 0, 'patch': 0}
    
    # Determine max steps
    total_batches = len(dataloader)
    if max_steps_per_epoch is not None:
        if max_steps_per_epoch > total_batches:
            max_steps = total_batches
        else:
            max_steps = max_steps_per_epoch
    else:
        max_steps = total_batches
    
    # Use itertools.islice to limit iterations (avoids KeyError with DataLoader)
    limited_dataloader = itertools.islice(dataloader, max_steps)
    
    desc = f"Epoch {epoch+1}/{num_epochs}"
    if max_steps < total_batches:
        desc += f" (capped at {max_steps}/{total_batches} steps)"
    
    progress_bar = tqdm(limited_dataloader, desc=desc, total=max_steps)
    
    batch_times = []
    data_times = []
    gpu_times = []
    prev_iter_time = time.time()
    steps_completed = 0
    
    for batch_idx, batch in enumerate(progress_bar):
        iter_start = time.time()
        data_load_time = iter_start - prev_iter_time if batch_idx > 0 else 0
        
        batch_start = time.time()
        
        # Handle multi-crop or single image
        if use_multi_crop and isinstance(batch, list):
            images = batch[0].to(device)  # Use first global crop
        else:
            images = batch.to(device)
        
        # Convert to channels_last if supported
        try:
            images = images.to(memory_format=torch.channels_last)
        except:
            pass
        
        optimizer.zero_grad()
        
        gpu_start = time.time()
        
        # Mixed precision training
        if device.type == 'cuda':
            if AUTOCAST_NEW_API:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    # Teacher forward (frozen) - with optional caching
                    cache_key = None
                    if cache_teacher_features:
                        # Use batch hash as cache key
                        cache_key = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
                    
                    teacher_cls, teacher_patches = extract_teacher_features(
                        teacher, images, use_cls_token=use_cls_token,
                        cache_dir=teacher_feature_dir, cache_key=cache_key
                    )
                    
                    # Student forward
                    student_cls, student_patches = extract_student_features(
                        student, images, use_cls_token=use_cls_token
                    )
                    
                    # Debug: Print feature statistics on first batch of first epoch
                    if batch_idx == 0 and epoch == 0:
                        print(f"\n🔍 Debug: Feature statistics (first batch, epoch {epoch+1})")
                        print(f"  Teacher CLS: shape={teacher_cls.shape}, std={teacher_cls.std().item():.6f}, mean={teacher_cls.mean().item():.6f}, norm={teacher_cls.norm(dim=-1).mean().item():.6f}")
                        print(f"  Student CLS: shape={student_cls.shape}, std={student_cls.std().item():.6f}, mean={student_cls.mean().item():.6f}, norm={student_cls.norm(dim=-1).mean().item():.6f}")
                        # Compute cosine similarity (handle dimension mismatch by projecting teacher)
                        if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                            teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                            teacher_cls_proj_norm = F.normalize(teacher_cls_proj, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            cosine_sim = (teacher_cls_proj_norm * student_cls_norm).sum(dim=-1).mean().item()
                        elif teacher_cls.shape[-1] == student_cls.shape[-1]:
                            teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            cosine_sim = (teacher_cls_norm * student_cls_norm).sum(dim=-1).mean().item()
                        else:
                            cosine_sim = None
                        if cosine_sim is not None:
                            print(f"  CLS cosine similarity: {cosine_sim:.6f} (1.0 = identical, 0.0 = orthogonal)")
                        else:
                            print(f"  CLS cosine similarity: N/A (dimension mismatch: {teacher_cls.shape[-1]} vs {student_cls.shape[-1]})")
                        print(f"  Teacher patches: shape={teacher_patches.shape}, std={teacher_patches.std().item():.6f}")
                        print(f"  Student patches: shape={student_patches.shape}, std={student_patches.std().item():.6f}")
                    
                    # Compute distillation loss
                    loss, metrics = compute_distillation_loss(
                        student_cls, student_patches,
                        teacher_cls, teacher_patches,
                        loss_weights=loss_weights,
                        distillation_loss_module=distillation_loss_module
                    )
            else:
                with autocast():
                    cache_key = None
                    if cache_teacher_features:
                        cache_key = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
                    
                    teacher_cls, teacher_patches = extract_teacher_features(
                        teacher, images, use_cls_token=use_cls_token,
                        cache_dir=teacher_feature_dir, cache_key=cache_key
                    )
                    
                    student_cls, student_patches = extract_student_features(
                        student, images, use_cls_token=use_cls_token
                    )
                    
                    # Debug: Print feature statistics on first batch of first epoch
                    if batch_idx == 0 and epoch == 0:
                        print(f"\n🔍 Debug: Feature statistics (first batch, epoch {epoch+1})")
                        print(f"  Teacher CLS: shape={teacher_cls.shape}, std={teacher_cls.std().item():.6f}, mean={teacher_cls.mean().item():.6f}, norm={teacher_cls.norm(dim=-1).mean().item():.6f}")
                        print(f"  Student CLS: shape={student_cls.shape}, std={student_cls.std().item():.6f}, mean={student_cls.mean().item():.6f}, norm={student_cls.norm(dim=-1).mean().item():.6f}")
                        # Compute cosine similarity (handle dimension mismatch by projecting teacher)
                        if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                            teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                            teacher_cls_proj_norm = F.normalize(teacher_cls_proj, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            cosine_sim = (teacher_cls_proj_norm * student_cls_norm).sum(dim=-1).mean().item()
                        elif teacher_cls.shape[-1] == student_cls.shape[-1]:
                            teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            cosine_sim = (teacher_cls_norm * student_cls_norm).sum(dim=-1).mean().item()
                        else:
                            cosine_sim = None
                        if cosine_sim is not None:
                            print(f"  CLS cosine similarity: {cosine_sim:.6f} (1.0 = identical, 0.0 = orthogonal)")
                        else:
                            print(f"  CLS cosine similarity: N/A (dimension mismatch: {teacher_cls.shape[-1]} vs {student_cls.shape[-1]})")
                        print(f"  Teacher patches: shape={teacher_patches.shape}, std={teacher_patches.std().item():.6f}")
                        print(f"  Student patches: shape={student_patches.shape}, std={student_patches.std().item():.6f}")
                    
                    loss, metrics = compute_distillation_loss(
                        student_cls, student_patches,
                        teacher_cls, teacher_patches,
                        loss_weights=loss_weights,
                        distillation_loss_module=distillation_loss_module
                    )
        else:
            # CPU: no autocast
            cache_key = None
            if cache_teacher_features:
                cache_key = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
            
            teacher_cls, teacher_patches = extract_teacher_features(
                teacher, images, use_cls_token=use_cls_token,
                cache_dir=teacher_feature_dir, cache_key=cache_key
            )
            
            student_cls, student_patches = extract_student_features(
                student, images, use_cls_token=use_cls_token
            )
            
            # Debug: Print feature statistics on first batch of first epoch
            if batch_idx == 0 and epoch == 0:
                print(f"\n🔍 Debug: Feature statistics (first batch, epoch {epoch+1})")
                print(f"  Teacher CLS: shape={teacher_cls.shape}, std={teacher_cls.std().item():.6f}, mean={teacher_cls.mean().item():.6f}, norm={teacher_cls.norm(dim=-1).mean().item():.6f}")
                print(f"  Student CLS: shape={student_cls.shape}, std={student_cls.std().item():.6f}, mean={student_cls.mean().item():.6f}, norm={student_cls.norm(dim=-1).mean().item():.6f}")
                # Compute cosine similarity (handle dimension mismatch by projecting teacher)
                if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                    teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                    teacher_cls_proj_norm = F.normalize(teacher_cls_proj, dim=-1)
                    student_cls_norm = F.normalize(student_cls, dim=-1)
                    cosine_sim = (teacher_cls_proj_norm * student_cls_norm).sum(dim=-1).mean().item()
                elif teacher_cls.shape[-1] == student_cls.shape[-1]:
                    teacher_cls_norm = F.normalize(teacher_cls, dim=-1)
                    student_cls_norm = F.normalize(student_cls, dim=-1)
                    cosine_sim = (teacher_cls_norm * student_cls_norm).sum(dim=-1).mean().item()
                else:
                    cosine_sim = None
                if cosine_sim is not None:
                    print(f"  CLS cosine similarity: {cosine_sim:.6f} (1.0 = identical, 0.0 = orthogonal)")
                else:
                    print(f"  CLS cosine similarity: N/A (dimension mismatch: {teacher_cls.shape[-1]} vs {student_cls.shape[-1]})")
                print(f"  Teacher patches: shape={teacher_patches.shape}, std={teacher_patches.std().item():.6f}")
                print(f"  Student patches: shape={student_patches.shape}, std={student_patches.std().item():.6f}")
            
            loss, metrics = compute_distillation_loss(
                student_cls, student_patches,
                teacher_cls, teacher_patches,
                loss_weights=loss_weights,
                distillation_loss_module=distillation_loss_module
            )
        
        gpu_time = time.time() - gpu_start
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_metrics['cls'] += metrics['cls']
        total_metrics['patch'] += metrics['patch']
        steps_completed += 1
        global_step += 1
        
        # Track times
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        data_times.append(data_load_time)
        gpu_times.append(gpu_time)
        if len(batch_times) > 10:
            batch_times.pop(0)
            data_times.pop(0)
            gpu_times.pop(0)
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_data_time = sum(data_times) / len(data_times) if data_times else 0
        avg_gpu_time = sum(gpu_times) / len(gpu_times)
        
        # Update progress bar with detailed timing
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{loss.item():.8f}',
            'cls': f'{metrics["cls"]:.8f}',
            'patch': f'{metrics["patch"]:.8f}',
            'lr': f'{current_lr:.6f}',
            'gpu': f'{avg_gpu_time:.2f}s',
            'data': f'{avg_data_time:.2f}s',
            'batch': f'{avg_batch_time:.2f}s',
            'step': f'{steps_completed}/{max_steps}'
        })
        
        # Step-based checkpointing
        if save_every > 0 and global_step % save_every == 0 and checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'student': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_step_{global_step}.pth")
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
        
        prev_iter_time = time.time()
    
    # Step scheduler at end of epoch
    scheduler.step()
    
    # Compute averages
    num_batches = steps_completed
    if num_batches == 0:
        num_batches = 1
    
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics, global_step


def train_distillation(teacher, student, train_loader, num_epochs, device,
                      lr=5e-4, weight_decay=0.04, warmup_epochs=10,
                      loss_weights=None, checkpoint_dir=None, resume_from=None,
                      compile_student=True, use_fused_adamw=True,
                      use_cls_token=True, use_multi_crop=False, 
                      max_steps_per_epoch=None, cache_teacher_features=False,
                      teacher_feature_dir=None, save_every=0):
    """
    Main training function with all optimizations
    """
    if loss_weights is None:
        loss_weights = {'cls': 1.0, 'patch': 0.5}
    
    # Enable TF32 for faster training
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Print step-capping info
    if max_steps_per_epoch is not None:
        total_batches = len(train_loader)
        print(f"✓ Step-capped training: {max_steps_per_epoch} steps/epoch")
        print(f"  Full dataset: {total_batches} batches")
        print(f"  Each epoch will process: {max_steps_per_epoch} batches")
    else:
        print("✓ Full epoch training (no step cap)")
    
    # Compile ONLY student model (never teacher)
    if compile_student and hasattr(torch, 'compile'):
        print("Compiling student model with torch.compile...")
        print("⚠️  First compilation may take 5-10 minutes - this is normal!")
        student = torch.compile(student, mode='reduce-overhead')
        print("✓ Student model compiled successfully")
        print("  ⚠️  Teacher is NOT compiled (frozen, no benefit)")
    
    # Create distillation loss module with learnable projections
    # Teacher (DINOv2 ViT-B): 768 dims, Student (ViT-S): 384 dims
    distillation_loss_module = DistillationLoss(
        teacher_dim=768,
        student_dim=384,
        loss_weights=loss_weights
    ).to(device)
    print("✓ Created DistillationLoss module with learnable projections (768→384)")
    
    # Build optimizer: include both student and projection layers
    # Combine student parameters and distillation loss projection parameters
    # Group parameters: weight decay for non-bias/norm, no weight decay for bias/norm
    params_with_wd = []
    params_without_wd = []
    for name, param in student.named_parameters():
        if any(nd in name for nd in ["bias", "norm", "ln"]):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)
    for name, param in distillation_loss_module.named_parameters():
        if any(nd in name for nd in ["bias", "norm", "ln"]):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)
    
    param_groups = [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
    
    # Use fused AdamW for better performance (requires CUDA)
    if use_fused_adamw and device.type == 'cuda':
        try:
            optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=False)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, fused=False)
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs,
                               warmup_epochs=warmup_epochs)
    
    # GradScaler for mixed precision (use new API if available to avoid deprecation warning)
    if GRADSCALER_NEW_API:
        if device.type == 'cuda':
            scaler = GradScaler('cuda')
        else:
            scaler = GradScaler('cpu')
    else:
        # Old API
        scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    global_step = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        student.load_state_dict(checkpoint['student'])
        if 'distillation_loss' in checkpoint:
            distillation_loss_module.load_state_dict(checkpoint['distillation_loss'])
            print("✓ Loaded distillation loss module state")
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"✓ Resumed from epoch {start_epoch}, step {global_step}")
    
    # Initialize scheduler (call after optimizer.step() in first iteration, not before)
    # We'll call scheduler.step() at the end of each epoch instead
    
    # Check if dataset supports set_epoch (for random subset sampling)
    dataset = train_loader.dataset
    has_set_epoch = hasattr(dataset, 'set_epoch')
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Resample random subset if using RandomSubsetCachedDataset
        if has_set_epoch:
            dataset.set_epoch(epoch)
            print(f"  ↻ Resampled random subset for epoch {epoch+1}")
        
        avg_loss, avg_metrics, global_step = train_epoch(
            teacher, student, train_loader, optimizer, scheduler,
            device, scaler, epoch, num_epochs, loss_weights,
            use_cls_token=use_cls_token, use_multi_crop=use_multi_crop,
            max_steps_per_epoch=max_steps_per_epoch,
            cache_teacher_features=cache_teacher_features,
            teacher_feature_dir=teacher_feature_dir,
            save_every=save_every, global_step=global_step,
            checkpoint_dir=checkpoint_dir,
            distillation_loss_module=distillation_loss_module
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.8f} "
              f"(CLS: {avg_metrics['cls']:.8f}, Patch: {avg_metrics['patch']:.8f})")
        
        # Save checkpoint at end of epoch
        # Note: scheduler.step() is called at the end of train_epoch() after all optimizer.step() calls
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'student': student.state_dict(),
                'distillation_loss': distillation_loss_module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"  ✓ Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    return student


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoint_latest.pth first
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    if os.path.exists(latest_path):
        return latest_path
    
    # Otherwise, find the highest epoch checkpoint
    checkpoint_files = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            try:
                epoch_num = int(f.replace('checkpoint_epoch_', '').replace('.pth', ''))
                checkpoint_files.append((epoch_num, os.path.join(checkpoint_dir, f)))
            except ValueError:
                continue
    
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: x[0], reverse=True)
        return checkpoint_files[0][1]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training (with Cached Support)")
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data config YAML')
    parser.add_argument('--train_config', type=str, required=True,
                       help='Path to training config YAML')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, default='',
                       help='Path to checkpoint to resume from (auto-detects latest if not provided)')
    parser.add_argument('--no_resume', action='store_true',
                       help='Disable auto-resume from latest checkpoint')
    parser.add_argument('--max_steps_per_epoch', type=int, default=None,
                       help='Override max_steps_per_epoch from config')
    parser.add_argument('--precompute_cache_only', action='store_true',
                       help='Only precompute cache and exit (no training)')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)
    model_cfg = load_config(args.model_config)
    
    # Override max_steps_per_epoch from CLI if provided
    if args.max_steps_per_epoch is not None:
        train_cfg['max_steps_per_epoch'] = args.max_steps_per_epoch
        print(f"✓ Overrode max_steps_per_epoch to {args.max_steps_per_epoch}")
    
    # Handle precompute-only mode
    if args.precompute_cache_only:
        print("Precompute cache mode: Running cache precomputation only...")
        from precompute_cache import precompute_cache
        precompute_cache(data_cfg, train_cfg, batch_size=256)
        print("✓ Cache precomputation complete. Exiting.")
        return
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: Running on CPU - training will be very slow!")
    
    # Check cache mode (new setting takes precedence)
    use_cached_tensors = data_cfg.get('use_cached_tensors', data_cfg.get('use_cached', None))
    
    # Auto-detect cache if not explicitly set
    if use_cached_tensors is None:
        cache_root = data_cfg.get('cache_root', data_cfg.get('cache_dir', './cache_images'))
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
        cache_root = data_cfg.get('cache_root', data_cfg.get('cache_dir', './cache_images'))
        cache_root = os.path.expandvars(cache_root)
        cache_image_size = data_cfg.get('cache_image_size', 256)
        cache_dtype = data_cfg.get('cache_dtype', 'float32')
        
        print(f"✓ Using cached tensor mode")
        print(f"  Cache root: {cache_root}")
        print(f"  Cache image size: {cache_image_size}x{cache_image_size}")
        print(f"  Cache dtype: {cache_dtype}")
        print(f"  Applying full DINO-style augmentations during training")
        
        # Verify cache exists (check for sharded format first, then legacy)
        cache_path = Path(cache_root)
        shard_index_files = list(cache_path.glob("index_shard_*.json"))
        legacy_index_path = cache_path / "index.json"
        
        if not shard_index_files and not legacy_index_path.exists():
            raise FileNotFoundError(
                f"Cache index not found in {cache_root}\n"
                f"Expected either:\n"
                f"  - New format: index_shard_*.json files\n"
                f"  - Legacy format: index.json\n"
                f"Please run precompute_cache.py first to create the cache."
            )
        
        # Load and display cache metadata
        import json
        if shard_index_files:
            # New sharded format: merge metadata from all shard indices
            total_samples = 0
            total_cache_files = 0
            cache_image_size = None
            for shard_index_path in sorted(shard_index_files):
                with open(shard_index_path, 'r') as f:
                    shard_meta = json.load(f)
                total_samples += shard_meta.get('num_samples_in_slice', 0)
                total_cache_files += len(shard_meta.get('shards', []))
                if cache_image_size is None:
                    cache_image_size = shard_meta.get('cache_image_size', 'unknown')
            print(f"  Cache metadata (sharded format):")
            print(f"    Total samples: {total_samples:,}")
            print(f"    Processing shards: {len(shard_index_files)}")
            print(f"    Total cache files: {total_cache_files}")
            print(f"    Cached image size: {cache_image_size}")
        else:
            # Legacy format
            with open(legacy_index_path, 'r') as f:
                cache_meta = json.load(f)
            print(f"  Cache metadata (legacy format):")
            print(f"    Total samples: {cache_meta.get('num_samples', 'unknown'):,}")
            print(f"    Total cache files: {len(cache_meta.get('shards', []))}")
            print(f"    Cached image size: {cache_meta.get('cache_image_size', 'unknown')}")
    else:
        print(f"✓ Using original dataset mode (HuggingFace/raw images)")
    
    # Load teacher model
    teacher_name = model_cfg.get('teacher_name', 'dinov2_vitb14')
    teacher = load_teacher_model(teacher_name, device=device)
    
    # Build student model
    student_name = model_cfg.get('student_name', 'vit_small_patch16_224')
    student_img_size = model_cfg.get('student_img_size', 224)
    student = build_student_model(student_name, student_img_size, device=device)
    
    # Build DataLoader using factory function (handles cached vs original mode)
    print("\nBuilding DataLoader...")
    dataloader = build_pretraining_dataloader(data_cfg, train_cfg)
    
    # Loss weights
    loss_weights = {
        'cls': train_cfg.get('distill_loss_weights', {}).get('cls', 1.0),
        'patch': train_cfg.get('distill_loss_weights', {}).get('patch', 0.5)
    }
    
    # Training settings
    use_cls_token = model_cfg.get('use_cls_token', True)
    max_steps_per_epoch = train_cfg.get('max_steps_per_epoch', None)
    cache_teacher_features = train_cfg.get('cache_teacher_features', False)
    teacher_feature_dir = train_cfg.get('teacher_feature_dir', None)
    if teacher_feature_dir:
        teacher_feature_dir = os.path.expandvars(teacher_feature_dir)
    
    # Checkpoint directory
    checkpoint_dir = train_cfg.get('checkpoint_dir', './checkpoints')
    checkpoint_dir = os.path.expandvars(checkpoint_dir)
    save_every = train_cfg.get('save_every', 0)  # 0 = only at end of epoch
    
    # Auto-detect latest checkpoint if resume_from not provided and auto-resume enabled
    resume_from = args.resume_from if args.resume_from else None
    if resume_from is None and not args.no_resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_from = latest_checkpoint
            print(f"✓ Auto-detected latest checkpoint: {resume_from}")
            print(f"  To disable auto-resume, use --no_resume flag")
    
    if cache_teacher_features:
        print(f"✓ Teacher feature caching enabled: {teacher_feature_dir}")
    else:
        print("✓ Teacher feature caching disabled")
    
    train_distillation(
        teacher=teacher,
        student=student,
        train_loader=dataloader,
        num_epochs=train_cfg['num_epochs'],
        device=device,
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        warmup_epochs=train_cfg['warmup_epochs'],
        loss_weights=loss_weights,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        compile_student=train_cfg.get('compile_student', True),
        use_fused_adamw=train_cfg.get('use_fused_adamw', True),
        use_cls_token=use_cls_token,
        use_multi_crop=train_cfg.get('use_multi_crop', False),
        max_steps_per_epoch=max_steps_per_epoch,
        cache_teacher_features=cache_teacher_features,
        teacher_feature_dir=teacher_feature_dir,
        save_every=save_every
    )
    
    print("✓ Training completed!")


if __name__ == '__main__':
    main()

