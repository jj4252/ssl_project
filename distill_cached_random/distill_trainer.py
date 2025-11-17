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


class ProjectionHead(nn.Module):
    """
    Simple MLP head for SSL representation learning.
    Operates on student CLS embeddings (dim = student_dim, e.g., 384).
    
    Note: Does NOT L2-normalize outputs - Barlow Twins loss does its own normalization.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    
    def forward(self, x):
        # x: [B, in_dim]
        z = self.net(x)
        # Don't normalize - Barlow Twins will normalize by mean/std per feature dimension
        return z


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Extract off-diagonal elements from a square matrix.
    
    Args:
        x: [d, d] square matrix
    
    Returns:
        Flattened off-diagonal elements
    """
    n, m = x.shape
    assert n == m
    # Flatten, remove diagonal, reshape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    """
    Barlow Twins loss for self-supervised learning.
    
    Args:
        z1, z2: [B, d], raw projection outputs (NOT pre-normalized)
        lambd: Weight for off-diagonal term (default: 5e-3)
    
    Returns:
        Barlow Twins loss
    
    Note:
        This function normalizes inputs by mean/std per feature dimension internally.
        The ProjectionHead should NOT L2-normalize its outputs.
    """
    assert z1.shape == z2.shape
    B, D = z1.shape
    
    # Normalize per feature dimension: zero mean and unit variance
    # This is the standard Barlow Twins normalization
    z1_norm = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True) + 1e-8)
    z2_norm = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True) + 1e-8)
    
    # Cross-correlation matrix: [D, D]
    # Each element c_ij = mean over batch of (z1_norm[:, i] * z2_norm[:, j])
    c = (z1_norm.T @ z2_norm) / B
    
    # On-diagonal should be 1, off-diagonal should be 0
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    
    loss = on_diag + lambd * off_diag
    
    return loss


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss with student-side projection to match teacher dimension.
    
    Instead of projecting teacher features down to student dimension (which can collapse),
    we project student features up to teacher dimension to match raw DINOv2 features directly.
    This preserves teacher feature diversity and prevents collapse.
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
        
        # Student-side projection layers (only needed if dimensions differ)
        # Project student features UP to teacher dimension to match raw DINOv2 features
        if teacher_dim != student_dim:
            self.student_proj_cls = nn.Linear(student_dim, teacher_dim, bias=False)
            self.student_proj_patch = nn.Linear(student_dim, teacher_dim, bias=False)
            # Initialize with orthogonal matrix to preserve feature diversity
            nn.init.orthogonal_(self.student_proj_cls.weight)
            nn.init.orthogonal_(self.student_proj_patch.weight)
        else:
            self.student_proj_cls = None
            self.student_proj_patch = None
    
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
        # Track if diagnostics have been printed (only print once)
        if not hasattr(self, '_diagnostic_printed'):
            self._diagnostic_printed = False
        
        # CLS token loss
        # Use raw teacher features (frozen, diverse) as target
        teacher_cls_target = F.normalize(teacher_cls, dim=-1)  # [B, D_t] - raw DINOv2 features
        
        # Project student features UP to teacher dimension to match teacher
        if self.student_proj_cls is not None:
            student_cls_proj = self.student_proj_cls(student_cls)  # [B, D_t]
            student_cls_norm = F.normalize(student_cls_proj, dim=-1)  # [B, D_t]
        else:
            # Same dimensions: direct normalization
            student_cls_norm = F.normalize(student_cls, dim=-1)  # [B, D_s]
        
        # DIAGNOSTIC: Check teacher CLS targets actually used in loss
        if not self._diagnostic_printed:
            print(f"\n🔍 DistillationLoss.forward() - Teacher CLS Target Diagnostics:")
            print(f"  Shape: {teacher_cls_target.shape}")
            print(f"  Mean: {teacher_cls_target.mean().item():.6f}")
            print(f"  Std: {teacher_cls_target.std().item():.6f}")
            # Per-dimension variance (over batch, per feature dim)
            cls_var = teacher_cls_target.var(dim=0)  # [D] variance over batch dimension
            print(f"  Per-dim variance (over batch): mean={cls_var.mean().item():.6f}, "
                  f"min={cls_var.min().item():.6f}, max={cls_var.max().item():.6f}")
            # Pairwise similarity over batch (should be < 0.3 for diverse features)
            teacher_cls_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
            print(f"  Pairwise similarity (batch): {teacher_cls_pairwise_sim:.6f} "
                  f"{'⚠️ COLLAPSED!' if teacher_cls_pairwise_sim > 0.9 else '✓ diverse' if teacher_cls_pairwise_sim < 0.3 else '⚠️ suspicious'}")
            # Verify no batch dimension reduction
            print(f"  ✓ Verified: Normalization uses dim=-1 (feature dim), NOT dim=0 (batch dim)")
            print(f"  ✓ Using cosine similarity loss (1 - cosine_sim) instead of MSE")
        
        # CLS loss: per-sample cosine, then mean
        # Use F.cosine_similarity for explicit per-sample computation
        # Note: Both should now be in teacher dimension (D_t)
        assert student_cls_norm.shape[-1] == teacher_cls_target.shape[-1], \
            f"Dimension mismatch: student_cls_norm {student_cls_norm.shape[-1]} vs teacher_cls_target {teacher_cls_target.shape[-1]}"
        cosine_sim_cls = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1)  # [B]
        loss_cls = (1.0 - cosine_sim_cls).mean()
        
        # Patch embeddings loss
        B_s, N_s, D_s = student_patches.shape
        B_t, N_t, D_t = teacher_patches.shape
        
        # Handle patch number mismatch first
        if N_s != N_t:
            if N_s < N_t:
                teacher_patches = teacher_patches[:, :N_s, :]  # Truncate teacher [B, N_s, D_t]
            else:
                student_patches = student_patches[:, :N_t, :]  # Truncate student [B, N_t, D_s]
        
        # Handle dimension mismatch
        # Use raw teacher patch features (frozen, diverse) as target
        teacher_patches_target = F.normalize(teacher_patches, dim=-1)  # [B, N, D_t] - raw DINOv2 features
        
        # Project student patches UP to teacher dimension to match teacher
        if self.student_proj_patch is not None:
            student_patches_proj = self.student_proj_patch(student_patches)  # [B, N, D_t]
            student_patches_norm = F.normalize(student_patches_proj, dim=-1)  # [B, N, D_t]
        else:
            # Same dimensions: direct normalization
            student_patches_norm = F.normalize(student_patches, dim=-1)  # [B, N, D_s]
        
        # DIAGNOSTIC: Check teacher patch targets actually used in loss
        if not self._diagnostic_printed:
            print(f"\n🔍 DistillationLoss.forward() - Teacher Patch Target Diagnostics:")
            print(f"  Shape: {teacher_patches_target.shape}")
            print(f"  Mean: {teacher_patches_target.mean().item():.6f}")
            print(f"  Std: {teacher_patches_target.std().item():.6f}")
            # Per-dimension variance (over batch and patches, per feature dim)
            patch_var = teacher_patches_target.var(dim=(0, 1))  # [D] variance over batch and patches
            print(f"  Per-dim variance (over batch+patches): mean={patch_var.mean().item():.6f}, "
                  f"min={patch_var.min().item():.6f}, max={patch_var.max().item():.6f}")
            # Pairwise similarity: flatten to [B*N, D] then compute
            B, N, D = teacher_patches_target.shape
            teacher_patches_flat = teacher_patches_target.view(B * N, D)
            patch_pairwise_sim = (teacher_patches_flat @ teacher_patches_flat.T).mean().item()
            print(f"  Pairwise similarity (batch*patches): {patch_pairwise_sim:.6f} "
                  f"{'⚠️ COLLAPSED!' if patch_pairwise_sim > 0.9 else '✓ diverse' if patch_pairwise_sim < 0.3 else '⚠️ suspicious'}")
            # Verify aggregation: loss should average over batch, patches, and features
            print(f"  ✓ Verified: Loss aggregates over batch (dim=0), patches (dim=1), features (dim=2)")
            print(f"  ✓ Verified: NO batch dimension reduction before loss computation")
            print(f"  ✓ Using cosine similarity loss (1 - cosine_sim) instead of MSE")
            self._diagnostic_printed = True
        
        # PATCHES loss: per-sample, per-patch cosine, then mean
        # Flatten to [B*N, D] for per-patch cosine similarity
        # Note: Both should now be in teacher dimension (D_t)
        B, N, D = student_patches_norm.shape
        assert D == teacher_patches_target.shape[-1], \
            f"Dimension mismatch: student_patches_norm {D} vs teacher_patches_target {teacher_patches_target.shape[-1]}"
        cosine_sim_patch = F.cosine_similarity(
            student_patches_norm.view(-1, D),
            teacher_patches_target.view(-1, D),
            dim=-1,
        )  # [B*N]
        loss_patch = (1.0 - cosine_sim_patch).mean()
        
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
                checkpoint_dir=None, distillation_loss_module=None,
                student_projection_head=None, ssl_config=None):
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
    total_metrics = {'cls': 0, 'patch': 0, 'ssl': 0}
    
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
        
        # Handle batch format: either (view1, view2) for SSL or single image
        # When SSL is enabled, DataLoader collates tuples into (batch_view1, batch_view2)
        # Check for tuple first (SSL mode returns tuple of two batched views)
        if isinstance(batch, tuple) and len(batch) == 2:
            # Two views for SSL (Barlow Twins) - batch is (batch_view1, batch_view2)
            images_view1 = batch[0].to(device)
            images_view2 = batch[1].to(device)
            images = images_view1  # Use view1 for KD (can also use view2, doesn't matter)
            
            # Convert to channels_last if supported
            try:
                images_view1 = images_view1.to(memory_format=torch.channels_last)
                images_view2 = images_view2.to(memory_format=torch.channels_last)
                images = images.to(memory_format=torch.channels_last)
            except:
                pass
        elif isinstance(batch, list):
            # Multi-crop mode or list of crops
            if use_multi_crop:
                images = batch[0].to(device)  # Use first global crop
            else:
                # List but not multi-crop - might be a single-element list or unexpected format
                # Try to handle it gracefully
                if len(batch) > 0:
                    images = batch[0].to(device) if isinstance(batch[0], torch.Tensor) else torch.stack(batch).to(device)
                else:
                    raise ValueError(f"Empty batch list encountered")
            images_view1 = images  # For compatibility
            images_view2 = images  # Fallback (won't be used for SSL)
            
            # Convert to channels_last if supported
            try:
                images = images.to(memory_format=torch.channels_last)
            except:
                pass
        elif isinstance(batch, torch.Tensor):
            # Single image tensor (normal case when SSL is disabled)
            images = batch.to(device)
            images_view1 = images  # For compatibility
            images_view2 = images  # Fallback (won't be used for SSL)
            
            # Convert to channels_last if supported
            try:
                images = images.to(memory_format=torch.channels_last)
            except:
                pass
        else:
            # Unexpected batch format
            raise TypeError(f"Unexpected batch type: {type(batch)}. Expected tuple (for SSL), list (for multi-crop), or Tensor (for single view).")
        
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
                    
                    # Student forward on view1 (for KD)
                    student_cls, student_patches = extract_student_features(
                        student, images, use_cls_token=use_cls_token
                    )
                    
                    # Student forward on view2 (for SSL) - only if SSL is enabled
                    if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None:
                        student_cls_view2, _ = extract_student_features(
                            student, images_view2, use_cls_token=use_cls_token
                        )
                    else:
                        student_cls_view2 = None
                    
                    # Sanity debug: log statistics for initial batches
                    # This helps verify teacher targets are diverse and student is learning
                    if (batch_idx % 50 == 0 or (batch_idx == 0 and epoch < 5)):
                        with torch.no_grad():
                            # Get normalized teacher and student features for debugging
                            # Teacher features are now used directly (no projection)
                            teacher_cls_target = F.normalize(teacher_cls, dim=-1)  # Raw DINOv2 features
            
                            # Student features are projected UP to teacher dimension
                            if distillation_loss_module is not None and hasattr(distillation_loss_module, 'student_proj_cls') and distillation_loss_module.student_proj_cls is not None:
                                student_cls_proj = distillation_loss_module.student_proj_cls(student_cls)
                                student_cls_norm = F.normalize(student_cls_proj, dim=-1)
                            else:
                                student_cls_norm = F.normalize(student_cls, dim=-1)
                            
                            # For diagnostics: teacher is now the same as "raw" (no projection)
                            teacher_raw_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            
                            # Compute statistics
                            teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                            mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                            
                            # Per-dimension variance (should be ~1/D for healthy features)
                            teacher_var = teacher_cls_target.var(dim=0).mean().item()
                            student_var = student_cls_norm.var(dim=0).mean().item()
                            expected_var = 1.0 / teacher_cls_target.shape[-1]  # 1/D for unit-norm vectors
                            
                            print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                            print(f"    Teacher pairwise sim: {teacher_raw_pairwise_sim:.6f} (should be 0.0-0.3 for diverse DINOv2, no projection)")
                            print(f"    Teacher TARGET pairwise sim: {teacher_pairwise_sim:.6f} (same as raw, no projection)")
                            print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f} (should stay <0.7)")
                            print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f} (should increase)")
                            print(f"    Teacher per-dim variance: {teacher_var:.6f} (expected ~{expected_var:.6f})")
                            print(f"    Student per-dim variance: {student_var:.6f} (expected ~{expected_var:.6f})")
                            
                            if teacher_raw_pairwise_sim > 0.9:
                                print(f"    ⚠️  CRITICAL: Raw teacher features are collapsed! Check data loading.")
                            elif teacher_raw_pairwise_sim > 0.5:
                                print(f"    ⚠️  WARNING: Raw teacher features show reduced diversity")
                            if teacher_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Teacher targets (after projection) are collapsing!")
                            if student_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Student features are collapsing!")
                            if student_var < expected_var * 0.1:
                                print(f"    ⚠️  WARNING: Student variance is very low (collapse risk)")
                    
                    # Compute distillation loss
                    loss_kd, metrics = compute_distillation_loss(
                        student_cls, student_patches,
                        teacher_cls, teacher_patches,
                        loss_weights=loss_weights,
                        distillation_loss_module=distillation_loss_module
                    )
                    
                    # Compute SSL loss (Barlow Twins) if enabled
                    loss_ssl = None
                    if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None and student_cls_view2 is not None:
                        # Project both views through projection head
                        z1 = student_projection_head(student_cls)  # [B, proj_dim]
                        z2 = student_projection_head(student_cls_view2)  # [B, proj_dim]
                        
                        # Barlow Twins loss
                        barlow_lambd = ssl_config.get('barlow_lambd', 5e-3)
                        loss_ssl = barlow_twins_loss(z1, z2, lambd=barlow_lambd)
                        
                        # Combine losses
                        ssl_weight = ssl_config.get('weight', 0.5)
                        loss = loss_kd + ssl_weight * loss_ssl
                    else:
                        loss = loss_kd
                        loss_ssl = torch.tensor(0.0, device=device)
                    
                    # Update metrics
                    if loss_ssl is not None:
                        metrics['ssl'] = loss_ssl.item()
                    else:
                        metrics['ssl'] = 0.0
            else:
                with autocast():
                    cache_key = None
                    if cache_teacher_features:
                        cache_key = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
                    
                    teacher_cls, teacher_patches = extract_teacher_features(
                        teacher, images, use_cls_token=use_cls_token,
                        cache_dir=teacher_feature_dir, cache_key=cache_key
                    )
                    
                    # Student forward on view1 (for KD)
                    student_cls, student_patches = extract_student_features(
                        student, images, use_cls_token=use_cls_token
                    )
                    
                    # Student forward on view2 (for SSL) - only if SSL is enabled
                    if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None:
                        student_cls_view2, _ = extract_student_features(
                            student, images_view2, use_cls_token=use_cls_token
                        )
                    else:
                        student_cls_view2 = None
                    
                    # Sanity debug: log statistics for initial batches
                    # This helps verify teacher targets are diverse and student is learning
                    if (batch_idx % 50 == 0 or (batch_idx == 0 and epoch < 5)):
                        with torch.no_grad():
                            # Get normalized teacher and student features for debugging
                            # Teacher features are now used directly (no projection)
                            teacher_cls_target = F.normalize(teacher_cls, dim=-1)  # Raw DINOv2 features
            
                            # Student features are projected UP to teacher dimension
                            if distillation_loss_module is not None and hasattr(distillation_loss_module, 'student_proj_cls') and distillation_loss_module.student_proj_cls is not None:
                                student_cls_proj = distillation_loss_module.student_proj_cls(student_cls)
                                student_cls_norm = F.normalize(student_cls_proj, dim=-1)
                            else:
                                student_cls_norm = F.normalize(student_cls, dim=-1)
                            
                            # For diagnostics: teacher is now the same as "raw" (no projection)
                            teacher_raw_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            
                            # Compute statistics
                            teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                            mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                            
                            # Per-dimension variance (should be ~1/D for healthy features)
                            teacher_var = teacher_cls_target.var(dim=0).mean().item()
                            student_var = student_cls_norm.var(dim=0).mean().item()
                            expected_var = 1.0 / teacher_cls_target.shape[-1]  # 1/D for unit-norm vectors
                            
                            print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                            print(f"    Teacher pairwise sim: {teacher_raw_pairwise_sim:.6f} (should be 0.0-0.3 for diverse DINOv2, no projection)")
                            print(f"    Teacher TARGET pairwise sim: {teacher_pairwise_sim:.6f} (same as raw, no projection)")
                            print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f} (should stay <0.7)")
                            print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f} (should increase)")
                            print(f"    Teacher per-dim variance: {teacher_var:.6f} (expected ~{expected_var:.6f})")
                            print(f"    Student per-dim variance: {student_var:.6f} (expected ~{expected_var:.6f})")
                            
                            if teacher_raw_pairwise_sim > 0.9:
                                print(f"    ⚠️  CRITICAL: Raw teacher features are collapsed! Check data loading.")
                            elif teacher_raw_pairwise_sim > 0.5:
                                print(f"    ⚠️  WARNING: Raw teacher features show reduced diversity")
                            if teacher_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Teacher targets (after projection) are collapsing!")
                            if student_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Student features are collapsing!")
                            if student_var < expected_var * 0.1:
                                print(f"    ⚠️  WARNING: Student variance is very low (collapse risk)")
                    
                    # Compute distillation loss
                    loss_kd, metrics = compute_distillation_loss(
                        student_cls, student_patches,
                        teacher_cls, teacher_patches,
                        loss_weights=loss_weights,
                        distillation_loss_module=distillation_loss_module
                    )
                    
                    # Compute SSL loss (Barlow Twins) if enabled
                    loss_ssl = None
                    if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None and student_cls_view2 is not None:
                        # Project both views through projection head
                        z1 = student_projection_head(student_cls)  # [B, proj_dim]
                        z2 = student_projection_head(student_cls_view2)  # [B, proj_dim]
                        
                        # Barlow Twins loss
                        barlow_lambd = ssl_config.get('barlow_lambd', 5e-3)
                        loss_ssl = barlow_twins_loss(z1, z2, lambd=barlow_lambd)
                        
                        # Combine losses
                        ssl_weight = ssl_config.get('weight', 0.5)
                        loss = loss_kd + ssl_weight * loss_ssl
                    else:
                        loss = loss_kd
                        loss_ssl = torch.tensor(0.0, device=device)
                    
                    # Update metrics
                    if loss_ssl is not None:
                        metrics['ssl'] = loss_ssl.item()
                    else:
                        metrics['ssl'] = 0.0
        else:
            # CPU: no autocast
            cache_key = None
            if cache_teacher_features:
                cache_key = hashlib.md5(images.cpu().numpy().tobytes()).hexdigest()
            
            teacher_cls, teacher_patches = extract_teacher_features(
                teacher, images, use_cls_token=use_cls_token,
                cache_dir=teacher_feature_dir, cache_key=cache_key
            )
            
            # Student forward on view1 (for KD)
            student_cls, student_patches = extract_student_features(
                student, images, use_cls_token=use_cls_token
            )
            
            # Student forward on view2 (for SSL) - only if SSL is enabled
            if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None:
                student_cls_view2, _ = extract_student_features(
                    student, images_view2, use_cls_token=use_cls_token
                )
            else:
                student_cls_view2 = None
            
            # Sanity debug: log statistics for initial batches
            # This helps verify teacher targets are diverse and student is learning
            if (batch_idx % 50 == 0 or (batch_idx == 0 and epoch < 5)):
                with torch.no_grad():
                    # Get normalized teacher and student features for debugging
                    # Teacher features are now used directly (no projection)
                    teacher_cls_target = F.normalize(teacher_cls, dim=-1)  # Raw DINOv2 features
            
                    # Student features are projected UP to teacher dimension
                    if distillation_loss_module is not None and hasattr(distillation_loss_module, 'student_proj_cls') and distillation_loss_module.student_proj_cls is not None:
                        student_cls_proj = distillation_loss_module.student_proj_cls(student_cls)
                        student_cls_norm = F.normalize(student_cls_proj, dim=-1)
                    else:
                        student_cls_norm = F.normalize(student_cls, dim=-1)
                    
                    # For diagnostics: teacher is now the same as "raw" (no projection)
                    teacher_raw_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                    
                    # Compute statistics
                    teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                    student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                    mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                    
                    # Per-dimension variance (should be ~1/D for healthy features)
                    teacher_var = teacher_cls_target.var(dim=0).mean().item()
                    student_var = student_cls_norm.var(dim=0).mean().item()
                    expected_var = 1.0 / teacher_cls_target.shape[-1]  # 1/D for unit-norm vectors
                    
                    print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                    print(f"    Teacher RAW pairwise sim: {teacher_raw_pairwise_sim:.6f} (should be 0.0-0.3 for diverse DINOv2)")
                    print(f"    Teacher TARGET pairwise sim: {teacher_pairwise_sim:.6f} (after projection)")
                    print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f} (should stay <0.7)")
                    print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f} (should increase)")
                    print(f"    Teacher per-dim variance: {teacher_var:.6f} (expected ~{expected_var:.6f})")
                    print(f"    Student per-dim variance: {student_var:.6f} (expected ~{expected_var:.6f})")
                    
                    if teacher_raw_pairwise_sim > 0.9:
                        print(f"    ⚠️  CRITICAL: Raw teacher features are collapsed! Check data loading.")
                    elif teacher_raw_pairwise_sim > 0.5:
                        print(f"    ⚠️  WARNING: Raw teacher features show reduced diversity")
                    if teacher_pairwise_sim > 0.9:
                        print(f"    ⚠️  WARNING: Teacher targets (after projection) are collapsing!")
                    if student_pairwise_sim > 0.9:
                        print(f"    ⚠️  WARNING: Student features are collapsing!")
                    if student_var < expected_var * 0.1:
                        print(f"    ⚠️  WARNING: Student variance is very low (collapse risk)")
            
            # Compute distillation loss
            loss_kd, metrics = compute_distillation_loss(
                student_cls, student_patches,
                teacher_cls, teacher_patches,
                loss_weights=loss_weights,
                distillation_loss_module=distillation_loss_module
            )
            
            # Compute SSL loss (Barlow Twins) if enabled
            loss_ssl = None
            if ssl_config and ssl_config.get('enabled', False) and student_projection_head is not None and student_cls_view2 is not None:
                # Project both views through projection head
                z1 = student_projection_head(student_cls)  # [B, proj_dim]
                z2 = student_projection_head(student_cls_view2)  # [B, proj_dim]
                
                # Barlow Twins loss
                barlow_lambd = ssl_config.get('barlow_lambd', 5e-3)
                loss_ssl = barlow_twins_loss(z1, z2, lambd=barlow_lambd)
                
                # Combine losses
                ssl_weight = ssl_config.get('weight', 0.5)
                loss = loss_kd + ssl_weight * loss_ssl
            else:
                loss = loss_kd
                loss_ssl = torch.tensor(0.0, device=device)
            
            # Update metrics
            if loss_ssl is not None:
                metrics['ssl'] = loss_ssl.item()
            else:
                metrics['ssl'] = 0.0
        
        gpu_time = time.time() - gpu_start
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_metrics['cls'] += metrics['cls']
        total_metrics['patch'] += metrics['patch']
        total_metrics['ssl'] += metrics.get('ssl', 0.0)
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
        postfix_dict = {
            'loss': f'{loss.item():.8f}',
            'cls': f'{metrics["cls"]:.8f}',
            'patch': f'{metrics["patch"]:.8f}',
            'lr': f'{current_lr:.6f}',
            'gpu': f'{avg_gpu_time:.2f}s',
            'data': f'{avg_data_time:.2f}s',
            'batch': f'{avg_batch_time:.2f}s',
            'step': f'{steps_completed}/{max_steps}'
        }
        if ssl_config and ssl_config.get('enabled', False) and metrics.get('ssl', 0.0) > 0:
            postfix_dict['ssl'] = f'{metrics["ssl"]:.8f}'
        progress_bar.set_postfix(postfix_dict)
        
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
                      teacher_feature_dir=None, save_every=0, ssl_config=None):
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
    
    # Create distillation loss module with student-side projection
    # Teacher (DINOv2 ViT-B): 768 dims, Student (ViT-S): 384 dims
    # NEW: Project student UP to teacher dimension (384→768) to match raw DINOv2 features
    # This prevents teacher feature collapse that occurred with teacher-side projection
    distillation_loss_module = DistillationLoss(
        teacher_dim=768,
        student_dim=384,
        loss_weights=loss_weights
    ).to(device)
    print("✓ Created DistillationLoss module with student-side projection (384→768)")
    print("  ✓ Student features projected UP to match raw DINOv2 features (preserves teacher diversity)")
    print("  ✓ Projection layers are TRAINABLE (will learn optimal 384→768 mapping)")
    
    # Create SSL projection head if SSL is enabled
    student_projection_head = None
    if ssl_config and ssl_config.get('enabled', False):
        student_dim = 384  # ViT-S embedding dimension
        proj_hidden_dim = ssl_config.get('proj_hidden_dim', 1024)
        proj_out_dim = ssl_config.get('proj_out_dim', 256)
        student_projection_head = ProjectionHead(
            in_dim=student_dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim
        ).to(device)
        print(f"✓ Created SSL projection head: {student_dim} → {proj_hidden_dim} → {proj_out_dim}")
    else:
        print("✓ SSL disabled (no projection head)")
    
    # Build optimizer: student parameters + distillation projection layers + projection head (if SSL enabled)
    # Group parameters: weight decay for non-bias/norm, no weight decay for bias/norm
    # CRITICAL: Ensure all parameters have the same dtype and device
    params_with_wd = []
    params_without_wd = []
    
    # Get reference dtype and device from student (in case it's compiled)
    # Extract from actual parameter to handle compiled models
    try:
        ref_param = next(iter(student.parameters()))
        ref_dtype = ref_param.dtype
        ref_device = ref_param.device
    except StopIteration:
        # Fallback if student has no parameters (shouldn't happen)
        ref_dtype = torch.float32
        ref_device = device
    
    # Ensure distillation_loss_module parameters match student's dtype and device
    # Move the entire module to ensure consistency
    distillation_loss_module = distillation_loss_module.to(device=ref_device, dtype=ref_dtype)
    
    # Ensure SSL projection head parameters match student's dtype and device
    if student_projection_head is not None:
        student_projection_head = student_projection_head.to(device=ref_device, dtype=ref_dtype)
    
    # Student parameters
    for name, param in student.named_parameters():
        if any(nd in name for nd in ["bias", "norm", "ln"]):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)
    
    # Distillation loss module parameters (projection layers) - NOW TRAINABLE
    for name, param in distillation_loss_module.named_parameters():
        if any(nd in name for nd in ["bias", "norm", "ln"]):
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)
    
    # Add projection head parameters if SSL is enabled
    if student_projection_head is not None:
        for name, param in student_projection_head.named_parameters():
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
        if student_projection_head is not None and 'student_projection_head' in checkpoint:
            student_projection_head.load_state_dict(checkpoint['student_projection_head'])
            print("✓ Loaded SSL projection head state")
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
            distillation_loss_module=distillation_loss_module,
            student_projection_head=student_projection_head, ssl_config=ssl_config
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
            if student_projection_head is not None:
                checkpoint['student_projection_head'] = student_projection_head.state_dict()
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
    
    # SSL configuration
    ssl_config = train_cfg.get('ssl', None)
    
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
        save_every=save_every,
        ssl_config=ssl_config
    )
    
    print("✓ Training completed!")


if __name__ == '__main__':
    main()

