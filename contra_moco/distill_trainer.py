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
import sys
from pathlib import Path

# Add current directory to Python path for local imports
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

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


class MoCoModel(nn.Module):
    """
    MoCo-v3 style contrastive learning with flexible backbone support.
    
    Supports both Vision Transformers (ViT) and ResNet backbones.
    Uses momentum encoder and a queue of negative samples for contrastive learning.
    """
    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        image_size: int = 96,
        proj_dim: int = 256,
        proj_hidden_dim: int = 0,  # 0 = use embed_dim (backward compatible)
        queue_size: int = 65536,
        momentum: float = 0.99,
        temperature: float = 0.2,
        use_queue: bool = True,
        backbone_type: str = "auto",  # "auto", "vit", "resnet" - auto-detect from backbone_name
    ):
        super().__init__()
        
        # Auto-detect backbone type if not specified
        if backbone_type == "auto":
            if "vit" in backbone_name.lower() or "deit" in backbone_name.lower():
                backbone_type = "vit"
            elif "resnet" in backbone_name.lower() or "res" in backbone_name.lower():
                backbone_type = "resnet"
            else:
                # Default to ViT for timm models
                backbone_type = "vit"
        
        self.backbone_type = backbone_type
        
        # Query encoder (trainable)
        if backbone_type == "vit":
            self.encoder_q = timm.create_model(
                backbone_name,
                img_size=image_size,
                num_classes=0,  # We use CLS features, no classifier
                pretrained=False,  # Start from scratch
                global_pool="",  # Don't pool, return all tokens for ViT
            )
        else:  # ResNet
            # ResNet models don't accept img_size parameter
            self.encoder_q = timm.create_model(
                backbone_name,
                num_classes=0,  # We use global pooling features, no classifier
                pretrained=False,  # Start from scratch
                global_pool="avg",  # Global average pooling for ResNet
            )
        
        # Key encoder (momentum encoder, same architecture)
        if backbone_type == "vit":
            self.encoder_k = timm.create_model(
                backbone_name,
                img_size=image_size,
                num_classes=0,
                pretrained=False,
                global_pool="",  # Don't pool, return all tokens for ViT
            )
        else:  # ResNet
            # ResNet models don't accept img_size parameter
            self.encoder_k = timm.create_model(
                backbone_name,
                num_classes=0,
                pretrained=False,
                global_pool="avg",  # Global average pooling for ResNet
            )
        
        # Get embedding dimension from encoder
        embed_dim = self.encoder_q.num_features
        
        # Projection heads: MLP on top of features
        # Use proj_hidden_dim if provided (>0), otherwise use embed_dim (backward compatible)
        if proj_hidden_dim <= 0:
            proj_hidden_dim = embed_dim
        
        self.proj_q = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_dim),
        )
        
        self.proj_k = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Linear(proj_hidden_dim, proj_dim),
        )
        
        # Initialize encoder_k params = encoder_q params, and stop gradients
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # Initialize projection heads the same way
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # MoCo queue: [proj_dim, queue_size]
        # Each column is a key vector, so normalize along dim=0 (across proj_dim for each column)
        self.register_buffer("queue", torch.randn(proj_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)  # Normalize each key vector (column) to unit norm
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        self.momentum = momentum
        self.temperature = temperature
        self.queue_size = queue_size
        self.proj_dim = proj_dim
        self.use_queue = use_queue  # Can disable queue for batch-only contrastive learning (debugging)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        EMA update: param_k = m * param_k + (1-m) * param_q
        
        CRITICAL: encoder_k and proj_k must have requires_grad=False
        """
        # Verify encoder_k is frozen (no gradients)
        for name, param_k in self.encoder_k.named_parameters():
            assert not param_k.requires_grad, \
                f"CRITICAL: encoder_k.{name} has requires_grad=True! Key encoder must be frozen."
        
        for name, param_k in self.proj_k.named_parameters():
            assert not param_k.requires_grad, \
                f"CRITICAL: proj_k.{name} has requires_grad=True! Key projection must be frozen."
        
        # EMA update: param_k = m * param_k + (1-m) * param_q
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        # Also update projection head
        for param_q, param_k in zip(self.proj_q.parameters(), self.proj_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        Update the queue with new keys.
        
        CRITICAL: Must receive k (key vectors), NEVER q (query vectors)!
        
        Args:
            keys: [B, proj_dim], should be normalized key vectors (k), not query vectors (q)
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # CRITICAL VERIFICATION: Ensure keys are detached (no gradients)
        assert not keys.requires_grad, \
            "CRITICAL: Keys must be detached before enqueueing! Use k, not q, and ensure it's in no_grad context."
        
        # Ensure keys are normalized (safety check)
        keys = F.normalize(keys, dim=-1)
        
        # Replace entries in [ptr : ptr + batch_size]
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[:, ptr:end] = keys.T  # [proj_dim, batch_size]
        else:
            # Wrap around
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].T
            self.queue[:, :end - self.queue_size] = keys[first:].T
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def _extract_features(self, encoder, images):
        """
        Extract features from encoder (backbone-agnostic).
        
        For ViT: extracts CLS token features
        For ResNet: extracts global average pooled features
        """
        if self.backbone_type == "vit":
            # ViT: extract CLS token
            features = encoder.forward_features(images)
            if isinstance(features, torch.Tensor):
                # If features is a tensor, CLS is first token
                feat = features[:, 0]  # [B, embed_dim]
            else:
                # If features is a dict, extract CLS token
                feat = features.get('x', features.get('tokens', None))[:, 0]
        else:
            # ResNet: forward_features already returns pooled features with global_pool="avg"
            feat = encoder.forward_features(images)
            # If it's still a tensor with spatial dimensions, do global pooling
            if len(feat.shape) > 2:
                feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return feat
    
    def forward_moco(self, im_q: torch.Tensor, im_k: torch.Tensor, 
                     im_local1: torch.Tensor = None, im_local2: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for MoCo-v3 contrastive learning with optional multi-crop support.
        
        Args:
            im_q: Query images (global crop 1) [B, C, H, W]
            im_k: Key images (global crop 2) [B, C, H, W] (different augmentation of same images)
            im_local1: Optional local crop 1 [B, C, H, W]
            im_local2: Optional local crop 2 [B, C, H, W]
        
        Returns:
            Contrastive loss (InfoNCE), averaged over all crops
        """
        batch_size = im_q.shape[0]
        
        # Assertions for debugging
        assert im_q.shape == im_k.shape, f"im_q and im_k must have same shape, got {im_q.shape} vs {im_k.shape}"
        assert batch_size > 0, "Batch size must be positive"
        
        # -----------------
        # Key branch (with momentum update) - computed once for all queries
        # -----------------
        with torch.no_grad():
            # Update momentum encoder (EMA update)
            self._momentum_update_key_encoder()
            
            # Forward through momentum encoder
            k_features = self._extract_features(self.encoder_k, im_k)  # [B, embed_dim]
            k = self.proj_k(k_features)  # [B, proj_dim]
            k = F.normalize(k, dim=-1)  # L2 normalize
        
        # Count number of crops for averaging
        num_crops = 1  # Start with global crop
        if im_local1 is not None:
            num_crops += 1
        if im_local2 is not None:
            num_crops += 1
        
        # -----------------
        # Global crop 1 (query) vs Global crop 2 (key)
        # -----------------
        q_features = self._extract_features(self.encoder_q, im_q)  # [B, embed_dim]
        q = self.proj_q(q_features)  # [B, proj_dim]
        q = F.normalize(q, dim=-1)  # L2 normalize
        
        loss_global = self._compute_contrastive_loss(q, k, batch_size)
        # Accumulate loss incrementally (divide by num_crops to get average)
        total_loss = loss_global / num_crops
        
        # Free memory: delete intermediate tensors for global crop
        del q_features, q
        
        # -----------------
        # Local crops (if provided) - process sequentially to save memory
        # Each local crop contrasts against the global key
        # -----------------
        if im_local1 is not None:
            assert im_local1.shape == im_q.shape, f"im_local1 shape mismatch: {im_local1.shape} vs {im_q.shape}"
            q_local1_features = self._extract_features(self.encoder_q, im_local1)  # [B, embed_dim]
            q_local1 = self.proj_q(q_local1_features)  # [B, proj_dim]
            q_local1 = F.normalize(q_local1, dim=-1)  # L2 normalize
            
            loss_local1 = self._compute_contrastive_loss(q_local1, k, batch_size)
            # Accumulate loss incrementally
            total_loss = total_loss + loss_local1 / num_crops
            
            # Free memory: delete intermediate tensors (process sequentially to reduce peak memory)
            del q_local1_features, q_local1, loss_local1
        
        if im_local2 is not None:
            assert im_local2.shape == im_q.shape, f"im_local2 shape mismatch: {im_local2.shape} vs {im_q.shape}"
            q_local2_features = self._extract_features(self.encoder_q, im_local2)  # [B, embed_dim]
            q_local2 = self.proj_q(q_local2_features)  # [B, proj_dim]
            q_local2 = F.normalize(q_local2, dim=-1)  # L2 normalize
            
            loss_local2 = self._compute_contrastive_loss(q_local2, k, batch_size)
            # Accumulate loss incrementally
            total_loss = total_loss + loss_local2 / num_crops
            
            # Free memory: delete intermediate tensors
            del q_local2_features, q_local2, loss_local2
        
        # Update queue with new keys (detached, normalized) - only if using queue
        # CRITICAL: Must use k (key), NEVER q (query)
        if self.use_queue:
            with torch.no_grad():
                # Explicitly verify we're enqueueing k, not q
                assert k.requires_grad == False, "k must be detached before enqueueing!"
                self._dequeue_and_enqueue(k)  # k is the key, q is the query - NEVER enqueue q!
        
        return total_loss
    
    def _compute_contrastive_loss(self, q: torch.Tensor, k: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss between queries and keys.
        
        Args:
            q: Query features [B, proj_dim], normalized
            k: Key features [B, proj_dim], normalized
            batch_size: Batch size
        
        Returns:
            InfoNCE loss
        """
        # Positive: q · k+ (same index in batch)
        # Element-wise dot product: [B] -> [B, 1]
        logits_pos = torch.einsum("bd,bd->b", [q, k]).unsqueeze(-1)  # [B, 1]
        
        if self.use_queue:
            # Negative: q · queue
            # q: [B, proj_dim], queue: [proj_dim, queue_size]
            # Result: [B, queue_size]
            logits_neg = torch.einsum("bd,dk->bk", [q, self.queue])  # [B, queue_size]
            
            # Concatenate: [B, 1 + queue_size]
            # Positive is at index 0, negatives at indices 1:queue_size+1
            logits = torch.cat([logits_pos, logits_neg], dim=1)
            
            # Assertion: verify positive is at index 0
            assert logits.shape == (batch_size, 1 + self.queue_size), \
                f"Logits shape mismatch: expected {(batch_size, 1 + self.queue_size)}, got {logits.shape}"
        else:
            # Batch-only contrastive learning (for debugging)
            # Use other samples in the batch as negatives
            # q: [B, proj_dim], k: [B, proj_dim]
            # Compute all pairwise similarities: [B, B]
            logits_all = torch.einsum("bd,cd->bc", [q, k])  # [B, B]
            
            # Positive is on diagonal, negatives are off-diagonal
            logits = logits_all  # [B, B]
            
            # Targets: diagonal indices (0, 1, 2, ..., B-1)
            targets = torch.arange(batch_size, dtype=torch.long, device=logits.device)
        
        # Apply temperature
        logits = logits / self.temperature
        
        # Set targets based on mode
        if not self.use_queue:
            # For batch-only mode, targets are already set above (diagonal indices)
            pass
        else:
            # Targets: all positives are at index 0
            targets = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # CRITICAL VERIFICATION: Assert logits[:,0] is positive (q·k) and labels are correct
        if self.use_queue:
            # Verify positive logits are at index 0
            assert torch.allclose(logits[:, 0], logits_pos.squeeze(-1) / self.temperature), \
                "CRITICAL: Positive logits must be at index 0!"
            # Verify targets are all zeros (positive at index 0)
            assert torch.all(targets == 0), \
                "CRITICAL: All targets must be 0 (positive at index 0)!"
            # Runtime check: positive should be larger than negatives (after some training)
            # This is a sanity check that will warn if the model is learning backwards
            if hasattr(self, '_check_pos_neg_ratio'):
                pos_mean = logits[:, 0].mean().item()
                neg_mean = logits[:, 1:].mean().item()
                if pos_mean <= neg_mean:
                    print(f"⚠️  WARNING: Positive logits ({pos_mean:.4f}) <= Negative logits ({neg_mean:.4f})!")
                    print(f"    This suggests the model is learning backwards - check augmentations/labels!")
        
        # InfoNCE loss
        loss = F.cross_entropy(logits, targets)
        return loss


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
        # Track if diagnostics have been printed (only print once)
        if not hasattr(self, '_diagnostic_printed'):
            self._diagnostic_printed = False
        
        # CLS token loss
        if self.teacher_proj_cls is not None:
            # Project teacher to student dimension
            teacher_cls_proj = self.teacher_proj_cls(teacher_cls)  # [B, D_s]
            teacher_cls_target = F.normalize(teacher_cls_proj, dim=-1)  # [B, D_s] - final target used in loss
        else:
            # Same dimensions: direct normalization
            teacher_cls_target = F.normalize(teacher_cls, dim=-1)  # [B, D_t] - final target used in loss
        
        student_cls_norm = F.normalize(student_cls, dim=-1)
        
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
        if self.teacher_proj_patch is not None:
            # Project teacher patches to student dimension
            teacher_patches_proj = self.teacher_proj_patch(teacher_patches)  # [B, N, D_s]
            teacher_patches_target = F.normalize(teacher_patches_proj, dim=-1)  # [B, N, D_s] - final target
        else:
            # Same dimensions: direct normalization
            teacher_patches_target = F.normalize(teacher_patches, dim=-1)  # [B, N, D_t] - final target
        
        student_patches_norm = F.normalize(student_patches, dim=-1)
        
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
        B, N, D = student_patches_norm.shape
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
        # DataLoader may convert tuple to list during collation, so check both
        if (isinstance(batch, (tuple, list)) and len(batch) == 2):
            # Two views for SSL (Barlow Twins) - batch is (batch_view1, batch_view2) or [batch_view1, batch_view2]
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
        elif isinstance(batch, list) and len(batch) > 2:
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
                    
                    # Teacher forward (frozen) - use view1 for KD
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
                    
                    # Step 1 Diagnostic: Verify representation consistency (training vs evaluation)
                    # Print feature statistics that should match evaluation
                    if batch_idx == 0 and epoch == 0:
                        with torch.no_grad():
                            # Extract features exactly as evaluation does
                            eval_features = student.forward_features(images)
                            if isinstance(eval_features, torch.Tensor):
                                eval_cls = eval_features[:, 0]  # CLS token [B, D]
                            else:
                                eval_cls = eval_features.get('x', eval_features.get('tokens', None))[:, 0]
                            eval_cls_norm = F.normalize(eval_cls, dim=-1, p=2)
                            
                            # Compare with training extraction
                            train_cls_norm = F.normalize(student_cls, dim=-1, p=2)
                            
                            print(f"\n🔍 Step 1 Diagnostic: Training vs Evaluation Representation Consistency")
                            print(f"  Training CLS (extract_student_features): shape={student_cls.shape}, "
                                  f"norm={train_cls_norm.norm(dim=-1).mean().item():.6f}, "
                                  f"std={train_cls_norm.std().item():.6f}")
                            print(f"  Eval CLS (forward_features[:, 0]): shape={eval_cls.shape}, "
                                  f"norm={eval_cls_norm.norm(dim=-1).mean().item():.6f}, "
                                  f"std={eval_cls_norm.std().item():.6f}")
                            
                            # Check if they match
                            diff = (train_cls_norm - eval_cls_norm).abs().mean().item()
                            if diff < 1e-5:
                                print(f"  ✓ Representations match (diff={diff:.8f})")
                            else:
                                print(f"  ⚠️  WARNING: Representations differ (diff={diff:.8f})")
                                print(f"     This could cause train/eval mismatch!")
                    
                    # Sanity debug: log statistics for initial batches
                    # This helps verify teacher targets are diverse and student is learning
                    if (batch_idx % 50 == 0 or (batch_idx == 0 and epoch < 5)):
                        with torch.no_grad():
                            # Get normalized teacher and student features for debugging
                            if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                                teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                                teacher_cls_target = F.normalize(teacher_cls_proj, dim=-1)
                            else:
                                teacher_cls_target = F.normalize(teacher_cls, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            
                            # Compute statistics
                            teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                            mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                            
                            print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                            print(f"    Teacher CLS pairwise sim: {teacher_pairwise_sim:.6f}")
                            print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f}")
                            print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f}")
                            if student_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Student features are collapsing!")
                            if teacher_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Teacher targets are collapsing!")
                    
                    # Compute distillation loss (unchanged)
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
                    
                    # Teacher forward (frozen) - use view1 for KD
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
                            if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                                teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                                teacher_cls_target = F.normalize(teacher_cls_proj, dim=-1)
                            else:
                                teacher_cls_target = F.normalize(teacher_cls, dim=-1)
                            student_cls_norm = F.normalize(student_cls, dim=-1)
                            
                            # Compute statistics
                            teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                            student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                            mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                            
                            print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                            print(f"    Teacher CLS pairwise sim: {teacher_pairwise_sim:.6f}")
                            print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f}")
                            print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f}")
                            if student_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Student features are collapsing!")
                            if teacher_pairwise_sim > 0.9:
                                print(f"    ⚠️  WARNING: Teacher targets are collapsing!")
                    
                    # Compute distillation loss (unchanged)
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
            
            # Teacher forward (frozen) - use view1 for KD
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
                    if distillation_loss_module is not None and hasattr(distillation_loss_module, 'teacher_proj_cls') and distillation_loss_module.teacher_proj_cls is not None:
                        teacher_cls_proj = distillation_loss_module.teacher_proj_cls(teacher_cls)
                        teacher_cls_target = F.normalize(teacher_cls_proj, dim=-1)
                    else:
                        teacher_cls_target = F.normalize(teacher_cls, dim=-1)
                    student_cls_norm = F.normalize(student_cls, dim=-1)
                    
                    # Compute statistics
                    teacher_pairwise_sim = (teacher_cls_target @ teacher_cls_target.T).mean().item()
                    student_pairwise_sim = (student_cls_norm @ student_cls_norm.T).mean().item()
                    mean_cosine_st_te = F.cosine_similarity(student_cls_norm, teacher_cls_target, dim=-1).mean().item()
                    
                    print(f"\n  🔍 Debug (epoch {epoch+1}, batch {batch_idx}):")
                    print(f"    Teacher CLS pairwise sim: {teacher_pairwise_sim:.6f}")
                    print(f"    Student CLS pairwise sim: {student_pairwise_sim:.6f}")
                    print(f"    Mean cosine(student, teacher): {mean_cosine_st_te:.6f}")
                    if student_pairwise_sim > 0.9:
                        print(f"    ⚠️  WARNING: Student features are collapsing!")
                    if teacher_pairwise_sim > 0.9:
                        print(f"    ⚠️  WARNING: Teacher targets are collapsing!")
            
            # Compute distillation loss (unchanged)
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
            if student_projection_head is not None:
                checkpoint['student_projection_head'] = student_projection_head.state_dict()
            if distillation_loss_module is not None:
                checkpoint['distillation_loss'] = distillation_loss_module.state_dict()
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
    
    # Print epoch summary
    if ssl_config and ssl_config.get('enabled', False):
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.8f} "
              f"(KD: {avg_metrics['cls']:.8f} + {avg_metrics['patch']:.8f}, "
              f"SSL: {avg_metrics.get('ssl', 0.0):.8f})")
    else:
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.8f} "
              f"(CLS: {avg_metrics['cls']:.8f}, Patch: {avg_metrics['patch']:.8f})")
    
    return avg_loss, avg_metrics, global_step


def train_moco(model, train_loader, num_epochs, device,
               lr=5e-4, weight_decay=0.05, warmup_epochs=10,
               checkpoint_dir=None, resume_from=None,
               compile_student=True, use_fused_adamw=True,
               save_every=0, max_grad_norm=1.0,
               momentum_anneal=False, initial_momentum=0.99):
    """
    Train MoCo-v3 model with contrastive learning.
    
    Args:
        model: MoCoModel (supports both ViT and ResNet backbones)
        train_loader: DataLoader returning (view1, view2) tuples
        num_epochs: Number of training epochs
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs for learning rate
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        compile_student: Whether to compile encoder_q
        use_fused_adamw: Whether to use fused AdamW
        save_every: Save checkpoint every N steps (0 = only at end of epoch)
        max_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        momentum_anneal: If True, anneal momentum from initial_momentum to 0.99 over training
        initial_momentum: Initial momentum value (will anneal to 0.99 if momentum_anneal=True)
    """
    # Enable TF32 for faster training
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Compile ONLY encoder_q (never encoder_k)
    if compile_student and hasattr(torch, 'compile'):
        print("Compiling encoder_q with torch.compile...")
        print("⚠️  First compilation may take 5-10 minutes - this is normal!")
        model.encoder_q = torch.compile(model.encoder_q, mode='reduce-overhead')
        print("✓ Encoder_q compiled successfully")
        print("  ⚠️  Encoder_k is NOT compiled (momentum encoder, no benefit)")
    
    # Build optimizer: ONLY encoder_q + proj_q parameters (encoder_k + proj_k are frozen)
    params_with_wd = []
    params_without_wd = []
    
    # Only trainable parameters (encoder_q + proj_q)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
    
    # GradScaler for mixed precision
    if GRADSCALER_NEW_API:
        if device.type == 'cuda':
            scaler = GradScaler('cuda')
        else:
            scaler = GradScaler('cpu')
    else:
        scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    global_step = 0
    
    # Resume from checkpoint if provided
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.encoder_q.load_state_dict(checkpoint['encoder_q'])
        model.proj_q.load_state_dict(checkpoint['proj_q'])
        # Load momentum encoder and projection (if available, for backward compatibility)
        if 'encoder_k' in checkpoint:
            model.encoder_k.load_state_dict(checkpoint['encoder_k'])
        else:
            # Backward compatibility: copy from encoder_q if not saved
            print("  ⚠️  Warning: encoder_k not found in checkpoint, copying from encoder_q")
            for param_q, param_k in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
        
        if 'proj_k' in checkpoint:
            model.proj_k.load_state_dict(checkpoint['proj_k'])
        else:
            # Backward compatibility: copy from proj_q if not saved
            print("  ⚠️  Warning: proj_k not found in checkpoint, copying from proj_q")
            for param_q, param_k in zip(model.proj_q.parameters(), model.proj_k.parameters()):
                param_k.data.copy_(param_q.data)
        
        # Load MoCo queue state (if available, for backward compatibility)
        if 'queue' in checkpoint:
            model.queue.copy_(checkpoint['queue'])
        else:
            print("  ⚠️  Warning: queue not found in checkpoint, using random initialization")
        
        if 'queue_ptr' in checkpoint:
            model.queue_ptr.copy_(checkpoint['queue_ptr'])
        else:
            print("  ⚠️  Warning: queue_ptr not found in checkpoint, resetting to 0")
            model.queue_ptr.zero_()
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"✓ Resumed from epoch {start_epoch}, step {global_step}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Momentum annealing: linearly interpolate from initial_momentum to 0.99
        if momentum_anneal:
            # Anneal over the course of training (from epoch 0 to num_epochs-1)
            progress = epoch / max(num_epochs - 1, 1)  # 0.0 to 1.0
            current_momentum = initial_momentum + (0.99 - initial_momentum) * progress
            model.momentum = current_momentum
            if epoch % 10 == 0 or epoch == start_epoch:
                print(f"  Momentum: {current_momentum:.4f} (annealing from {initial_momentum} to 0.99)")
        
        model.train()
        # Set encoder_k to eval mode (it's always in eval mode, but be explicit)
        model.encoder_k.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Augmentation sanity check (once, at the very start of training)
        if epoch == start_epoch:
            print("\n🔍 Augmentation Sanity Check:")
            print("  Verifying that crops are genuinely different views...")
            # Get first batch from a fresh iterator (don't consume progress_bar)
            first_batch = next(iter(train_loader))
            if isinstance(first_batch, (tuple, list)):
                if len(first_batch) == 2:
                    # Standard 2-crop MoCo
                    im_q_check = first_batch[0][:4].to(device)  # First 4 samples
                    im_k_check = first_batch[1][:4].to(device)
                    
                    # Compute pixel-wise difference
                    pixel_diff = (im_q_check - im_k_check).abs().mean().item()
                    pixel_std_q = im_q_check.std().item()
                    pixel_std_k = im_k_check.std().item()
                    
                    print(f"  Pixel difference (|im_q - im_k|): {pixel_diff:.6f}")
                    print(f"  im_q std: {pixel_std_q:.6f}, im_k std: {pixel_std_k:.6f}")
                    
                    if pixel_diff < 0.01:
                        print(f"  ⚠️  CRITICAL: im_q and im_k are nearly identical! Check augmentations!")
                    elif pixel_diff < 0.05:
                        print(f"  ⚠️  WARNING: im_q and im_k are very similar - augmentations may be too weak")
                    else:
                        print(f"  ✓ Augmentations are producing different views")
                elif len(first_batch) == 4:
                    # Multi-crop MoCo (2 global + 2 local)
                    im_global1_check = first_batch[0][:4].to(device)
                    im_global2_check = first_batch[1][:4].to(device)
                    im_local1_check = first_batch[2][:4].to(device)
                    im_local2_check = first_batch[3][:4].to(device)
                    
                    pixel_diff_g = (im_global1_check - im_global2_check).abs().mean().item()
                    pixel_diff_l1 = (im_global1_check - im_local1_check).abs().mean().item()
                    pixel_diff_l2 = (im_global1_check - im_local2_check).abs().mean().item()
                    
                    print(f"  Pixel difference (|global1 - global2|): {pixel_diff_g:.6f}")
                    print(f"  Pixel difference (|global1 - local1|): {pixel_diff_l1:.6f}")
                    print(f"  Pixel difference (|global1 - local2|): {pixel_diff_l2:.6f}")
                    
                    if pixel_diff_g < 0.01 or pixel_diff_l1 < 0.01 or pixel_diff_l2 < 0.01:
                        print(f"  ⚠️  CRITICAL: Some crops are nearly identical! Check augmentations!")
                    else:
                        print(f"  ✓ Multi-crop augmentations are producing different views")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Batch can be either (view1, view2) for standard MoCo or (global1, global2, local1, local2) for multi-crop
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    # Standard 2-crop MoCo
                    im_q = batch[0].to(device, non_blocking=True)
                    im_k = batch[1].to(device, non_blocking=True)
                    im_local1 = None
                    im_local2 = None
                elif len(batch) == 4:
                    # Multi-crop MoCo (2 global + 2 local)
                    im_q = batch[0].to(device, non_blocking=True)  # global1
                    im_k = batch[1].to(device, non_blocking=True)  # global2
                    im_local1 = batch[2].to(device, non_blocking=True)  # local1
                    im_local2 = batch[3].to(device, non_blocking=True)  # local2
                else:
                    raise ValueError(f"Expected batch to be tuple/list of 2 or 4 views, got length {len(batch)}")
            else:
                raise ValueError(f"Expected batch to be tuple/list, got {type(batch)}")
            
            # Convert to channels_last if supported
            try:
                im_q = im_q.to(memory_format=torch.channels_last)
                im_k = im_k.to(memory_format=torch.channels_last)
                if im_local1 is not None:
                    im_local1 = im_local1.to(memory_format=torch.channels_last)
                if im_local2 is not None:
                    im_local2 = im_local2.to(memory_format=torch.channels_last)
            except:
                pass
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if device.type == 'cuda':
                if AUTOCAST_NEW_API:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        loss = model.forward_moco(im_q, im_k, im_local1, im_local2)
                else:
                    with autocast():
                        loss = model.forward_moco(im_q, im_k, im_local1, im_local2)
            else:
                loss = model.forward_moco(im_q, im_k, im_local1, im_local2)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping (if enabled)
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.encoder_q.parameters(), max_norm=max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.proj_q.parameters(), max_norm=max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Periodic diagnostics (every 50 batches or first batch of first 5 epochs)
            if (batch_idx % 50 == 0) or (epoch < 5 and batch_idx == 0):
                with torch.no_grad():
                    # Extract features for diagnostics (recompute to avoid modifying forward_moco)
                    q_features = model._extract_features(model.encoder_q, im_q)
                    q = model.proj_q(q_features)
                    q_norm = F.normalize(q, dim=-1)
                    
                    k_features = model._extract_features(model.encoder_k, im_k)
                    k = model.proj_k(k_features)
                    k_norm = F.normalize(k, dim=-1)
                    
                    # Compute positive similarity (q · k) - should increase during training
                    pos_sim = (q_norm * k_norm).sum(dim=-1).mean().item()
                    
                    # Compute average negative similarity (q · queue) - should stay low
                    batch_size = q_norm.shape[0]  # Get batch size from features
                    if model.use_queue:
                        neg_sim = (q_norm @ model.queue).mean().item()
                    else:
                        # Batch-only mode: use other samples in batch as negatives
                        neg_sim = (q_norm @ k_norm.T).mean().item() - pos_sim / batch_size  # Approximate off-diagonal mean
                    
                    # Compute feature diversity (pairwise similarity of q) - should be moderate (0.1-0.5)
                    q_pairwise_sim = (q_norm @ q_norm.T).mean().item()
                    
                    # Per-dimension variance (should be ~1/D for healthy features)
                    q_var = q_norm.var(dim=0).mean().item()
                    expected_var = 1.0 / q_norm.shape[-1]  # 1/D for unit-norm vectors
                    
                    # Runtime check: positive should be larger than negatives
                    # This verifies the model is learning correctly
                    if model.use_queue:
                        # Recompute logits for verification
                        logits_pos_check = (q_norm * k_norm).sum(dim=-1, keepdim=True)  # [B, 1]
                        logits_neg_check = q_norm @ model.queue  # [B, queue_size]
                        pos_mean = logits_pos_check.mean().item()
                        neg_mean = logits_neg_check.mean().item()
                    else:
                        pos_mean = pos_sim
                        neg_mean = neg_sim
                    
                    # FIXED: Compare directly using difference (ratio breaks when neg_mean < 0)
                    # When neg_mean is negative, that's actually GOOD (negatives are far apart)
                    pos_neg_diff = pos_mean - neg_mean  # Should be positive and large
                    # Use absolute value for ratio to handle negative neg_mean correctly
                    pos_neg_ratio = pos_mean / (abs(neg_mean) + 1e-8) if neg_mean != 0 else float('inf')
                    
                    print(f"\n🔍 MoCo Diagnostics (epoch {epoch+1}, batch {batch_idx}):")
                    print(f"  Positive similarity (q·k): {pos_sim:.4f} (should increase, target >0.7)")
                    print(f"  Negative similarity (q·{'queue' if model.use_queue else 'batch'}): {neg_sim:.4f} (should stay low/negative, target <0.1)")
                    print(f"  Pos - Neg difference: {pos_neg_diff:.4f} (should be >0.5, ideally >0.8)")
                    print(f"  Pos/|Neg| ratio: {pos_neg_ratio:.4f} (should be >1.0, ideally >2.0)")
                    print(f"  Query diversity (q pairwise sim): {q_pairwise_sim:.4f} (should be 0.1-0.5)")
                    print(f"  Query per-dim variance: {q_var:.6f} (expected ~{expected_var:.6f})")
                    
                    # Warnings - FIXED: Check difference instead of ratio
                    if pos_neg_diff < 0:
                        print(f"  ⚠️  CRITICAL: Positive < Negative! Model learning backwards - check labels/augmentations!")
                    elif pos_neg_diff < 0.5:
                        print(f"  ⚠️  WARNING: Pos-Neg difference is low - model may not be learning contrastive signal well")
                    elif pos_neg_ratio < 1.5:
                        print(f"  ⚠️  WARNING: Pos/|Neg| ratio is low - model may not be learning contrastive signal well")
                    
                    if q_pairwise_sim > 0.9:
                        print(f"  ⚠️  WARNING: Features are collapsing! (pairwise sim >0.9)")
                    elif q_pairwise_sim < 0.05:
                        print(f"  ⚠️  WARNING: Features are over-dispersed! (pairwise sim <0.05)")
                    else:
                        print(f"  ✓ Features have good diversity")
                    
                    if q_var < expected_var * 0.1:
                        print(f"  ⚠️  WARNING: Per-dim variance is very low (collapse risk)!")
                    
                    # Enable runtime pos/neg check for future batches
                    if not hasattr(model, '_check_pos_neg_ratio'):
                        model._check_pos_neg_ratio = True
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # Step-based checkpointing
            if save_every > 0 and global_step % save_every == 0 and checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = {
                    'encoder_q': model.encoder_q.state_dict(),
                    'proj_q': model.proj_q.state_dict(),
                    'encoder_k': model.encoder_k.state_dict(),  # Save momentum encoder
                    'proj_k': model.proj_k.state_dict(),         # Save momentum projection
                    'queue': model.queue,                        # Save MoCo queue buffer
                    'queue_ptr': model.queue_ptr,                # Save queue pointer
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                }
                torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_step_{global_step}.pth")
                torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
        
        # Step scheduler at end of epoch
        scheduler.step()
        
        # Print epoch summary
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")
        
        # Save checkpoint at end of epoch
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'encoder_q': model.encoder_q.state_dict(),
                'proj_q': model.proj_q.state_dict(),
                'encoder_k': model.encoder_k.state_dict(),  # Save momentum encoder
                'proj_k': model.proj_k.state_dict(),         # Save momentum projection
                'queue': model.queue,                        # Save MoCo queue buffer
                'queue_ptr': model.queue_ptr,                 # Save queue pointer
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
            print(f"  ✓ Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    return model


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
    
    # Create distillation loss module with learnable projections
    # Teacher (DINOv2 ViT-B): 768 dims, Student (ViT-S): 384 dims
    distillation_loss_module = DistillationLoss(
        teacher_dim=768,
        student_dim=384,
        loss_weights=loss_weights
    ).to(device)
    print("✓ Created DistillationLoss module with learnable projections (768→384)")
    
    # FIX: Freeze projection layers to prevent collapse
    # The projection layers should be fixed, not learnable
    for name, param in distillation_loss_module.named_parameters():
        param.requires_grad = False
        print(f"  ✓ Frozen projection layer: {name}")
    
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
    
    # Build optimizer: student parameters + projection head (if SSL enabled)
    # Group parameters: weight decay for non-bias/norm, no weight decay for bias/norm
    params_with_wd = []
    params_without_wd = []
    for name, param in student.named_parameters():
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
    
    # DO NOT add distillation_loss_module parameters (they're frozen)
    
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
    
    # Get training mode (default to "kd" for backward compatibility)
    training_mode = train_cfg.get('training_mode', 'kd')
    print(f"✓ Training mode: {training_mode}")
    
    # Override max_steps_per_epoch from CLI if provided
    if args.max_steps_per_epoch is not None:
        train_cfg['max_steps_per_epoch'] = args.max_steps_per_epoch
        print(f"✓ Overrode max_steps_per_epoch to {args.max_steps_per_epoch}")
    
    # Handle precompute-only mode (not used for CIFAR10, but kept for compatibility)
    if args.precompute_cache_only:
        print("⚠️  Precompute cache mode not supported for CIFAR10 experiment")
        print("   CIFAR10 uses direct dataset loading, not cached tensors")
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
    
    # Branch based on training mode
    if training_mode == "moco_v3":
        run_moco_training(data_cfg, train_cfg, model_cfg, device, args)
    elif training_mode == "kd":
        run_kd_training(data_cfg, train_cfg, model_cfg, device, args)
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}. Use 'moco_v3' or 'kd'")
    
    print("✓ Training completed!")


def run_moco_training(data_cfg, train_cfg, model_cfg, device, args):
    """Run MoCo-v3 contrastive learning training"""
    print("\n" + "="*60)
    print("MoCo-v3 Contrastive Learning Training")
    print("="*60)
    
    # Get MoCo config
    moco_cfg = train_cfg.get('moco', {})
    proj_dim = moco_cfg.get('proj_dim', 256)
    proj_hidden_dim = moco_cfg.get('proj_hidden_dim', 0)  # 0 = use embed_dim
    queue_size = moco_cfg.get('queue_size', 65536)
    momentum = moco_cfg.get('momentum', 0.99)
    momentum_anneal = moco_cfg.get('momentum_anneal', False)
    temperature = moco_cfg.get('temperature', 0.2)
    use_queue = moco_cfg.get('use_queue', True)
    
    print(f"MoCo-v3 Configuration:")
    print(f"  Projection dimension: {proj_dim}")
    if proj_hidden_dim > 0:
        print(f"  Projection hidden dimension: {proj_hidden_dim}")
    print(f"  Queue size: {queue_size:,}")
    print(f"  Momentum: {momentum}" + (f" (will anneal to 0.99)" if momentum_anneal else ""))
    print(f"  Temperature: {temperature}")
    print(f"  Use queue: {use_queue}" + (" (batch-only mode)" if not use_queue else ""))
    
    # Build model
    # Support both 'student_name' (legacy) and 'backbone_name' (new)
    backbone_name = model_cfg.get('backbone_name', model_cfg.get('student_name', 'vit_small_patch16_224'))
    image_size = model_cfg.get('image_size', data_cfg.get('image_size', 96))
    
    print(f"\nBuilding MoCo-v3 model...")
    print(f"  Backbone: {backbone_name}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Get backbone type from model config (for explicit control)
    backbone_type = model_cfg.get('backbone_type', 'auto')
    
    # Warn if ResNet is being used with non-standard image size
    if backbone_type == "resnet" or ("resnet" in backbone_name.lower()):
        if image_size != 224:
            print(f"  ⚠️  WARNING: ResNet models are typically trained on 224x224 images.")
            print(f"     Using {image_size}x{image_size} will work (ResNet is fully convolutional),")
            print(f"     but performance may be suboptimal. Consider using image_size: 224.")
    
    model = MoCoModel(
        backbone_name=backbone_name,
        image_size=image_size,
        proj_dim=proj_dim,
        proj_hidden_dim=proj_hidden_dim,
        queue_size=queue_size,
        momentum=momentum,
        temperature=temperature,
        use_queue=use_queue,
        backbone_type=backbone_type,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"  (encoder_k + proj_k are frozen, updated via momentum)")
    
    # Warn if trainable parameters exceed 100M
    if trainable_params > 100e6:
        print(f"  ⚠️  WARNING: Trainable parameters ({trainable_params / 1e6:.2f}M) exceed 100M limit!")
        print(f"     Consider using a smaller model or reducing projection head size.")
    
    # Build DataLoader (should return two views per image)
    print("\nBuilding DataLoader...")
    dataloader = build_pretraining_dataloader(data_cfg, train_cfg)
    
    # Checkpoint directory - automatically append model name
    base_checkpoint_dir = train_cfg.get('checkpoint_dir', './checkpoints')
    base_checkpoint_dir = os.path.expandvars(base_checkpoint_dir)
    
    # Extract model name from backbone_name (e.g., "vit_small_patch16_224" -> "vit_small")
    # Handle various naming patterns: "vit_small_patch16_224", "resnet50", etc.
    if '_' in backbone_name:
        parts = backbone_name.split('_')
        # For ViT models: take first two parts (e.g., "vit_small")
        # For other models: take first part (e.g., "resnet")
        if 'vit' in backbone_name.lower():
            model_name = '_'.join(parts[:2]) if len(parts) >= 2 else parts[0]
        else:
            model_name = parts[0]
    else:
        model_name = backbone_name
    
    # Clean up model name for directory (remove special chars)
    model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Create model-specific checkpoint directory
    checkpoint_dir = os.path.join(base_checkpoint_dir, model_name)
    print(f"\nCheckpoint directory: {checkpoint_dir}")
    
    save_every = train_cfg.get('save_every', 0)  # 0 = only at end of epoch
    
    # Auto-detect latest checkpoint if resume_from not provided and auto-resume enabled
    resume_from = args.resume_from if args.resume_from else None
    if resume_from is None and not args.no_resume:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_from = latest_checkpoint
            print(f"✓ Auto-detected latest checkpoint: {resume_from}")
            print(f"  To disable auto-resume, use --no_resume flag")
    
    # Start training
    train_moco(
        model=model,
        train_loader=dataloader,
        num_epochs=train_cfg['num_epochs'],
        device=device,
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        warmup_epochs=train_cfg['warmup_epochs'],
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        compile_student=train_cfg.get('compile_student', True),
        use_fused_adamw=train_cfg.get('use_fused_adamw', True),
        save_every=save_every,
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        momentum_anneal=momentum_anneal,
        initial_momentum=momentum,
    )


def run_kd_training(data_cfg, train_cfg, model_cfg, device, args):
    """Run knowledge distillation training (original KD pipeline)"""
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
    
    # Get SSL config from train_config
    ssl_config = train_cfg.get('ssl', {})
    if ssl_config.get('enabled', False):
        print(f"✓ SSL enabled: {ssl_config.get('type', 'barlow')}")
        print(f"  SSL weight: {ssl_config.get('weight', 0.5)}")
        print(f"  Barlow lambda: {ssl_config.get('barlow_lambd', 5e-3)}")
    
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

