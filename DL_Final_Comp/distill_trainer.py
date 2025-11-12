"""
Knowledge Distillation Trainer for Self-Supervised Learning

Trains a lightweight ViT student to mimic a frozen DINOv2 teacher
using unlabeled data only.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
from pathlib import Path

import timm
from data_loader import PretrainDataset
from transforms import MultiCropTransform, EvalTransform
from optimizer import build_optimizer, build_scheduler


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_teacher_model(teacher_name="dinov2_vitb14", device="cuda"):
    """
    Load pretrained DINOv2 teacher model from torch.hub
    
    Args:
        teacher_name: Model name (e.g., "dinov2_vitb14")
        device: Device to load model on
    
    Returns:
        Frozen teacher model
    """
    print(f"Loading teacher model: {teacher_name}")
    try:
        # Load DINOv2 from torch.hub
        teacher = torch.hub.load("facebookresearch/dinov2", teacher_name)
        teacher = teacher.to(device)
        teacher.eval()
        
        # Freeze all parameters
        for param in teacher.parameters():
            param.requires_grad = False
        
        print(f"✓ Teacher loaded: {teacher_name}")
        print(f"  Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M")
        print(f"  Frozen: True")
        
        return teacher
    except Exception as e:
        raise RuntimeError(f"Failed to load teacher model: {e}")


def build_student_model(model_name="vit_small_patch16_224", 
                       img_size=224, 
                       device="cuda"):
    """
    Build student ViT model from timm (random initialization)
    
    Args:
        model_name: timm model name (e.g., "vit_small_patch16_224")
        img_size: Input image size
        device: Device to load model on
    
    Returns:
        Student model (trainable)
    """
    print(f"Building student model: {model_name}")
    student = timm.create_model(
        model_name,
        pretrained=False,  # Random initialization
        img_size=img_size,
        num_classes=0,  # No classification head
    )
    student = student.to(device)
    student.train()
    
    num_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    
    print(f"✓ Student created: {model_name}")
    print(f"  Parameters: {num_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    
    return student


def extract_teacher_features(teacher, images, use_cls_token=True):
    """
    Extract features from frozen teacher model
    
    Args:
        teacher: DINOv2 teacher model
        images: Input images [B, 3, H, W]
        use_cls_token: Whether to use CLS token or mean-pool patches
    
    Returns:
        cls_embedding: [B, D] CLS token embedding
        patch_embeddings: [B, N, D] patch token embeddings
    """
    with torch.no_grad():
        # DINOv2 forward_features may return dict or tensor
        features = teacher.forward_features(images)
        
        # Handle DINOv2 output format (dict with keys)
        if isinstance(features, dict):
            # DINOv2 returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
            if 'x_norm_clstoken' in features:
                cls_embedding = features['x_norm_clstoken']  # [B, D]
            elif 'cls_token' in features:
                cls_embedding = features['cls_token']
            else:
                # Fallback: use first token if available
                cls_embedding = features.get('x', features.get('tokens', None))
                if cls_embedding is not None:
                    cls_embedding = cls_embedding[:, 0]
                else:
                    raise ValueError("Could not extract CLS token from DINOv2 output")
            
            if 'x_norm_patchtokens' in features:
                patch_embeddings = features['x_norm_patchtokens']  # [B, N, D]
            elif 'patch_tokens' in features:
                patch_embeddings = features['patch_tokens']
            else:
                # Fallback: use remaining tokens
                patch_embeddings = features.get('x', features.get('tokens', None))
                if patch_embeddings is not None:
                    patch_embeddings = patch_embeddings[:, 1:]
                else:
                    raise ValueError("Could not extract patch tokens from DINOv2 output")
        else:
            # Assume tensor format [B, N+1, D] (CLS + patches)
            if use_cls_token:
                cls_embedding = features[:, 0]  # CLS token [B, D]
            else:
                cls_embedding = features[:, 1:].mean(dim=1)  # Mean-pool patches [B, D]
            
            patch_embeddings = features[:, 1:]  # Patch tokens [B, N, D]
        
        # Normalize embeddings (DINOv2 may already normalize, but ensure it)
        cls_embedding = F.normalize(cls_embedding, dim=-1, p=2)
        patch_embeddings = F.normalize(patch_embeddings, dim=-1, p=2)
    
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


def compute_distillation_loss(student_cls, student_patches, 
                             teacher_cls, teacher_patches,
                             loss_weights=None):
    """
    Compute distillation loss between student and teacher embeddings
    
    Args:
        student_cls: Student CLS embedding [B, D_s]
        student_patches: Student patch embeddings [B, N_s, D_s]
        teacher_cls: Teacher CLS embedding [B, D_t]
        teacher_patches: Teacher patch embeddings [B, N_t, D_t]
        loss_weights: Dict with 'cls' and 'patch' weights
    
    Returns:
        Total loss and component losses
    """
    if loss_weights is None:
        loss_weights = {'cls': 1.0, 'patch': 0.5}
    
    # CLS token loss: MSE between normalized embeddings
    # Note: Teacher and student may have different embedding dimensions
    # We use a projection if needed, but for simplicity, we'll use cosine similarity
    # or MSE if dimensions match
    
    # For CLS: Use MSE if dims match, else use cosine distance
    if student_cls.shape[-1] == teacher_cls.shape[-1]:
        loss_cls = F.mse_loss(student_cls, teacher_cls)
    else:
        # Different dimensions: use cosine similarity loss
        # Cosine similarity = dot product of normalized vectors
        # Loss = 1 - cosine_similarity
        cosine_sim = F.cosine_similarity(student_cls, teacher_cls, dim=-1)
        loss_cls = (1 - cosine_sim).mean()
    
    # Patch embeddings loss
    # Handle different number of patches or dimensions
    B_s, N_s, D_s = student_patches.shape
    B_t, N_t, D_t = teacher_patches.shape
    
    if N_s == N_t and D_s == D_t:
        # Same shape: direct MSE
        loss_patch = F.mse_loss(student_patches, teacher_patches)
    elif D_s == D_t:
        # Same embedding dim, different num patches: interpolate or crop
        if N_s < N_t:
            # Student has fewer patches: crop teacher
            teacher_patches = teacher_patches[:, :N_s, :]
            loss_patch = F.mse_loss(student_patches, teacher_patches)
        else:
            # Student has more patches: crop student
            student_patches = student_patches[:, :N_t, :]
            loss_patch = F.mse_loss(student_patches, teacher_patches)
    else:
        # Different dimensions: use mean-pooled cosine similarity
        student_pooled = student_patches.mean(dim=1)  # [B, D_s]
        teacher_pooled = teacher_patches.mean(dim=1)  # [B, D_t]
        if D_s == D_t:
            loss_patch = F.mse_loss(student_pooled, teacher_pooled)
        else:
            cosine_sim = F.cosine_similarity(student_pooled, teacher_pooled, dim=-1)
            loss_patch = (1 - cosine_sim).mean()
    
    # Weighted combination
    total_loss = loss_weights['cls'] * loss_cls + loss_weights['patch'] * loss_patch
    
    return total_loss, {
        'total': total_loss.item(),
        'cls': loss_cls.item(),
        'patch': loss_patch.item()
    }


def train_epoch(teacher, student, dataloader, optimizer, scheduler, 
                device, scaler, epoch, num_epochs, loss_weights, 
                use_cls_token=True, use_multi_crop=True):
    """
    Train one epoch of knowledge distillation
    
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
    
    Returns:
        Average loss and metrics
    """
    student.train()
    total_loss = 0
    total_metrics = {'cls': 0, 'patch': 0}
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    batch_times = []
    data_times = []
    prev_iter_time = time.time()
    
    for batch_idx, batch in enumerate(progress_bar):
        iter_start = time.time()
        
        if batch_idx > 0:
            data_load_time = iter_start - prev_iter_time
        else:
            data_load_time = 0
        
        batch_start = time.time()
        
        # Handle multi-crop or single image
        if use_multi_crop and isinstance(batch, list):
            # Multi-crop: use first global crop for distillation
            # (can extend to use multiple crops, but for simplicity use one)
            images = batch[0].to(device)  # First global crop
        else:
            images = batch.to(device)
        
        # Convert to channels_last if supported
        try:
            images = images.to(memory_format=torch.channels_last)
        except:
            pass
        
        optimizer.zero_grad()
        
        # Mixed precision training
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
        
        with autocast(device_type=device_type, dtype=dtype):
            # Teacher forward (frozen)
            teacher_cls, teacher_patches = extract_teacher_features(
                teacher, images, use_cls_token=use_cls_token
            )
            
            # Student forward
            student_cls, student_patches = extract_student_features(
                student, images, use_cls_token=use_cls_token
            )
            
            # Compute distillation loss
            loss, metrics = compute_distillation_loss(
                student_cls, student_patches,
                teacher_cls, teacher_patches,
                loss_weights=loss_weights
            )
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_metrics['cls'] += metrics['cls']
        total_metrics['patch'] += metrics['patch']
        
        # Track times
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        data_times.append(data_load_time)
        if len(batch_times) > 10:
            batch_times.pop(0)
            data_times.pop(0)
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_data_time = sum(data_times) / len(data_times) if data_times else 0
        
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'cls': f'{metrics["cls"]:.4f}',
            'patch': f'{metrics["patch"]:.4f}',
            'lr': f'{current_lr:.6f}',
            'gpu': f'{avg_batch_time:.2f}s',
            'data': f'{avg_data_time:.2f}s'
        })
        
        prev_iter_time = time.time()
    
    # Step scheduler at end of epoch
    scheduler.step()
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    return avg_loss, avg_metrics


def train_distillation(teacher, student, train_loader, num_epochs, device,
                      lr=5e-4, weight_decay=0.04, warmup_epochs=10,
                      loss_weights=None, checkpoint_dir=None, resume_from=None,
                      use_torch_compile=True, use_fused_adamw=True,
                      use_cls_token=True, use_multi_crop=True, save_freq=10):
    """
    Main training function for knowledge distillation
    
    Args:
        teacher: Frozen teacher model
        student: Trainable student model
        train_loader: Training data loader
        num_epochs: Number of epochs
        device: Device
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs
        loss_weights: Loss weights dict
        checkpoint_dir: Checkpoint directory
        resume_from: Path to checkpoint to resume from
        use_torch_compile: Whether to compile model
        use_fused_adamw: Whether to use fused AdamW
        use_cls_token: Whether to use CLS token
        use_multi_crop: Whether to use multi-crop
        save_freq: Save checkpoint every N epochs
    """
    if loss_weights is None:
        loss_weights = {'cls': 1.0, 'patch': 0.5}
    
    # Enable TF32 for faster training
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Compile student model for faster execution
    compiled_student = None
    if use_torch_compile and hasattr(torch, 'compile'):
        print("Compiling student model with torch.compile...")
        print("⚠️  First compilation may take 5-10 minutes - this is normal!")
        compiled_student = torch.compile(student, mode='reduce-overhead')
        print("✓ Student model compiled successfully")
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(student, lr=lr, weight_decay=weight_decay,
                               fused=use_fused_adamw)
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs,
                               warmup_epochs=warmup_epochs)
    
    # GradScaler for mixed precision
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        student.load_state_dict(checkpoint['student'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Use compiled model if available
    if compiled_student is not None:
        student = compiled_student
    
    # Initialize scheduler
    if start_epoch == 0:
        scheduler.step()
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        avg_loss, avg_metrics = train_epoch(
            teacher, student, train_loader, optimizer, scheduler,
            device, scaler, epoch, num_epochs, loss_weights,
            use_cls_token=use_cls_token, use_multi_crop=use_multi_crop
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} "
              f"(CLS: {avg_metrics['cls']:.4f}, Patch: {avg_metrics['patch']:.4f})")
        
        # Save checkpoint
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'student': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
            }
            # Always save latest
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
            
            # Save numbered checkpoint
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                torch.save(checkpoint, 
                          f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
                print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    return student


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training")
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data config YAML')
    parser.add_argument('--train_config', type=str, required=True,
                       help='Path to training config YAML')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, default='',
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)
    model_cfg = load_config(args.model_config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: Running on CPU - training will be very slow!")
    
    # Load teacher model
    teacher_name = model_cfg.get('teacher_name', 'dinov2_vitb14')
    teacher = load_teacher_model(teacher_name, device=device)
    
    # Build student model
    student_name = model_cfg.get('student_name', 'vit_small_patch16_224')
    student_img_size = model_cfg.get('student_img_size', 224)
    student = build_student_model(student_name, student_img_size, device=device)
    
    # Dataset and DataLoader
    # For KD, we can use simpler augmentation (single crop) or multi-crop
    # Note: Both teacher (DINOv2) and student should use same image size (224)
    use_multi_crop = train_cfg.get('use_multi_crop', False)
    if use_multi_crop:
        transform = MultiCropTransform(
            global_crops_scale=tuple(data_cfg.get('global_crops_scale', [0.4, 1.0])),
            local_crops_scale=tuple(data_cfg.get('local_crops_scale', [0.05, 0.4])),
            local_crops_number=data_cfg.get('local_crops_number', 8),
            image_size=student_img_size  # Use student image size (224 for DINOv2 compatibility)
        )
    else:
        # Simple augmentation for KD
        # Resize to student_img_size (224) to match DINOv2 teacher input size
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        transform = transforms.Compose([
            transforms.RandomResizedCrop(student_img_size, scale=(0.2, 1.0),
                                       interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    dataset = PretrainDataset(transform=transform)
    num_workers = data_cfg.get('num_workers', 4)
    persistent_workers = data_cfg.get('persistent_workers', False) and num_workers > 0
    prefetch_factor = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    
    print(f"DataLoader settings: {num_workers} workers, persistent={persistent_workers}, prefetch={prefetch_factor}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=data_cfg.get('pin_memory', True),
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Test data loading
    print("Testing data loading...")
    try:
        test_batch = next(iter(dataloader))
        if isinstance(test_batch, list):
            print(f"✓ Data loading works! Batch has {len(test_batch)} crops")
        else:
            print(f"✓ Data loading works! Batch shape: {test_batch.shape}")
    except Exception as e:
        print(f"⚠️  Data loading test failed: {e}")
        print("   Try reducing num_workers or setting persistent_workers=false")
    
    # Loss weights
    loss_weights = {
        'cls': train_cfg.get('distill_loss_weights', {}).get('cls', 1.0),
        'patch': train_cfg.get('distill_loss_weights', {}).get('patch', 0.5)
    }
    
    # Training
    use_cls_token = model_cfg.get('use_cls_token', True)
    
    # Expand checkpoint directory path (handle $USER variable)
    checkpoint_dir = train_cfg.get('checkpoint_dir', './checkpoints')
    checkpoint_dir = os.path.expandvars(checkpoint_dir)
    
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
        resume_from=args.resume_from if args.resume_from else None,
        use_torch_compile=train_cfg.get('use_torch_compile', True),
        use_fused_adamw=train_cfg.get('use_fused_adamw', True),
        use_cls_token=use_cls_token,
        use_multi_crop=use_multi_crop,
        save_freq=train_cfg.get('save_freq', 10)
    )
    
    print("✓ Training completed!")


if __name__ == '__main__':
    main()

