"""
Evaluate multiple checkpoints on CIFAR10/100 using linear probing.

This script:
1. Loads checkpoints from different epochs (1, 100, 200, 250, best)
2. Extracts features from CIFAR10/100 datasets
3. Trains linear probes for each checkpoint
4. Compares and reports performance
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import timm


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_backbone_type_from_state_dict(state_dict):
    """
    Detect backbone type (ViT or ResNet) from state dict keys.
    
    Returns:
        'vit' if ViT architecture detected, 'resnet' if ResNet detected, None if unclear
    """
    keys = list(state_dict.keys())
    
    # ViT indicators
    vit_indicators = ['cls_token', 'pos_embed', 'patch_embed', 'blocks.0.norm1', 'blocks.0.attn']
    # ResNet indicators
    resnet_indicators = ['conv1.weight', 'bn1.weight', 'layer1.0.conv1', 'layer2.0.conv1']
    
    vit_count = sum(1 for key in keys if any(indicator in key for indicator in vit_indicators))
    resnet_count = sum(1 for key in keys if any(indicator in key for indicator in resnet_indicators))
    
    if vit_count > resnet_count and vit_count > 0:
        return 'vit'
    elif resnet_count > vit_count and resnet_count > 0:
        return 'resnet'
    else:
        return None


def load_checkpoint(checkpoint_path, device, mode='kd'):
    """
    Load checkpoint and return model state dict.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        mode: 'kd' (knowledge distillation) or 'moco_v3' (contrastive learning)
    
    Returns:
        Tuple of (state_dict, detected_backbone_type)
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except RuntimeError as e:
        if "failed reading zip archive" in str(e) or "central directory" in str(e):
            raise RuntimeError(f"Checkpoint file is corrupted or incomplete: {checkpoint_path}")
        else:
            raise
    
    # Handle different checkpoint formats based on mode
    if mode == 'moco_v3':
        # MoCo checkpoints contain 'encoder_q' and 'proj_q'
        if 'encoder_q' in checkpoint:
            state_dict = checkpoint['encoder_q']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
    else:
        # KD mode: original logic
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint
    
    # Handle torch.compile() prefix: compiled models have '_orig_mod.' prefix
    # Strip this prefix if present to match uncompiled model keys
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print(f"  Detected compiled model checkpoint (has '_orig_mod.' prefix), stripping prefix...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    # Detect backbone type from state dict
    detected_backbone = detect_backbone_type_from_state_dict(state_dict)
    
    return state_dict, detected_backbone


def build_student_model(model_config, device):
    """Build student model architecture"""
    # Support both 'student_name' (legacy) and 'backbone_name' (new)
    model_name = model_config.get('backbone_name', model_config.get('student_name', 'vit_small_patch16_224'))
    img_size = model_config.get('image_size', model_config.get('student_img_size', 224))
    
    # Get backbone type from config (for explicit control)
    backbone_type = model_config.get('backbone_type', 'auto')
    
    # Auto-detect if not explicitly set
    if backbone_type == 'auto':
        if 'resnet' in model_name.lower():
            backbone_type = 'resnet'
        elif 'vit' in model_name.lower():
            backbone_type = 'vit'
        else:
            backbone_type = 'vit'  # Default to ViT
    
    print(f"Building student model: {model_name} with img_size={img_size} (backbone_type={backbone_type})")
    
    # ResNet models don't accept img_size parameter
    if backbone_type == 'resnet':
        student = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # No classification head
            global_pool="avg",  # Global average pooling for ResNet
        )
    else:
        # ViT models: Try to create model with custom img_size
        try:
            student = timm.create_model(
                model_name,
                pretrained=False,
                img_size=img_size,
                num_classes=0,  # No classification head
                global_pool="",  # Don't pool, return all tokens for ViT
            )
        except Exception as e:
            print(f"  Warning: Could not create model with img_size={img_size}, trying default size first...")
            student = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
    
    # Always patch patch_embed to ensure it matches desired img_size
    if hasattr(student, 'patch_embed'):
        if hasattr(student.patch_embed, 'patch_size'):
            patch_size = student.patch_embed.patch_size
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
        else:
            patch_size = 16
        
        current_img_size = None
        if hasattr(student.patch_embed, 'img_size'):
            current_img_size = student.patch_embed.img_size
            if isinstance(current_img_size, (list, tuple)):
                current_img_size = current_img_size[0]
        
        grid_size = img_size // patch_size
        
        if current_img_size != img_size:
            print(f"  Patching patch_embed from {current_img_size}x{current_img_size} to {img_size}x{img_size}")
            if hasattr(student.patch_embed, 'img_size'):
                student.patch_embed.img_size = (img_size, img_size)
            if hasattr(student.patch_embed, 'grid_size'):
                student.patch_embed.grid_size = (grid_size, grid_size)
            if hasattr(student.patch_embed, 'num_patches'):
                student.patch_embed.num_patches = grid_size * grid_size
    
    student = student.to(device)
    student.eval()
    return student


def get_cifar_dataset(dataset_name='cifar10', train=True, image_size=96):
    """
    Get CIFAR10 or CIFAR100 dataset with proper transforms.
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        train: Whether to get training or test set
        image_size: Target image size (will resize from 32x32)
    
    Returns:
        Dataset
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == 'cifar100':
        dataset = datasets.CIFAR100(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def extract_features(model, dataloader, device, use_cls_token=True):
    """
    Extract features from frozen encoder model.
    
    Args:
        model: Frozen encoder model
        dataloader: DataLoader for dataset
        device: Device to run on
        use_cls_token: Whether to use CLS token or mean-pool patches
    
    Returns:
        features: [N, D] tensor of features
        labels: [N] tensor of labels
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Forward through frozen backbone
            outputs = model.forward_features(images)
            
            # Handle DINOv2 dict output format (teacher model) or tensor format (student model)
            if isinstance(outputs, dict):
                # DINOv2 format: extract from dict
                if 'x_norm_clstoken' in outputs:
                    cls_embedding = outputs['x_norm_clstoken']  # [B, D] - already normalized
                    if use_cls_token:
                        feat = cls_embedding
                    else:
                        # Use normalized patch tokens
                        if 'x_norm_patchtokens' in outputs:
                            feat = outputs['x_norm_patchtokens'].mean(dim=1)  # [B, D]
                        else:
                            feat = cls_embedding  # Fallback to CLS
                elif 'cls_token' in outputs:
                    cls_embedding = outputs['cls_token']
                    if use_cls_token:
                        feat = cls_embedding
                    else:
                        if 'patch_tokens' in outputs:
                            feat = outputs['patch_tokens'].mean(dim=1)
                        else:
                            feat = cls_embedding
                else:
                    # Fallback: try to get from 'x' or 'tokens'
                    tokens = outputs.get('x', outputs.get('tokens', None))
                    if tokens is not None:
                        if use_cls_token:
                            feat = tokens[:, 0]  # CLS token
                        else:
                            feat = tokens[:, 1:].mean(dim=1)  # Mean-pool patches
                    else:
                        raise ValueError("Could not extract features from DINOv2 dict output")
                # DINOv2 features are already normalized, but normalize again to be safe
                feat = F.normalize(feat, dim=-1, p=2)
            else:
                # Tensor format (timm models)
                # For ViT: [B, N+1, D] (CLS + patches)
                # For ResNet: [B, D] (already pooled)
                if len(outputs.shape) == 3:
                    # ViT format: [B, N+1, D]
                    if use_cls_token:
                        feat = outputs[:, 0]  # CLS token [B, D]
                    else:
                        feat = outputs[:, 1:].mean(dim=1)  # Mean-pool patches [B, D]
                elif len(outputs.shape) == 2:
                    # ResNet format: [B, D] (already pooled)
                    feat = outputs
                else:
                    # Handle spatial dimensions (ResNet before pooling)
                    if len(outputs.shape) == 4:
                        feat = F.adaptive_avg_pool2d(outputs, (1, 1)).flatten(1)
                    else:
                        raise ValueError(f"Unexpected feature shape: {outputs.shape}")
                
                # Normalize features
                feat = F.normalize(feat, dim=-1, p=2)
            
            features.append(feat.cpu())
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def analyze_features(features, name="Features"):
    """
    Analyze feature quality and diversity.
    
    Args:
        features: [N, D] tensor of features
        name: Name for the feature set (for printing)
    
    Returns:
        Dict with statistics
    """
    features = features.float()  # Ensure float type
    
    stats = {
        'mean': features.mean().item(),
        'std': features.std().item(),
        'min': features.min().item(),
        'max': features.max().item(),
        'norm': features.norm(dim=-1).mean().item(),
    }
    
    # Per-dimension variance
    feature_var = features.var(dim=0)
    stats['var_mean'] = feature_var.mean().item()
    stats['var_min'] = feature_var.min().item()
    stats['var_max'] = feature_var.max().item()
    
    # Pairwise similarity (sample-based for efficiency)
    sample_size = min(1000, len(features))
    sample_features = features[:sample_size]
    pairwise_sim = (sample_features @ sample_features.T).mean().item()
    stats['pairwise_sim'] = pairwise_sim
    stats['sample_size'] = sample_size
    
    # Print diagnostics
    print(f"  🔍 {name} Quality Analysis:")
    print(f"    Shape: {features.shape}")
    print(f"    Mean: {stats['mean']:.6f}")
    print(f"    Std: {stats['std']:.6f}")
    print(f"    Min: {stats['min']:.6f}")
    print(f"    Max: {stats['max']:.6f}")
    print(f"    Feature norm (should be ~1.0): {stats['norm']:.6f}")
    print(f"    Per-dim variance: mean={stats['var_mean']:.6f}, min={stats['var_min']:.6f}, max={stats['var_max']:.6f}")
    print(f"    Avg pairwise similarity (first {sample_size}): {stats['pairwise_sim']:.6f}")
    
    # Warnings
    if stats['pairwise_sim'] > 0.9:
        print(f"    ⚠️  WARNING: Features are highly similar (COLLAPSED)!")
        print(f"    This explains why accuracy is ~10% (random chance)")
    elif stats['pairwise_sim'] > 0.7:
        print(f"    ⚠️  WARNING: Features are too similar (partial collapse)")
    else:
        print(f"    ✓ Features have good diversity")
    
    if stats['var_mean'] < 0.01:
        print(f"    ⚠️  WARNING: Very low variance across dimensions (COLLAPSED)!")
    elif stats['var_mean'] < 0.05:
        print(f"    ⚠️  WARNING: Low variance across dimensions (partial collapse)")
    
    if abs(stats['norm'] - 1.0) > 0.1:
        print(f"    ⚠️  WARNING: Feature norm is not ~1.0 (normalization issue?)")
    
    return stats


def train_linear_probe(train_features, train_labels, test_features, test_labels, C=1.0, max_iter=1000):
    """
    Train linear probe (logistic regression) on features.
    
    Args:
        train_features: [N_train, D] training features
        train_labels: [N_train] training labels
        test_features: [N_test, D] test features
        test_labels: [N_test] test labels
        C: Regularization strength
        max_iter: Maximum iterations
    
    Returns:
        accuracy: Test accuracy
        clf: Trained classifier
    """
    # Convert to numpy
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.numpy()
    if isinstance(test_features, torch.Tensor):
        test_features = test_features.numpy()
    if isinstance(test_labels, torch.Tensor):
        test_labels = test_labels.numpy()
    
    # Train logistic regression
    # Note: multi_class='multinomial' is deprecated in sklearn 1.5+, removed in 1.7+
    # It's now the default behavior, so we omit it
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver='lbfgs',
        n_jobs=-1
    )
    clf.fit(train_features, train_labels)
    
    # Evaluate
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy, clf


def find_checkpoint_files(checkpoint_dir, epochs=None, include_best=True):
    """
    Find checkpoint files for specified epochs.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        epochs: List of epoch numbers to find (e.g., [1, 100, 200, 250])
        include_best: Whether to include 'best' checkpoint (checkpoint_latest.pth)
    
    Returns:
        Dict mapping epoch number to checkpoint path
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}
    
    # Find best checkpoint (checkpoint_latest.pth)
    if include_best:
        latest_path = checkpoint_dir / 'checkpoint_latest.pth'
        if latest_path.exists():
            checkpoints['best'] = latest_path
        else:
            print(f"⚠️  Warning: Best checkpoint (checkpoint_latest.pth) not found")
    
    # Find epoch-specific checkpoints
    if epochs:
        for epoch in epochs:
            epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            if epoch_path.exists():
                checkpoints[epoch] = epoch_path
            else:
                print(f"⚠️  Warning: Checkpoint for epoch {epoch} not found: {epoch_path}")
    
    return checkpoints


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoints on CIFAR10/100')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                       help='Dataset to evaluate on (cifar10 or cifar100)')
    parser.add_argument('--epochs', type=int, nargs='+', default=[1, 100, 200, 250],
                       help='Epoch numbers to evaluate (default: 1 100 200 250)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--linear_probe_C', type=float, default=1.0,
                       help='Regularization strength for linear probe')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results (JSON)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--eval_teacher', action='store_true',
                       help='Also evaluate teacher model for comparison')
    parser.add_argument('--eval_untrained', action='store_true',
                       help='Also evaluate untrained student (random init) for comparison')
    parser.add_argument('--mode', type=str, default='kd', choices=['kd', 'moco_v3'],
                       help='Training mode: kd (knowledge distillation) or moco_v3 (contrastive learning)')
    
    args = parser.parse_args()
    
    # Load config
    model_config = load_config(args.model_config)
    
    # Auto-detect use_cls_token based on backbone type
    backbone_type = model_config.get('backbone_type', 'auto')
    if backbone_type == 'auto':
        model_name = model_config.get('backbone_name', model_config.get('student_name', ''))
        if 'resnet' in model_name.lower():
            backbone_type = 'resnet'
        elif 'vit' in model_name.lower():
            backbone_type = 'vit'
    
    # ResNet doesn't use CLS token, ViT does
    if backbone_type == 'resnet':
        use_cls_token = False
    else:
        use_cls_token = model_config.get('use_cls_token', True)
    
    # Print mode
    print(f"✓ Evaluation mode: {args.mode}")
    if args.mode == 'moco_v3':
        print("  Using encoder_q from MoCo-v3 checkpoints")
    else:
        print("  Using student model from KD checkpoints")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoints
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = find_checkpoint_files(checkpoint_dir, epochs=args.epochs)
    
    if not checkpoints:
        print(f"❌ Error: No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\n✓ Found {len(checkpoints)} checkpoints:")
    for name, path in checkpoints.items():
        print(f"  - {name}: {path}")
    
    # Build model architecture
    student = build_student_model(model_config, device)
    
    # Load CIFAR datasets
    print(f"\n📦 Loading {args.dataset.upper()} datasets...")
    # Support both 'image_size' (new) and 'student_img_size' (legacy)
    img_size = model_config.get('image_size', model_config.get('student_img_size', 224))
    train_dataset = get_cifar_dataset(args.dataset, train=True, image_size=img_size)
    test_dataset = get_cifar_dataset(args.dataset, train=False, image_size=img_size)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate teacher model if requested
    if args.eval_teacher:
        print(f"\n{'='*60}")
        print("Evaluating TEACHER model (DINOv2)")
        print(f"{'='*60}")
        try:
            # Import teacher loading function
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from distill_trainer import load_teacher_model
            
            teacher = load_teacher_model(
                teacher_name=model_config.get('teacher_name', 'dinov2_vitb14'),
                device=device
            )
            teacher.eval()
            
            # Create a wrapper class that upscales images to 224x224 for teacher
            class UpscaleDataLoader:
                """Wrapper dataloader that upscales images to target_size"""
                def __init__(self, dataloader, target_size=224, device='cuda'):
                    self.dataloader = dataloader
                    self.target_size = target_size
                    self.device = device
                
                def __iter__(self):
                    for images, labels in self.dataloader:
                        images = images.to(self.device)
                        # Upscale to teacher's expected size (96x96 -> 224x224)
                        if images.shape[-1] != self.target_size or images.shape[-2] != self.target_size:
                            images = F.interpolate(
                                images,
                                size=(self.target_size, self.target_size),
                                mode='bilinear',
                                align_corners=False
                            )
                        yield images, labels
                
                def __len__(self):
                    return len(self.dataloader)
            
            teacher_train_loader = UpscaleDataLoader(train_loader, target_size=224, device=device)
            teacher_test_loader = UpscaleDataLoader(test_loader, target_size=224, device=device)
            
            print("🔍 Extracting teacher training features (upscaling 96x96 -> 224x224)...")
            teacher_train_features, teacher_train_labels = extract_features(
                teacher, teacher_train_loader, device, use_cls_token=use_cls_token
            )
            analyze_features(teacher_train_features, name="Teacher Features")
            
            print("🔍 Extracting teacher test features (upscaling 96x96 -> 224x224)...")
            teacher_test_features, teacher_test_labels = extract_features(
                teacher, teacher_test_loader, device, use_cls_token=use_cls_token
            )
            
            print("🎯 Training linear probe on teacher features...")
            teacher_accuracy, _ = train_linear_probe(
                teacher_train_features, teacher_train_labels,
                teacher_test_features, teacher_test_labels,
                C=args.linear_probe_C
            )
            print(f"✓ Teacher linear probe accuracy: {teacher_accuracy:.4f} ({teacher_accuracy*100:.2f}%)")
            
            # Store teacher result
            results['teacher'] = {
                'checkpoint_path': 'teacher_model',
                'accuracy': float(teacher_accuracy),
                'accuracy_percent': float(teacher_accuracy * 100),
                'train_samples': len(teacher_train_features),
                'test_samples': len(teacher_test_features),
                'feature_dim': teacher_train_features.shape[1]
            }
            
            # DIAGNOSTIC: Test projected teacher accuracy
            print(f"\n{'='*60}")
            print("🔬 DIAGNOSTIC: Testing Projected Teacher Accuracy")
            print(f"{'='*60}")
            print("This tests if the frozen projection layers preserve discriminative information.")
            
            # Try to load distillation loss module from a checkpoint
            distillation_loss_module = None
            if checkpoints:
                # Try to load from latest checkpoint (prefer 'best', then highest epoch number)
                latest_checkpoint_path = None
                if 'best' in checkpoints:
                    latest_checkpoint_path = checkpoints['best']
                else:
                    # Find highest epoch number
                    int_keys = [k for k in checkpoints.keys() if isinstance(k, int)]
                    if int_keys:
                        latest_checkpoint_path = checkpoints[max(int_keys)]
                
                if latest_checkpoint_path:
                    try:
                        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                        if 'distillation_loss' in checkpoint:
                            # Import DistillationLoss class
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent))
                            from distill_trainer import DistillationLoss
                            
                            # Create module and load state
                            distillation_loss_module = DistillationLoss(
                                teacher_dim=768,
                                student_dim=384,
                                loss_weights={'cls': 1.0, 'patch': 0.5}
                            ).to(device)
                            distillation_loss_module.load_state_dict(checkpoint['distillation_loss'])
                            distillation_loss_module.eval()
                            print(f"✓ Loaded distillation loss module from checkpoint: {latest_checkpoint_path}")
                        else:
                            print(f"⚠️  Checkpoint does not contain 'distillation_loss' key, creating fresh module...")
                            # Create fresh module with default initialization
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent))
                            from distill_trainer import DistillationLoss
                            distillation_loss_module = DistillationLoss(
                                teacher_dim=768,
                                student_dim=384,
                                loss_weights={'cls': 1.0, 'patch': 0.5}
                            ).to(device)
                            distillation_loss_module.eval()
                            print(f"✓ Created fresh distillation loss module (Xavier uniform initialization)")
                    except Exception as e:
                        print(f"⚠️  Could not load distillation loss module: {e}")
                        print(f"   Creating fresh module...")
                        try:
                            import sys
                            sys.path.insert(0, str(Path(__file__).parent))
                            from distill_trainer import DistillationLoss
                            distillation_loss_module = DistillationLoss(
                                teacher_dim=768,
                                student_dim=384,
                                loss_weights={'cls': 1.0, 'patch': 0.5}
                            ).to(device)
                            distillation_loss_module.eval()
                            print(f"✓ Created fresh distillation loss module (Xavier uniform initialization)")
                        except Exception as e2:
                            print(f"❌ Could not create distillation loss module: {e2}")
                            distillation_loss_module = None
                else:
                    # No checkpoint found, create fresh module
                    try:
                        import sys
                        sys.path.insert(0, str(Path(__file__).parent))
                        from distill_trainer import DistillationLoss
                        distillation_loss_module = DistillationLoss(
                            teacher_dim=768,
                            student_dim=384,
                            loss_weights={'cls': 1.0, 'patch': 0.5}
                        ).to(device)
                        distillation_loss_module.eval()
                        print(f"✓ Created fresh distillation loss module (Xavier uniform initialization)")
                    except Exception as e:
                        print(f"❌ Could not create distillation loss module: {e}")
                        distillation_loss_module = None
            
            if distillation_loss_module is not None and distillation_loss_module.teacher_proj_cls is not None:
                print(f"\n📊 Projection Layer Statistics:")
                proj_weight = distillation_loss_module.teacher_proj_cls.weight.data
                print(f"  Weight shape: {proj_weight.shape}")
                print(f"  Weight norm: {proj_weight.norm().item():.6f}")
                print(f"  Weight std: {proj_weight.std().item():.6f}")
                print(f"  Weight mean: {proj_weight.mean().item():.6f}")
                print(f"  Weight min: {proj_weight.min().item():.6f}")
                print(f"  Weight max: {proj_weight.max().item():.6f}")
                
                # Project teacher features
                print(f"\n🔍 Projecting teacher features through projection layer...")
                with torch.no_grad():
                    # Move features to device
                    teacher_train_features_proj = distillation_loss_module.teacher_proj_cls(
                        teacher_train_features.to(device)
                    ).cpu()
                    teacher_test_features_proj = distillation_loss_module.teacher_proj_cls(
                        teacher_test_features.to(device)
                    ).cpu()
                    
                    # Normalize projected features (same as in loss computation)
                    teacher_train_features_proj = F.normalize(teacher_train_features_proj, dim=-1, p=2)
                    teacher_test_features_proj = F.normalize(teacher_test_features_proj, dim=-1, p=2)
                
                print(f"  Original teacher features: {teacher_train_features.shape}")
                print(f"  Projected teacher features: {teacher_train_features_proj.shape}")
                
                # Analyze projected features
                analyze_features(teacher_train_features_proj, name="Projected Teacher Features")
                
                # Train linear probe on projected features
                print(f"\n🎯 Training linear probe on PROJECTED teacher features...")
                teacher_proj_accuracy, _ = train_linear_probe(
                    teacher_train_features_proj, teacher_train_labels,
                    teacher_test_features_proj, teacher_test_labels,
                    C=args.linear_probe_C
                )
                print(f"✓ Projected teacher linear probe accuracy: {teacher_proj_accuracy:.4f} ({teacher_proj_accuracy*100:.2f}%)")
                
                # Compare accuracies
                print(f"\n📊 Comparison:")
                print(f"  Original teacher accuracy: {teacher_accuracy:.4f} ({teacher_accuracy*100:.2f}%)")
                print(f"  Projected teacher accuracy: {teacher_proj_accuracy:.4f} ({teacher_proj_accuracy*100:.2f}%)")
                accuracy_drop = teacher_accuracy - teacher_proj_accuracy
                accuracy_drop_pct = (accuracy_drop / teacher_accuracy) * 100 if teacher_accuracy > 0 else 0
                print(f"  Accuracy drop: {accuracy_drop:.4f} ({accuracy_drop_pct:.2f}%)")
                
                if accuracy_drop > 0.1:  # More than 10% drop
                    print(f"  ⚠️  WARNING: Large accuracy drop! The projection layer is losing discriminative information.")
                    print(f"     This could explain why the student model performs poorly.")
                elif accuracy_drop > 0.05:  # More than 5% drop
                    print(f"  ⚠️  Moderate accuracy drop. The projection may be affecting performance.")
                else:
                    print(f"  ✓ Projection preserves most discriminative information.")
                
                # Store projected teacher result
                results['teacher_projected'] = {
                    'checkpoint_path': 'teacher_model_projected',
                    'accuracy': float(teacher_proj_accuracy),
                    'accuracy_percent': float(teacher_proj_accuracy * 100),
                    'accuracy_drop': float(accuracy_drop),
                    'accuracy_drop_percent': float(accuracy_drop_pct),
                    'train_samples': len(teacher_train_features_proj),
                    'test_samples': len(teacher_test_features_proj),
                    'feature_dim': teacher_train_features_proj.shape[1]
                }
            else:
                print(f"⚠️  Could not create/load distillation loss module. Skipping projected teacher test.")
        except Exception as e:
            print(f"❌ Error evaluating teacher: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate untrained student if requested
    if args.eval_untrained:
        print(f"\n{'='*60}")
        print("Evaluating UNTRAINED student (random initialization)")
        print(f"{'='*60}")
        
        # Build fresh untrained student
        untrained_student = build_student_model(model_config, device)
        untrained_student.eval()
        
        print("🔍 Extracting untrained student training features...")
        untrained_train_features, untrained_train_labels = extract_features(untrained_student, train_loader, device, use_cls_token=use_cls_token)
        analyze_features(untrained_train_features, name="Untrained Student Features")
        
        print("🔍 Extracting untrained student test features...")
        untrained_test_features, untrained_test_labels = extract_features(untrained_student, test_loader, device, use_cls_token=use_cls_token)
        
        print("🎯 Training linear probe on untrained student features...")
        untrained_accuracy, _ = train_linear_probe(
            untrained_train_features, untrained_train_labels,
            untrained_test_features, untrained_test_labels,
            C=args.linear_probe_C
        )
        print(f"✓ Untrained student linear probe accuracy: {untrained_accuracy:.4f} ({untrained_accuracy*100:.2f}%)")
        
        # Store untrained result
        results['untrained'] = {
            'checkpoint_path': 'untrained_student',
            'accuracy': float(untrained_accuracy),
            'accuracy_percent': float(untrained_accuracy * 100),
            'train_samples': len(untrained_train_features),
            'test_samples': len(untrained_test_features),
            'feature_dim': untrained_train_features.shape[1]
        }
    
    # Evaluate each checkpoint
    # Sort checkpoints: handle mixed int/str keys by converting to comparable format
    def sort_key(item):
        name, path = item
        if isinstance(name, int):
            return (0, name)  # Int keys come first, sorted by value
        else:
            return (1, str(name))  # String keys come after, sorted alphabetically
    
    for checkpoint_name, checkpoint_path in sorted(checkpoints.items(), key=sort_key):
        print(f"\n{'='*60}")
        print(f"Evaluating checkpoint: {checkpoint_name} ({checkpoint_path.name})")
        print(f"{'='*60}")
        
        # Load checkpoint
        try:
            # Check file size first (corrupted files are often incomplete)
            file_size = checkpoint_path.stat().st_size
            if file_size < 1024:  # Less than 1KB is suspicious
                print(f"  ⚠️  Warning: Checkpoint file is very small ({file_size} bytes), may be corrupted")
            elif file_size < 10 * 1024 * 1024:  # Less than 10MB is suspicious for a model checkpoint
                print(f"  ⚠️  Warning: Checkpoint file seems small ({file_size / 1024 / 1024:.2f} MB)")
            
            state_dict, detected_backbone = load_checkpoint(checkpoint_path, device, mode=args.mode)
            
            # Check if checkpoint architecture matches config
            config_backbone_type = model_config.get('backbone_type', 'auto')
            if config_backbone_type == 'auto':
                model_name = model_config.get('backbone_name', model_config.get('student_name', ''))
                if 'resnet' in model_name.lower():
                    config_backbone_type = 'resnet'
                elif 'vit' in model_name.lower():
                    config_backbone_type = 'vit'
            
            if detected_backbone and config_backbone_type != 'auto':
                if detected_backbone != config_backbone_type:
                    print(f"\n  ❌ CRITICAL ERROR: Architecture mismatch!")
                    print(f"     Checkpoint contains {detected_backbone.upper()} weights")
                    print(f"     But config specifies {config_backbone_type.upper()} architecture")
                    print(f"     This will cause loading to fail. Please use the correct model config.")
                    print(f"     For {detected_backbone.upper()} checkpoints, use:")
                    if detected_backbone == 'vit':
                        print(f"       --model_config model_config_moco_vit.yaml")
                    else:
                        print(f"       --model_config model_config_moco_resnet.yaml")
                    print(f"\n  ⚠️  Skipping this checkpoint due to architecture mismatch.")
                    print(f"  Please re-run evaluation with the correct model config.\n")
                    continue  # Skip this checkpoint
                else:
                    print(f"  ✓ Checkpoint architecture matches config ({detected_backbone.upper()})")
            elif detected_backbone:
                print(f"  ℹ️  Detected checkpoint architecture: {detected_backbone.upper()}")
            
            # ============================================================
            # DIAGNOSTIC: Checkpoint loading verification (MoCo-v3)
            # ============================================================
            if args.mode == 'moco_v3':
                print(f"\n🔬 DIAGNOSTIC: MoCo-v3 Checkpoint Loading")
                # Load full checkpoint to inspect structure
                full_checkpoint = torch.load(checkpoint_path, map_location=device)
                print(f"  Checkpoint keys: {list(full_checkpoint.keys())}")
                
                if 'encoder_q' in full_checkpoint:
                    encoder_q_keys = list(full_checkpoint['encoder_q'].keys())[:5]
                    print(f"  encoder_q keys (first 5): {encoder_q_keys}")
                    print(f"  encoder_q total keys: {len(full_checkpoint['encoder_q'])}")
                
                if 'proj_q' in full_checkpoint:
                    proj_q_keys = list(full_checkpoint['proj_q'].keys())
                    print(f"  proj_q keys: {proj_q_keys}")
                
                # Verify state_dict matches model
                model_keys = set(student.state_dict().keys())
                checkpoint_keys = set(state_dict.keys())
                
                missing_in_checkpoint = model_keys - checkpoint_keys
                extra_in_checkpoint = checkpoint_keys - model_keys
                
                if missing_in_checkpoint:
                    print(f"  ⚠️  Model keys missing in checkpoint (first 5): {list(missing_in_checkpoint)[:5]}")
                if extra_in_checkpoint:
                    print(f"  ⚠️  Extra keys in checkpoint (first 5): {list(extra_in_checkpoint)[:5]}")
                if not missing_in_checkpoint and not extra_in_checkpoint:
                    print(f"  ✓ All model keys match checkpoint keys")
            
            # Debug: print checkpoint keys
            if checkpoint_name == list(checkpoints.keys())[0]:  # Only for first checkpoint
                print(f"  Debug: Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")
                print(f"  Debug: Model keys (first 10): {list(student.state_dict().keys())[:10]}")
            
            # Try strict loading first
            try:
                student.load_state_dict(state_dict, strict=True)
                print(f"✓ Loaded checkpoint: {checkpoint_name} (strict mode)")
            except RuntimeError as e:
                # If strict loading fails, try with strict=False and report missing/unexpected keys
                print(f"  ⚠️  Strict loading failed, trying non-strict mode...")
                print(f"  Error: {str(e)[:200]}")  # Print first part of error
                missing_keys, unexpected_keys = student.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"  ⚠️  Missing {len(missing_keys)} keys (first 5: {missing_keys[:5]})")
                if unexpected_keys:
                    print(f"  ⚠️  Unexpected {len(unexpected_keys)} keys (first 5: {unexpected_keys[:5]})")
                if not missing_keys and not unexpected_keys:
                    print(f"  ✓ All keys matched (non-strict mode)")
                else:
                    print(f"  ⚠️  Loaded with mismatched keys - model may not work correctly")
            
            student.eval()
            
            # ============================================================
            # DIAGNOSTIC: Check if model parameters actually changed
            # ============================================================
            if args.mode == 'moco_v3' and args.eval_untrained:
                print(f"\n{'='*60}")
                print("🔬 DIAGNOSTIC: Parameter Change Analysis (MoCo-v3)")
                print(f"{'='*60}")
                
                # Build fresh untrained model for comparison
                untrained_for_diag = build_student_model(model_config, device)
                untrained_for_diag.eval()
                
                # Compare key parameters
                print("Comparing key parameters between untrained and trained models:")
                param_changes = []
                total_params = 0
                changed_params = 0
                
                trained_params = dict(student.named_parameters())
                untrained_params = dict(untrained_for_diag.named_parameters())
                
                # Check first few layers (most likely to change)
                key_layers = [
                    'patch_embed.proj.weight',
                    'patch_embed.proj.bias',
                    'blocks.0.norm1.weight',
                    'blocks.0.attn.qkv.weight',
                    'blocks.0.attn.proj.weight',
                    'blocks.0.mlp.fc1.weight',
                    'blocks.0.mlp.fc2.weight',
                ]
                
                for layer_name in key_layers:
                    if layer_name in trained_params and layer_name in untrained_params:
                        trained_param = trained_params[layer_name]
                        untrained_param = untrained_params[layer_name]
                        
                        # Compute statistics
                        mean_diff = (trained_param - untrained_param).abs().mean().item()
                        max_diff = (trained_param - untrained_param).abs().max().item()
                        param_norm_trained = trained_param.norm().item()
                        param_norm_untrained = untrained_param.norm().item()
                        
                        param_changes.append({
                            'layer': layer_name,
                            'mean_diff': mean_diff,
                            'max_diff': max_diff,
                            'norm_trained': param_norm_trained,
                            'norm_untrained': param_norm_untrained,
                            'relative_change': mean_diff / (param_norm_untrained + 1e-8)
                        })
                
                # Check all parameters
                for name in trained_params.keys():
                    if name in untrained_params:
                        total_params += 1
                        diff = (trained_params[name] - untrained_params[name]).abs().mean().item()
                        if diff > 1e-6:  # Threshold for "changed"
                            changed_params += 1
                
                # Print results
                print(f"\n  Overall Statistics:")
                print(f"    Total parameters checked: {total_params}")
                print(f"    Parameters that changed (diff > 1e-6): {changed_params} ({100*changed_params/total_params:.1f}%)")
                
                print(f"\n  Key Layer Analysis:")
                for change_info in param_changes:
                    print(f"    {change_info['layer']}:")
                    print(f"      Mean absolute diff: {change_info['mean_diff']:.8f}")
                    print(f"      Max absolute diff: {change_info['max_diff']:.8f}")
                    print(f"      Norm (untrained): {change_info['norm_untrained']:.6f}")
                    print(f"      Norm (trained): {change_info['norm_trained']:.6f}")
                    print(f"      Relative change: {change_info['relative_change']*100:.4f}%")
                    
                    if change_info['mean_diff'] < 1e-6:
                        print(f"      ⚠️  WARNING: This layer didn't change!")
                    elif change_info['relative_change'] < 0.001:
                        print(f"      ⚠️  WARNING: Very small relative change (<0.1%)")
                
                # Check if model collapsed (all parameters very similar)
                if changed_params / total_params < 0.1:
                    print(f"\n  ⚠️  CRITICAL: Less than 10% of parameters changed!")
                    print(f"      This suggests the model may not have learned anything.")
                elif changed_params / total_params < 0.5:
                    print(f"\n  ⚠️  WARNING: Less than 50% of parameters changed.")
                    print(f"      This may indicate partial learning or early stopping.")
                else:
                    print(f"\n  ✓ Most parameters changed - model appears to have learned.")
                
                # Clean up
                del untrained_for_diag
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            
            # ============================================================
            # DIAGNOSTIC: Model state verification
            # ============================================================
            print(f"\n🔬 DIAGNOSTIC: Model State Verification")
            print(f"  Model is in eval mode: {not student.training}")
            print(f"  Device: {next(student.parameters()).device}")
            print(f"  Dtype: {next(student.parameters()).dtype}")
            
            # Check if model has dropout (should be disabled in eval)
            has_dropout = False
            for name, module in student.named_modules():
                if 'dropout' in name.lower() or isinstance(module, torch.nn.Dropout):
                    has_dropout = True
                    print(f"  Found dropout layer: {name} (p={module.p if hasattr(module, 'p') else 'N/A'})")
            if not has_dropout:
                print(f"  No dropout layers found (expected for ViT)")
            
            # For MoCo-v3: Verify encoder_q structure
            if args.mode == 'moco_v3':
                print(f"\n  MoCo-v3 Specific Checks:")
                # Check if we can access forward_features
                try:
                    # Support both 'image_size' (new) and 'student_img_size' (legacy)
                    img_size = model_config.get('image_size', model_config.get('student_img_size', 224))
                    test_input = torch.randn(1, 3, img_size, img_size).to(device)
                    with torch.no_grad():
                        test_output = student.forward_features(test_input)
                    print(f"    ✓ forward_features() works correctly")
                    print(f"    Output shape: {test_output.shape if isinstance(test_output, torch.Tensor) else 'dict'}")
                except Exception as e:
                    print(f"    ❌ ERROR: forward_features() failed: {e}")
            
        except RuntimeError as e:
            error_msg = str(e)
            if "corrupted" in error_msg.lower() or "zip archive" in error_msg.lower() or "central directory" in error_msg.lower():
                print(f"❌ Checkpoint file is corrupted or incomplete: {checkpoint_path.name}")
                print(f"   File size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
                print(f"   This checkpoint will be skipped. You may need to delete it and re-train.")
            else:
                print(f"❌ Error loading checkpoint {checkpoint_name}: {e}")
                import traceback
                traceback.print_exc()
            continue
        except Exception as e:
            print(f"❌ Unexpected error loading checkpoint {checkpoint_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Extract training features
        print(f"🔍 Extracting training features...")
        train_features, train_labels = extract_features(student, train_loader, device, use_cls_token=use_cls_token)
        print(f"  Train features shape: {train_features.shape}")
        
        # ============================================================
        # DIAGNOSTIC: Compare features between untrained and trained
        # ============================================================
        if args.mode == 'moco_v3' and args.eval_untrained and 'untrained' in results:
            print(f"\n{'='*60}")
            print("🔬 DIAGNOSTIC: Feature Comparison (Untrained vs Trained)")
            print(f"{'='*60}")
            
            # Get untrained features (already computed)
            # We need to extract them again for the same samples, or compare statistics
            # For now, compare on a small subset
            print("  Comparing features on first 1000 training samples...")
            
            # Get subset of training data
            subset_indices = list(range(min(1000, len(train_dataset))))
            subset_dataset = torch.utils.data.Subset(train_dataset, subset_indices)
            subset_loader = DataLoader(
                subset_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 workers for small subset
                pin_memory=False
            )
            
            # Extract features from trained model
            trained_subset_features, _ = extract_features(student, subset_loader, device, use_cls_token=use_cls_token)
            
            # Extract features from untrained model
            untrained_for_feat = build_student_model(model_config, device)
            untrained_for_feat.eval()
            untrained_subset_features, _ = extract_features(untrained_for_feat, subset_loader, device, use_cls_token=use_cls_token)
            
            # Compare statistics
            print(f"\n  Feature Statistics Comparison:")
            print(f"    Trained features:")
            print(f"      Mean: {trained_subset_features.mean().item():.6f}")
            print(f"      Std: {trained_subset_features.std().item():.6f}")
            print(f"      Min: {trained_subset_features.min().item():.6f}")
            print(f"      Max: {trained_subset_features.max().item():.6f}")
            print(f"    Untrained features:")
            print(f"      Mean: {untrained_subset_features.mean().item():.6f}")
            print(f"      Std: {untrained_subset_features.std().item():.6f}")
            print(f"      Min: {untrained_subset_features.min().item():.6f}")
            print(f"      Max: {untrained_subset_features.max().item():.6f}")
            
            # Compute pairwise similarity for both
            sample_size = min(100, len(trained_subset_features))
            trained_sim = (trained_subset_features[:sample_size] @ trained_subset_features[:sample_size].T).mean().item()
            untrained_sim = (untrained_subset_features[:sample_size] @ untrained_subset_features[:sample_size].T).mean().item()
            
            print(f"\n  Pairwise Similarity (first {sample_size} samples):")
            print(f"    Trained: {trained_sim:.6f}")
            print(f"    Untrained: {untrained_sim:.6f}")
            
            # Compute feature difference
            feature_diff = (trained_subset_features - untrained_subset_features).abs().mean().item()
            print(f"\n  Mean absolute feature difference: {feature_diff:.6f}")
            
            if feature_diff < 1e-4:
                print(f"    ⚠️  CRITICAL: Features are nearly identical to untrained model!")
                print(f"        This suggests the model didn't learn meaningful representations.")
            elif feature_diff < 0.01:
                print(f"    ⚠️  WARNING: Features are very similar to untrained model.")
                print(f"        The model may not have learned much.")
            else:
                print(f"    ✓ Features differ from untrained model - model appears to have learned.")
            
            # Check if trained features are more collapsed than untrained
            if trained_sim > untrained_sim + 0.1:
                print(f"\n  ⚠️  WARNING: Trained features are MORE collapsed than untrained!")
                print(f"      Trained pairwise sim: {trained_sim:.6f}")
                print(f"      Untrained pairwise sim: {untrained_sim:.6f}")
                print(f"      This suggests training made features worse!")
            elif trained_sim < untrained_sim - 0.1:
                print(f"\n  ✓ Trained features are LESS collapsed than untrained (good!)")
            
            # Clean up
            del untrained_for_feat, trained_subset_features, untrained_subset_features
            torch.cuda.empty_cache() if device.type == 'cuda' else None
        
        # Extract test features FOR THIS CHECKPOINT (after loading weights)
        print(f"🔍 Extracting test features...")
        test_features, test_labels = extract_features(student, test_loader, device, use_cls_token=use_cls_token)
        print(f"  Test features shape: {test_features.shape}")
        
        # Analyze feature quality
        feature_stats = analyze_features(train_features, name=f"Checkpoint {checkpoint_name} Features")
        
        # Train linear probe
        print(f"🎯 Training linear probe (C={args.linear_probe_C})...")
        accuracy, clf = train_linear_probe(
            train_features, train_labels,
            test_features, test_labels,
            C=args.linear_probe_C
        )
        
        print(f"✓ Linear probe accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Store results
        results[checkpoint_name] = {
            'checkpoint_path': str(checkpoint_path),
            'accuracy': float(accuracy),
            'accuracy_percent': float(accuracy * 100),
            'train_samples': len(train_features),
            'test_samples': len(test_features),
            'feature_dim': train_features.shape[1],
            'feature_stats': feature_stats  # Include feature statistics
        }
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Linear Probe C: {args.linear_probe_C}")
    print(f"\nResults:")
    print(f"{'Checkpoint':<15} {'Accuracy':<12} {'Accuracy %':<12}")
    print(f"{'-'*40}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, result in sorted_results:
        print(f"{str(name):<15} {result['accuracy']:<12.4f} {result['accuracy_percent']:<12.2f}%")
    
    # Find best checkpoint
    best_checkpoint = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best checkpoint: {best_checkpoint[0]} ({best_checkpoint[1]['accuracy_percent']:.2f}%)")
    
    # Save results to JSON
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'linear_probe_C': args.linear_probe_C,
            'use_cls_token': use_cls_token,
            'results': results,
            'best_checkpoint': best_checkpoint[0]
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()

