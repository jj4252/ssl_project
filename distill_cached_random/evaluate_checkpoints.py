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


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return student model state dict"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except RuntimeError as e:
        if "failed reading zip archive" in str(e) or "central directory" in str(e):
            raise RuntimeError(f"Checkpoint file is corrupted or incomplete: {checkpoint_path}")
        else:
            raise
    
    # Handle different checkpoint formats
    # Note: New checkpoints may also contain 'distillation_loss' key, which is ignored here
    # (only the student model weights are needed for evaluation)
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
    
    return state_dict


def build_student_model(model_config, device):
    """Build student model architecture"""
    model_name = model_config['student_name']
    img_size = model_config['student_img_size']
    
    print(f"Building student model: {model_name} with img_size={img_size}")
    
    # Try to create model with custom img_size
    try:
        student = timm.create_model(
            model_name,
            pretrained=False,
            img_size=img_size,
            num_classes=0,  # No classification head
        )
    except Exception as e:
        print(f"  Warning: Could not create model with img_size={img_size}, trying default size first...")
        student = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
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
                # Tensor format (timm models): [B, N+1, D] (CLS + patches)
                if use_cls_token:
                    feat = outputs[:, 0]  # CLS token [B, D]
                else:
                    feat = outputs[:, 1:].mean(dim=1)  # Mean-pool patches [B, D]
                
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
    
    args = parser.parse_args()
    
    # Load config
    model_config = load_config(args.model_config)
    use_cls_token = model_config.get('use_cls_token', True)
    
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
    train_dataset = get_cifar_dataset(args.dataset, train=True, image_size=model_config['student_img_size'])
    test_dataset = get_cifar_dataset(args.dataset, train=False, image_size=model_config['student_img_size'])
    
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
    
    # Extract test features once (same for all checkpoints)
    print(f"\n🔍 Extracting test features...")
    student.eval()
    test_features, test_labels = extract_features(student, test_loader, device, use_cls_token=use_cls_token)
    print(f"  Test features shape: {test_features.shape}")
    
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
    if not results:
        results = {}
    
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
            
            state_dict = load_checkpoint(checkpoint_path, device)
            
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

