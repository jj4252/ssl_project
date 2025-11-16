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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if 'student' in checkpoint:
        return checkpoint['student']
    elif 'model' in checkpoint:
        return checkpoint['model']
    elif 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    else:
        # Assume the checkpoint itself is the state dict
        return checkpoint


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
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class='multinomial',
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
    
    # Evaluate each checkpoint
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
            state_dict = load_checkpoint(checkpoint_path, device)
            student.load_state_dict(state_dict)
            student.eval()
            print(f"✓ Loaded checkpoint: {checkpoint_name}")
        except Exception as e:
            print(f"❌ Error loading checkpoint {checkpoint_name}: {e}")
            continue
        
        # Extract training features
        print(f"🔍 Extracting training features...")
        train_features, train_labels = extract_features(student, train_loader, device, use_cls_token=use_cls_token)
        print(f"  Train features shape: {train_features.shape}")
        
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
            'feature_dim': train_features.shape[1]
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

