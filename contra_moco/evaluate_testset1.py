"""
Evaluate MoCo model on testset_1 using linear probing.

This script:
1. Loads a trained MoCo checkpoint
2. Extracts features from testset_1 train/val/test splits
3. Trains linear probes with different C values
4. Reports validation and test accuracy
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
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


def load_checkpoint(checkpoint_path, device, mode='moco_v3'):
    """
    Load checkpoint and return model state dict.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        mode: 'kd' or 'moco_v3'
    
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
    
    # Handle MoCo checkpoints
    if mode == 'moco_v3':
        if 'encoder_q' in checkpoint:
            state_dict = checkpoint['encoder_q']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        # KD mode
        if 'student' in checkpoint:
            state_dict = checkpoint['student']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    
    # Handle torch.compile() prefix
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        print(f"  Detected compiled model checkpoint, stripping '_orig_mod.' prefix...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key[len('_orig_mod.'):]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    return state_dict


def build_model(model_config, device):
    """Build model architecture from config"""
    model_name = model_config.get('backbone_name', model_config.get('student_name', 'vit_small_patch16_224'))
    img_size = model_config.get('image_size', model_config.get('student_img_size', 96))
    backbone_type = model_config.get('backbone_type', 'auto')
    
    if backbone_type == 'auto':
        if 'resnet' in model_name.lower():
            backbone_type = 'resnet'
        elif 'vit' in model_name.lower():
            backbone_type = 'vit'
        else:
            backbone_type = 'vit'
    
    print(f"Building model: {model_name} with img_size={img_size} (backbone_type={backbone_type})")
    
    if backbone_type == 'resnet':
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )
    else:
        try:
            model = timm.create_model(
                model_name,
                pretrained=False,
                img_size=img_size,
                num_classes=0,
                global_pool="",
            )
        except Exception as e:
            print(f"  Warning: Could not create model with img_size={img_size}, trying default...")
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool="",
            )
    
    # Patch patch_embed if needed
    if hasattr(model, 'patch_embed'):
        if hasattr(model.patch_embed, 'patch_size'):
            patch_size = model.patch_embed.patch_size
            if isinstance(patch_size, (list, tuple)):
                patch_size = patch_size[0]
        else:
            patch_size = 16
        
        grid_size = img_size // patch_size
        if hasattr(model.patch_embed, 'img_size'):
            model.patch_embed.img_size = (img_size, img_size)
        if hasattr(model.patch_embed, 'grid_size'):
            model.patch_embed.grid_size = (grid_size, grid_size)
        if hasattr(model.patch_embed, 'num_patches'):
            model.patch_embed.num_patches = grid_size * grid_size
    
    model = model.to(device)
    model.eval()
    return model


class TestSet1Dataset(Dataset):
    """Dataset for testset_1 with CSV labels"""
    
    def __init__(self, image_dir, labels_csv, image_size=96, transform=None):
        """
        Args:
            image_dir: Directory containing images (train/, val/, or test/)
            labels_csv: Path to CSV file with labels (filename, class_id, class_name)
            image_size: Target image size
            transform: Optional transform
        """
        self.image_dir = Path(image_dir)
        self.labels_df = pd.read_csv(labels_csv)
        self.image_size = image_size
        
        # Create default transform if not provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Create mapping from filename to class_id
        self.filename_to_class = {}
        for _, row in self.labels_df.iterrows():
            self.filename_to_class[row['filename']] = row['class_id']
        
        # Get all image files
        self.image_files = []
        self.labels = []
        
        for img_file in sorted(self.image_dir.glob('*.jpg')):
            filename = img_file.name
            if filename in self.filename_to_class:
                self.image_files.append(img_file)
                self.labels.append(self.filename_to_class[filename])
        
        print(f"  Loaded {len(self.image_files)} images from {image_dir}")
        print(f"  Number of classes: {len(set(self.labels))}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def extract_features(model, dataloader, device, use_cls_token=True):
    """Extract features from model"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            
            # Forward through model
            outputs = model.forward_features(images)
            
            # Handle different output formats
            if isinstance(outputs, dict):
                if 'x_norm_clstoken' in outputs:
                    feat = outputs['x_norm_clstoken'] if use_cls_token else outputs['x_norm_patchtokens'].mean(dim=1)
                elif 'cls_token' in outputs:
                    feat = outputs['cls_token'] if use_cls_token else outputs['patch_tokens'].mean(dim=1)
                else:
                    tokens = outputs.get('x', outputs.get('tokens', None))
                    if tokens is not None:
                        feat = tokens[:, 0] if use_cls_token else tokens[:, 1:].mean(dim=1)
                    else:
                        raise ValueError("Could not extract features from dict output")
            else:
                # Tensor output: assume [B, N, D] or [B, D]
                if len(outputs.shape) == 3:
                    feat = outputs[:, 0] if use_cls_token else outputs[:, 1:].mean(dim=1)
                else:
                    feat = outputs
            
            # Normalize features
            feat = F.normalize(feat, dim=-1)
            
            features.append(feat.cpu())
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    
    return features, labels


def train_linear_probe(train_features, train_labels, test_features, test_labels, C=1.0, max_iter=1000):
    """Train linear probe and return accuracy"""
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
        solver='lbfgs',
        n_jobs=-1
    )
    clf.fit(train_features, train_labels)
    
    # Evaluate
    predictions = clf.predict(test_features)
    accuracy = accuracy_score(test_labels, predictions)
    
    return accuracy, clf


def find_best_C(train_features, train_labels, val_features, val_labels, C_values=None, max_iter=1000):
    """Try multiple C values and return the best one"""
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0]
    
    best_accuracy = -1.0
    best_C = None
    best_clf = None
    all_results = {}
    
    print(f"  Trying {len(C_values)} C values: {C_values}")
    for C in C_values:
        accuracy, clf = train_linear_probe(
            train_features, train_labels,
            val_features, val_labels,
            C=C, max_iter=max_iter
        )
        all_results[C] = accuracy
        print(f"    C={C:6.2f}: accuracy={accuracy:.4f} ({accuracy*100:.2f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
            best_clf = clf
    
    return best_accuracy, best_C, best_clf, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate MoCo model on testset_1')
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, 
                       default='/scratch/jj4252/Nov_14_distill/fall2025_finalproject/testset_1/data',
                       help='Path to testset_1/data directory')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=96,
                       help='Image size for preprocessing')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--mode', type=str, default='moco_v3', choices=['kd', 'moco_v3'],
                       help='Training mode: kd or moco_v3')
    parser.add_argument('--output_file', type=str, default='testset1_evaluation_results.json',
                       help='Output file for results (JSON)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    model_config = load_config(args.model_config)
    
    # Auto-detect use_cls_token
    backbone_type = model_config.get('backbone_type', 'auto')
    if backbone_type == 'auto':
        model_name = model_config.get('backbone_name', model_config.get('student_name', ''))
        if 'resnet' in model_name.lower():
            backbone_type = 'resnet'
        elif 'vit' in model_name.lower():
            backbone_type = 'vit'
    
    use_cls_token = model_config.get('use_cls_token', True) if backbone_type != 'resnet' else False
    
    # Build model
    model = build_model(model_config, device)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint, device, mode=args.mode)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict, strict=True)
        print("  ✓ Checkpoint loaded successfully (strict mode)")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load checkpoint in strict mode: {e}")
        print("  Trying non-strict mode...")
        model.load_state_dict(state_dict, strict=False)
        print("  ✓ Checkpoint loaded (non-strict mode)")
    
    model.eval()
    
    # Setup data paths
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    train_labels_csv = data_dir / 'train_labels.csv'
    val_labels_csv = data_dir / 'val_labels.csv'
    test_labels_csv = data_dir / 'test_labels_INTERNAL.csv'
    
    # Create datasets
    print(f"\nLoading datasets from {data_dir}")
    train_dataset = TestSet1Dataset(train_dir, train_labels_csv, image_size=args.image_size)
    val_dataset = TestSet1Dataset(val_dir, val_labels_csv, image_size=args.image_size)
    test_dataset = TestSet1Dataset(test_dir, test_labels_csv, image_size=args.image_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Extract features
    print("\n🔍 Extracting features...")
    print("  Training set...")
    train_features, train_labels = extract_features(model, train_loader, device, use_cls_token=use_cls_token)
    print(f"    Train features shape: {train_features.shape}")
    print(f"    Train labels shape: {train_labels.shape}")
    print(f"    Number of classes: {len(torch.unique(train_labels))}")
    
    print("  Validation set...")
    val_features, val_labels = extract_features(model, val_loader, device, use_cls_token=use_cls_token)
    print(f"    Val features shape: {val_features.shape}")
    
    print("  Test set...")
    test_features, test_labels = extract_features(model, test_loader, device, use_cls_token=use_cls_token)
    print(f"    Test features shape: {test_features.shape}")
    
    # Find best C on validation set
    print("\n🎯 Finding best C value on validation set...")
    val_accuracy, best_C, best_clf, all_C_results = find_best_C(
        train_features, train_labels,
        val_features, val_labels,
        C_values=[0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0],
        max_iter=1000
    )
    print(f"\n✓ Best validation accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%) with C={best_C}")
    
    # Evaluate on test set with best C
    print(f"\n🎯 Evaluating on test set with best C={best_C}...")
    test_accuracy, test_clf = train_linear_probe(
        train_features, train_labels,
        test_features, test_labels,
        C=best_C, max_iter=1000
    )
    print(f"✓ Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Print classification report
    print("\n📊 Classification Report (Test Set):")
    test_pred = test_clf.predict(test_features.numpy() if isinstance(test_features, torch.Tensor) else test_features)
    test_labels_np = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else test_labels
    print(classification_report(test_labels_np, test_pred))
    
    # Save results
    results = {
        'checkpoint': str(args.checkpoint),
        'model_config': str(args.model_config),
        'validation_accuracy': float(val_accuracy),
        'validation_accuracy_percent': float(val_accuracy * 100),
        'test_accuracy': float(test_accuracy),
        'test_accuracy_percent': float(test_accuracy * 100),
        'best_C': float(best_C),
        'all_C_results': {str(k): float(v) for k, v in all_C_results.items()},
        'train_samples': int(len(train_features)),
        'val_samples': int(len(val_features)),
        'test_samples': int(len(test_features)),
        'num_classes': int(len(torch.unique(train_labels))),
        'feature_dim': int(train_features.shape[1])
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {Path(args.checkpoint).name}")
    print(f"Validation Accuracy: {val_accuracy*100:.2f}% (C={best_C})")
    print(f"Test Accuracy: {test_accuracy*100:.2f}% (C={best_C})")
    print(f"Number of classes: {len(torch.unique(train_labels))}")
    print(f"Feature dimension: {train_features.shape[1]}")
    print("="*60)


if __name__ == '__main__':
    main()

