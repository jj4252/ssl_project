import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader

from data_loader import EvalDataset
from transforms import EvalTransform
from vit_model import build_vit
from dino_wrapper import DINO
from extract_features import extract_features


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--eval_config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./features')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    eval_cfg = load_config(args.eval_config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Model
    backbone = build_vit(
        model_name=model_cfg['model_name'],
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size']
    )
    model = DINO(backbone, use_cls_token=model_cfg['use_cls_token'])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    backbone = model.get_backbone()
    backbone.eval()
    
    # Transforms
    transform = EvalTransform(image_size=data_cfg['image_size'])
    
    # Datasets
    train_dataset = EvalDataset(split='train', transform=transform)
    test_dataset = EvalDataset(split='test', transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=eval_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_cfg['batch_size'],
        shuffle=False,
        num_workers=data_cfg['num_workers'],
        pin_memory=data_cfg['pin_memory']
    )
    
    # Extract features
    print("Extracting train features...")
    train_features, train_labels = extract_features(
        backbone, train_loader, device, 
        use_cls_token=eval_cfg['use_cls_token']
    )
    
    print("Extracting test features...")
    test_features, test_labels = extract_features(
        backbone, test_loader, device,
        use_cls_token=eval_cfg['use_cls_token']
    )
    
    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'train_features': train_features,
        'train_labels': train_labels,
        'test_features': test_features,
        'test_labels': test_labels
    }, f"{args.output_dir}/features.pt")
    
    print(f"Features saved to {args.output_dir}/features.pt")


if __name__ == '__main__':
    main()

