import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data_loader import PretrainDataset
from transforms import MultiCropTransform
from vit_model import build_vit
from dino_wrapper import DINO
from train_dino import train_dino


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--train_config', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_from', type=str, default='')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: Running on CPU - training will be very slow!")
    
    # Dataset
    transform = MultiCropTransform(
        global_crops_scale=tuple(data_cfg['global_crops_scale']),
        local_crops_scale=tuple(data_cfg['local_crops_scale']),
        local_crops_number=data_cfg['local_crops_number'],
        image_size=data_cfg['image_size']
    )
    dataset = PretrainDataset(transform=transform)
    # Setup DataLoader with safe defaults
    num_workers = data_cfg['num_workers']
    persistent_workers = data_cfg.get('persistent_workers', False) and num_workers > 0
    prefetch_factor = data_cfg.get('prefetch_factor', 2) if num_workers > 0 else None
    
    print(f"DataLoader settings: {num_workers} workers, persistent={persistent_workers}, prefetch={prefetch_factor}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=data_cfg['pin_memory'],
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    # Test data loading with first batch
    print("Testing data loading...")
    try:
        test_batch = next(iter(dataloader))
        print(f"✓ Data loading works! Batch has {len(test_batch)} crops")
    except Exception as e:
        print(f"⚠️  Data loading test failed: {e}")
        print("   Try reducing num_workers or setting persistent_workers=false")
    
    # Model
    backbone = build_vit(
        model_name=model_cfg['model_name'],
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        drop_path_rate=model_cfg['drop_path_rate']
    )
    model = DINO(
        backbone,
        out_dim=train_cfg['out_dim'],
        use_cls_token=model_cfg['use_cls_token']
    )
    model = model.to(device)
    
    # Convert to channels_last memory format for better performance
    if train_cfg.get('use_channels_last', False) and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("✓ Model converted to channels_last format")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # Train with optimizations
    num_global = 2
    num_local = data_cfg['local_crops_number']
    
    train_dino(
        model=model,
        train_loader=dataloader,
        num_epochs=train_cfg['num_epochs'],
        device=device,
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        warmup_epochs=train_cfg['warmup_epochs'],
        teacher_temp=train_cfg['teacher_temp'],
        student_temp=train_cfg['student_temp'],
        warmup_teacher_temp=train_cfg['warmup_teacher_temp'],
        warmup_teacher_temp_epochs=train_cfg['warmup_teacher_temp_epochs'],
        checkpoint_dir=train_cfg['checkpoint_dir'],
        resume_from=args.resume_from if args.resume_from else None,
        use_torch_compile=train_cfg.get('use_torch_compile', True),
        use_fused_adamw=train_cfg.get('use_fused_adamw', True),
        num_global=num_global,
        num_local=num_local,
        save_freq=train_cfg.get('save_freq', 10)
    )


if __name__ == '__main__':
    main()

