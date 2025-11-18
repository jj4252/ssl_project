"""
VICReg Trainer for Self-Supervised Learning

Implements VICReg (Bardes et al., 2022) with ViT-S/16 backbone.
Uses invariance, variance, and covariance terms for SSL.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import os
import time
from tqdm import tqdm
import timm

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

from data_loader import build_pretraining_dataloader
from optimizer import build_optimizer, build_scheduler


class VICRegViT(nn.Module):
    """
    VICReg model with ViT-S/16 backbone and projection head.
    
    Architecture:
    - ViT-S/16 encoder (CLS token output)
    - 3-layer MLP projector: 384 -> 2048 -> 2048 -> 2048
    """
    def __init__(
        self,
        backbone_name: str = "vit_small_patch16_224",
        image_size: int = 96,
        proj_dim: int = 2048,
        proj_hidden_dim: int = 2048,
    ):
        super().__init__()
        
        # ViT-S/16 encoder
        self.encoder = timm.create_model(
            backbone_name,
            img_size=image_size,
            num_classes=0,  # We use CLS features, no classifier
            pretrained=False,  # Start from scratch
            global_pool="",  # Don't pool, return all tokens
        )
        
        # Get embedding dimension from encoder
        embed_dim = self.encoder.num_features
        
        # Projection head: 3-layer MLP with LayerNorm (BatchNorm causes issues in SSL)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.LayerNorm(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_dim),
        )
        
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
    
    def forward(self, x):
        """
        Forward pass: encoder -> CLS token -> projector
        
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Projected embeddings [B, proj_dim]
        """
        # Extract CLS token features
        features = self.encoder.forward_features(x)
        if isinstance(features, torch.Tensor):
            # If features is a tensor, CLS is first token
            cls_features = features[:, 0]  # [B, embed_dim]
        else:
            # If features is a dict, extract CLS token
            cls_features = features.get('x', features.get('tokens', None))[:, 0]
        
        # Project to higher dimension
        z = self.proj(cls_features)  # [B, proj_dim]
        
        return z


def invariance_loss(z1, z2):
    """
    Invariance term: MSE between two views of the same image.
    
    Args:
        z1, z2: Projected embeddings [B, D]
    
    Returns:
        Scalar loss
    """
    return torch.mean((z1 - z2) ** 2)


def variance_loss(z, gamma=1.0, eps=1e-4):
    """
    Variance term: penalize dimensions with std < gamma.
    
    Args:
        z: Projected embeddings [B, D]
        gamma: Target standard deviation (default: 1.0)
        eps: Small epsilon for numerical stability
    
    Returns:
        Scalar loss
    """
    # Compute per-dimension std over batch
    std = torch.sqrt(z.var(dim=0) + eps)  # [D]
    # Penalty for dimensions with std < gamma
    penalty = torch.relu(gamma - std)
    return penalty.mean()


def covariance_loss(z, eps=1e-4):
    """
    Covariance term: penalize off-diagonal elements of covariance matrix.
    
    Args:
        z: Projected embeddings [B, D]
        eps: Small epsilon for numerical stability
    
    Returns:
        Scalar loss
    """
    B, D = z.shape
    
    # Mean-center z
    z = z - z.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix: [D, D]
    # cov = (z.T @ z) / (B - 1)
    cov = (z.T @ z) / (B - 1 + eps)
    
    # Extract off-diagonal elements
    # Create mask for off-diagonal elements
    mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
    off_diag = cov[mask]
    
    # Return mean squared off-diagonal elements
    return (off_diag ** 2).mean()


def compute_vicreg_loss(z1, z2, lambda_inv=25.0, mu_var=25.0, nu_cov=1.0, gamma=1.0):
    """
    Compute total VICReg loss.
    
    Args:
        z1, z2: Projected embeddings [B, D]
        lambda_inv: Weight for invariance term
        mu_var: Weight for variance term
        nu_cov: Weight for covariance term
        gamma: Target standard deviation for variance term
    
    Returns:
        Dictionary with total loss and individual components
    """
    # Invariance term
    inv_loss = invariance_loss(z1, z2)
    
    # Variance term (average over both views)
    var_loss_z1 = variance_loss(z1, gamma=gamma)
    var_loss_z2 = variance_loss(z2, gamma=gamma)
    var_loss = 0.5 * (var_loss_z1 + var_loss_z2)
    
    # Covariance term (average over both views)
    cov_loss_z1 = covariance_loss(z1)
    cov_loss_z2 = covariance_loss(z2)
    cov_loss = 0.5 * (cov_loss_z1 + cov_loss_z2)
    
    # Total loss
    total_loss = (
        lambda_inv * inv_loss +
        mu_var * var_loss +
        nu_cov * cov_loss
    )
    
    return {
        'total': total_loss,
        'invariance': inv_loss.item(),
        'variance': var_loss.item(),
        'covariance': cov_loss.item(),
    }


def train_vicreg(
    model,
    train_loader,
    num_epochs,
    device,
    lr=1e-3,
    weight_decay=1e-4,
    warmup_epochs=10,
    checkpoint_dir=None,
    resume_from=None,
    compile_model=True,
    use_fused_adamw=True,
    save_every=0,
    lambda_inv=25.0,
    mu_var=25.0,
    nu_cov=1.0,
    gamma=1.0,
    max_grad_norm=1.0,
):
    """
    Train VICReg model.
    
    Args:
        model: VICRegViT model
        train_loader: DataLoader returning (view1, view2) tuples
        num_epochs: Number of training epochs
        device: Device to train on
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs for learning rate
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
        compile_model: Whether to compile model with torch.compile
        use_fused_adamw: Whether to use fused AdamW
        save_every: Save checkpoint every N steps (0 = only at end of epoch)
        lambda_inv: Weight for invariance term
        mu_var: Weight for variance term
        nu_cov: Weight for covariance term
        gamma: Target standard deviation for variance term
        max_grad_norm: Maximum gradient norm for clipping
    """
    # Enable TF32 for faster training
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Compile model if requested
    if compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        print("⚠️  First compilation may take 5-10 minutes - this is normal!")
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled successfully")
    
    # Build optimizer
    params_with_wd = []
    params_without_wd = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "norm", "ln", "bn"]):
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
    
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=warmup_epochs)
    
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
        model.load_state_dict(checkpoint['model'])
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
        model.train()
        total_loss = 0.0
        total_inv = 0.0
        total_var = 0.0
        total_cov = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Data loading verification (first batch of first epoch)
        if epoch == start_epoch:
            # Get first batch for verification
            first_batch_iter = iter(train_loader)
            first_batch = next(first_batch_iter)
            if isinstance(first_batch, (tuple, list)) and len(first_batch) == 2:
                x1_check = first_batch[0][:4].to(device)
                x2_check = first_batch[1][:4].to(device)
                pixel_diff = (x1_check - x2_check).abs().mean().item()
                print("\n🔍 Data Loading Verification:")
                print(f"  x1 shape: {x1_check.shape}, x2 shape: {x2_check.shape}")
                print(f"  x1 mean: {x1_check.mean():.4f}, x2 mean: {x2_check.mean():.4f}")
                print(f"  x1-x2 pixel diff: {pixel_diff:.6f} (should be >0.1 for different views)")
                if pixel_diff < 0.01:
                    print(f"  ⚠️  CRITICAL: Views are nearly identical! Check augmentations!")
                elif pixel_diff < 0.05:
                    print(f"  ⚠️  WARNING: Views are very similar - augmentations may be too weak")
                else:
                    print(f"  ✓ Views are sufficiently different")
            # Recreate iterator for actual training
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Batch should be (view1, view2) tuple
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x1 = batch[0].to(device, non_blocking=True)
                x2 = batch[1].to(device, non_blocking=True)
            else:
                raise ValueError(f"Expected batch to be tuple/list of 2 views, got {type(batch)}")
            
            # Convert to channels_last if supported
            try:
                x1 = x1.to(memory_format=torch.channels_last)
                x2 = x2.to(memory_format=torch.channels_last)
            except:
                pass
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if device.type == 'cuda':
                if AUTOCAST_NEW_API:
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        z1 = model(x1)
                        z2 = model(x2)
                        loss_dict = compute_vicreg_loss(
                            z1, z2,
                            lambda_inv=lambda_inv,
                            mu_var=mu_var,
                            nu_cov=nu_cov,
                            gamma=gamma
                        )
                        loss = loss_dict['total']
                else:
                    with autocast():
                        z1 = model(x1)
                        z2 = model(x2)
                        loss_dict = compute_vicreg_loss(
                            z1, z2,
                            lambda_inv=lambda_inv,
                            mu_var=mu_var,
                            nu_cov=nu_cov,
                            gamma=gamma
                        )
                        loss = loss_dict['total']
            else:
                z1 = model(x1)
                z2 = model(x2)
                loss_dict = compute_vicreg_loss(
                    z1, z2,
                    lambda_inv=lambda_inv,
                    mu_var=mu_var,
                    nu_cov=nu_cov,
                    gamma=gamma
                )
                loss = loss_dict['total']
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping and monitoring
            if max_grad_norm > 0 or (batch_idx % 50 == 0) or (epoch < 5 and batch_idx == 0):
                scaler.unscale_(optimizer)
                
                # Gradient monitoring (every 50 batches or first batch of first 5 epochs)
                if (batch_idx % 50 == 0) or (epoch < 5 and batch_idx == 0):
                    total_grad_norm = 0.0
                    param_count = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                            param_count += 1
                    if param_count > 0:
                        total_grad_norm = total_grad_norm ** (1. / 2)
                        if total_grad_norm < 1e-6:
                            print(f"  ⚠️  WARNING: Very small gradient norm ({total_grad_norm:.8f}) - model may not be learning!")
                        elif total_grad_norm < 0.001:
                            print(f"  ⚠️  WARNING: Low gradient norm ({total_grad_norm:.6f}) - check learning rate")
                        else:
                            print(f"  ✓ Gradient norm: {total_grad_norm:.6f}")
                
                # Gradient clipping
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_inv += loss_dict['invariance']
            total_var += loss_dict['variance']
            total_cov += loss_dict['covariance']
            num_batches += 1
            global_step += 1
            
            # Enhanced loss logging (every 10 batches for first epoch, every 50 otherwise)
            if (epoch < 1 and batch_idx % 10 == 0) or (batch_idx % 50 == 0):
                print(f"\n  Loss breakdown (step {global_step}):")
                print(f"    Invariance: {loss_dict['invariance']:.6f} (weighted: {lambda_inv * loss_dict['invariance']:.4f})")
                print(f"    Variance: {loss_dict['variance']:.6f} (weighted: {mu_var * loss_dict['variance']:.4f})")
                print(f"    Covariance: {loss_dict['covariance']:.6f} (weighted: {nu_cov * loss_dict['covariance']:.4f})")
                print(f"    Total: {loss.item():.6f}")
            
            # Periodic diagnostics (every 50 batches or first batch of first 5 epochs)
            if (batch_idx % 50 == 0) or (epoch < 5 and batch_idx == 0):
                with torch.no_grad():
                    # Compute diagnostics on z1
                    z1_np = z1.detach()
                    
                    # Per-dimension std
                    std_per_dim = torch.sqrt(z1_np.var(dim=0) + 1e-4)  # [D]
                    mean_std = std_per_dim.mean().item()
                    min_std = std_per_dim.min().item()
                    max_std = std_per_dim.max().item()
                    
                    # Pairwise cosine similarity (sample subset if batch is large)
                    B = z1_np.shape[0]
                    sample_size = min(512, B)
                    if B > sample_size:
                        indices = torch.randperm(B, device=device)[:sample_size]
                        z1_sample = z1_np[indices]
                    else:
                        z1_sample = z1_np
                    
                    # Normalize for cosine similarity
                    z1_norm = F.normalize(z1_sample, dim=-1)
                    pairwise_sim = (z1_norm @ z1_norm.T).mean().item()
                    
                    # Approximate covariance magnitude
                    z1_centered = z1_np - z1_np.mean(dim=0, keepdim=True)
                    cov_approx = (z1_centered.T @ z1_centered) / (B - 1 + 1e-4)
                    off_diag_mask = ~torch.eye(cov_approx.shape[0], dtype=torch.bool, device=device)
                    cov_magnitude = (cov_approx[off_diag_mask] ** 2).mean().item()
                    
                    print(f"\n🔍 VICReg Diagnostics (epoch {epoch+1}, batch {batch_idx}):")
                    print(f"  Total loss: {loss.item():.6f}")
                    print(f"  Invariance: {loss_dict['invariance']:.6f}")
                    print(f"  Variance: {loss_dict['variance']:.6f}")
                    print(f"  Covariance: {loss_dict['covariance']:.6f}")
                    print(f"  Per-dim std: mean={mean_std:.6f}, min={min_std:.6f}, max={max_std:.6f}")
                    print(f"  Pairwise cosine sim: {pairwise_sim:.4f} (should be 0.1-0.5, not drift to 1.0)")
                    print(f"  Covariance magnitude: {cov_magnitude:.6f}")
                    
                    # Warnings
                    if min_std < 0.01:
                        print(f"  ⚠️  CRITICAL: Very low min std ({min_std:.6f}) - risk of collapse!")
                    elif min_std < 0.1:
                        print(f"  ⚠️  WARNING: Low min std ({min_std:.6f}) - monitor for collapse")
                    
                    if pairwise_sim > 0.9:
                        print(f"  ⚠️  CRITICAL: Features collapsing! (pairwise sim >0.9)")
                    elif pairwise_sim > 0.7:
                        print(f"  ⚠️  WARNING: Features may be collapsing (pairwise sim >0.7)")
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'inv': f'{loss_dict["invariance"]:.4f}',
                'var': f'{loss_dict["variance"]:.4f}',
                'cov': f'{loss_dict["covariance"]:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # Step-based checkpointing
            if save_every > 0 and global_step % save_every == 0 and checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                }
                torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pth'))
        
        # End of epoch checkpoint
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_latest.pth'))
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        avg_inv = total_inv / num_batches
        avg_var = total_var / num_batches
        avg_cov = total_cov / num_batches
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Avg loss: {avg_loss:.6f}")
        print(f"  Avg invariance: {avg_inv:.6f}")
        print(f"  Avg variance: {avg_var:.6f}")
        print(f"  Avg covariance: {avg_cov:.6f}")
        print(f"  Learning rate: {optimizer.param_groups[0]['lr']:.6f}")


def load_config(config_path):
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_vicreg_training(data_cfg, train_cfg, model_cfg, device, args):
    """Run VICReg self-supervised learning training"""
    print("\n" + "="*60)
    print("VICReg Self-Supervised Learning Training")
    print("="*60)
    
    # Get VICReg config
    vicreg_cfg = train_cfg.get('vicreg', {})
    lambda_inv = vicreg_cfg.get('lambda_invariance', vicreg_cfg.get('inv_weight', 25.0))
    mu_var = vicreg_cfg.get('mu_variance', vicreg_cfg.get('var_weight', 25.0))
    nu_cov = vicreg_cfg.get('nu_covariance', vicreg_cfg.get('cov_weight', 1.0))
    gamma = vicreg_cfg.get('gamma', 1.0)
    proj_dim = vicreg_cfg.get('proj_dim', 2048)
    proj_hidden_dim = vicreg_cfg.get('proj_hidden_dim', 2048)
    
    print(f"VICReg Configuration:")
    print(f"  Lambda (invariance): {lambda_inv}")
    print(f"  Mu (variance): {mu_var}")
    print(f"  Nu (covariance): {nu_cov}")
    print(f"  Gamma (target std): {gamma}")
    print(f"  Projection dim: {proj_dim}")
    print(f"  Projection hidden dim: {proj_hidden_dim}")
    
    # Build model
    backbone_name = model_cfg.get('backbone_name', 'vit_small_patch16_224')
    image_size = data_cfg.get('image_size', 96)
    
    print(f"\nBuilding VICReg model...")
    print(f"  Backbone: {backbone_name}")
    print(f"  Image size: {image_size}x{image_size}")
    
    model = VICRegViT(
        backbone_name=backbone_name,
        image_size=image_size,
        proj_dim=proj_dim,
        proj_hidden_dim=proj_hidden_dim,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Build DataLoader (should return two views per image)
    print("\nBuilding DataLoader...")
    dataloader = build_pretraining_dataloader(data_cfg, train_cfg)
    
    # Checkpoint directory
    checkpoint_dir = train_cfg.get('checkpoint_dir', './checkpoints')
    checkpoint_dir = os.path.expandvars(checkpoint_dir)
    save_every = train_cfg.get('save_every', 0)  # 0 = only at end of epoch
    
    # Auto-detect latest checkpoint if resume_from not provided
    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        # Try to find latest checkpoint
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
            if os.path.exists(latest_path):
                resume_from = latest_path
                print(f"✓ Auto-detected latest checkpoint: {resume_from}")
    
    # Training hyperparameters
    num_epochs = train_cfg.get('num_epochs', 200)
    lr = train_cfg.get('learning_rate', 1e-3)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    warmup_epochs = train_cfg.get('warmup_epochs', 10)
    compile_model = train_cfg.get('compile_model', True)
    use_fused_adamw = train_cfg.get('use_fused_adamw', True)
    max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
    
    # Train
    train_vicreg(
        model=model,
        train_loader=dataloader,
        num_epochs=num_epochs,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        compile_model=compile_model,
        use_fused_adamw=use_fused_adamw,
        save_every=save_every,
        lambda_inv=lambda_inv,
        mu_var=mu_var,
        nu_cov=nu_cov,
        gamma=gamma,
        max_grad_norm=max_grad_norm,
    )


def main():
    parser = argparse.ArgumentParser(description="VICReg SSL Training")
    parser.add_argument('--data_config', type=str, required=True,
                        help='Path to data config YAML')
    parser.add_argument('--train_config', type=str, required=True,
                        help='Path to training config YAML')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no_resume', action='store_true',
                        help='Disable auto-resume from latest checkpoint')
    args = parser.parse_args()
    
    # Load configs
    data_cfg = load_config(args.data_config)
    train_cfg = load_config(args.train_config)
    model_cfg = load_config(args.model_config)
    
    # Get training mode (default to "vicreg" for this script, but support others)
    training_mode = train_cfg.get('training_mode', 'vicreg')
    print(f"✓ Training mode: {training_mode}")
    
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
    if training_mode == "vicreg":
        run_vicreg_training(data_cfg, train_cfg, model_cfg, device, args)
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}. Use 'vicreg'")
    
    print("✓ Training completed!")


if __name__ == '__main__':
    main()

