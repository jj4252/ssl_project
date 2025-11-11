import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from optimizer import cosine_schedule

def dino_loss(student_outputs, teacher_outputs, center, 
              teacher_temp=0.04, student_temp=0.1, 
              num_global=2, num_local=2):
    """
    DINO loss with restricted pairings: student only matches teacher global+local
    Avoids expensive local-to-local comparisons for speed.
    
    Args:
        student_outputs: List of student outputs [global1, global2, local1, local2, ...]
        teacher_outputs: List of teacher outputs [global1, global2, local1, local2, ...]
        num_global: Number of global crops (default 2)
        num_local: Number of local crops (default 2)
    """
    total_loss = 0
    
    # Teacher: average over global + local views only (not all crops)
    # This reduces computation by avoiding local-to-local comparisons
    teacher_views = teacher_outputs[:num_global + num_local]  # Only global + local
    teacher_out = torch.stack(teacher_views)
    teacher_out = teacher_out.detach()
    teacher_out = torch.softmax((teacher_out - center) / teacher_temp, dim=-1)
    teacher_out = teacher_out.mean(dim=0)  # Average over teacher views
    
    # Student: match all student crops against teacher global+local average
    for student_out in student_outputs:
        student_out = torch.softmax(student_out / student_temp, dim=-1)
        loss = -torch.sum(teacher_out * torch.log(student_out + 1e-10), dim=-1)
        total_loss += loss.mean()
    
    return total_loss / len(student_outputs)


def train_epoch(model, dataloader, optimizer, scheduler, 
                center, device, scaler, epoch, num_epochs,
                teacher_temp=0.04, student_temp=0.1,
                warmup_teacher_temp=0.04, warmup_teacher_temp_epochs=30,
                num_global=2, num_local=2):
    model.train()
    total_loss = 0
    
    # Temperature schedule
    if epoch < warmup_teacher_temp_epochs:
        current_teacher_temp = warmup_teacher_temp + (
            teacher_temp - warmup_teacher_temp
        ) * epoch / warmup_teacher_temp_epochs
    else:
        current_teacher_temp = teacher_temp
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    # Get initial LR for display
    initial_lr = optimizer.param_groups[0]['lr']
    
    import time
    batch_times = []
    data_times = []
    
    # Track iteration start time (data loading happens in background)
    prev_iter_time = time.time()
    
    for batch_idx, crops in enumerate(progress_bar):
        iter_start = time.time()
        
        # Data loading time = time since last iteration (includes prefetch wait)
        if batch_idx > 0:
            data_load_time = iter_start - prev_iter_time
        else:
            data_load_time = 0  # First batch already loaded
        
        batch_start = time.time()
        # crops: list of tensors [batch_size, 3, 96, 96]
        # Convert to channels_last for better performance (if supported)
        try:
            crops = [c.to(device, memory_format=torch.channels_last) for c in crops]
        except:
            # Fallback if channels_last not supported
            crops = [c.to(device) for c in crops]
        
        optimizer.zero_grad()
        
        # Use BF16 autocast for better performance on modern GPUs
        device_type = 'cuda' if device.type == 'cuda' else 'cpu'
        dtype = torch.bfloat16 if device_type == 'cuda' else torch.float32
        with torch.amp.autocast(device_type=device_type, dtype=dtype):
            # Student forward
            student_outputs = model(crops, is_teacher=False)
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = model(crops, is_teacher=True)
            
            # Loss with restricted pairings
            loss = dino_loss(student_outputs, teacher_outputs, center,
                           teacher_temp=current_teacher_temp,
                           student_temp=student_temp,
                           num_global=num_global,
                           num_local=num_local)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Note: scheduler.step() moved to after epoch to step per epoch, not per batch
        
        # Update teacher with momentum (only once per epoch, not per batch)
        if batch_idx == 0:  # Only update once per epoch
            momentum = cosine_schedule(epoch, max_epochs=num_epochs, 
                                      base_value=0.996, final_value=1.0)
            model.update_teacher(momentum)
        
        # Update center
        with torch.no_grad():
            teacher_out = torch.stack(teacher_outputs)
            center = 0.9 * center + 0.1 * teacher_out.mean(dim=0).mean(dim=0)
        
        total_loss += loss.item()
        
        # Track times
        batch_time = time.time() - batch_start
        total_time = time.time() - iter_start
        
        batch_times.append(batch_time)
        data_times.append(data_load_time)
        if len(batch_times) > 10:
            batch_times.pop(0)
            data_times.pop(0)
        
        avg_batch_time = sum(batch_times) / len(batch_times)
        if len(data_times) > 0:
            avg_data_time = sum(data_times) / len(data_times)
        else:
            avg_data_time = 0
        avg_total_time = avg_batch_time + avg_data_time
        
        # Update for next iteration
        prev_iter_time = time.time()
        
        # Get current LR from optimizer
        current_lr = optimizer.param_groups[0]['lr']
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'lr': f'{current_lr:.6f}',
            'gpu': f'{avg_batch_time:.2f}s',
            'data': f'{avg_data_time:.2f}s',
            'total': f'{avg_total_time:.2f}s'
        })
    
    # Step scheduler at END of epoch (after all optimizer steps)
    scheduler.step()
    
    return total_loss / len(dataloader), center


def train_dino(model, train_loader, num_epochs, device, 
               lr=0.0005, weight_decay=0.04, warmup_epochs=5,
               teacher_temp=0.04, student_temp=0.1,
               warmup_teacher_temp=0.04, warmup_teacher_temp_epochs=30,
               checkpoint_dir=None, resume_from=None,
               use_torch_compile=True, use_fused_adamw=True,
               num_global=2, num_local=2, save_freq=10):
    """Main training function with performance optimizations"""
    from optimizer import build_optimizer, build_scheduler
    
    # Enable TF32 for faster training on Ampere+ GPUs
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Initialize center BEFORE compiling (need to access model attributes)
    out_dim = model.student_head.last_layer.weight.shape[0]
    center = torch.zeros(out_dim, device=device)
    
    # Compile model for faster execution (PyTorch 2.0+)
    # NOTE: First compilation can take 5-10 minutes, but subsequent runs are fast
    compiled_model = None
    if use_torch_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        print("⚠️  First compilation may take 5-10 minutes - this is normal!")
        print("    Subsequent runs will be much faster.")
        compiled_model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled successfully")
    
    # Use fused AdamW for better performance
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, 
                               fused=use_fused_adamw)
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs, 
                               warmup_epochs=warmup_epochs)
    
    # Use BF16 scaler for better performance
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from)
        # Load state dict before compiling (compiled models can't load directly)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        center = checkpoint['center']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Use compiled model if available (after loading checkpoint)
    if compiled_model is not None:
        model = compiled_model
    
    # Initialize scheduler to get correct LR for first epoch
    # LambdaLR starts at epoch 0, so we need to step it once to get epoch 0's LR
    if start_epoch == 0:
        # Step scheduler once to initialize LR for epoch 0
        scheduler.step()
    
    for epoch in range(start_epoch, num_epochs):
        avg_loss, center = train_epoch(
            model, train_loader, optimizer, scheduler, center, device, scaler,
            epoch, num_epochs, teacher_temp, student_temp,
            warmup_teacher_temp, warmup_teacher_temp_epochs,
            num_global=num_global, num_local=num_local
        )
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint (every save_freq epochs, or always save latest)
        if checkpoint_dir:
            import os
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'center': center,
                'epoch': epoch,
            }
            # Always save latest checkpoint
            torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_latest.pth")
            
            # Save numbered checkpoint every save_freq epochs or at the last epoch
            if (epoch + 1) % save_freq == 0 or (epoch + 1) == num_epochs:
                torch.save(checkpoint, 
                          f"{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth")
                print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
    
    return model

