import torch
import math

def build_optimizer(model, lr=0.0005, weight_decay=0.04, fused=True):
    """Build AdamW optimizer with optional fused implementation"""
    # Exclude bias and normalization from weight decay
    params_groups = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in ["bias", "norm", "ln"])],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in ["bias", "norm", "ln"])],
            "weight_decay": 0.0,
        },
    ]
    # Use fused AdamW for better performance (requires CUDA)
    if fused and torch.cuda.is_available():
        try:
            optimizer = torch.optim.AdamW(params_groups, lr=lr, fused=True)
        except TypeError:
            # Fused not available, fall back to regular
            optimizer = torch.optim.AdamW(params_groups, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params_groups, lr=lr)
    return optimizer


def build_scheduler(optimizer, num_epochs=200, warmup_epochs=10, 
                   min_lr=1e-6, base_lr=None):
    """Build cosine learning rate scheduler with warmup"""
    if base_lr is None:
        base_lr = optimizer.param_groups[0]['lr']
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: use (epoch + 1) to avoid zero LR at epoch 0
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def cosine_schedule(epoch, max_epochs, base_value=0.996, final_value=1.0):
    """Cosine schedule for teacher momentum"""
    progress = epoch / max_epochs
    return final_value - (final_value - base_value) * (1 + math.cos(math.pi * progress)) / 2

