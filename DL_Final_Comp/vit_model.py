import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

def build_vit(model_name="vit_small_patch16_224", 
              img_size=96,
              patch_size=16,
              num_classes=0,  # 0 = no classification head
              drop_path_rate=0.1):
    """
    Build ViT-S/16 or ViT-B/16
    
    Options:
    - vit_small_patch16_224 (ViT-S/16, ~22M params)
    - vit_base_patch16_224 (ViT-B/16, ~86M params)
    """
    if "small" in model_name:
        embed_dim = 384
        depth = 12
        num_heads = 6
    elif "base" in model_name:
        embed_dim = 768
        depth = 12
        num_heads = 12
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
        global_pool='',
    )
    
    return model

