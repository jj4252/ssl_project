import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_features(model, dataloader, device, use_cls_token=True):
    """
    Extract features from frozen encoder
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
                feat = outputs[:, 0]  # CLS token
            else:
                feat = outputs[:, 1:].mean(dim=1)  # Mean-pool patches
            
            feat = nn.functional.normalize(feat, dim=-1, p=2)
            
            features.append(feat.cpu())
            labels.append(targets)
    
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels

