import torch
import torch.nn as nn
from copy import deepcopy

class DINOHead(nn.Module):
    """Projection head for DINO"""
    def __init__(self, in_dim, out_dim=32768, hidden_dim=1536, 
                 bottleneck_dim=256, n_layers=3):
        super().__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        # Final layer
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINO(nn.Module):
    """DINO teacher-student wrapper"""
    def __init__(self, backbone, out_dim=65536, use_cls_token=True):
        super().__init__()
        embed_dim = backbone.embed_dim
        self.use_cls_token = use_cls_token
        
        # Student
        self.student = backbone
        self.student_head = DINOHead(embed_dim, out_dim)
        
        # Teacher (EMA)
        self.teacher = deepcopy(backbone)
        self.teacher_head = DINOHead(embed_dim, out_dim)
        
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False
    
    def forward(self, x, is_teacher=False):
        """x: list of crops - batch all crops together for efficiency"""
        if is_teacher:
            backbone = self.teacher
            head = self.teacher_head
        else:
            backbone = self.student
            head = self.student_head
        
        # Batch all crops together for much better GPU utilization
        # x is a list of [batch_size, 3, H, W] tensors
        # Concatenate all crops: [batch_size * num_crops, 3, H, W]
        all_crops = torch.cat(x, dim=0)
        
        # Single forward pass for all crops
        features = backbone.forward_features(all_crops)
        
        # Use CLS token or mean-pool
        if self.use_cls_token:
            cls_tokens = features[:, 0]  # CLS token
        else:
            cls_tokens = features[:, 1:].mean(dim=1)  # Mean-pool patches
        
        # Project through head
        all_outputs = head(cls_tokens)
        
        # Split back into per-crop outputs
        batch_size = x[0].shape[0]
        num_crops = len(x)
        outputs = []
        for i in range(num_crops):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            outputs.append(all_outputs[start_idx:end_idx])
        
        return outputs
    
    @torch.no_grad()
    def update_teacher(self, momentum):
        """EMA update of teacher"""
        for student_param, teacher_param in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            teacher_param.data = (
                momentum * teacher_param.data + 
                (1 - momentum) * student_param.data
            )
        for student_param, teacher_param in zip(
            self.student_head.parameters(), 
            self.teacher_head.parameters()
        ):
            teacher_param.data = (
                momentum * teacher_param.data + 
                (1 - momentum) * student_param.data
            )
    
    def get_backbone(self):
        """Return the student backbone for feature extraction"""
        return self.student

