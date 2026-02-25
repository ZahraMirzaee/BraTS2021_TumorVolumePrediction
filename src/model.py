# This module defines the model architecture using nnU-Net as backbone with a scalar head for volume prediction.

import torch
import torch.nn as nn

class NNUNetScalarHead(nn.Module):
    def __init__(self, backbone: nn.Module, n_backbone_channels: int, hidden: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_backbone_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        feat_summed = feat.sum(dim=(-3, -2, -1))
        return self.head(feat_summed)

def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad