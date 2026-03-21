from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models.video as video_models

class R3D18Baseline(nn.Module):
    """
    3D ResNet-18 baseline.
    Input: (B,C,T,H,W)
    Output: logits (B,)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)                # (B, D)
        logits = self.classifier(feats).squeeze(1)
        return logits
