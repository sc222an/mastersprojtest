from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models

class FrameMeanPoolBaseline(nn.Module):
    """
    Apply 2D CNN to each frame, mean-pool features over time, then classify.
    Input: (B, T, C, H, W)
    Output: logits (B,)
    """
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = net.fc.in_features
            net.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = net
        self.classifier = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feats = self.backbone(x)              # (B*T, D)
        feats = feats.view(B, T, -1).mean(1)  # (B, D)
        logits = self.classifier(feats).squeeze(1)  # (B,)
        return logits
