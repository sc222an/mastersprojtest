from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as models


class FrameLSTM(nn.Module):
    """
    Frame-based model: 2D backbone per frame + LSTM temporal pooling.
    Input: (B, T, C, H, W)
    Output: logits (B,)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        lstm_hidden_size: int = 512,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = False,
        lstm_dropout: float = 0.0,
    ):
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
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )

        proj_dim = lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.classifier = nn.Linear(proj_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)

        feats = self.backbone(x)  # (B*T, D)
        feats = feats.view(B, T, -1)  # (B, T, D)

        lstm_out, _ = self.lstm(feats)  # (B, T, H*dirs)
        last = lstm_out[:, -1, :]  # (B, H*dirs)

        logits = self.classifier(last).squeeze(1)  # (B,)
        return logits
