import torch
import torch.nn as nn
from torchvision import models


class SkinScoringModel(nn.Module):
    """EfficientNet-B0 based model for skin scoring with 5 output values."""

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Get the input features of the original classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier head with 5 outputs
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, 5),
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.sigmoid(x)
        return x
