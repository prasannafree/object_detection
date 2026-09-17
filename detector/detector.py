"""
Full Lightweight Object Detector.

Assembles Backbone + FPN Neck + FCOS Head into a single model.

Architecture:
    Input (B, 3, 640, 640)
      → GhostNet Backbone → C3, C4, C5
      → Simple FPN         → P3, P4, P5
      → FCOS Head          → cls, reg, ctr per level
"""

import torch
import torch.nn as nn

from detector.backbone import GhostNetBackbone
from detector.neck import SimpleFPN
from detector.head import FCOSHead


class LightDetector(nn.Module):
    """
    Lightweight anchor-free object detector.

    Combines GhostNet-0.5x backbone, simple FPN, and FCOS detection head.
    Total parameters: ~2-3M (lightweight enough for edge deployment).
    """

    def __init__(self, num_classes=80, fpn_channels=128):
        super(LightDetector, self).__init__()

        # Backbone: extracts multi-scale features
        self.backbone = GhostNetBackbone(width_mult=0.5)

        # Neck: combines multi-scale features via top-down pathway
        self.neck = SimpleFPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=fpn_channels
        )

        # Head: per-pixel classification, regression, centerness
        self.head = FCOSHead(
            in_channels=fpn_channels,
            num_classes=num_classes,
            num_convs=4,
            num_levels=3
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) input tensor

        Returns:
            cls_outputs: List of (B, num_classes, H_i, W_i) per level
            reg_outputs: List of (B, 4, H_i, W_i) per level
            ctr_outputs: List of (B, 1, H_i, W_i) per level
        """
        # Extract multi-scale backbone features
        c3, c4, c5 = self.backbone(images)

        # Build feature pyramid
        fpn_features = self.neck(c3, c4, c5)

        # Predict detections at each level
        cls_outputs, reg_outputs, ctr_outputs = self.head(fpn_features)

        return cls_outputs, reg_outputs, ctr_outputs
