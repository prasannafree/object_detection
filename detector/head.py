"""
FCOS Detection Head (Anchor-Free).

Paper: "FCOS: Fully Convolutional One-Stage Object Detection" (ICCV 2019)

For each pixel on each FPN level, the head predicts:
  - Classification: which of the 80 classes (or background)
  - Regression:     (l, t, r, b) distances to the 4 box edges
  - Centerness:     how close this pixel is to the object center (0-1)

Architecture per FPN level:
    P_i ──→ 4× (Conv 3x3 + GroupNorm + ReLU) ──→ shared features
                  │              │              │
              cls conv       reg conv       ctr conv
              (80)            (4)            (1)

All FPN levels share the same head weights (parameter sharing),
but each level gets its own learnable scale factor for regression.
"""

import torch
import torch.nn as nn
import math


class ScaleExp(nn.Module):
    """
    Learnable per-level scale factor for regression outputs.

    The regression branch predicts (l, t, r, b) distances. Different FPN levels
    detect objects of different sizes, so the predicted distances vary wildly
    in magnitude. A per-level learnable scale + exp() ensures:
      1. The network always predicts positive distances (exp is always > 0)
      2. Each level can learn its own natural scale
    """

    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


class FCOSHead(nn.Module):
    """
    FCOS-style detection head with shared convolutions across FPN levels.

    Input:  List of FPN features [P3, P4, P5], each (B, in_channels, H_i, W_i)
    Output: For each level:
              cls_logits (B, num_classes, H_i, W_i) — raw class scores
              reg_preds  (B, 4, H_i, W_i)          — (l, t, r, b) distances
              ctr_logits (B, 1, H_i, W_i)          — centerness score
    """

    def __init__(self, in_channels=128, num_classes=80, num_convs=4,
                 num_levels=3):
        super(FCOSHead, self).__init__()
        self.num_classes = num_classes

        # Shared classification tower: 4× Conv + GN + ReLU
        cls_tower = []
        for _ in range(num_convs):
            cls_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
        self.cls_tower = nn.Sequential(*cls_tower)

        # Shared regression tower: 4× Conv + GN + ReLU
        reg_tower = []
        for _ in range(num_convs):
            reg_tower.append(
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))
        self.reg_tower = nn.Sequential(*reg_tower)

        # Final prediction layers
        self.cls_logits = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.ctr_logits = nn.Conv2d(in_channels, 1, 3, padding=1)

        # Per-level scale factors for regression
        self.scales = nn.ModuleList(
            [ScaleExp(init_value=1.0) for _ in range(num_levels)])

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following FCOS paper recommendations."""
        for module in [self.cls_tower, self.reg_tower]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, std=0.01)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Classification bias: initialize to -log((1-pi)/pi) where pi=0.01
        # This prevents the model from predicting everything as foreground
        # at the start of training (when weights are random)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)
        nn.init.normal_(self.cls_logits.weight, std=0.01)

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)
        nn.init.normal_(self.ctr_logits.weight, std=0.01)
        nn.init.zeros_(self.ctr_logits.bias)

    def forward(self, features):
        """
        Args:
            features: List of FPN feature maps [P3, P4, P5]

        Returns:
            cls_outputs: List of (B, num_classes, H_i, W_i) per level
            reg_outputs: List of (B, 4, H_i, W_i) per level
            ctr_outputs: List of (B, 1, H_i, W_i) per level
        """
        cls_outputs = []
        reg_outputs = []
        ctr_outputs = []

        for i, feat in enumerate(features):
            # Classification branch
            cls_feat = self.cls_tower(feat)
            cls_out = self.cls_logits(cls_feat)

            # Regression branch
            reg_feat = self.reg_tower(feat)
            reg_out = self.scales[i](self.reg_pred(reg_feat))

            # Centerness branch (shares features with classification)
            ctr_out = self.ctr_logits(cls_feat)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            ctr_outputs.append(ctr_out)

        return cls_outputs, reg_outputs, ctr_outputs
