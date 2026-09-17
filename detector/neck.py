"""
Simple FPN (Feature Pyramid Network) Neck.

Paper: "Feature Pyramid Networks for Object Detection" (CVPR 2017)

Takes multi-scale features from backbone (C3, C4, C5) and produces
semantically strong features at all scales (P3, P4, P5) by combining
deep (semantic) features with shallow (spatial) features via a top-down
pathway with lateral connections.

Flow:
    C5 ──→ 1×1 conv ──────────────────────────────→ P5
                │
            upsample 2×
                │
    C4 ──→ 1×1 conv ──→ + ──→ 3×3 conv ──→ P4
                              │
                          upsample 2×
                              │
    C3 ──→ 1×1 conv ──→ + ──→ 3×3 conv ──→ P3

All P-levels have `out_channels` (default 128) channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """
    Simple Feature Pyramid Network.

    Input:  C3 (B, c3, 80, 80), C4 (B, c4, 40, 40), C5 (B, c5, 20, 20)
    Output: P3 (B, 128, 80, 80), P4 (B, 128, 40, 40), P5 (B, 128, 20, 20)
    """

    def __init__(self, in_channels_list, out_channels=128):
        """
        Args:
            in_channels_list: List of channel counts from backbone [c3, c4, c5].
                              e.g., [40, 112, 160] for GhostNet-0.5x.
            out_channels: Number of channels in every P-level output.
        """
        super(SimpleFPN, self).__init__()

        # Lateral connections: 1×1 conv to reduce each C-level to out_channels
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Smoothing layers: 3×3 conv after merging to reduce aliasing
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c3, c4, c5):
        # Top-down pathway
        # Start from the deepest level (C5)
        p5 = self.lateral_c5(c5)

        # Upsample P5 and add to lateral C4
        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode='nearest')
        p4 = self.smooth_p4(p4)

        # Upsample P4 and add to lateral C3
        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode='nearest')
        p3 = self.smooth_p3(p3)

        return [p3, p4, p5]
