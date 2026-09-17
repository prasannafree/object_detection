"""
GhostNet Backbone for Object Detection.

Architecture inspired by:
  - GhostNet (CVPR 2020): "More Features from Cheap Operations"
  - MobileNetV2 (CVPR 2018): Inverted Residual blocks
  - ResNet (CVPR 2015): Skip/residual connections

The backbone extracts multi-scale features (C3, C4, C5) from an input image.
These are fed into an FPN neck for object detection.
"""

import torch
import torch.nn as nn
import math


class GhostModule(nn.Module):
    """
    Ghost Module: generates feature maps using fewer parameters.

    Instead of a full convolution producing `out_channels` features,
    we produce `out_channels // ratio` intrinsic features via standard conv,
    then generate the remaining "ghost" features via cheap depthwise conv.
    Finally, concatenate both to get `out_channels` total.

    Tensor shapes (example with in=16, out=32, ratio=2):
        Input:     (B, 16, H, W)
        Intrinsic: (B, 16, H, W)  via 1x1 conv
        Ghosts:    (B, 16, H, W)  via 3x3 depthwise conv on intrinsic
        Output:    (B, 32, H, W)  via concatenation
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2,
                 dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels

        # Number of intrinsic features (the "real" convolution output)
        init_channels = math.ceil(out_channels / ratio)
        # Number of ghost features to generate cheaply
        new_channels = init_channels * (ratio - 1)

        # Step 1: Standard convolution → intrinsic features
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # Step 2: Cheap depthwise convolution → ghost features
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)       # Intrinsic features
        x2 = self.cheap_operation(x1)    # Ghost features
        out = torch.cat([x1, x2], dim=1) # Concatenate along channel dim
        # Trim to exact out_channels (math.ceil may produce 1 extra)
        return out[:, :self.out_channels, :, :]


class GhostBottleneck(nn.Module):
    """
    Ghost Bottleneck: the residual block of GhostNet.

    Structure (inspired by MobileNetV2 inverted residual):
        Input
          │
          GhostModule (expand: in_ch → mid_ch)
          │
          [If stride=2: Depthwise Conv 3x3 stride=2]
          │
          GhostModule (squeeze: mid_ch → out_ch, no ReLU)
          │
          + ← Residual shortcut (identity if in==out & stride==1,
          │                       else DW conv + 1x1 conv)
          Output

    The expansion ratio is controlled by mid_ch:
        mid_ch = out_ch * expand_ratio (typically 1-6x)
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 dw_kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # First ghost module: expand channels
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # Depthwise convolution for spatial downsampling (only if stride=2)
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dw_conv = nn.Sequential()

        # Second ghost module: squeeze channels (no ReLU — added after residual)
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        # Shortcut connection
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x

        out = self.ghost1(x)
        out = self.dw_conv(out)
        out = self.ghost2(out)

        out = out + self.shortcut(residual)
        return out


class GhostNetBackbone(nn.Module):
    """
    GhostNet-0.5x Backbone producing multi-scale features.

    Input:  (B, 3, 640, 640)
    Output: C3 (B, 40, 80, 80)   — stride 8,  small objects
            C4 (B, 112, 40, 40)  — stride 16, medium objects
            C5 (B, 160, 20, 20)  — stride 32, large objects

    Architecture (GhostNet-0.5x channel widths):
        Stem:   Conv 3x3 stride 2 → 16 channels  (320x320)
        Stage1: GhostBottleneck × 1, 16→16        (320x320)
        Stage2: GhostBottleneck × 1, 16→24 s=2    (160x160)
                GhostBottleneck × 1, 24→24         (160x160)
        Stage3: GhostBottleneck × 1, 24→40 s=2    (80x80)   → C3
                GhostBottleneck × 1, 40→40         (80x80)
        Stage4: GhostBottleneck × 3, 40→80 s=2    (40x40)
                GhostBottleneck × 3, 80→112        (40x40)   → C4
        Stage5: GhostBottleneck × 3, 112→160 s=2  (20x20)   → C5
    """

    def __init__(self, width_mult=0.5):
        super(GhostNetBackbone, self).__init__()

        def _make_divisible(v, divisor=8):
            """Ensure channel count is divisible by 8 for hardware efficiency."""
            return int(math.ceil(v * width_mult / divisor) * divisor)

        # Channel configs: (in, mid_expand_ratio, out, stride)
        # mid = out * expand_ratio
        configs = [
            # Stage 1: stride 2 (from stem), total stride = 2
            (16,  1, 16,  1),
            # Stage 2: total stride = 4
            (16,  1, 24,  2),
            (24,  1, 24,  1),
            # Stage 3: total stride = 8 → outputs C3
            (24,  1, 40,  2),
            (40,  1, 40,  1),
            # Stage 4: total stride = 16 → outputs C4
            (40,  3, 80,  2),
            (80,  3, 80,  1),
            (80,  3, 80,  1),
            (80,  3, 112, 1),
            # Stage 5: total stride = 32 → outputs C5
            (112, 6, 160, 2),
            (160, 6, 160, 1),
            (160, 6, 160, 1),
        ]

        # Stem: initial conv to go from RGB to first feature channels
        first_channels = _make_divisible(16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, first_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_channels),
            nn.ReLU(inplace=True),
        )

        # Build all stages
        layers = []
        in_ch = first_channels
        for (inp, expand_ratio, out, stride) in configs:
            inp_ch = _make_divisible(inp)
            out_ch = _make_divisible(out)
            mid_ch = out_ch * expand_ratio
            # Use in_ch from previous layer output, not config inp
            layers.append(GhostBottleneck(in_ch, mid_ch, out_ch, stride=stride))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)

        # Record which layer indices produce C3, C4, C5
        # After stem (stride 2), cumulative strides at each layer:
        # Layer 0: s=1 → total=2   (Stage 1)
        # Layer 1: s=2 → total=4   (Stage 2 start)
        # Layer 2: s=1 → total=4
        # Layer 3: s=2 → total=8   (Stage 3 start) → C3 after layer 4
        # Layer 4: s=1 → total=8                    → C3
        # Layer 5: s=2 → total=16  (Stage 4 start)
        # Layer 6-7: s=1 → total=16
        # Layer 8: s=1 → total=16                   → C4
        # Layer 9: s=2 → total=32  (Stage 5 start)
        # Layer 10-11: s=1 → total=32               → C5
        self.c3_idx = 4   # End of stage 3
        self.c4_idx = 8   # End of stage 4
        self.c5_idx = 11  # End of stage 5

        # Store actual output channel counts for FPN
        # These are computed dynamically based on width_mult
        c3_out = _make_divisible(configs[self.c3_idx][2])
        c4_out = _make_divisible(configs[self.c4_idx][2])
        c5_out = _make_divisible(configs[self.c5_idx][2])
        self.out_channels = [c3_out, c4_out, c5_out]

    def forward(self, x):
        x = self.stem(x)

        c3 = c4 = c5 = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.c3_idx:
                c3 = x
            elif i == self.c4_idx:
                c4 = x
            elif i == self.c5_idx:
                c5 = x

        return c3, c4, c5
