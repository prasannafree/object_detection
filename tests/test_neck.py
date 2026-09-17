"""Tests for Simple FPN Neck."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.backbone import GhostNetBackbone
from detector.neck import SimpleFPN


def test_fpn_shapes():
    """Test FPN with actual backbone output channels."""
    print("--- Test: SimpleFPN output shapes ---")
    backbone = GhostNetBackbone(width_mult=0.5)
    in_ch = backbone.out_channels
    print(f"  Backbone out_channels: {in_ch}")

    fpn = SimpleFPN(in_channels_list=in_ch, out_channels=128)

    c3 = torch.randn(1, in_ch[0], 80, 80)
    c4 = torch.randn(1, in_ch[1], 40, 40)
    c5 = torch.randn(1, in_ch[2], 20, 20)

    p3, p4, p5 = fpn(c3, c4, c5)

    print(f"  C3 {c3.shape} → P3 {p3.shape}")
    print(f"  C4 {c4.shape} → P4 {p4.shape}")
    print(f"  C5 {c5.shape} → P5 {p5.shape}")

    assert p3.shape == (1, 128, 80, 80), f"P3 wrong: {p3.shape}"
    assert p4.shape == (1, 128, 40, 40), f"P4 wrong: {p4.shape}"
    assert p5.shape == (1, 128, 20, 20), f"P5 wrong: {p5.shape}"
    print("  PASSED\n")


def test_fpn_batch():
    print("--- Test: FPN with batch size > 1 ---")
    backbone = GhostNetBackbone(width_mult=0.5)
    in_ch = backbone.out_channels
    fpn = SimpleFPN(in_channels_list=in_ch, out_channels=128)
    B = 4
    c3 = torch.randn(B, in_ch[0], 80, 80)
    c4 = torch.randn(B, in_ch[1], 40, 40)
    c5 = torch.randn(B, in_ch[2], 20, 20)

    p3, p4, p5 = fpn(c3, c4, c5)
    assert p3.shape[0] == B
    assert p4.shape[0] == B
    assert p5.shape[0] == B
    print(f"  Batch size {B}: all shapes correct")
    print("  PASSED\n")


if __name__ == "__main__":
    test_fpn_shapes()
    test_fpn_batch()
    print("All FPN tests passed!")
