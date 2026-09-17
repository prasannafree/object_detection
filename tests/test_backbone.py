"""Tests for GhostNet Backbone."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.backbone import GhostModule, GhostBottleneck, GhostNetBackbone


def test_ghost_module():
    print("--- Test: GhostModule ---")
    module = GhostModule(in_channels=16, out_channels=32, ratio=2)
    x = torch.randn(1, 16, 80, 80)
    out = module(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (1, 32, 80, 80), f"Expected (1,32,80,80), got {out.shape}"

    # Count parameters
    params = sum(p.numel() for p in module.parameters())
    print(f"  Parameters: {params}")
    print("  PASSED\n")


def test_ghost_bottleneck_stride1():
    print("--- Test: GhostBottleneck stride=1 ---")
    block = GhostBottleneck(in_channels=40, mid_channels=120, out_channels=40, stride=1)
    x = torch.randn(1, 40, 80, 80)
    out = block(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (1, 40, 80, 80), f"Expected (1,40,80,80), got {out.shape}"
    print("  PASSED\n")


def test_ghost_bottleneck_stride2():
    print("--- Test: GhostBottleneck stride=2 ---")
    block = GhostBottleneck(in_channels=24, mid_channels=72, out_channels=40, stride=2)
    x = torch.randn(1, 24, 160, 160)
    out = block(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (1, 40, 80, 80), f"Expected (1,40,80,80), got {out.shape}"
    print("  PASSED\n")


def test_backbone_shapes():
    print("--- Test: GhostNetBackbone output shapes ---")
    model = GhostNetBackbone(width_mult=0.5)
    x = torch.randn(1, 3, 640, 640)
    c3, c4, c5 = model(x)
    print(f"  Input: {x.shape}")
    print(f"  C3:    {c3.shape}  (expected stride 8  → 80×80)")
    print(f"  C4:    {c4.shape}  (expected stride 16 → 40×40)")
    print(f"  C5:    {c5.shape}  (expected stride 32 → 20×20)")
    assert c3.shape[2:] == (80, 80), f"C3 spatial dims wrong: {c3.shape}"
    assert c4.shape[2:] == (40, 40), f"C4 spatial dims wrong: {c4.shape}"
    assert c5.shape[2:] == (20, 20), f"C5 spatial dims wrong: {c5.shape}"
    print("  PASSED\n")


def test_backbone_param_count():
    print("--- Test: Backbone parameter count ---")
    model = GhostNetBackbone(width_mult=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable:,}")
    assert total_params < 5_000_000, f"Backbone too heavy: {total_params:,} params"
    print(f"  PASSED (< 5M params)\n")


if __name__ == "__main__":
    test_ghost_module()
    test_ghost_bottleneck_stride1()
    test_ghost_bottleneck_stride2()
    test_backbone_shapes()
    test_backbone_param_count()
    print("All backbone tests passed!")
