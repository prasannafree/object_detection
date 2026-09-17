"""Tests for FCOS Detection Head."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.head import FCOSHead


def test_head_output_shapes():
    print("--- Test: FCOSHead output shapes ---")
    head = FCOSHead(in_channels=128, num_classes=80, num_convs=4, num_levels=3)

    features = [
        torch.randn(2, 128, 80, 80),  # P3
        torch.randn(2, 128, 40, 40),  # P4
        torch.randn(2, 128, 20, 20),  # P5
    ]

    cls_out, reg_out, ctr_out = head(features)

    print(f"  Level 0 (P3): cls={cls_out[0].shape}, reg={reg_out[0].shape}, ctr={ctr_out[0].shape}")
    print(f"  Level 1 (P4): cls={cls_out[1].shape}, reg={reg_out[1].shape}, ctr={ctr_out[1].shape}")
    print(f"  Level 2 (P5): cls={cls_out[2].shape}, reg={reg_out[2].shape}, ctr={ctr_out[2].shape}")

    assert cls_out[0].shape == (2, 80, 80, 80)
    assert reg_out[0].shape == (2, 4, 80, 80)
    assert ctr_out[0].shape == (2, 1, 80, 80)
    assert cls_out[1].shape == (2, 80, 40, 40)
    assert reg_out[1].shape == (2, 4, 40, 40)
    assert ctr_out[1].shape == (2, 1, 40, 40)
    assert cls_out[2].shape == (2, 80, 20, 20)
    assert reg_out[2].shape == (2, 4, 20, 20)
    assert ctr_out[2].shape == (2, 1, 20, 20)
    print("  PASSED\n")


def test_head_regression_positive():
    print("--- Test: Regression outputs are positive (after ScaleExp) ---")
    head = FCOSHead(in_channels=128, num_classes=80, num_convs=4, num_levels=3)
    features = [torch.randn(1, 128, 10, 10) for _ in range(3)]
    _, reg_out, _ = head(features)

    for i, reg in enumerate(reg_out):
        assert (reg > 0).all(), f"Level {i} has non-positive regression values"
        print(f"  Level {i}: min={reg.min().item():.4f}, max={reg.max().item():.4f} — all positive")
    print("  PASSED\n")


if __name__ == "__main__":
    test_head_output_shapes()
    test_head_regression_positive()
    print("All head tests passed!")
