"""Tests for FCOS Target Assignment."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.target import FCOSTargetAssigner


def test_single_box_assignment():
    """
    Place a single box at the center of the image.
    Verify that the correct pixels are assigned as positive.
    """
    print("--- Test: Single box target assignment ---")
    assigner = FCOSTargetAssigner(
        strides=(8, 16, 32),
        sizes=((0, 64), (64, 128), (128, 1e8))
    )

    # A small box near center of 640x640 image
    # box size ~40x40 → max(l,t,r,b) should be ~20 → assigned to P3 (range 0-64)
    gt_boxes = torch.tensor([[300.0, 300.0, 340.0, 340.0]])  # 40x40 box
    gt_labels = torch.tensor([5])  # class 5

    feat_sizes = [(80, 80), (40, 40), (20, 20)]
    cls_targets, reg_targets, ctr_targets = assigner.assign(
        feat_sizes, gt_boxes, gt_labels, device='cpu')

    # Check P3 (level 0) has some positive pixels
    num_pos_p3 = (cls_targets[0] >= 0).sum().item()
    print(f"  P3 positive pixels: {num_pos_p3}")
    assert num_pos_p3 > 0, "No positive pixels assigned at P3!"

    # Check that positive pixels have class 5
    pos_mask = cls_targets[0] >= 0
    assert (cls_targets[0][pos_mask] == 5).all(), "Wrong class assignment"
    print(f"  All positive pixels have correct class: 5")

    # Check centerness: center pixel should have centerness close to 1.0
    pos_ctr = ctr_targets[0][pos_mask]
    max_ctr = pos_ctr.max().item()
    print(f"  Max centerness: {max_ctr:.4f} (should be close to 1.0)")
    assert max_ctr > 0.5, f"Max centerness too low: {max_ctr}"

    # Check regression targets are positive for positive pixels
    pos_reg = reg_targets[0][pos_mask]
    assert (pos_reg > 0).all(), "Regression targets should be positive inside box"
    print(f"  All regression targets positive: min={pos_reg.min().item():.2f}")
    print("  PASSED\n")


def test_no_boxes():
    print("--- Test: No GT boxes ---")
    assigner = FCOSTargetAssigner()

    gt_boxes = torch.zeros((0, 4))
    gt_labels = torch.zeros((0,), dtype=torch.long)

    feat_sizes = [(80, 80), (40, 40), (20, 20)]
    cls_targets, reg_targets, ctr_targets = assigner.assign(
        feat_sizes, gt_boxes, gt_labels, device='cpu')

    # All pixels should be background (-1)
    for i in range(3):
        assert (cls_targets[i] == -1).all(), f"Level {i} should all be background"
    print("  All levels are background — PASSED\n")


def test_large_box_assigned_to_p5():
    print("--- Test: Large box assigned to P5 ---")
    assigner = FCOSTargetAssigner(
        strides=(8, 16, 32),
        sizes=((0, 64), (64, 128), (128, 1e8))
    )

    # A large 400x400 box → max(l,t,r,b) >> 128 → should go to P5
    gt_boxes = torch.tensor([[100.0, 100.0, 500.0, 500.0]])
    gt_labels = torch.tensor([0])

    feat_sizes = [(80, 80), (40, 40), (20, 20)]
    cls_targets, _, _ = assigner.assign(feat_sizes, gt_boxes, gt_labels, device='cpu')

    num_pos_p5 = (cls_targets[2] >= 0).sum().item()
    print(f"  P5 positive pixels: {num_pos_p5}")
    assert num_pos_p5 > 0, "Large box should be assigned to P5"
    print("  PASSED\n")


if __name__ == "__main__":
    test_single_box_assignment()
    test_no_boxes()
    test_large_box_assigned_to_p5()
    print("All target assignment tests passed!")
