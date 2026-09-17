"""Tests for Loss Functions."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.losses import focal_loss, iou_loss, centerness_loss


def test_focal_loss_perfect_prediction():
    """When predictions are very confident and correct, loss should be near 0."""
    print("--- Test: Focal loss with perfect predictions ---")
    # 10 pixels, 5 classes. Target = class 2 for all.
    cls_logits = torch.full((10, 5), -10.0)  # low prob for all classes
    cls_logits[:, 2] = 10.0                  # very high prob for class 2
    cls_targets = torch.full((10,), 2, dtype=torch.long)

    loss = focal_loss(cls_logits, cls_targets)
    print(f"  Loss: {loss.item():.6f} (should be near 0)")
    assert loss.item() < 0.01, f"Loss too high for perfect prediction: {loss.item()}"
    print("  PASSED\n")


def test_focal_loss_ignores_neg1():
    """Pixels marked with -1 should be ignored."""
    print("--- Test: Focal loss ignores -1 targets ---")
    cls_logits = torch.randn(10, 5)
    cls_targets = torch.full((10,), -1, dtype=torch.long)

    loss = focal_loss(cls_logits, cls_targets)
    print(f"  Loss: {loss.item():.6f} (should be 0 — all ignored)")
    assert loss.item() == 0.0
    print("  PASSED\n")


def test_iou_loss_perfect():
    """When predicted == target, IoU = 1 so loss = 0."""
    print("--- Test: IoU loss with perfect predictions ---")
    reg = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                        [5.0, 5.0, 5.0, 5.0]])
    loss = iou_loss(reg, reg)
    print(f"  Loss: {loss.item():.6f} (should be 0)")
    assert loss.item() < 1e-5
    print("  PASSED\n")


def test_iou_loss_imperfect():
    """With different predictions and targets, loss should be > 0."""
    print("--- Test: IoU loss with imperfect predictions ---")
    pred = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    target = torch.tensor([[20.0, 20.0, 20.0, 20.0]])
    loss = iou_loss(pred, target)
    print(f"  Loss: {loss.item():.4f} (should be > 0)")
    assert loss.item() > 0
    print("  PASSED\n")


def test_centerness_loss_perfect():
    """With perfect centerness predictions, loss ≈ 0."""
    print("--- Test: Centerness loss with perfect predictions ---")
    ctr_targets = torch.tensor([1.0, 0.5, 0.0])
    # Logits that would produce these after sigmoid
    # sigmoid(10) ≈ 1.0, sigmoid(0) = 0.5, sigmoid(-10) ≈ 0.0
    ctr_logits = torch.tensor([10.0, 0.0, -10.0])
    loss = centerness_loss(ctr_logits, ctr_targets)
    print(f"  Loss: {loss.item():.6f} (should be near 0)")
    assert loss.item() < 0.5, f"Loss too high: {loss.item()}"
    print("  PASSED\n")


if __name__ == "__main__":
    test_focal_loss_perfect_prediction()
    test_focal_loss_ignores_neg1()
    test_iou_loss_perfect()
    test_iou_loss_imperfect()
    test_centerness_loss_perfect()
    print("All loss tests passed!")
