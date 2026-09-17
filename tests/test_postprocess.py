"""Tests for Post-processing (Box Decoding + NMS)."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.postprocess import decode_predictions, postprocess


def test_box_decoding():
    """Hand-craft known predictions and verify decoded boxes."""
    print("--- Test: Box decoding ---")
    # Single pixel at grid position (4, 4) on P3 (stride=8)
    # Maps to input coords: x=4*8+4=36, y=4*8+4=36
    # If predicted (l=10, t=10, r=10, b=10):
    #   x1=36-10=26, y1=36-10=26, x2=36+10=46, y2=36+10=46

    H, W = 10, 10
    num_classes = 5

    cls_logits = torch.full((num_classes, H, W), -10.0)  # all background
    cls_logits[2, 4, 4] = 5.0  # class 2 at pixel (4,4), high confidence

    reg_preds = torch.ones((4, H, W)) * 0.01  # near-zero everywhere
    reg_preds[:, 4, 4] = 10.0  # l=t=r=b=10 at pixel (4,4)

    ctr_logits = torch.full((1, H, W), -10.0)  # low centerness everywhere
    ctr_logits[0, 4, 4] = 5.0  # high centerness at (4,4)

    boxes, scores, labels = decode_predictions(
        [cls_logits], [reg_preds], [ctr_logits],
        strides=[8], score_thresh=0.01
    )

    print(f"  Decoded boxes: {boxes}")
    print(f"  Scores: {scores}")
    print(f"  Labels: {labels}")

    assert boxes.shape[0] >= 1, "Should have at least 1 detection"
    # Find the detection for our pixel
    best = scores.argmax()
    box = boxes[best]
    print(f"  Best box: {box.tolist()}")
    assert labels[best] == 2, f"Expected class 2, got {labels[best]}"
    # x_center = 4*8 + 4 = 36
    assert abs(box[0].item() - 26.0) < 1.0, f"x1 wrong: {box[0]}"
    assert abs(box[1].item() - 26.0) < 1.0, f"y1 wrong: {box[1]}"
    assert abs(box[2].item() - 46.0) < 1.0, f"x2 wrong: {box[2]}"
    assert abs(box[3].item() - 46.0) < 1.0, f"y2 wrong: {box[3]}"
    print("  PASSED\n")


def test_nms_removes_duplicates():
    """Create overlapping boxes and verify NMS keeps only the best."""
    print("--- Test: NMS removes duplicates ---")
    H, W = 10, 10
    num_classes = 3

    # Two adjacent high-confidence pixels predicting the same class
    cls_logits = torch.full((num_classes, H, W), -10.0)
    cls_logits[1, 4, 4] = 5.0  # class 1
    cls_logits[1, 4, 5] = 4.0  # class 1 (slightly lower)

    # Both predict large overlapping boxes
    reg_preds = torch.ones((4, H, W)) * 0.01
    reg_preds[:, 4, 4] = 30.0  # big box
    reg_preds[:, 4, 5] = 30.0  # big overlapping box

    ctr_logits = torch.full((1, H, W), -10.0)
    ctr_logits[0, 4, 4] = 5.0
    ctr_logits[0, 4, 5] = 4.0

    boxes, scores, labels = postprocess(
        [cls_logits], [reg_preds], [ctr_logits],
        strides=[8], score_thresh=0.01, nms_thresh=0.5
    )

    print(f"  Detections after NMS: {boxes.shape[0]}")
    # The two boxes overlap heavily, NMS should suppress one
    assert boxes.shape[0] == 1, f"Expected 1 after NMS, got {boxes.shape[0]}"
    assert labels[0] == 1
    print("  PASSED\n")


def test_empty_predictions():
    print("--- Test: Empty predictions ---")
    cls_logits = torch.full((5, 10, 10), -10.0)  # all very low
    reg_preds = torch.ones((4, 10, 10))
    ctr_logits = torch.full((1, 10, 10), -10.0)

    boxes, scores, labels = postprocess(
        [cls_logits], [reg_preds], [ctr_logits],
        strides=[8], score_thresh=0.5
    )

    print(f"  Detections: {boxes.shape[0]} (should be 0)")
    assert boxes.shape[0] == 0
    print("  PASSED\n")


if __name__ == "__main__":
    test_box_decoding()
    test_nms_removes_duplicates()
    test_empty_predictions()
    print("All post-processing tests passed!")
