"""End-to-end test for the full detector model."""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from detector.detector import LightDetector


def test_detector_forward():
    print("--- Test: Full Detector forward pass ---")
    model = LightDetector(num_classes=80, fpn_channels=128)
    model.eval()

    x = torch.randn(2, 3, 640, 640)

    with torch.no_grad():
        cls_out, reg_out, ctr_out = model(x)

    print(f"  Input: {x.shape}")
    for i in range(3):
        print(f"  Level {i}: cls={cls_out[i].shape}, reg={reg_out[i].shape}, ctr={ctr_out[i].shape}")

    # Verify shapes
    assert cls_out[0].shape == (2, 80, 80, 80)
    assert reg_out[0].shape == (2, 4, 80, 80)
    assert ctr_out[0].shape == (2, 1, 80, 80)
    print("  PASSED\n")


def test_detector_param_count():
    print("--- Test: Full model parameter count ---")
    model = LightDetector(num_classes=80, fpn_channels=128)

    total = sum(p.numel() for p in model.parameters())
    backbone = sum(p.numel() for p in model.backbone.parameters())
    neck = sum(p.numel() for p in model.neck.parameters())
    head = sum(p.numel() for p in model.head.parameters())

    print(f"  Backbone: {backbone:>10,} params")
    print(f"  Neck:     {neck:>10,} params")
    print(f"  Head:     {head:>10,} params")
    print(f"  Total:    {total:>10,} params")
    print("  PASSED\n")


def test_detector_speed():
    print("--- Test: Forward pass speed (CPU) ---")
    model = LightDetector(num_classes=80, fpn_channels=128)
    model.eval()

    x = torch.randn(1, 3, 640, 640)

    # Warmup
    with torch.no_grad():
        model(x)

    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            model(x)
        times.append(time.time() - start)

    avg_ms = sum(times) / len(times) * 1000
    print(f"  Average forward pass: {avg_ms:.1f}ms (CPU, batch=1)")
    print("  PASSED\n")


if __name__ == "__main__":
    test_detector_forward()
    test_detector_param_count()
    test_detector_speed()
    print("All detector tests passed!")
