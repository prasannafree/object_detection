"""
Single-image inference and visualization.

Usage:
    python infer.py --image data/val2017/000000000139.jpg --checkpoint checkpoints/best.pth
    python infer.py --image data/val2017/000000000139.jpg  # (untrained model, for testing)
"""

import os   # object 
import sys
import argparse

import cv2
import numpy as np
import torch

from config import cfg
from detector.detector import LightDetector
from detector.postprocess import postprocess

# COCO class names (80 categories)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Random colors for visualization
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3)).tolist()


def preprocess_image(image_path, img_size=640):
    """Load and preprocess a single image for inference."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize preserving aspect ratio
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))

    # Pad to square
    pad_h = img_size - new_h
    pad_w = img_size - new_w
    top = pad_h // 2
    left = pad_w // 2
    image = cv2.copyMakeBorder(image, top, pad_h - top, left, pad_w - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Normalize and to tensor
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

    return tensor, original, scale, top, left


def draw_detections(image, boxes, scores, labels, score_thresh=0.3):
    """Draw bounding boxes on the image."""
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_idx = int(label)
        class_name = COCO_CLASSES[label_idx] if label_idx < len(
            COCO_CLASSES) else f"class_{label_idx}"
        color = COLORS[label_idx % len(COLORS)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        text = f"{class_name}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--score-thresh', type=float, default=0.3,
                        help='Score threshold for visualization')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: outputs/infer_result.jpg)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = LightDetector(
        num_classes=cfg.num_classes, fpn_channels=cfg.fpn_channels)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Preprocess
    tensor, original, scale, pad_top, pad_left = preprocess_image(
        args.image, cfg.img_size)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        cls_outputs, reg_outputs, ctr_outputs = model(tensor)

    # Post-process (single image — remove batch dim)
    cls_out = [c[0] for c in cls_outputs]
    reg_out = [r[0] for r in reg_outputs]
    ctr_out = [c[0] for c in ctr_outputs]

    boxes, scores, labels = postprocess(
        cls_out, reg_out, ctr_out, strides=cfg.strides,
        score_thresh=cfg.score_threshold, nms_thresh=cfg.nms_threshold)

    print(f"Detections: {boxes.shape[0]}")

    # Map boxes back to original image coordinates
    if boxes.shape[0] > 0:
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        # Remove padding offset
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top
        # Undo resize
        boxes /= scale
        # Clip to image boundaries
        h, w = original.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    else:
        boxes = np.array([])
        scores = np.array([])
        labels = np.array([])

    # Visualize
    result = draw_detections(original, boxes, scores, labels,
                             args.score_thresh)

    os.makedirs('outputs', exist_ok=True)
    output_path = args.output or 'outputs/infer_result.jpg'
    cv2.imwrite(output_path, result)
    print(f"Result saved to: {output_path}")


if __name__ == '__main__':
    main()
