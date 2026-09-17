"""
Post-processing: Box Decoding + Non-Maximum Suppression.

After the model predicts per-pixel (cls, reg, ctr) values, we need to:
  1. Decode (l, t, r, b) distances back into (x1, y1, x2, y2) boxes
  2. Combine classification score with centerness score
  3. Filter low-confidence detections
  4. Apply NMS to remove duplicate detections
"""

import torch
import torchvision


def decode_predictions(cls_outputs, reg_outputs, ctr_outputs, strides,
                       score_thresh=0.05, max_detections_per_level=1000):
    """
    Decode raw model outputs into bounding boxes for a single image.

    Args:
        cls_outputs: List of (num_classes, H_i, W_i) per level
        reg_outputs: List of (4, H_i, W_i) per level
        ctr_outputs: List of (1, H_i, W_i) per level
        strides:     List of stride values [8, 16, 32]
        score_thresh: Minimum score to keep a detection
        max_detections_per_level: Max boxes per FPN level before NMS

    Returns:
        boxes:  (K, 4) tensor [x1, y1, x2, y2]
        scores: (K,) tensor
        labels: (K,) tensor of class indices
    """
    all_boxes = []
    all_scores = []
    all_labels = []

    for level_idx, stride in enumerate(strides):
        cls_logit = cls_outputs[level_idx]   # (num_classes, H, W)
        reg_pred = reg_outputs[level_idx]    # (4, H, W)
        ctr_logit = ctr_outputs[level_idx]   # (1, H, W)

        H, W = cls_logit.shape[1], cls_logit.shape[2]
        device = cls_logit.device

        # Generate grid points
        shifts_x = torch.arange(0, W, device=device, dtype=torch.float32)
        shifts_y = torch.arange(0, H, device=device, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        points_x = (shift_x * stride + stride // 2).flatten()  # (H*W,)
        points_y = (shift_y * stride + stride // 2).flatten()

        # Flatten predictions
        cls_prob = torch.sigmoid(cls_logit.reshape(cls_logit.shape[0], -1).T)
        # cls_prob: (H*W, num_classes)
        ctr_prob = torch.sigmoid(ctr_logit.reshape(-1))  # (H*W,)
        reg_pred = reg_pred.reshape(4, -1).T  # (H*W, 4)

        # Combined score = cls_score * centerness
        scores = cls_prob * ctr_prob[:, None]  # (H*W, num_classes)

        # Find max class score per location
        max_scores, max_labels = scores.max(dim=1)  # (H*W,)

        # Filter by score threshold
        keep = max_scores > score_thresh
        if keep.sum() == 0:
            continue

        max_scores = max_scores[keep]
        max_labels = max_labels[keep]
        reg_pred = reg_pred[keep]
        points_x = points_x[keep]
        points_y = points_y[keep]

        # Keep top-k per level to limit NMS compute
        if max_scores.shape[0] > max_detections_per_level:
            topk_indices = max_scores.topk(max_detections_per_level).indices
            max_scores = max_scores[topk_indices]
            max_labels = max_labels[topk_indices]
            reg_pred = reg_pred[topk_indices]
            points_x = points_x[topk_indices]
            points_y = points_y[topk_indices]

        # Decode (l, t, r, b) → (x1, y1, x2, y2)
        l, t, r, b = reg_pred[:, 0], reg_pred[:, 1], reg_pred[:, 2], reg_pred[:, 3]
        x1 = points_x - l
        y1 = points_y - t
        x2 = points_x + r
        y2 = points_y + b
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        all_boxes.append(boxes)
        all_scores.append(max_scores)
        all_labels.append(max_labels)

    if len(all_boxes) == 0:
        device = cls_outputs[0].device
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device),
                torch.zeros((0,), dtype=torch.long, device=device))

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_boxes, all_scores, all_labels


def postprocess(cls_outputs, reg_outputs, ctr_outputs, strides,
                score_thresh=0.05, nms_thresh=0.5, max_detections=100):
    """
    Full post-processing pipeline for a single image.

    1. Decode predictions from all FPN levels
    2. Apply per-class NMS
    3. Keep top-k detections

    Args:
        cls_outputs: List of (num_classes, H_i, W_i) per level
        reg_outputs: List of (4, H_i, W_i) per level
        ctr_outputs: List of (1, H_i, W_i) per level
        strides:     [8, 16, 32]
        score_thresh: Minimum detection confidence
        nms_thresh:   IoU threshold for NMS
        max_detections: Maximum number of final detections

    Returns:
        boxes:  (M, 4) [x1, y1, x2, y2]
        scores: (M,)
        labels: (M,)
    """
    boxes, scores, labels = decode_predictions(
        cls_outputs, reg_outputs, ctr_outputs, strides, score_thresh)

    if boxes.shape[0] == 0:
        return boxes, scores, labels

    # Per-class NMS using torchvision's batched_nms
    keep = torchvision.ops.batched_nms(boxes, scores, labels, nms_thresh)

    # Limit to max_detections
    keep = keep[:max_detections]

    return boxes[keep], scores[keep], labels[keep]
