"""
Detection Losses for FCOS.

Three loss components:
  1. Focal Loss (classification) — handles massive class imbalance
  2. IoU Loss (regression)       — scale-invariant box regression
  3. BCE Loss (centerness)       — quality score for each prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(cls_logits, cls_targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for dense classification.

    Standard cross-entropy treats all examples equally. In detection, 99%+
    of pixels are background (easy negatives). Focal loss down-weights
    easy examples so the model focuses on hard ones.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        cls_logits:  (N, num_classes) raw logits
        cls_targets: (N,) class indices. -1 = ignore, 0+ = class index
        alpha: balance factor for positive vs negative (default 0.25)
        gamma: focusing parameter (default 2.0)

    Returns:
        Scalar loss (mean over positive samples)
    """
    num_classes = cls_logits.shape[1]

    # Ignore pixels marked with -1
    valid_mask = cls_targets >= 0
    cls_logits = cls_logits[valid_mask]
    cls_targets = cls_targets[valid_mask]

    if cls_logits.numel() == 0:
        return cls_logits.sum() * 0.0  # Return 0 with grad

    # Convert class indices to one-hot: (N,) → (N, num_classes)
    target_one_hot = F.one_hot(cls_targets, num_classes).float()

    # Sigmoid probabilities
    p = torch.sigmoid(cls_logits)

    # Binary cross entropy per class (no reduction)
    ce_loss = F.binary_cross_entropy_with_logits(
        cls_logits, target_one_hot, reduction='none')

    # p_t: probability of the correct class
    p_t = p * target_one_hot + (1 - p) * (1 - target_one_hot)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting: alpha for positives, (1-alpha) for negatives
    alpha_t = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)

    loss = alpha_t * focal_weight * ce_loss

    # Normalize by total number of positive samples across the batch
    num_pos = (cls_targets >= 0).sum().clamp(min=1)
    return loss.sum() / num_pos


def iou_loss(reg_preds, reg_targets):
    """
    IoU Loss for bounding box regression.

    Unlike L1/L2 loss, IoU loss is scale-invariant: a 5px error on a 20px
    box is penalized more than a 5px error on a 200px box.

    We compute IoU directly from (l, t, r, b) distance predictions.

    Args:
        reg_preds:   (N, 4) predicted (l, t, r, b)
        reg_targets: (N, 4) target (l, t, r, b)

    Returns:
        Scalar loss (1 - mean IoU)
    """
    if reg_preds.numel() == 0:
        return reg_preds.sum() * 0.0

    # Predicted box dimensions
    pred_l, pred_t, pred_r, pred_b = reg_preds.unbind(dim=1)
    tgt_l, tgt_t, tgt_r, tgt_b = reg_targets.unbind(dim=1)

    # Areas
    pred_area = (pred_l + pred_r) * (pred_t + pred_b)
    tgt_area = (tgt_l + tgt_r) * (tgt_t + tgt_b)

    # Intersection
    inter_w = torch.min(pred_l, tgt_l) + torch.min(pred_r, tgt_r)
    inter_h = torch.min(pred_t, tgt_t) + torch.min(pred_b, tgt_b)
    inter_area = inter_w * inter_h

    # Union
    union_area = pred_area + tgt_area - inter_area

    # IoU
    iou = inter_area / union_area.clamp(min=1e-6)

    return (1 - iou).mean()


def centerness_loss(ctr_logits, ctr_targets):
    """
    Binary Cross-Entropy loss for centerness prediction.

    Centerness ranges from 0 (pixel at box edge) to 1 (pixel at box center).
    Only computed on positive (foreground) pixels.

    Args:
        ctr_logits:  (N,) raw centerness logits
        ctr_targets: (N,) target centerness values in [0, 1]

    Returns:
        Scalar loss
    """
    if ctr_logits.numel() == 0:
        return ctr_logits.sum() * 0.0

    return F.binary_cross_entropy_with_logits(ctr_logits, ctr_targets)


class FCOSLoss(nn.Module):
    """
    Combined FCOS loss: Focal + IoU + Centerness.

    total_loss = focal_loss + lambda_reg * iou_loss + lambda_ctr * ctr_loss
    """

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0,
                 lambda_reg=1.0, lambda_ctr=1.0):
        super(FCOSLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.lambda_ctr = lambda_ctr

    def forward(self, cls_outputs, reg_outputs, ctr_outputs,
                cls_targets, reg_targets, ctr_targets):
        """
        Compute total loss across all FPN levels.

        Args:
            cls_outputs: List of (B, num_classes, H_i, W_i) per level
            reg_outputs: List of (B, 4, H_i, W_i) per level
            ctr_outputs: List of (B, 1, H_i, W_i) per level
            cls_targets: List of (B, H_i*W_i) per level
            reg_targets: List of (B, H_i*W_i, 4) per level
            ctr_targets: List of (B, H_i*W_i) per level

        Returns:
            total_loss, loss_dict with individual components
        """
        all_cls_logits = []
        all_cls_targets = []
        all_reg_preds = []
        all_reg_targets = []
        all_ctr_logits = []
        all_ctr_targets = []

        for level_idx in range(len(cls_outputs)):
            B = cls_outputs[level_idx].shape[0]

            # Reshape predictions: (B, C, H, W) → (B*H*W, C)
            cls_logit = cls_outputs[level_idx].permute(0, 2, 3, 1).reshape(
                -1, self.num_classes)
            reg_pred = reg_outputs[level_idx].permute(0, 2, 3, 1).reshape(-1, 4)
            ctr_logit = ctr_outputs[level_idx].squeeze(1).reshape(-1)

            # Reshape targets: (B, H*W) → (B*H*W,)
            cls_tgt = cls_targets[level_idx].reshape(-1)
            reg_tgt = reg_targets[level_idx].reshape(-1, 4)
            ctr_tgt = ctr_targets[level_idx].reshape(-1)

            all_cls_logits.append(cls_logit)
            all_cls_targets.append(cls_tgt)
            all_reg_preds.append(reg_pred)
            all_reg_targets.append(reg_tgt)
            all_ctr_logits.append(ctr_logit)
            all_ctr_targets.append(ctr_tgt)

        # Concatenate all levels
        all_cls_logits = torch.cat(all_cls_logits, dim=0)
        all_cls_targets = torch.cat(all_cls_targets, dim=0)
        all_reg_preds = torch.cat(all_reg_preds, dim=0)
        all_reg_targets = torch.cat(all_reg_targets, dim=0)
        all_ctr_logits = torch.cat(all_ctr_logits, dim=0)
        all_ctr_targets = torch.cat(all_ctr_targets, dim=0)

        # Classification: all valid pixels (positive + negative)
        loss_cls = focal_loss(all_cls_logits, all_cls_targets,
                              self.alpha, self.gamma)

        # Regression and centerness: only on positive pixels
        pos_mask = all_cls_targets >= 0
        pos_reg_preds = all_reg_preds[pos_mask]
        pos_reg_targets = all_reg_targets[pos_mask]
        pos_ctr_logits = all_ctr_logits[pos_mask]
        pos_ctr_targets = all_ctr_targets[pos_mask]

        loss_reg = iou_loss(pos_reg_preds, pos_reg_targets)
        loss_ctr = centerness_loss(pos_ctr_logits, pos_ctr_targets)

        total_loss = loss_cls + self.lambda_reg * loss_reg + \
            self.lambda_ctr * loss_ctr

        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_ctr': loss_ctr.item(),
            'loss_total': total_loss.item(),
        }

        return total_loss, loss_dict
