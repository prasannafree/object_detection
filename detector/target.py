"""
FCOS Target Assignment.

For each pixel on each FPN feature level, this module decides:
  1. Is this pixel responsible for detecting an object? (positive vs negative)
  2. If yes, which ground-truth box should it predict?
  3. What are the regression targets (l, t, r, b) and centerness?

Assignment rules (FCOS-style):
  - A pixel is positive if its mapped input-image location falls inside a GT box
  - Among overlapping GT boxes, pick the one with smallest area
  - Scale constraint: each FPN level only handles objects within a size range
      P3: max(l,t,r,b) ∈ [0, 64]     → small objects
      P4: max(l,t,r,b) ∈ [64, 128]   → medium objects
      P5: max(l,t,r,b) ∈ [128, ∞]    → large objects
"""

import torch


class FCOSTargetAssigner:
    """
    Assigns ground-truth targets to each pixel location on each FPN level.
    """

    def __init__(self, strides=(8, 16, 32),
                 sizes=((0, 64), (64, 128), (128, 1e8))):
        """
        Args:
            strides: Stride of each FPN level relative to input image.
            sizes: (min, max) range of max(l,t,r,b) for each level.
        """
        self.strides = strides
        self.sizes = sizes

    def _compute_points(self, feat_h, feat_w, stride, device):
        """
        Generate grid of (x, y) coordinates in input image space.

        For stride=8 and feat_size=80×80, pixel (j, i) maps to:
            x = j * 8 + 4  (center of the stride-8 cell)
            y = i * 8 + 4

        Returns: (H*W, 2) tensor of (x, y) points
        """
        shifts_x = torch.arange(0, feat_w, device=device, dtype=torch.float32)
        shifts_y = torch.arange(0, feat_h, device=device, dtype=torch.float32)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')

        points_x = shift_x * stride + stride // 2
        points_y = shift_y * stride + stride // 2

        points = torch.stack([points_x.flatten(), points_y.flatten()], dim=1)
        return points  # (H*W, 2)

    def assign(self, feat_sizes, gt_boxes, gt_labels, device):
        """
        Compute targets for all FPN levels for a single image.

        Args:
            feat_sizes: List of (H, W) for each FPN level
            gt_boxes:   (N, 4) tensor [x1, y1, x2, y2]
            gt_labels:  (N,) tensor of class indices
            device:     torch device

        Returns:
            cls_targets_per_level:  List of (H*W,) tensors. -1=ignore, 0-79=class
            reg_targets_per_level:  List of (H*W, 4) tensors. (l, t, r, b)
            ctr_targets_per_level:  List of (H*W,) tensors. centerness in [0, 1]
        """
        num_gts = gt_boxes.shape[0]
        cls_targets_all = []
        reg_targets_all = []
        ctr_targets_all = []

        for level_idx, (feat_h, feat_w) in enumerate(feat_sizes):
            stride = self.strides[level_idx]
            size_min, size_max = self.sizes[level_idx]
            num_points = feat_h * feat_w

            # Grid of pixel centers in input-image coordinates
            points = self._compute_points(feat_h, feat_w, stride, device)
            # points: (H*W, 2) — each row is (x, y)

            if num_gts == 0:
                cls_targets_all.append(
                    torch.full((num_points,), -1, dtype=torch.long, device=device))
                reg_targets_all.append(
                    torch.zeros((num_points, 4), dtype=torch.float32, device=device))
                ctr_targets_all.append(
                    torch.zeros((num_points,), dtype=torch.float32, device=device))
                continue

            # Compute (l, t, r, b) for every point against every GT box
            # points: (H*W, 2), gt_boxes: (N, 4)
            xs, ys = points[:, 0], points[:, 1]  # (H*W,)
            x1 = gt_boxes[:, 0]  # (N,)
            y1 = gt_boxes[:, 1]
            x2 = gt_boxes[:, 2]
            y2 = gt_boxes[:, 3]

            # Broadcasting: (H*W, 1) vs (1, N) → (H*W, N)
            l = xs[:, None] - x1[None, :]
            t = ys[:, None] - y1[None, :]
            r = x2[None, :] - xs[:, None]
            b = y2[None, :] - ys[:, None]

            reg_targets = torch.stack([l, t, r, b], dim=2)  # (H*W, N, 4)

            # Condition 1: point must be inside the GT box (all of l,t,r,b > 0)
            is_inside = reg_targets.min(dim=2).values > 0  # (H*W, N)

            # Condition 2: max(l,t,r,b) must be within this level's size range
            max_reg = reg_targets.max(dim=2).values  # (H*W, N)
            is_in_level = (max_reg >= size_min) & (max_reg < size_max)

            # Combined valid mask
            is_valid = is_inside & is_in_level  # (H*W, N)

            # For each point, pick the GT with smallest area (if multiple match)
            areas = (x2 - x1) * (y2 - y1)  # (N,)
            areas = areas[None, :].expand(num_points, num_gts).clone()  # (H*W, N)
            areas[~is_valid] = 1e8  # Invalidate non-matching pairs

            min_area, min_area_idx = areas.min(dim=1)  # (H*W,)

            # Gather targets for the best-matching GT
            cls_targets = gt_labels[min_area_idx]  # (H*W,)
            cls_targets[min_area >= 1e8] = -1  # Mark as background

            # Gather regression targets: pick the (l,t,r,b) for the matched GT
            reg_targets = reg_targets[
                torch.arange(num_points, device=device), min_area_idx]  # (H*W, 4)

            # Compute centerness
            l_t = reg_targets[:, 0]
            t_t = reg_targets[:, 1]
            r_t = reg_targets[:, 2]
            b_t = reg_targets[:, 3]

            lr_min = torch.min(l_t, r_t)
            lr_max = torch.max(l_t, r_t)
            tb_min = torch.min(t_t, b_t)
            tb_max = torch.max(t_t, b_t)

            centerness = torch.sqrt(
                (lr_min / lr_max.clamp(min=1e-6)) *
                (tb_min / tb_max.clamp(min=1e-6))
            )
            centerness[cls_targets == -1] = 0  # Background has 0 centerness

            cls_targets_all.append(cls_targets)
            reg_targets_all.append(reg_targets)
            ctr_targets_all.append(centerness)

        return cls_targets_all, reg_targets_all, ctr_targets_all
