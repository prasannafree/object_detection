"""
COCO mAP Evaluation.

Runs the model on the entire val2017 set and computes:
  - AP @ IoU=0.50:0.95 (primary COCO metric)
  - AP @ IoU=0.50
  - AP @ IoU=0.75
  - AP for small, medium, large objects

Usage:
    python eval.py --checkpoint checkpoints/best.pth
"""

import os
import sys
import json
import argparse
import time

import torch
import numpy as np
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config import cfg
from datapipeline.coco_dataset import COCODataset, collate_fn
from detector.detector import LightDetector
from detector.postprocess import postprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on COCO val2017')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=str, default='outputs/eval_results.json')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = LightDetector(
        num_classes=cfg.num_classes, fpn_channels=cfg.fpn_channels)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Dataset (validation, no augmentation)
    val_dataset = COCODataset(
        json_file=cfg.val_json, img_dir=cfg.val_img_dir,
        img_size=cfg.img_size, is_train=False)

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    print(f"Evaluating on {len(val_dataset)} images...")

    # Build category ID reverse mapping (contiguous 0-79 → COCO cat IDs)
    coco_gt = COCO(cfg.val_json)
    cat_ids = coco_gt.getCatIds()
    idx_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}

    # Run inference
    all_results = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)

            cls_outputs, reg_outputs, ctr_outputs = model(images)

            batch_size = images.shape[0]
            for i in range(batch_size):
                # Get image ID from dataset
                dataset_idx = batch_idx * args.batch_size + i
                if dataset_idx >= len(val_dataset):
                    break
                img_id = val_dataset.img_ids[dataset_idx]

                # Post-process single image
                cls_out = [c[i] for c in cls_outputs]
                reg_out = [r[i] for r in reg_outputs]
                ctr_out = [c[i] for c in ctr_outputs]

                boxes, scores, labels = postprocess(
                    cls_out, reg_out, ctr_out, strides=cfg.strides,
                    score_thresh=cfg.score_threshold,
                    nms_thresh=cfg.nms_threshold)

                if boxes.shape[0] == 0:
                    continue

                boxes = boxes.cpu().numpy()
                scores = scores.cpu().numpy()
                labels = labels.cpu().numpy()

                for j in range(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes[j]
                    # Convert back to COCO format: [x, y, w, h]
                    w = x2 - x1
                    h = y2 - y1

                    # Map contiguous label index back to COCO category ID
                    cat_id = idx_to_cat_id.get(int(labels[j]), int(labels[j]))

                    all_results.append({
                        'image_id': int(img_id),
                        'category_id': cat_id,
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(scores[j]),
                    })

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {(batch_idx + 1) * args.batch_size}/"
                      f"{len(val_dataset)} images "
                      f"({elapsed:.1f}s, "
                      f"{len(all_results)} detections so far)")

    elapsed = time.time() - start_time
    print(f"\nInference complete: {len(all_results)} detections in {elapsed:.1f}s")

    if len(all_results) == 0:
        print("No detections! Model may need more training.")
        return

    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f)
    print(f"Results saved to: {args.output}")

    # Run COCO evaluation
    print("\n--- COCO Evaluation ---")
    coco_dt = coco_gt.loadRes(args.output)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print summary
    metrics = {
        'AP@[.50:.95]': coco_eval.stats[0],
        'AP@.50': coco_eval.stats[1],
        'AP@.75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5],
    }
    print(f"\nSummary: {json.dumps(metrics, indent=2)}")


if __name__ == '__main__':
    main()
