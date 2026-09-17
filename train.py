"""
Training script for the Lightweight Object Detector.

Usage:
    python train.py                        # Full training
    python train.py --epochs 1 --debug     # Quick smoke test
"""

import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import cfg
from datapipeline.coco_dataset import COCODataset, collate_fn
from detector.detector import LightDetector
from detector.target import FCOSTargetAssigner
from detector.losses import FCOSLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train Lightweight Detector')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode: train on val set, small iterations')
    parser.add_argument('--max-iters', type=int, default=None,
                        help='Max iterations per epoch (for debugging)')
    return parser.parse_args()


def warmup_cosine_lr(optimizer, current_iter, warmup_iters, total_iters,
                     base_lr):
    """Linear warmup followed by cosine annealing."""
    if current_iter < warmup_iters:
        lr = base_lr * (current_iter / max(warmup_iters, 1))
    else:
        progress = (current_iter - warmup_iters) / max(
            total_iters - warmup_iters, 1)
        lr = base_lr * 0.5 * (1.0 + __import__('math').cos(
            __import__('math').pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_one_epoch(model, dataloader, optimizer, loss_fn, target_assigner,
                    scaler, device, epoch, total_iters_so_far, total_iters,
                    warmup_iters, base_lr, max_iters=None):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_loss_cls = 0.0
    epoch_loss_reg = 0.0
    epoch_loss_ctr = 0.0
    num_iters = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_iters is not None and batch_idx >= max_iters:
            break

        current_iter = total_iters_so_far + batch_idx
        lr = warmup_cosine_lr(optimizer, current_iter, warmup_iters,
                              total_iters, base_lr)

        images = images.to(device)

        # Forward pass with AMP
        with autocast('cuda', enabled=(device.type == 'cuda')):
            cls_outputs, reg_outputs, ctr_outputs = model(images)

            # Compute feature sizes from output shapes
            feat_sizes = [(c.shape[2], c.shape[3]) for c in cls_outputs]

            # Compute targets for each image in the batch
            batch_cls_targets = [[] for _ in range(len(feat_sizes))]
            batch_reg_targets = [[] for _ in range(len(feat_sizes))]
            batch_ctr_targets = [[] for _ in range(len(feat_sizes))]

            for t in targets:
                gt_boxes = t['boxes'].to(device)
                gt_labels = t['labels'].to(device)

                cls_tgts, reg_tgts, ctr_tgts = target_assigner.assign(
                    feat_sizes, gt_boxes, gt_labels, device)

                for lvl in range(len(feat_sizes)):
                    batch_cls_targets[lvl].append(cls_tgts[lvl])
                    batch_reg_targets[lvl].append(reg_tgts[lvl])
                    batch_ctr_targets[lvl].append(ctr_tgts[lvl])

            # Stack per-level targets into batch tensors
            for lvl in range(len(feat_sizes)):
                batch_cls_targets[lvl] = torch.stack(
                    batch_cls_targets[lvl], dim=0)
                batch_reg_targets[lvl] = torch.stack(
                    batch_reg_targets[lvl], dim=0)
                batch_ctr_targets[lvl] = torch.stack(
                    batch_ctr_targets[lvl], dim=0)

            # Compute loss
            total_loss, loss_dict = loss_fn(
                cls_outputs, reg_outputs, ctr_outputs,
                batch_cls_targets, batch_reg_targets, batch_ctr_targets)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss_dict['loss_total']
        epoch_loss_cls += loss_dict['loss_cls']
        epoch_loss_reg += loss_dict['loss_reg']
        epoch_loss_ctr += loss_dict['loss_ctr']
        num_iters += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch [{epoch}] Iter [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss_dict['loss_total']:.4f} "
                  f"(cls: {loss_dict['loss_cls']:.4f}, "
                  f"reg: {loss_dict['loss_reg']:.4f}, "
                  f"ctr: {loss_dict['loss_ctr']:.4f}) "
                  f"LR: {lr:.6f}")

    avg_loss = epoch_loss / max(num_iters, 1)
    return {
        'loss': avg_loss,
        'loss_cls': epoch_loss_cls / max(num_iters, 1),
        'loss_reg': epoch_loss_reg / max(num_iters, 1),
        'loss_ctr': epoch_loss_ctr / max(num_iters, 1),
        'num_iters': num_iters,
    }


def main():
    args = parse_args()

    # Override config with CLI args
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    base_lr = args.lr or cfg.lr

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    if args.debug:
        json_file = cfg.val_json
        img_dir = cfg.val_img_dir
        print("DEBUG MODE: Training on val set")
    else:
        json_file = cfg.train_json
        img_dir = cfg.train_img_dir

    train_dataset = COCODataset(
        json_file=json_file, img_dir=img_dir,
        img_size=cfg.img_size, is_train=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
        drop_last=True)

    print(f"Dataset: {len(train_dataset)} images, "
          f"{len(train_loader)} batches/epoch")

    # Model
    model = LightDetector(
        num_classes=cfg.num_classes, fpn_channels=cfg.fpn_channels)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), lr=base_lr,
        momentum=0.9, weight_decay=cfg.weight_decay)

    # Loss
    loss_fn = FCOSLoss(num_classes=cfg.num_classes)

    # Target assigner
    target_assigner = FCOSTargetAssigner(strides=cfg.strides)

    # AMP scaler
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Training schedule
    iters_per_epoch = len(train_loader)
    if args.max_iters:
        iters_per_epoch = min(iters_per_epoch, args.max_iters)
    total_iters = epochs * iters_per_epoch
    warmup_iters = min(500, total_iters // 5)

    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))

    # Checkpoint dir
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print(f"  Warmup: {warmup_iters} iters")
    print(f"  Total:  {total_iters} iters\n")

    training_log = []

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        total_iters_so_far = epoch * iters_per_epoch

        metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, target_assigner,
            scaler, device, epoch, total_iters_so_far, total_iters,
            warmup_iters, base_lr, max_iters=args.max_iters)

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch [{epoch}/{epochs}] completed in {epoch_time:.1f}s")
        print(f"  Avg Loss: {metrics['loss']:.4f} "
              f"(cls: {metrics['loss_cls']:.4f}, "
              f"reg: {metrics['loss_reg']:.4f}, "
              f"ctr: {metrics['loss_ctr']:.4f})")

        training_log.append({
            'epoch': epoch,
            'loss': metrics['loss'],
            'loss_cls': metrics['loss_cls'],
            'loss_reg': metrics['loss_reg'],
            'loss_ctr': metrics['loss_ctr'],
            'time': epoch_time,
        })

        # Save checkpoint
        is_best = metrics['loss'] < best_loss
        if is_best:
            best_loss = metrics['loss']

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'config': cfg.__dict__,
        }

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(checkpoint,
                       f'checkpoints/checkpoint_epoch_{epoch}.pth')

        if is_best:
            torch.save(checkpoint, 'checkpoints/best.pth')
            print("  → Saved best checkpoint")

        # Save training log
        with open('outputs/training_log.json', 'w') as f:
            json.dump(training_log, f, indent=2)

    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Best model saved to: checkpoints/best.pth")


if __name__ == '__main__':
    main()
