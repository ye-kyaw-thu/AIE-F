#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - Training script for MSL Sign Language Recognition

Features
────────
  - Automatic Mixed Precision (AMP) for RTX 3090 Ti
  - Early stopping with patience
  - Cosine / Step / Plateau LR scheduler
  - TensorBoard logging
  - Label smoothing
  - Class-weighted loss for imbalanced data
  - Top-k checkpoint saving
  - Gradient clipping

Usage
─────
  python src/train.py \
      --config config/config.yaml \
      --model  bilstm \
      --exp    exp01_bilstm
"""

import argparse
import heapq
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent))
from dataset import build_dataloaders
from models  import build_model
from utils   import (
    get_logger, set_seed, get_device, count_parameters,
    parse_annotation_file, build_label_vocabulary,
    match_videos_to_annotations, load_splits,
    save_label_map, save_checkpoint, compute_class_weights,
)
from augment import RandomAugmentor


# ─── Focal Loss (optional) ────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma   = gamma
        self.weight  = weight
        self.ls      = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits, targets, weight=self.weight,
            label_smoothing=self.ls, reduction='none'
        )
        pt  = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, top_k=(1, 5)):
    """Compute top-k accuracy."""
    results = {}
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = labels.size(0)
        _, pred = logits.topk(min(max_k, logits.size(1)), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        for k in top_k:
            if k <= logits.size(1):
                correct_k = correct[:k].reshape(-1).float().sum()
                results[f'top{k}'] = correct_k.item() / batch_size
            else:
                results[f'top{k}'] = 0.0
    return results


# ─── Train / Validate one epoch ───────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, logger):
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_batches  = 0
    log_interval = cfg['logging'].get('log_interval', 20)

    for batch_idx, batch in enumerate(loader):
        kp     = batch['keypoints'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        mask   = batch['mask'].to(device, non_blocking=True)
        lengths= batch['length'].to(device, non_blocking=True)

        with autocast('cuda', enabled=cfg['training']['use_amp']):
            if hasattr(model, 'lstm'):
                logits = model(kp, lengths=lengths, mask=mask)
            elif hasattr(model, 'encoder'):
                logits = model(kp, mask=mask)
            else:
                logits = model(kp)
            loss = criterion(logits, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg['training'].get('grad_clip', 1.0)
        )
        scaler.step(optimizer)
        scaler.update()

        acc = compute_accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_top1 += acc['top1']
        total_top5 += acc.get('top5', 0.0)
        n_batches  += 1

        if (batch_idx + 1) % log_interval == 0:
            logger.debug(
                f"  batch {batch_idx+1}/{len(loader)}  "
                f"loss={loss.item():.4f}  top1={acc['top1']*100:.1f}%"
            )

    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top5': total_top5 / n_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_batches  = 0

    for batch in loader:
        kp     = batch['keypoints'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        mask   = batch['mask'].to(device, non_blocking=True)
        lengths= batch['length'].to(device, non_blocking=True)

        with autocast('cuda', enabled=cfg['training']['use_amp']):
            if hasattr(model, 'lstm'):
                logits = model(kp, lengths=lengths, mask=mask)
            elif hasattr(model, 'encoder'):
                logits = model(kp, mask=mask)
            else:
                logits = model(kp)
            loss = criterion(logits, labels)

        acc = compute_accuracy(logits, labels)
        total_loss += loss.item()
        total_top1 += acc['top1']
        total_top5 += acc.get('top5', 0.0)
        n_batches  += 1

    return {
        'loss': total_loss / n_batches,
        'top1': total_top1 / n_batches,
        'top5': total_top5 / n_batches,
    }


# ─── Main training loop ───────────────────────────────────────────────────────

def train(args):
    # Config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg['training']
    dcfg = cfg['data']
    set_seed(tcfg['seed'])

    # Experiment directories
    exp_dir  = Path('results') / args.exp
    ckpt_dir = exp_dir / 'checkpoints'
    log_dir  = exp_dir / 'logs'
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    with open(exp_dir / 'config_used.yaml', 'w') as f:
        yaml.dump(cfg, f)

    logger = get_logger('train', log_file=str(log_dir / 'train.log'))
    logger.info(f"Experiment: {args.exp}  Model: {args.model}")

    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    records              = parse_annotation_file(dcfg['annotation_file'])
    label2idx, idx2label = build_label_vocabulary(records)
    num_classes          = len(label2idx)
    logger.info(f"Classes: {num_classes}")

    save_label_map(label2idx, idx2label, dcfg['label_map_file'])

    records = match_videos_to_annotations(
        dcfg['video_dir'], records, dcfg['keypoint_dir']
    )

    # ── Load splits from all-class augmented manifest ─────────────────────────
    # The new design: augmented_manifest.json has keys 'train', 'val', 'test'
    # where ALL 556 classes are present in every split.  This is the only
    # correct approach when there is 1 original video per class.
    aug_manifest_path = Path(dcfg['augmented_dir']) / 'augmented_manifest.json'

    if not aug_manifest_path.exists():
        logger.error(
            f"Augmented manifest not found: {aug_manifest_path}\n"
            "Run:  bash scripts/03_augment_data.sh"
        )
        raise FileNotFoundError(str(aug_manifest_path))

    with open(aug_manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    # Support both new dict format {'train':[], 'val':[], 'test':[]}
    # and old flat-list format (legacy, just in case)
    if isinstance(manifest, dict) and 'train' in manifest:
        train_records = manifest['train']
        val_records   = manifest['val']
        test_records  = manifest['test']
        logger.info(
            f"Using all-class manifest: "
            f"train={len(train_records)}, val={len(val_records)}, test={len(test_records)}"
        )
    else:
        # Old flat-list format fallback (pre-fix manifest)
        logger.warning(
            "Old flat manifest format detected. Re-run 03_augment_data.sh "
            "to regenerate with the all-class design."
        )
        train_records = manifest
        splits        = load_splits(dcfg['split_file'])
        val_records   = [records[i] for i in splits['val']]
        test_records  = [records[i] for i in splits['test']]

    # No on-the-fly augmentation needed — training data is already augmented
    augmentor = None

    train_loader, val_loader, test_loader = build_dataloaders(
        train_records, val_records, test_records,
        label2idx, cfg,
        augmentor  = augmentor,
        model_type = args.model,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.model, cfg, num_classes).to(device)
    n_params = count_parameters(model)
    logger.info(f"Model: {args.model}  Params: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # All classes have equal aug counts → uniform weights; still pass for API compat
    class_weights = compute_class_weights(
        train_records, label2idx, device
    ) if args.weighted_loss else None

    if tcfg.get('loss', 'cross_entropy') == 'focal':
        criterion = FocalLoss(
            gamma          = tcfg.get('focal_gamma', 2.0),
            weight         = class_weights,
            label_smoothing= tcfg.get('label_smoothing', 0.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight          = class_weights,
            label_smoothing = tcfg.get('label_smoothing', 0.0),
        )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = tcfg['learning_rate'],
        weight_decay = tcfg['weight_decay'],
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    sched_name = tcfg.get('scheduler', 'cosine')
    if sched_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = tcfg.get('cosine_T_max', tcfg['num_epochs']),
            eta_min = tcfg.get('cosine_eta_min', 1e-6),
        )
    elif sched_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5
        )

    # Warmup wrapper — use ChainedScheduler pattern to avoid SequentialLR
    # internal epoch-passing deprecation warning
    warmup_epochs = tcfg.get('warmup_epochs', 0)
    if warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, scheduler], milestones=[warmup_epochs]
        )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = GradScaler('cuda', enabled=tcfg['use_amp'])

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(log_dir)) if cfg['logging']['use_tensorboard'] else None

    # ── Training loop ─────────────────────────────────────────────────────────
    # Start at -1 so the very first epoch always triggers a checkpoint save,
    # even when val accuracy is 0%.  Without this, a model that never exceeds
    # 0% accuracy (common early in training with 556 classes) never writes
    # best.pth and the evaluation script fails with "checkpoint not found".
    best_val_top1  = -1.0
    patience_count = 0
    patience       = tcfg['patience']
    save_top_k     = tcfg.get('save_top_k', 3)
    top_k_heap     = []  # min-heap of (val_top1, epoch, filepath)
    history        = []

    logger.info("Starting training…")
    for epoch in range(1, tcfg['num_epochs'] + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, cfg, logger
        )
        val_metrics = validate(model, val_loader, criterion, device, cfg)

        # Scheduler step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_metrics['top1'])
        else:
            scheduler.step()

        lr  = optimizer.param_groups[0]['lr']
        dur = time.time() - t0

        logger.info(
            f"Epoch {epoch:3d}/{tcfg['num_epochs']} | "
            f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']*100:.2f}% | "
            f"val loss={val_metrics['loss']:.4f} top1={val_metrics['top1']*100:.2f}% "
            f"top5={val_metrics['top5']*100:.2f}% | "
            f"lr={lr:.2e} | {dur:.0f}s"
        )

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val',   val_metrics['loss'],   epoch)
            writer.add_scalar('Acc/train_top1', train_metrics['top1'], epoch)
            writer.add_scalar('Acc/val_top1',   val_metrics['top1'],   epoch)
            writer.add_scalar('Acc/val_top5',   val_metrics['top5'],   epoch)
            writer.add_scalar('LR', lr, epoch)

        history.append({
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}':   v for k, v in val_metrics.items()},
            'lr': lr,
        })

        # Checkpoint (save top-k)
        ckpt_path = ckpt_dir / f"epoch{epoch:03d}_val{val_metrics['top1']:.4f}.pth"
        ckpt_state = {
            'epoch':      epoch,
            'model_type': args.model,
            'state_dict': model.state_dict(),
            'optimizer':  optimizer.state_dict(),
            'val_top1':   val_metrics['top1'],
            'num_classes': num_classes,
            'config':     cfg,
        }

        is_best = val_metrics['top1'] > best_val_top1
        if is_best:
            best_val_top1  = val_metrics['top1']
            patience_count = 0
            save_checkpoint(ckpt_state, str(ckpt_path),
                            is_best=True,
                            best_filepath=str(ckpt_dir / 'best.pth'))
            logger.info(f"  ★ New best val top-1: {best_val_top1*100:.2f}%")
        else:
            patience_count += 1

        # Top-k heap management
        heapq.heappush(top_k_heap, (val_metrics['top1'], epoch, str(ckpt_path)))
        if len(top_k_heap) > save_top_k:
            worst_score, worst_epoch, worst_path = heapq.heappop(top_k_heap)
            if Path(worst_path).exists() and worst_path != str(ckpt_dir / 'best.pth'):
                Path(worst_path).unlink()

        if not is_best:
            save_checkpoint(ckpt_state, str(ckpt_path))

        # Early stopping
        if patience_count >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save training history
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()

    logger.info(f"Training complete. Best val top-1: {best_val_top1*100:.2f}%")
    logger.info(f"Best checkpoint: {ckpt_dir / 'best.pth'}")
    return best_val_top1


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Train MSL Sign Language Recognition Model')
    parser.add_argument('--config',        default='config/config.yaml')
    parser.add_argument('--model',         default='bilstm',
                        choices=['bilstm', 'transformer', 'stgcn'])
    parser.add_argument('--exp',           default='exp01',
                        help='Experiment name (results saved under results/<exp>/)')
    parser.add_argument('--weighted_loss', action='store_true',
                        help='Use inverse-frequency class weights in loss')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

