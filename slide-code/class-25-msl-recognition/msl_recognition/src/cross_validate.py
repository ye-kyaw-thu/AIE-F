#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_validate.py - K-Fold Cross-Validation for MSL Recognition

Correct experimental design for 1-sample-per-class datasets
─────────────────────────────────────────────────────────────
The same split-design problem that caused 0% in train/val/test also
affects naive K-Fold: splitting 558 raw records into K folds gives
~446 train / ~112 val with ZERO class overlap → 0% every fold.

Fix: fold over the AUGMENTED training pool from the manifest.
  - The manifest['train'] has 10,602 records (558 classes × 19 aug copies)
  - Every class has 19 copies → every class appears in BOTH train and val
    within every fold
  - Final test is always manifest['test'] (558 originals, strictest eval)

K-Fold structure (e.g. K=5, 10,602 train samples):
  Each fold:
    train = 4/5 × 10,602 ≈ 8,481 augmented samples  (all 558 classes)
    val   = 1/5 × 10,602 ≈ 2,121 augmented samples  (all 558 classes)
    test  = 558 originals (same across all folds, for final comparison)

Usage
─────
  python src/cross_validate.py \
      --config config/config.yaml \
      --model  bilstm \
      --exp    cv_bilstm \
      --folds  5
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold
from torch.amp import GradScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from dataset import MSLDataset, MSLDatasetGCN
from models  import build_model
from utils   import (
    get_logger, set_seed, get_device,
    load_label_map, compute_class_weights, save_checkpoint,
)
from train   import train_one_epoch, validate, FocalLoss


# ─── Single fold ──────────────────────────────────────────────────────────────

def run_fold(
    fold_idx:   int,
    train_recs: list,
    val_recs:   list,
    test_recs:  list,
    label2idx:  dict,
    cfg:        dict,
    model_type: str,
    exp_dir:    Path,
    device:     torch.device,
    logger,
) -> dict:
    """Train and evaluate one fold. Returns result dict."""

    tcfg     = cfg['training']
    dcfg     = cfg['data']
    fold_dir = exp_dir / f'fold_{fold_idx:02d}'
    fold_dir.mkdir(parents=True, exist_ok=True)

    set_seed(tcfg['seed'] + fold_idx)

    DatasetClass = MSLDatasetGCN if model_type == 'stgcn' else MSLDataset
    flatten      = (model_type != 'stgcn')
    max_T        = dcfg['max_seq_len']

    # No on-the-fly augmentation — data is already augmented from manifest
    train_ds = DatasetClass(train_recs, label2idx, max_seq_len=max_T,
                            flatten=flatten, augmentor=None)
    val_ds   = DatasetClass(val_recs,   label2idx, max_seq_len=max_T,
                            flatten=flatten, augmentor=None)
    test_ds  = DatasetClass(test_recs,  label2idx, max_seq_len=max_T,
                            flatten=flatten, augmentor=None)

    train_loader = DataLoader(train_ds, batch_size=tcfg['batch_size'], shuffle=True,
                              drop_last=True,  num_workers=tcfg['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg['batch_size'], shuffle=False,
                              drop_last=False, num_workers=tcfg['num_workers'], pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=tcfg['batch_size'], shuffle=False,
                              drop_last=False, num_workers=tcfg['num_workers'], pin_memory=True)

    logger.info(
        f"Fold {fold_idx} | "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    num_classes   = len(label2idx)
    model         = build_model(model_type, cfg, num_classes).to(device)
    class_weights = compute_class_weights(train_recs, label2idx, device)

    if tcfg.get('loss', 'cross_entropy') == 'focal':
        criterion = FocalLoss(
            gamma=tcfg.get('focal_gamma', 2.0),
            weight=class_weights,
            label_smoothing=tcfg.get('label_smoothing', 0.0),
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=tcfg.get('label_smoothing', 0.0),
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg['learning_rate'], weight_decay=tcfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg['num_epochs'], eta_min=tcfg.get('cosine_eta_min', 1e-6)
    )
    scaler = GradScaler('cuda', enabled=tcfg['use_amp'])

    # Start at -1 so epoch 1 always writes a checkpoint even at 0% accuracy
    best_val_top1  = -1.0
    patience_count = 0
    patience       = tcfg['patience']
    history        = []

    for epoch in range(1, tcfg['num_epochs'] + 1):
        t0            = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion,
                                        scaler, device, cfg, logger)
        val_metrics   = validate(model, val_loader, criterion, device, cfg)
        scheduler.step()

        history.append({
            'epoch': epoch,
            **train_metrics,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })

        is_best = val_metrics['top1'] > best_val_top1
        if is_best:
            best_val_top1  = val_metrics['top1']
            patience_count = 0
            save_checkpoint(
                {
                    'epoch': epoch, 'model_type': model_type,
                    'state_dict': model.state_dict(),
                    'val_top1': best_val_top1,
                    'num_classes': num_classes, 'config': cfg,
                },
                str(fold_dir / 'best.pth'),
                is_best=False,
            )
        else:
            patience_count += 1

        if epoch % 10 == 0 or is_best:
            logger.info(
                f"  Fold {fold_idx} | Epoch {epoch:3d} | "
                f"train={train_metrics['top1']*100:.1f}% | "
                f"val={val_metrics['top1']*100:.2f}%"
                f"{'  ★' if is_best else ''} | {time.time()-t0:.0f}s"
            )

        if patience_count >= patience:
            logger.info(f"  Fold {fold_idx}: early stopping at epoch {epoch}")
            break

    with open(fold_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate best checkpoint on test set
    test_ckpt = torch.load(str(fold_dir / 'best.pth'), map_location=device)
    model.load_state_dict(test_ckpt['state_dict'])
    test_metrics = validate(model, test_loader, criterion, device, cfg)
    logger.info(
        f"  Fold {fold_idx} FINAL | "
        f"best_val={best_val_top1*100:.2f}% | "
        f"test={test_metrics['top1']*100:.2f}%"
    )

    return {
        'fold':          fold_idx,
        'best_val_top1': best_val_top1,
        'test_top1':     test_metrics['top1'],
        'best_epoch':    max(history, key=lambda h: h.get('val_top1', 0))['epoch'],
        'checkpoint':    str(fold_dir / 'best.pth'),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='K-Fold Cross-Validation for MSL Recognition '
                    '(folds over augmented manifest, all-class design)'
    )
    parser.add_argument('--config',     default='config/config.yaml')
    parser.add_argument('--model',      default='bilstm',
                        choices=['bilstm', 'transformer', 'stgcn'])
    parser.add_argument('--exp',        default='cv_exp01')
    parser.add_argument('--folds',      type=int, default=None)
    parser.add_argument('--fold_start', type=int, default=0,
                        help='Resume from this fold index')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg    = cfg['data']
    set_seed(cfg['training']['seed'])

    exp_dir = Path('results') / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger('cross_validate', log_file=str(exp_dir / 'cv.log'))
    device = get_device()
    n_folds = args.folds or dcfg.get('n_folds', 5)

    # ── Load label map ────────────────────────────────────────────────────────
    label2idx, idx2label = load_label_map(dcfg['label_map_file'])

    # ── Load augmented manifest ───────────────────────────────────────────────
    aug_manifest_path = Path(dcfg['augmented_dir']) / 'augmented_manifest.json'
    if not aug_manifest_path.exists():
        logger.error(
            f"Augmented manifest not found: {aug_manifest_path}\n"
            "Run: bash scripts/03_augment_data.sh"
        )
        sys.exit(1)

    with open(aug_manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    if not (isinstance(manifest, dict) and 'train' in manifest):
        logger.error(
            "Old manifest format detected. "
            "Re-run: bash scripts/03_augment_data.sh --force"
        )
        sys.exit(1)

    all_train_recs = manifest['train']   # 10,602 aug samples, all 558 classes
    test_recs      = manifest['test']    # 558 originals — used as test in every fold

    logger.info("=" * 60)
    logger.info(f"Cross-Validation | model={args.model} | folds={n_folds}")
    logger.info("Folding over augmented training pool (all-class design)")
    logger.info(f"  Aug train pool : {len(all_train_recs)} samples")
    logger.info(f"  Test set       : {len(test_recs)} originals (fixed, all folds)")
    logger.info("=" * 60)

    # ── Build K-Fold indices over the augmented train pool ───────────────────
    # Every class has 19 aug copies in the pool → every class appears in
    # both train and val within each fold → meaningful accuracy signal.
    indices = np.arange(len(all_train_recs))
    kf      = KFold(n_splits=n_folds, shuffle=True,
                    random_state=cfg['training']['seed'])

    fold_splits = [
        {'fold': i, 'train': tr.tolist(), 'val': va.tolist()}
        for i, (tr, va) in enumerate(kf.split(indices))
    ]

    # Save the fold splits for reproducibility
    with open(exp_dir / 'cv_fold_splits.json', 'w') as f:
        json.dump(fold_splits, f, indent=2)

    fold_results = []

    for fold_data in fold_splits:
        fold_idx = fold_data['fold']
        if fold_idx < args.fold_start:
            logger.info(f"Skipping fold {fold_idx}")
            continue

        train_recs = [all_train_recs[i] for i in fold_data['train']]
        val_recs   = [all_train_recs[i] for i in fold_data['val']]

        result = run_fold(
            fold_idx   = fold_idx,
            train_recs = train_recs,
            val_recs   = val_recs,
            test_recs  = test_recs,
            label2idx  = label2idx,
            cfg        = cfg,
            model_type = args.model,
            exp_dir    = exp_dir,
            device     = device,
            logger     = logger,
        )
        fold_results.append(result)
        logger.info(
            f"Fold {fold_idx} done | "
            f"val={result['best_val_top1']*100:.2f}% | "
            f"test={result['test_top1']*100:.2f}%"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    val_scores  = [r['best_val_top1'] for r in fold_results]
    test_scores = [r['test_top1']     for r in fold_results]
    best_fold   = max(fold_results, key=lambda r: r['best_val_top1'])

    summary = {
        'model':             args.model,
        'n_folds':           len(fold_results),
        'fold_val_scores':   [round(s * 100, 2) for s in val_scores],
        'fold_test_scores':  [round(s * 100, 2) for s in test_scores],
        'mean_val_top1':     round(float(np.mean(val_scores))  * 100, 2),
        'std_val_top1':      round(float(np.std(val_scores))   * 100, 2),
        'mean_test_top1':    round(float(np.mean(test_scores)) * 100, 2),
        'std_test_top1':     round(float(np.std(test_scores))  * 100, 2),
        'best_fold':         best_fold['fold'],
        'best_top1':         round(best_fold['best_val_top1']  * 100, 2),
        # Keep these keys for plot_results.py compatibility
        'fold_scores':       [round(s * 100, 2) for s in val_scores],
        'mean_top1':         round(float(np.mean(val_scores))  * 100, 2),
        'std_top1':          round(float(np.std(val_scores))   * 100, 2),
        'best_checkpoint':   best_fold['checkpoint'],
        'fold_details':      fold_results,
    }

    with open(exp_dir / 'cv_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Cross-Validation Summary  [{args.model}]")
    print(f"{'='*65}")
    print(f"  {'Fold':<8} {'Val Top-1':>10} {'Test Top-1':>12}")
    print(f"  {'─'*34}")
    for r in fold_results:
        print(f"  Fold {r['fold']}   {r['best_val_top1']*100:>8.2f}%   "
              f"{r['test_top1']*100:>9.2f}%")
    print(f"  {'─'*34}")
    print(f"  {'Mean±Std':<8} "
          f"{summary['mean_val_top1']:>7.2f}±{summary['std_val_top1']:.2f}%   "
          f"{summary['mean_test_top1']:>7.2f}±{summary['std_test_top1']:.2f}%")
    print(f"  Best fold : {best_fold['fold']}  (val {summary['best_top1']:.2f}%)")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()

