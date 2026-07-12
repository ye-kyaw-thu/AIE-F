#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_results.py - Visualise training curves and compare experiments

Generates
─────────
  1. Training curves (loss + accuracy) per experiment
  2. Multi-experiment accuracy comparison bar chart (Validation & Test)
  3. Learning rate schedule overlay
  4. Cross-validation score distribution (if cv_summary.json present)

Usage
─────
  # Plot one experiment
  python src/plot_results.py --exp results/exp01_bilstm

  # Compare multiple experiments (Generates comparison_val.png AND comparison_test.png)
  python src/plot_results.py \
      --exp results/exp_bilstm results/exp_transformer results/exp_stgcn \
      --output results/comparison_val.png \
      --output_test results/comparison_test.png
"""

# === FIX: Handle Jupyter's matplotlib backend conflict ===
import os
_mpl_backend = os.environ.get('MPLBACKEND')
if _mpl_backend and _mpl_backend not in [
    'gtk3agg', 'gtk3cairo', 'gtk4agg', 'gtk4cairo', 'macosx',
    'nbagg', 'notebook', 'qtagg', 'qtcairo', 'qt5agg', 'qt5cairo',
    'tkagg', 'tkcairo', 'webagg', 'wx', 'wxagg', 'wxcairo',
    'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'
]:
    os.environ['MPLBACKEND'] = 'agg'
# === END FIX ===

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


PALETTE = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
           '#00BCD4', '#795548', '#607D8B']


def format_model_name(name: str) -> str:
    """Convert folder name to a readable model name."""
    for prefix in ['exp_', 'cv_', 'run_', 'test_']:
        if name.startswith(prefix):
            name = name[len(prefix):]
    
    acronyms = {
        'bilstm': 'BiLSTM', 'stgcn': 'ST-GCN', 'transformer': 'Transformer',
        'lstm': 'LSTM', 'gru': 'GRU', 'cnn': 'CNN', 'rnn': 'RNN',
        'gcn': 'GCN', 'tcn': 'TCN'
    }
    
    lower_name = name.lower()
    if lower_name in acronyms:
        return acronyms[lower_name]
    
    for key, val in acronyms.items():
        if lower_name.startswith(key):
            suffix = name[len(key):]
            if suffix.startswith('_'):
                suffix = suffix[1:]
            return f"{val} {suffix}" if suffix else val
            
    return name.replace('_', ' ').title()


def get_value(item: dict, key_options: list, default=None):
    """Try multiple key names and return the first one found."""
    for key in key_options:
        if key in item and item[key] is not None:
            return item[key]
    return default


def normalize_accuracy(value, default=0):
    """Convert accuracy to percentage (0-100) range."""
    if value is None:
        return default
    if value > 1.0:
        return value
    return value * 100


def load_history(exp_dir: Path) -> Optional[list]:
    h_path = exp_dir / 'history.json'
    if not h_path.exists():
        return None
    with open(h_path) as f:
        return json.load(f)


def load_cv_summary(exp_dir: Path) -> Optional[dict]:
    cv_path = exp_dir / 'cv_summary.json'
    if not cv_path.exists():
        return None
    with open(cv_path) as f:
        return json.load(f)


def plot_training_curves(exp_dir: Path, save_path: Optional[str] = None):
    history = load_history(exp_dir)
    if history is None:
        print(f"No history.json found in {exp_dir}")
        return

    epochs     = [get_value(h, ['epoch', 'Epoch', 'ep', 'step']) for h in history]
    train_loss = [get_value(h, ['loss', 'train_loss', 'trainLoss', 'Loss', 'train/loss']) for h in history]
    val_loss   = [get_value(h, ['val_loss', 'valid_loss', 'valLoss', 'validation_loss', 'val/loss']) for h in history]
    
    train_top1_raw = [get_value(h, ['top1', 'train_top1', 'train_acc', 'train_accuracy', 
                                    'trainAcc', 'accuracy', 'train/acc', 'train/top1']) for h in history]
    train_top1 = [normalize_accuracy(v) for v in train_top1_raw]
    
    val_top1_raw = [get_value(h, ['val_top1', 'val_acc', 'val_accuracy', 'valAcc', 
                                   'valid_acc', 'validation_accuracy', 'val/acc', 'val/top1']) for h in history]
    val_top1 = [normalize_accuracy(v) for v in val_top1_raw]
    
    val_top5_raw = [get_value(h, ['val_top5', 'val_top_5', 'val_top5_acc', 'val/top5']) for h in history]
    val_top5 = [normalize_accuracy(v) for v in val_top5_raw]
    
    lrs        = [get_value(h, ['lr', 'learning_rate', 'LR', 'train/lr']) for h in history]

    valid_indices = [i for i, e in enumerate(epochs) if e is not None]
    if not valid_indices:
        print(f"No valid epoch data in {exp_dir}")
        return

    epochs     = [epochs[i] for i in valid_indices]
    train_loss = [train_loss[i] if train_loss[i] is not None else float('nan') for i in valid_indices]
    val_loss   = [val_loss[i] if val_loss[i] is not None else float('nan') for i in valid_indices]
    train_top1 = [train_top1[i] for i in valid_indices]
    val_top1   = [val_top1[i] for i in valid_indices]
    val_top5   = [val_top5[i] for i in valid_indices]
    lrs        = [lrs[i] for i in valid_indices]

    has_train_loss = any(not (isinstance(v, float) and np.isnan(v)) for v in train_loss)
    has_val_loss = any(not (isinstance(v, float) and np.isnan(v)) for v in val_loss)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves: {format_model_name(exp_dir.name)}", fontsize=14, fontweight='bold')

    ax = axes[0]
    if has_train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', color=PALETTE[0], linewidth=2)
    else:
        ax.text(0.5, 0.5, 'No train loss data', ha='center', va='center', transform=ax.transAxes)
    if has_val_loss:
        ax.plot(epochs, val_loss, label='Val Loss', color=PALETTE[1], linewidth=2)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Loss'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    has_train_acc = any(v > 0 for v in train_top1)
    has_val_acc = any(v > 0 for v in val_top1)
    
    if has_train_acc:
        ax.plot(epochs, train_top1, label='Train Top-1', color=PALETTE[0], linewidth=2)
    if has_val_acc:
        ax.plot(epochs, val_top1,   label='Val Top-1',   color=PALETTE[1], linewidth=2)
        valid_val_top1 = [(i, v) for i, v in enumerate(val_top1) if v > 0]
        if valid_val_top1:
            best_idx = max(valid_val_top1, key=lambda x: x[1])
            best_epoch = epochs[best_idx[0]]
            best_acc = best_idx[1]
            ax.axvline(best_epoch, color='grey', linestyle=':', alpha=0.7)
            ax.annotate(f'Best {best_acc:.1f}%', xy=(best_epoch, best_acc),
                        xytext=(best_epoch + 2, best_acc - 5),
                        arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)
    
    has_val_top5 = any(v > 0 for v in val_top5)
    if has_val_top5:
        ax.plot(epochs, val_top5,   label='Val Top-5',   color=PALETTE[2], linewidth=2, linestyle='--')
    
    if not has_train_acc and not has_val_acc:
        ax.text(0.5, 0.5, 'No accuracy data', ha='center', va='center', transform=ax.transAxes)
    
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    valid_lrs = [v for v in lrs if v is not None and v > 0]
    if valid_lrs:
        ax.semilogy(epochs, lrs, color=PALETTE[3], linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate (log)')
        ax.set_title('LR Schedule'); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    out = save_path or str(exp_dir / 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {out}")


def plot_comparison(exp_dirs: List[Path], output: str, custom_names: Optional[List[str]] = None, split: str = 'val'):
    """Bar chart comparing best top1 across experiments for a specific split (val/test)."""
    names       = []
    best_top1s  = []
    cv_means    = []
    cv_stds     = []

    for i, exp_dir in enumerate(exp_dirs):
        # Read the precise evaluation metrics for the requested split
        metrics_path = exp_dir / 'evaluation' / f'metrics_{split}.json'
        
        # CV data is usually only relevant for validation
        cv = load_cv_summary(exp_dir) if split == 'val' else None

        if custom_names and i < len(custom_names):
            name = custom_names[i]
        else:
            name = format_model_name(exp_dir.name)

        found_data = False  # <--- ADD THIS FLAG

        # Priority 1: Use the exact metrics JSON
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            names.append(name)
            best_top1s.append(metrics.get('top1_accuracy', 0.0) * 100)
            cv_means.append(None)
            cv_stds.append(None)
            found_data = True  # <--- MARK AS FOUND
            
        # Priority 2: Use CV summary (val only)
        elif cv:
            names.append(name)
            best_top1s.append(cv.get('best_top1', cv.get('mean_top1', 0)))
            cv_means.append(cv.get('mean_top1', 0))
            cv_stds.append(cv.get('std_top1', 0))
            found_data = True  # <--- MARK AS FOUND
            
        # Priority 3: Fallback to history.json (val only, less accurate)
        elif split == 'val':
            history = load_history(exp_dir)
            if history:
                val_top1s = []
                for h in history:
                    v = get_value(h, ['val_top1', 'val_acc', 'val_accuracy', 'valAcc', 
                                       'valid_acc', 'validation_accuracy', 'val/acc', 'val/top1'], 0)
                    val_top1s.append(normalize_accuracy(v))
                
                if any(v > 0 for v in val_top1s):
                    names.append(name)
                    best_top1s.append(max(val_top1s))
                    cv_means.append(None)
                    cv_stds.append(None)
                    found_data = True  # <--- MARK AS FOUND
        
        # REPLACE THE OLD BROKEN CHECK WITH THIS:
        if not found_data:
            print(f"[WARNING] No valid {split} data in {exp_dir}, skipping")

    if not names:
        print(f"No experiment data found for split '{split}'.")
        return

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 2), 6))

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    bars   = ax.bar(x, best_top1s, color=colors, alpha=0.85, width=0.5)

    # CV error bars
    cv_label_added = False
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        if mean is not None:
            label = 'CV Mean±Std' if not cv_label_added else None
            ax.errorbar(x[i], mean, yerr=std, fmt='D', color='black',
                        markersize=6, capsize=5, label=label)
            cv_label_added = True

    # Value labels on top of bars
    for bar, val in zip(bars, best_top1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Model Comparison — {split.capitalize()} Top-1 Accuracy', fontsize=14, fontweight='bold')
    
    # Smart Y-axis limits: Zoom in if accuracy is very high (e.g., 99-100%)
    y_max = max(best_top1s)
    if y_max > 95.0:
        # Start Y-axis closer to the lowest score to make differences visible
        y_min = min(best_top1s) - 2.0
        ax.set_ylim(max(0, y_min), min(101, y_max + 1.5))
    else:
        ax.set_ylim(0, min(105, y_max + 10))
        
    ax.grid(axis='y', alpha=0.3)
    
    if cv_label_added:
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{split.capitalize()} comparison chart saved → {output}")


def plot_cv_distribution(exp_dir: Path):
    cv = load_cv_summary(exp_dir)
    if cv is None:
        return

    scores = cv.get('fold_scores', cv.get('scores', cv.get('fold_accuracies', None)))
    if scores is None:
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    folds = [f'Fold {i}' for i in range(len(scores))]
    bars  = ax.bar(folds, scores, color=PALETTE[:len(scores)], alpha=0.85)
    
    mean_top1 = cv.get('mean_top1', cv.get('mean_accuracy', np.mean(scores)))
    std_top1 = cv.get('std_top1', cv.get('std_accuracy', np.std(scores)))
    
    ax.axhline(mean_top1, color='red', linestyle='--', linewidth=2,
               label=f"Mean: {mean_top1:.1f}%")
    ax.fill_between(range(-1, len(scores)+1),
                    mean_top1 - std_top1,
                    mean_top1 + std_top1,
                    alpha=0.15, color='red', label=f"±Std: {std_top1:.1f}%")

    for bar, v in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax.set_ylim(0, min(105, max(scores) + 8))
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title(f"Cross-Validation Distribution — {format_model_name(exp_dir.name)}", fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    out = str(exp_dir / 'cv_distribution.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CV distribution saved → {out}")


def main():
    parser = argparse.ArgumentParser(description='Plot MSL training results')
    parser.add_argument('--exp',         nargs='+', required=True,
                        help='One or more experiment directories')
    parser.add_argument('--names',       nargs='+', default=None,
                        help='Custom display names for experiments (optional)')
    parser.add_argument('--output',      default='results/comparison_val.png',
                        help='Output path for validation comparison chart')
    parser.add_argument('--output_test', default='results/comparison_test.png',
                        help='Output path for test comparison chart')
    args = parser.parse_args()

    exp_dirs = [Path(e) for e in args.exp]

    # Per-experiment curves
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"Warning: {exp_dir} not found, skipping")
            continue
        plot_training_curves(exp_dir)
        plot_cv_distribution(exp_dir)

    # Multi-experiment comparison (Now generates BOTH Val and Test!)
    if len(exp_dirs) > 1:
        valid = [e for e in exp_dirs if e.exists()]
        if valid:
            plot_comparison(valid, args.output,      args.names, split='val')
            plot_comparison(valid, args.output_test,  args.names, split='test')


if __name__ == '__main__':
    main()

