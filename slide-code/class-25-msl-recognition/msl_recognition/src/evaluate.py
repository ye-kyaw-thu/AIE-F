#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py - Comprehensive evaluation for MSL Recognition models

Computes
────────
  Top-1, Top-5 Accuracy
  Precision, Recall, F1 (macro / weighted)
  Per-class metrics
  Confusion matrix (saved as PNG + JSON)
  Prediction CSV with ground-truth vs predicted labels

Usage
─────
  python src/evaluate.py \
      --checkpoint results/exp01_bilstm/checkpoints/best.pth \
      --config     config/config.yaml \
      --split      test
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
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).parent))
from dataset   import MSLDataset, MSLDatasetGCN
from models    import build_model
from utils     import (
    get_logger, set_seed, get_device,
    parse_annotation_file, build_label_vocabulary,
    match_videos_to_annotations, load_splits, load_label_map,
)

torch.backends.cudnn.deterministic = True


# ─── Prediction collection ────────────────────────────────────────────────────

@torch.no_grad()
def collect_predictions(model, loader, device, model_type: str, use_amp: bool):
    """Run model over loader; return all logits, labels, indices."""
    model.eval()
    all_logits = []
    all_labels = []
    all_idxs   = []

    for batch in loader:
        kp     = batch['keypoints'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        mask   = batch['mask'].to(device, non_blocking=True)
        lengths= batch['length'].to(device, non_blocking=True)

        with autocast('cuda', enabled=use_amp):
            if model_type == 'bilstm':
                logits = model(kp, lengths=lengths, mask=mask)
            elif model_type == 'transformer':
                logits = model(kp, mask=mask)
            else:  # stgcn
                logits = model(kp)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
        all_idxs.extend(batch['idx'] if isinstance(batch['idx'], list) else batch['idx'].tolist())

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_labels, dim=0),
        all_idxs,
    )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_all_metrics(
    logits:    torch.Tensor,
    labels:    torch.Tensor,
    idx2label: dict,
    output_dir: str,
    split_name: str = 'test',
    plot_cm:   bool = True,
    logger = None,
):
    """Compute and save comprehensive metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true  = labels.numpy()
    y_prob  = torch.softmax(logits, dim=1).numpy()
    y_pred  = logits.argmax(dim=1).numpy()
    n_cls   = logits.shape[1]

    # Top-k accuracy
    top1 = float((y_pred == y_true).mean())

    # top-5: only meaningful when n_cls >= 5 and each class has ≥1 sample in split
    try:
        present_labels = sorted(np.unique(y_true).tolist())
        k5 = min(5, len(present_labels))
        top5 = float(top_k_accuracy_score(y_true, y_prob, k=k5, labels=list(range(n_cls))))
    except Exception:
        # Fallback: manual top-5
        top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
        top5 = float(np.mean([y_true[i] in top5_preds[i] for i in range(len(y_true))]))

    # Macro and weighted precision/recall/F1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )

    summary = {
        'split':          split_name,
        'num_samples':    int(len(y_true)),
        'num_classes':    n_cls,
        'top1_accuracy':  float(top1),
        'top5_accuracy':  float(top5),
        'precision_macro':   float(prec_macro),
        'recall_macro':      float(rec_macro),
        'f1_macro':          float(f1_macro),
        'precision_weighted': float(prec_w),
        'recall_weighted':    float(rec_w),
        'f1_weighted':        float(f1_w),
    }

    # Print summary
    bar = '─' * 55
    print(f"\n{bar}")
    print(f"  Evaluation Results [{split_name.upper()}]")
    print(f"{bar}")
    print(f"  Samples       : {summary['num_samples']}")
    print(f"  Classes       : {summary['num_classes']}")
    print(f"  Top-1 Accuracy: {top1*100:.2f}%")
    print(f"  Top-5 Accuracy: {top5*100:.2f}%")
    print(f"  Precision (M) : {prec_macro*100:.2f}%")
    print(f"  Recall    (M) : {rec_macro*100:.2f}%")
    print(f"  F1        (M) : {f1_macro*100:.2f}%")
    print(f"{bar}\n")

    # Save summary JSON
    with open(output_dir / f'metrics_{split_name}.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Per-class report — only include classes actually present in this split
    present_labels = sorted(np.unique(y_true).tolist())
    present_names  = [idx2label.get(i, str(i)) for i in present_labels]
    report = classification_report(
        y_true, y_pred,
        labels        = present_labels,
        target_names  = present_names,
        zero_division = 0,
        output_dict   = True,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(output_dir / f'per_class_{split_name}.csv', encoding='utf-8-sig')

    # Pretty-print top / bottom classes
    per_cls = {k: v for k, v in report.items() if k not in ['accuracy', 'macro avg', 'weighted avg']}
    sorted_f1 = sorted(per_cls.items(), key=lambda x: x[1]['f1-score'], reverse=True)
    print("Top-10 classes by F1:")
    for name, m in sorted_f1[:10]:
        print(f"  {name[:35]:<35}  F1={m['f1-score']:.3f}  "
              f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
              f"Support={int(m['support'])}")
    print("\nBottom-10 classes by F1:")
    for name, m in sorted_f1[-10:]:
        print(f"  {name[:35]:<35}  F1={m['f1-score']:.3f}  "
              f"Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
              f"Support={int(m['support'])}")

    # Confusion matrix
    if plot_cm:
        # Only use classes that are actually present in this split
        present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        cm      = confusion_matrix(y_true, y_pred, labels=present_labels)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        np.save(output_dir / f'confusion_matrix_{split_name}.npy', cm)

        n_present = len(present_labels)

        # For large class counts (>60): plot a compact pixel-map without labels.
        # For small counts (<=60): full annotated heatmap.
        if n_present > 60:
            fig, ax = plt.subplots(figsize=(14, 12))
            im = ax.imshow(cm_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1,
                           interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.set_xlabel(f'Predicted class index (0–{n_present-1})', fontsize=11)
            ax.set_ylabel(f'True class index (0–{n_present-1})',      fontsize=11)
            ax.set_title(
                f'Confusion Matrix [{split_name}]  '
                f'({n_present} classes present, row-normalised)\n'
                f'Diagonal = correct predictions', fontsize=12
            )
            # Mark diagonal clearly
            diag_acc = np.diag(cm_norm).mean()
            ax.set_title(
                f'Confusion Matrix [{split_name}]  ({n_present}/{n_cls} classes)\n'
                f'Mean diagonal (accuracy): {diag_acc*100:.1f}%', fontsize=12
            )
        else:
            tick_labels = [idx2label.get(i, str(i))[:18] for i in present_labels]
            fig_size    = max(10, n_present // 2)
            fig, ax     = plt.subplots(figsize=(fig_size, fig_size))
            sns.heatmap(
                cm_norm, ax=ax,
                xticklabels=tick_labels, yticklabels=tick_labels,
                cmap='Blues', vmin=0, vmax=1,
                annot=True, fmt='.2f', linewidths=0.4,
                annot_kws={'size': max(5, 9 - n_present // 8)},
            )
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True',      fontsize=12)
            plt.xticks(rotation=90, fontsize=max(5, 8 - n_present // 10))
            plt.yticks(rotation=0,  fontsize=max(5, 8 - n_present // 10))
            ax.set_title(f'Confusion Matrix [{split_name}]  (row-normalised)', fontsize=13)

        plt.tight_layout()
        plt.savefig(output_dir / f'confusion_matrix_{split_name}.png', dpi=150)
        plt.close()
        print(f"Confusion matrix saved → {output_dir / f'confusion_matrix_{split_name}.png'}")

    return summary


# ─── Prediction CSV ───────────────────────────────────────────────────────────

def save_predictions(
    logits:    torch.Tensor,
    labels:    torch.Tensor,
    idxs:      list,
    records:   list,
    idx2label: dict,
    output_path: str,
):
    """Save per-sample predictions as CSV."""
    y_true = labels.numpy()
    y_pred = logits.argmax(dim=1).numpy()
    probs  = torch.softmax(logits, dim=1).numpy()

    rows = []
    for i, (true, pred, idx) in enumerate(zip(y_true, y_pred, idxs)):
        rec  = records[idx] if idx < len(records) else {}
        rows.append({
            'sample_idx':    int(idx),
            'true_label':    idx2label.get(int(true), str(true)),
            'pred_label':    idx2label.get(int(pred), str(pred)),
            'correct':       int(true) == int(pred),
            'true_idx':      int(true),
            'pred_idx':      int(pred),
            'confidence':    float(probs[i, int(pred)]),
            'video_path':    rec.get('video_path', ''),
            'msl_gloss':     rec.get('msl_gloss', ''),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Predictions saved → {output_path}  "
          f"({df['correct'].sum()}/{len(df)} correct)")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate MSL Recognition Model')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--config',     default='config/config.yaml')
    parser.add_argument('--split',      default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', default=None,
                        help='Where to save results (default: checkpoint directory)')
    parser.add_argument('--no_cm',      action='store_true', help='Skip confusion matrix')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    set_seed(42)
    device = get_device()

    logger = get_logger('evaluate')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_type = ckpt.get('model_type', 'bilstm')
    logger.info(f"Loaded checkpoint: {args.checkpoint}  Model: {model_type}")

    # Data — load from all-class augmented manifest
    records              = parse_annotation_file(dcfg['annotation_file'])
    label2idx, idx2label = load_label_map(dcfg['label_map_file'])
    records              = match_videos_to_annotations(
        dcfg['video_dir'], records, dcfg['keypoint_dir']
    )

    aug_manifest_path = Path(dcfg['augmented_dir']) / 'augmented_manifest.json'
    with open(aug_manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)

    if isinstance(manifest, dict) and args.split in manifest:
        subset = manifest[args.split]
    else:
        # Fallback: old flat format — only test split makes sense here
        logger.warning("Old manifest format — falling back to splits.json for this split")
        splits = load_splits(dcfg['split_file'])
        subset = [records[i] for i in splits[args.split]]

    logger.info(f"Evaluating {len(subset)} samples from '{args.split}' split")

    DatasetClass = MSLDatasetGCN if model_type == 'stgcn' else MSLDataset
    dataset = DatasetClass(
        subset, label2idx,
        max_seq_len = dcfg['max_seq_len'],
        flatten     = (model_type != 'stgcn'),
        augmentor   = None,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  = cfg['training']['batch_size'],
        shuffle     = False,
        num_workers = cfg['training']['num_workers'],
        pin_memory  = True,
    )

    # Model
    num_classes = ckpt.get('num_classes', len(label2idx))
    model = build_model(model_type, cfg, num_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    logger.info(f"Model loaded. Evaluating {len(subset)} samples…")

    # Predict
    logits, labels, idxs = collect_predictions(
        model, loader, device, model_type,
        use_amp = cfg['training']['use_amp']
    )

    # Output dir
    out_dir = args.output_dir or str(Path(args.checkpoint).parent.parent / 'evaluation')

    # Metrics
    summary = compute_all_metrics(
        logits, labels, idx2label,
        output_dir = out_dir,
        split_name = args.split,
        plot_cm    = not args.no_cm,
        logger     = logger,
    )

    # Prediction CSV
    save_predictions(
        logits, labels, idxs, records, idx2label,
        output_path = str(Path(out_dir) / f'predictions_{args.split}.csv'),
    )

    print(f"\nAll evaluation outputs saved to: {out_dir}")


if __name__ == '__main__':
    main()

