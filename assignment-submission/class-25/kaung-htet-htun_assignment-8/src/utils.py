#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py - Core utilities for MSL Recognition System

Handles:
  - Annotation file parsing
  - Label vocabulary construction
  - Train/val/test splitting
  - Reproducibility helpers
  - Logging setup
"""

import os
import re
import json
import random
import logging
import hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# ─── Logging ──────────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[str] = None, level=logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout and optionally to a file."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logger.addHandler(fh)

    return logger


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Annotation Parsing ───────────────────────────────────────────────────────

def parse_annotation_file(ann_path: str) -> List[Dict]:
    """
    Parse the MSL annotation file.

    Format (tab-delimited):
        normal_myanmar_text  <TAB>  msl_gloss

    Returns:
        List of dicts with keys: 'idx', 'normal_text', 'msl_gloss', 'label'
        where 'label' is the normal_text (used as class identifier).
    """
    records = []
    ann_path = Path(ann_path)

    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with open(ann_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f):
            line = line.rstrip('\n')
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                # Try space-separated as fallback
                parts = line.split('  ', 1)
            if len(parts) < 2:
                logging.warning(f"Line {line_no+1}: cannot parse → '{line}'")
                continue

            normal_text = parts[0].strip()
            msl_gloss   = parts[1].strip()

            records.append({
                'idx':         line_no,
                'normal_text': normal_text,
                'msl_gloss':   msl_gloss,
                'label':       normal_text,   # class = full Myanmar phrase
            })

    logging.info(f"Parsed {len(records)} annotation entries from {ann_path}")
    return records


def build_label_vocabulary(records: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build label↔index mappings from parsed annotation records.

    Returns:
        label2idx: {'မီးချိတ်': 0, 'သဲ အိတ်': 1, ...}
        idx2label: {0: 'မီးချိတ်', 1: 'သဲ အိတ်', ...}
    """
    # Preserve order of first occurrence
    seen = {}
    for rec in records:
        lbl = rec['label']
        if lbl not in seen:
            seen[lbl] = len(seen)

    label2idx = seen
    idx2label = {v: k for k, v in label2idx.items()}

    logging.info(f"Vocabulary size: {len(label2idx)} unique classes")
    return label2idx, idx2label


def save_label_map(label2idx: Dict[str, int], idx2label: Dict[int, str], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'label2idx': label2idx,
        'idx2label': {str(k): v for k, v in idx2label.items()},
        'num_classes': len(label2idx),
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logging.info(f"Label map saved → {out_path}")


def load_label_map(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    label2idx = payload['label2idx']
    idx2label = {int(k): v for k, v in payload['idx2label'].items()}
    return label2idx, idx2label


# ─── Video ↔ Annotation Matching ──────────────────────────────────────────────

def match_videos_to_annotations(
    video_dir: str,
    records: List[Dict],
    keypoint_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Associate each annotation record with its keypoint (.npy) or video file.

    Matching strategy:
      1. Videos sorted by name → matched 1-to-1 with sorted annotation records
         (assumes video names are numbered or alphabetically ordered to match
          annotation file order).
      2. If a video_id_file (CSV with video_name,ann_idx) exists, use that.

    Each returned record gets extra keys:
        'video_path'   : full path to .mp4 (or None)
        'keypoint_path': full path to .npy (or None)
    """
    video_dir = Path(video_dir)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4'}

    def _numeric_sort_key(path: Path):
        """
        Natural / numeric sort key so that:
          idx20-2.mp4 < idx20-10.mp4 < idx20-31.mp4 < idx20-100.mp4
        Without this, Python's default str sort gives:
          idx20-10 < idx20-100 < idx20-2 < idx20-31   ← WRONG
        which silently maps every video to the wrong annotation.
        """
        stem = path.stem   # e.g. "idx20-31"
        # Split on any run of digits and sort numerically on each piece
        parts = re.split(r'(\d+)', stem)
        return [int(p) if p.isdigit() else p.lower() for p in parts]

    # Collect all video files sorted NUMERICALLY (not alphabetically)
    all_videos = sorted(
        [p for p in video_dir.rglob('*') if p.suffix in video_extensions],
        key=_numeric_sort_key,
    )

    logging.info(f"Found {len(all_videos)} video files in {video_dir}")

    if len(all_videos) != len(records):
        logging.warning(
            f"Video count ({len(all_videos)}) ≠ annotation count ({len(records)}). "
            "Matching by position up to min length."
        )

    matched = []
    for i, rec in enumerate(records):
        entry = dict(rec)
        if i < len(all_videos):
            vp = all_videos[i]
            entry['video_path'] = str(vp)
            # Derive keypoint path
            if keypoint_dir:
                kp_path = Path(keypoint_dir) / vp.relative_to(video_dir).with_suffix('.npy')
                entry['keypoint_path'] = str(kp_path)
            else:
                entry['keypoint_path'] = None
        else:
            entry['video_path'] = None
            entry['keypoint_path'] = None

        matched.append(entry)

    return matched


# ─── Data Splitting ───────────────────────────────────────────────────────────

def create_splits(
    records: List[Dict],
    label2idx: Dict[str, int],
    test_ratio: float = 0.15,
    val_ratio: float  = 0.15,
    seed: int         = 42,
    output_path: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Create train/val/test splits.

    Strategy
    ────────
    For the MSL4Emergency dataset almost every class has only ONE sample,
    so sklearn's stratified split is impossible (requires ≥2 per class).

    We instead use a simple reproducible random shuffle split.
    This is the correct and honest approach:
      • All 558 raw videos are split randomly (no augmentation leakage).
      • Augmentation is applied ONLY to the training portion afterwards.
      • Val/test are always evaluated on original, un-augmented keypoints.

    Returns dict with keys 'train', 'val', 'test' containing record indices.
    """
    labels  = [label2idx[r['label']] for r in records]
    label_counts = Counter(labels)
    min_count    = min(label_counts.values())

    # Check whether stratified split is feasible
    n_min_needed = 2   # sklearn requires at least 2 per class for stratify
    can_stratify = (min_count >= n_min_needed)

    if not can_stratify:
        logging.warning(
            f"Most classes have only {min_count} sample(s) — "
            "stratified splitting is not possible. "
            "Falling back to RANDOM (non-stratified) split. "
            "This is expected for the MSL4Emergency dataset where "
            "each sign has exactly one video. "
            "Augmentation will be applied to training keypoints only."
        )

    indices = list(range(len(records)))

    # Seed for reproducibility
    rng = random.Random(seed)
    shuffled = list(indices)
    rng.shuffle(shuffled)

    n        = len(shuffled)
    n_test   = max(1, int(round(n * test_ratio)))
    n_val    = max(1, int(round(n * val_ratio)))
    n_train  = n - n_test - n_val

    train_idx = sorted(shuffled[:n_train])
    val_idx   = sorted(shuffled[n_train: n_train + n_val])
    test_idx  = sorted(shuffled[n_train + n_val:])

    splits = {
        'train': train_idx,
        'val':   val_idx,
        'test':  test_idx,
    }

    logging.info(
        f"Data split (random) → train: {len(train_idx)}, "
        f"val: {len(val_idx)}, test: {len(test_idx)}"
    )
    logging.info(
        f"  Classes in train: {len(set(labels[i] for i in train_idx))}, "
        f"val: {len(set(labels[i] for i in val_idx))}, "
        f"test: {len(set(labels[i] for i in test_idx))}"
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=2)
        logging.info(f"Splits saved → {output_path}")

    return splits


def create_kfold_splits(
    records: List[Dict],
    label2idx: Dict[str, int],
    n_folds: int = 5,
    seed: int    = 42,
    output_path: Optional[str] = None,
) -> List[Dict[str, List[int]]]:
    """
    Create K-Fold splits for cross-validation.

    Uses standard KFold (not StratifiedKFold) because the MSL4Emergency
    dataset has predominantly 1 sample per class, making stratification
    impossible.  The fold boundaries are reproducible via `seed`.

    Returns list of {'fold', 'train', 'val'} dicts, one per fold.
    """
    indices = np.arange(len(records))

    kf     = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds  = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        folds.append({
            'fold':  fold,
            'train': train_idx.tolist(),
            'val':   val_idx.tolist(),
        })
        logging.info(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(folds, f, indent=2)
        logging.info(f"K-fold splits saved → {output_path}")

    return folds


def load_splits(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


# ─── Dataset Statistics ───────────────────────────────────────────────────────

def print_dataset_stats(records: List[Dict], label2idx: Dict[str, int], split: str = "full"):
    """Print class distribution and dataset statistics."""
    labels        = [r['label'] for r in records]
    counter       = Counter(labels)
    # Only count classes actually present in this split
    present_cls   = len(counter)
    all_cls       = len(label2idx)
    counts        = list(counter.values())

    print(f"\n{'='*60}")
    print(f"Dataset Statistics [{split}]")
    print(f"{'='*60}")
    print(f"  Total samples    : {len(records)}")
    print(f"  Classes (present): {present_cls} / {all_cls} total")
    print(f"  Samples/class    : min={min(counts)}, "
          f"max={max(counts)}, "
          f"mean={np.mean(counts):.2f}")

    # Distribution of class sizes (important for 1-sample datasets)
    size_dist = Counter(counts)
    print(f"  Size distribution:")
    for size in sorted(size_dist.keys()):
        n = size_dist[size]
        pct = n / present_cls * 100
        print(f"    {size} sample(s) per class : {n:4d} classes  ({pct:.1f}%)")

    if split == 'full' or present_cls <= 30:
        print(f"\n  Sample listing (first 20 classes):")
        for label, cnt in counter.most_common(20):
            print(f"    {label[:40]:<40} {cnt:3d}")
    print(f"{'='*60}\n")


# ─── Sequence Utilities ────────────────────────────────────────────────────────

def pad_sequence_to_length(
    seq: np.ndarray,
    target_len: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Pad or truncate a sequence to target_len.
    seq shape: (T, ...) → (target_len, ...)
    """
    T = seq.shape[0]
    if T >= target_len:
        return seq[:target_len]
    pad_shape = (target_len - T,) + seq.shape[1:]
    padding   = np.full(pad_shape, pad_value, dtype=seq.dtype)
    return np.concatenate([seq, padding], axis=0)


def compute_sequence_lengths(keypoint_dir: str, records: List[Dict]) -> Dict[int, int]:
    """Load all .npy files and record their lengths (T dimension)."""
    lengths = {}
    for rec in records:
        kp_path = rec.get('keypoint_path')
        if kp_path and Path(kp_path).exists():
            kp = np.load(kp_path, mmap_mode='r')
            lengths[rec['idx']] = kp.shape[0]
    return lengths


# ─── Checkpoint Helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
):
    """Save model checkpoint."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    if is_best and best_filepath:
        import shutil
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath: str, device: torch.device) -> dict:
    """Load model checkpoint."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    return torch.load(filepath, map_location=device)


# ─── Misc ─────────────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select best available device."""
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logging.info(f"Using GPU: {gpu} ({mem:.1f} GB)")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    return device


def compute_class_weights(
    records: List[Dict],
    label2idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced data.

    For the MSL4Emergency dataset (1 sample/class in original data):
    after augmentation all training classes have equal counts, so
    weights will be uniform — that is correct and expected.
    Classes absent from the training split get weight=1.0 (neutral).
    """
    labels  = [label2idx[r['label']] for r in records]
    counter = Counter(labels)
    n_cls   = len(label2idx)
    total   = len(labels)

    weights = torch.ones(n_cls)   # default: neutral weight for unseen classes
    for cls_idx, cnt in counter.items():
        if cnt > 0:
            weights[cls_idx] = total / (n_cls * cnt)

    return weights.to(device)

