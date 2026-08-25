#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py - One-shot data preparation for MSL Recognition

Steps performed
───────────────
  1. Parse annotation file (TSV: normal_text <TAB> msl_gloss)
  2. Build label vocabulary and save label_map.json
  3. Match video files to annotation records (positional matching)
  4. Print dataset statistics
  5. Create stratified train / val / test splits  → splits.json
  6. Create stratified K-fold splits              → kfold_splits.json
  7. Verify keypoint files exist (if already extracted)

Usage
─────
  python src/prepare_data.py --config config/config.yaml
  python src/prepare_data.py --config config/config.yaml --verify_keypoints
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_logger,
    parse_annotation_file,
    build_label_vocabulary,
    save_label_map,
    match_videos_to_annotations,
    create_splits,
    create_kfold_splits,
    print_dataset_stats,
    set_seed,
)


def verify_keypoints(records: list, logger) -> dict:
    """Check which keypoint .npy files exist."""
    total   = len(records)
    found   = sum(1 for r in records if r.get('keypoint_path') and Path(r['keypoint_path']).exists())
    missing = total - found
    logger.info(f"Keypoint verification: {found}/{total} found, {missing} missing")
    if missing > 0:
        logger.warning("Run extract_keypoints.py before training!")
        missing_list = [r.get('video_path', '?') for r in records
                        if not (r.get('keypoint_path') and Path(r['keypoint_path']).exists())]
        for p in missing_list[:10]:
            logger.warning(f"  Missing keypoints for: {p}")
        if len(missing_list) > 10:
            logger.warning(f"  ... and {len(missing_list)-10} more")
    return {'total': total, 'found': found, 'missing': missing}


def main():
    parser = argparse.ArgumentParser(description='Prepare MSL dataset: splits, label map, stats')
    parser.add_argument('--config',           default='config/config.yaml')
    parser.add_argument('--verify_keypoints', action='store_true',
                        help='Check whether keypoint .npy files already exist')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    set_seed(dcfg.get('seed', 42))

    log_dir = Path(cfg['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger('prepare_data', log_file=str(log_dir / 'prepare_data.log'))

    logger.info("=" * 60)
    logger.info("MSL Data Preparation")
    logger.info("=" * 60)

    # ── 1. Parse annotations ──────────────────────────────────────────────────
    logger.info(f"Parsing annotation file: {dcfg['annotation_file']}")
    records = parse_annotation_file(dcfg['annotation_file'])
    logger.info(f"Total annotation records: {len(records)}")

    # ── 2. Label vocabulary ───────────────────────────────────────────────────
    label2idx, idx2label = build_label_vocabulary(records)
    num_classes = len(label2idx)
    logger.info(f"Vocabulary built: {num_classes} unique classes")

    Path(dcfg['label_map_file']).parent.mkdir(parents=True, exist_ok=True)
    save_label_map(label2idx, idx2label, dcfg['label_map_file'])
    logger.info(f"Label map saved → {dcfg['label_map_file']}")

    # ── 3. Match videos ───────────────────────────────────────────────────────
    logger.info(f"Matching videos from: {dcfg['video_dir']}")
    records = match_videos_to_annotations(
        video_dir    = dcfg['video_dir'],
        records      = records,
        keypoint_dir = dcfg.get('keypoint_dir'),
    )

    # ── 4. Stats ──────────────────────────────────────────────────────────────
    print_dataset_stats(records, label2idx, split='full')

    # Research note for paper
    logger.info(
        "NOTE: MSL4Emergency has ~1 video per sign class. "
        "The pipeline handles this via: "
        "(1) random (non-stratified) train/val/test split on raw videos, "
        "(2) heavy keypoint-space augmentation on training split only (20×), "
        "(3) val/test evaluated on original un-augmented keypoints. "
        "This is the standard approach for low-resource sign language research."
    )

    # ── 5. Train/val/test splits ──────────────────────────────────────────────
    logger.info("Creating random train/val/test splits (1-sample-per-class safe) …")
    splits = create_splits(
        records     = records,
        label2idx   = label2idx,
        test_ratio  = dcfg.get('test_ratio', 0.15),
        val_ratio   = dcfg.get('val_ratio',  0.15),
        seed        = dcfg.get('seed', 42),
        output_path = dcfg['split_file'],
    )

    # ── 6. K-Fold splits ──────────────────────────────────────────────────────
    n_folds     = dcfg.get('n_folds', 5)
    kfold_path  = str(Path(dcfg['split_file']).parent / 'kfold_splits.json')
    logger.info(f"Creating {n_folds}-fold CV splits (standard KFold) …")
    create_kfold_splits(
        records     = records,
        label2idx   = label2idx,
        n_folds     = n_folds,
        seed        = dcfg.get('seed', 42),
        output_path = kfold_path,
    )

    # ── 7. Verify keypoints (optional) ───────────────────────────────────────
    if args.verify_keypoints:
        logger.info("Verifying keypoint files …")
        kp_stats = verify_keypoints(records, logger)
        with open(log_dir / 'keypoint_verify.json', 'w') as f:
            json.dump(kp_stats, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Data preparation complete!")
    print("=" * 60)
    print(f"  Annotation records : {len(records)}")
    print(f"  Unique classes     : {num_classes}")
    print(f"  Train samples      : {len(splits['train'])}  (→ augmented ×20 in next step)")
    print(f"  Val   samples      : {len(splits['val'])}   (original keypoints only)")
    print(f"  Test  samples      : {len(splits['test'])}   (original keypoints only)")
    print(f"  K-Fold splits      : {n_folds} folds  →  {kfold_path}")
    print(f"  Label map          : {dcfg['label_map_file']}")
    print(f"  Splits file        : {dcfg['split_file']}")
    print(f"")
    print(f"  ⚠  Note: ~1 video/class → random (non-stratified) split used.")
    print(f"     Val/test may not cover all {num_classes} classes — this is expected.")
    print(f"     Evaluation metrics will reflect classes seen in each split.")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

