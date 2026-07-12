#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
augment.py - Keypoint-space data augmentation for MSL Recognition

Correct experimental design for 1-sample-per-class datasets
─────────────────────────────────────────────────────────────
With 558 videos and ~556 unique classes, a random train/val/test split
produces ZERO class overlap between splits — making val/test accuracy
permanently 0% (the model is asked to classify classes it never trained on).

The correct approach:
  For each of the 558 original videos, generate aug_factor augmented copies.
  Then split BY AUGMENTATION TYPE, not by sample:

    Train  : aug copies 1..aug_factor-1  (all 558 classes, many copies)
    Val    : aug copy   0                 (all 558 classes, 1 aug copy each)
    Test   : the original .npy            (all 558 classes, 1 original each)

  Result: 100% class overlap across all three splits.
  The final test evaluates on REAL original keypoints — the strictest test.

Augmentation strategies (10 total)
────────────────────────────────────
 1. GaussianNoise      - coordinate jitter
 2. TimeWarp           - non-linear temporal warp via cubic spline
 3. HorizontalFlip     - mirror x + swap L/R hand blocks
 4. SpeedPerturbation  - resample to random speed factor
 5. FrameDropout       - randomly drop frames then restore length
 6. SpatialRotate      - 2-D rotation in x-y plane
 7. JointMask          - zero-out random joints (occlusion simulation)
 8. ScaleJitter        - global scale perturbation
 9. TemporalShift      - circular time shift
10. CombinedRandom     - random subset of the above
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# ─── Node layout ──────────────────────────────────────────────────────────────

POSE_N           = 33
HAND_N           = 21
LEFT_HAND_START  = POSE_N           # 33
RIGHT_HAND_START = POSE_N + HAND_N  # 54

POSE_MIRROR_PAIRS = [
    (1,4),(2,5),(3,6),(7,8),(9,10),
    (11,12),(13,14),(15,16),(17,18),(19,20),(21,22),
    (23,24),(25,26),(27,28),(29,30),(31,32),
]


# ─── Individual augmentations ─────────────────────────────────────────────────

def augment_gaussian_noise(seq: np.ndarray, std: float = 0.008) -> np.ndarray:
    return (seq + np.random.normal(0, std, seq.shape)).astype(np.float32)


def augment_time_warp(seq: np.ndarray, sigma: float = 0.15, knot: int = 4) -> np.ndarray:
    T = seq.shape[0]
    if T < 4:
        return seq
    orig_steps = np.linspace(0, T - 1, num=knot + 2)
    warp_steps = orig_steps + np.random.normal(0, sigma * T, orig_steps.shape)
    warp_steps[0]  = 0
    warp_steps[-1] = T - 1
    warp_steps     = np.clip(warp_steps, 0, T - 1)
    warp_steps     = np.sort(warp_steps)
    # Enforce strictly increasing (CubicSpline requirement)
    eps = 1e-4
    for i in range(1, len(warp_steps)):
        if warp_steps[i] <= warp_steps[i - 1]:
            warp_steps[i] = warp_steps[i - 1] + eps
    if warp_steps[-1] > T - 1:
        return seq.copy()
    cs         = CubicSpline(warp_steps, orig_steps)
    new_steps  = np.linspace(0, T - 1, T)
    orig_times = np.clip(cs(new_steps), 0, T - 1)
    warped = np.zeros_like(seq)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            warped[:, j, c] = np.interp(orig_times, np.arange(T), seq[:, j, c])
    return warped.astype(np.float32)


def augment_horizontal_flip(seq: np.ndarray) -> np.ndarray:
    flipped = seq.copy()
    flipped[:, :, 0] = -flipped[:, :, 0]
    for i, j in POSE_MIRROR_PAIRS:
        flipped[:, [i, j], :] = flipped[:, [j, i], :]
    lh = flipped[:, LEFT_HAND_START:LEFT_HAND_START+HAND_N, :].copy()
    rh = flipped[:, RIGHT_HAND_START:RIGHT_HAND_START+HAND_N, :].copy()
    flipped[:, LEFT_HAND_START:LEFT_HAND_START+HAND_N, :]  = rh
    flipped[:, RIGHT_HAND_START:RIGHT_HAND_START+HAND_N, :] = lh
    return flipped.astype(np.float32)


def augment_speed_perturbation(seq: np.ndarray,
                                min_factor: float = 0.75,
                                max_factor: float = 1.35) -> np.ndarray:
    T      = seq.shape[0]
    factor = np.random.uniform(min_factor, max_factor)
    new_T  = max(4, int(round(T / factor)))
    idx    = np.linspace(0, T - 1, new_T)
    out    = np.zeros((new_T, seq.shape[1], seq.shape[2]), dtype=np.float32)
    for j in range(seq.shape[1]):
        for c in range(seq.shape[2]):
            out[:, j, c] = np.interp(idx, np.arange(T), seq[:, j, c])
    return out


def augment_frame_dropout(seq: np.ndarray, max_drop_ratio: float = 0.10) -> np.ndarray:
    T    = seq.shape[0]
    drop = max(1, int(T * np.random.uniform(0, max_drop_ratio)))
    keep = sorted(random.sample(range(T), T - drop))
    kept = seq[keep]
    idx  = np.round(np.linspace(0, len(keep) - 1, T)).astype(int)
    return kept[idx].astype(np.float32)


def augment_spatial_rotate(seq: np.ndarray, max_angle_deg: float = 12.0) -> np.ndarray:
    angle = np.radians(np.random.uniform(-max_angle_deg, max_angle_deg))
    c, s  = np.cos(angle), np.sin(angle)
    rot   = np.array([[c, -s], [s, c]], dtype=np.float32)
    out   = seq.copy()
    out[:, :, :2] = np.einsum('ij,tnj->tni', rot, seq[:, :, :2])
    return out.astype(np.float32)


def augment_joint_mask(seq: np.ndarray, max_mask_ratio: float = 0.10) -> np.ndarray:
    N       = seq.shape[1]
    n_mask  = max(1, int(N * np.random.uniform(0, max_mask_ratio)))
    indices = random.sample(range(N), n_mask)
    out     = seq.copy()
    out[:, indices, :] = 0.0
    return out.astype(np.float32)


def augment_scale_jitter(seq: np.ndarray,
                          min_scale: float = 0.85,
                          max_scale: float = 1.15) -> np.ndarray:
    return (seq * np.random.uniform(min_scale, max_scale)).astype(np.float32)


def augment_temporal_shift(seq: np.ndarray, max_shift: int = 5) -> np.ndarray:
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(seq, shift, axis=0).astype(np.float32)


# ─── Combined random augmentor ─────────────────────────────────────────────────

class RandomAugmentor:
    """Apply a random subset of augmentations; each with its own probability."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def _p(self, key: str) -> float:
        return self.cfg.get(key, {}).get('prob', 0.0)

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        aug = seq.copy()

        if random.random() < self._p('gaussian_noise'):
            aug = augment_gaussian_noise(aug, std=self.cfg['gaussian_noise'].get('std', 0.008))

        if random.random() < self._p('time_warp'):
            aug = augment_time_warp(aug,
                sigma=self.cfg['time_warp'].get('sigma', 0.15),
                knot =self.cfg['time_warp'].get('knot',  4))

        if random.random() < self._p('horizontal_flip'):
            aug = augment_horizontal_flip(aug)

        if random.random() < self._p('speed_perturbation'):
            sp  = self.cfg['speed_perturbation']
            aug = augment_speed_perturbation(aug,
                min_factor=sp.get('min_factor', 0.75),
                max_factor=sp.get('max_factor', 1.35))

        if random.random() < self._p('frame_dropout'):
            aug = augment_frame_dropout(aug,
                max_drop_ratio=self.cfg['frame_dropout'].get('max_drop_ratio', 0.10))

        if random.random() < self._p('spatial_rotate'):
            aug = augment_spatial_rotate(aug,
                max_angle_deg=self.cfg['spatial_rotate'].get('max_angle', 12))

        if random.random() < self._p('joint_mask'):
            aug = augment_joint_mask(aug,
                max_mask_ratio=self.cfg['joint_mask'].get('max_mask_ratio', 0.10))

        if random.random() < self._p('scale_jitter'):
            sj  = self.cfg['scale_jitter']
            aug = augment_scale_jitter(aug,
                min_scale=sj.get('min_scale', 0.85),
                max_scale=sj.get('max_scale', 1.15))

        if random.random() < self._p('temporal_shift'):
            aug = augment_temporal_shift(aug,
                max_shift=self.cfg['temporal_shift'].get('max_shift', 5))

        return aug


# ─── Main augmentation pipeline (correct design for 1-sample-per-class) ───────

def augment_dataset_all_classes(
    keypoint_dir: str,
    output_dir:   str,
    records:      list,       # ALL 558 matched records
    label2idx:    dict,
    aug_factor:   int  = 20,  # total augmented copies per original
    aug_cfg:      dict = None,
    seed:         int  = 42,
) -> dict:
    """
    Correct augmentation strategy for datasets with 1 sample per class.

    For EACH of the 558 original videos:
      - aug_000  → val split   (1 augmented copy, realistic noise)
      - aug_001 .. aug_{N-1} → train split  (N-1 augmented copies)
      - original .npy         → test split  (real, un-augmented)

    This guarantees 100% class overlap across all three splits.

    Returns:
        dict with keys 'train', 'val', 'test', each a list of record dicts.
        Also saves augmented_manifest.json in output_dir.
    """
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir)
    train_dir  = output_dir / 'train'
    val_dir    = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    augmentor = RandomAugmentor(aug_cfg) if aug_cfg else None

    train_records = []
    val_records   = []
    test_records  = []   # points to original keypoint files

    missing = 0
    for rec in tqdm(records, desc="Augmenting all classes", unit="sign"):
        kp_path = rec.get('keypoint_path')
        if not kp_path or not Path(kp_path).exists():
            missing += 1
            continue

        seq   = np.load(kp_path).astype(np.float32)  # (T, 75, 3)
        idx   = rec['idx']
        label = rec['label']

        # ── Test: original keypoint (no augmentation) ─────────────────────
        test_records.append({
            **rec,
            'keypoint_path': kp_path,
            'is_augmented':  False,
            'split':         'test',
        })

        # ── Val: first augmented copy (aug_000) ───────────────────────────
        aug0_path = val_dir / f"{idx:04d}_aug000.npy"
        aug0      = augmentor(seq) if augmentor else seq.copy()
        np.save(str(aug0_path), aug0)
        val_records.append({
            **rec,
            'keypoint_path': str(aug0_path),
            'is_augmented':  True,
            'aug_id':        0,
            'split':         'val',
        })

        # ── Train: aug_001 .. aug_{aug_factor-1} ─────────────────────────
        for aug_i in range(1, aug_factor):
            aug_path = train_dir / f"{idx:04d}_aug{aug_i:03d}.npy"
            aug_seq  = augmentor(seq) if augmentor else seq.copy()
            np.save(str(aug_path), aug_seq)
            train_records.append({
                **rec,
                'keypoint_path': str(aug_path),
                'is_augmented':  True,
                'aug_id':        aug_i,
                'split':         'train',
            })

    if missing:
        print(f"[WARNING] {missing} records had no keypoint file and were skipped.")

    manifest = {
        'train': train_records,
        'val':   val_records,
        'test':  test_records,
    }

    manifest_path = output_dir / 'augmented_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("  Augmentation Complete (all-class design)")
    print(f"{'='*60}")
    print(f"  Original videos   : {len(records)}")
    print(f"  Aug factor        : {aug_factor}  (train uses {aug_factor-1}, val uses 1)")
    print(f"  Train samples     : {len(train_records)}  (augmented, all classes)")
    print(f"  Val   samples     : {len(val_records)}   (augmented, all classes)")
    print(f"  Test  samples     : {len(test_records)}   (original,  all classes)")
    print(f"  Class overlap     : 100% across all splits")
    print(f"  Manifest          : {manifest_path}")
    print(f"{'='*60}\n")

    return manifest


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import yaml
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import (parse_annotation_file, build_label_vocabulary,
                       match_videos_to_annotations, get_logger)

    parser = argparse.ArgumentParser(
        description="MSL keypoint augmentation (all-class design for 1-sample-per-class datasets)"
    )
    parser.add_argument('--config',      required=True)
    parser.add_argument('--aug_factor',  type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    acfg = cfg['augmentation']
    if args.aug_factor:
        acfg['aug_factor'] = args.aug_factor

    records              = parse_annotation_file(dcfg['annotation_file'])
    label2idx, idx2label = build_label_vocabulary(records)
    records              = match_videos_to_annotations(
        dcfg['video_dir'], records, dcfg['keypoint_dir']
    )

    augment_dataset_all_classes(
        keypoint_dir = dcfg['keypoint_dir'],
        output_dir   = dcfg['augmented_dir'],
        records      = records,
        label2idx    = label2idx,
        aug_factor   = acfg.get('aug_factor', 20),
        aug_cfg      = acfg,
        seed         = dcfg.get('seed', 42),
    )


if __name__ == '__main__':
    main()

