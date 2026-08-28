#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_keypoints.py - MediaPipe keypoint extraction for MSL videos
                       (Pose + Hands approach, compatible with mediapipe >= 0.10.13)

Why no Holistic?
  MediaPipe deprecated solutions.holistic in 0.10.1 and removed it entirely
  from builds targeting Python 3.12 (earliest available: 0.10.13).
  This version uses separate Pose and Hands detectors which are still
  supported and produce the identical output tensor shape.

Features extracted per frame:
  - 33 Pose  landmarks x 3 (x, y, z) = 99  floats
  - 21 Left  hand      x 3 (x, y, z) = 63  floats
  - 21 Right hand      x 3 (x, y, z) = 63  floats
  Total : 75 nodes x 3 coords = 225 floats per frame

Output: .npy files shape (T, 75, 3)
        Companion _vis.npy shape (T, 33) - pose visibility scores
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
import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# === MediaPipe import: Pose + Hands (mediapipe >= 0.10.13) ====================

def _import_mp_modules():
    """Try all known import paths; return (pose_mod, hands_mod, drawing_mod)."""
    import importlib
    prefixes = ["mediapipe.solutions", "mediapipe.python.solutions"]
    for prefix in prefixes:
        try:
            pose  = importlib.import_module(f"{prefix}.pose")
            hands = importlib.import_module(f"{prefix}.hands")
            draw  = importlib.import_module(f"{prefix}.drawing_utils")
            logging.info(f"[mediapipe] loaded via '{prefix}'")
            return pose, hands, draw
        except (ImportError, ModuleNotFoundError):
            continue
    # Legacy attribute access (mediapipe <= 0.9)
    try:
        import mediapipe as mp
        logging.info("[mediapipe] loaded via legacy mp.solutions")
        return mp.solutions.pose, mp.solutions.hands, mp.solutions.drawing_utils
    except AttributeError:
        pass
    raise ImportError(
        "\nCannot import mediapipe Pose/Hands.\n"
        "Try:  pip uninstall mediapipe -y && pip install mediapipe==0.10.18\n"
        "Then: python tools/probe_mediapipe.py\n"
    )


mp_pose, mp_hands, mp_drawing = _import_mp_modules()

# Expose connection sets for infer.py webcam overlay
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
mp_holistic      = None   # Holistic removed in 0.10.13+ — kept as None for compat

# === Constants ================================================================

POSE_N   = 33
HAND_N   = 21
TOTAL_N  = POSE_N + HAND_N * 2   # 75
FEAT_DIM = TOTAL_N * 3            # 225

POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12
LEFT_HAND_START     = POSE_N           # 33
RIGHT_HAND_START    = POSE_N + HAND_N  # 54


# === Frame-level extraction ===================================================

def _lm_to_array(lm_obj, n: int) -> np.ndarray:
    if lm_obj is None:
        return np.zeros((n, 3), dtype=np.float32)
    return np.array([[lm.x, lm.y, lm.z] for lm in lm_obj.landmark], dtype=np.float32)


def extract_frame_keypoints(pose_results, hands_results) -> np.ndarray:
    """
    Build (75, 3) array for one frame from separate pose + hands results.

    Hand label convention:
      MediaPipe reports handedness from the *camera* perspective (mirrored).
      'Right' in the result = signer's LEFT hand, and vice-versa.
      We swap so our indices match the signer's anatomy.
    """
    pose = _lm_to_array(getattr(pose_results, 'pose_landmarks', None), POSE_N)
    lh   = np.zeros((HAND_N, 3), dtype=np.float32)
    rh   = np.zeros((HAND_N, 3), dtype=np.float32)

    if hands_results.multi_hand_landmarks:
        for lm_list, handedness in zip(
            hands_results.multi_hand_landmarks,
            hands_results.multi_handedness,
        ):
            label = handedness.classification[0].label   # 'Left' or 'Right'
            arr   = _lm_to_array(lm_list, HAND_N)
            if label == 'Right':
                lh = arr   # camera-right = signer's left
            else:
                rh = arr   # camera-left  = signer's right

    return np.concatenate([pose, lh, rh], axis=0)   # (75, 3)


def extract_frame_visibility(pose_results) -> np.ndarray:
    """Pose visibility scores. Shape: (33,)."""
    pl = getattr(pose_results, 'pose_landmarks', None)
    if pl is None:
        return np.zeros(POSE_N, dtype=np.float32)
    return np.array([lm.visibility for lm in pl.landmark], dtype=np.float32)


# === Normalization =============================================================

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Translate to shoulder-midpoint origin; scale by shoulder width.
    Input/output shape: (T, 75, 3).
    """
    l_sh = seq[:, POSE_LEFT_SHOULDER,  :]
    r_sh = seq[:, POSE_RIGHT_SHOULDER, :]
    mid  = (l_sh + r_sh) / 2.0
    dist = np.linalg.norm(r_sh - l_sh, axis=1, keepdims=True)
    dist = np.where(dist < 1e-6, 1.0, dist)
    return ((seq - mid[:, None, :]) / dist[:, None, :]).astype(np.float32)


# === Per-video processing =====================================================

def process_video(video_path: str, detectors: dict, normalize: bool = True) -> dict:
    """
    Run Pose + Hands on every frame.

    Args:
        video_path: path to video file
        detectors:  {'pose': <Pose context>, 'hands': <Hands context>}
        normalize:  apply shoulder-based normalization

    Returns:
        dict with 'keypoints' (T,75,3), 'visibility' (T,33),
        'num_frames', 'fps', 'hands_present' — or {'keypoints': None} on error.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.warning(f"Cannot open: {video_path}")
        return {'keypoints': None}

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    pose_det  = detectors['pose']
    hands_det = detectors['hands']
    kp_buf    = []
    vis_buf   = []
    lh_cnt    = rh_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        p_res = pose_det.process(rgb)
        h_res = hands_det.process(rgb)

        kp_buf.append(extract_frame_keypoints(p_res, h_res))
        vis_buf.append(extract_frame_visibility(p_res))

        if h_res.multi_hand_landmarks and h_res.multi_handedness:
            for hd in h_res.multi_handedness:
                if hd.classification[0].label == 'Right':
                    lh_cnt += 1
                else:
                    rh_cnt += 1

    cap.release()
    if not kp_buf:
        return {'keypoints': None}

    keypoints  = np.stack(kp_buf,  axis=0)   # (T, 75, 3)
    visibility = np.stack(vis_buf, axis=0)   # (T, 33)
    T          = keypoints.shape[0]

    if normalize:
        keypoints = normalize_sequence(keypoints)

    return {
        'keypoints':     keypoints,
        'visibility':    visibility,
        'num_frames':    T,
        'fps':           fps,
        'hands_present': (lh_cnt / T, rh_cnt / T),
    }


# === Batch extraction =========================================================

def extract_all_videos(
    video_dir:        str,
    output_dir:       str,
    normalize:        bool = True,
    model_complexity: int  = 2,
    overwrite:        bool = False,
) -> dict:
    """Process all videos; save .npy keypoints mirroring the folder structure."""
    video_dir  = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm',
                  '.MP4', '.AVI', '.MOV', '.MKV'}

    import re as _re
    def _num_key(p):
        parts = _re.split(r'(\d+)', p.stem)
        return [int(x) if x.isdigit() else x.lower() for x in parts]

    videos = sorted(
        [p for p in video_dir.rglob('*') if p.suffix in video_exts],
        key=_num_key,
    )

    logging.info(f"Found {len(videos)} videos in '{video_dir}'")
    if not videos:
        raise RuntimeError(f"No videos found in {video_dir}")

    stats = {
        'total': len(videos), 'processed': 0, 'failed': 0, 'skipped': 0,
        'seq_lengths': [], 'fps_values': [], 'lh_presence': [], 'rh_presence': [],
    }

    # Hands model_complexity only accepts 0 or 1
    hands_complexity = min(model_complexity, 1)

    with mp_pose.Pose(
        static_image_mode=False, model_complexity=model_complexity,
        smooth_landmarks=True, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as pose_det, mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=hands_complexity,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands_det:

        detectors = {'pose': pose_det, 'hands': hands_det}

        for vp in tqdm(videos, desc="Extracting keypoints", unit="video"):
            rel   = vp.relative_to(video_dir)
            out_p = output_dir / rel.with_suffix('.npy')
            out_v = output_dir / rel.parent / (rel.stem + '_vis.npy')
            out_p.parent.mkdir(parents=True, exist_ok=True)

            if out_p.exists() and not overwrite:
                stats['seq_lengths'].append(np.load(str(out_p), mmap_mode='r').shape[0])
                stats['skipped'] += 1
                continue

            result = process_video(str(vp), detectors, normalize=normalize)

            if result['keypoints'] is None:
                logging.error(f"FAILED: {vp}")
                stats['failed'] += 1
                continue

            np.save(str(out_p), result['keypoints'])
            np.save(str(out_v), result['visibility'])
            stats['processed']    += 1
            stats['seq_lengths'].append(result['num_frames'])
            stats['fps_values'].append(result['fps'])
            stats['lh_presence'].append(result['hands_present'][0])
            stats['rh_presence'].append(result['hands_present'][1])

    sl = stats['seq_lengths']
    summary = {'total': stats['total'], 'processed': stats['processed'],
               'failed': stats['failed'], 'skipped': stats['skipped']}
    if sl:
        summary.update({
            'seq_len_min':      int(np.min(sl)),
            'seq_len_max':      int(np.max(sl)),
            'seq_len_mean':     float(np.mean(sl)),
            'seq_len_std':      float(np.std(sl)),
            'seq_len_p50':      float(np.percentile(sl, 50)),
            'seq_len_p95':      float(np.percentile(sl, 95)),
            'lh_mean_presence': float(np.mean(stats['lh_presence'])) if stats['lh_presence'] else 0.0,
            'rh_mean_presence': float(np.mean(stats['rh_presence'])) if stats['rh_presence'] else 0.0,
        })

    with open(output_dir / 'extraction_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*55)
    print("  Keypoint Extraction Summary")
    print("="*55)
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    print("="*55 + "\n")
    return summary


# === CLI ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Pose+Hands keypoints from MSL videos"
    )
    parser.add_argument('--video_dir',    required=True)
    parser.add_argument('--output_dir',   required=True)
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--complexity',   type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--overwrite',    action='store_true')
    parser.add_argument('--log_file',     default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            *([] if not args.log_file else [logging.FileHandler(args.log_file)]),
        ],
    )
    extract_all_videos(
        video_dir        = args.video_dir,
        output_dir       = args.output_dir,
        normalize        = not args.no_normalize,
        model_complexity = args.complexity,
        overwrite        = args.overwrite,
    )

if __name__ == '__main__':
    main()

