#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer.py - Single-video / real-time inference for MSL Recognition

Modes
─────
  --video   : run on one video file, print top-k predictions
  --webcam  : live webcam inference (press Q to quit, R to reset buffer)
  --batch   : run on a directory of videos, produce CSV

Usage
─────
  python src/infer.py \
      --checkpoint results/exp_A_bilstm/checkpoints/best.pth \
      --config     config/config.yaml \
      --video      data/videos/sample.mp4 \
      --top_k      5

  python src/infer.py \
      --checkpoint results/exp_A_bilstm/checkpoints/best.pth \
      --config     config/config.yaml \
      --webcam 0

  python src/infer.py \
      --checkpoint results/exp_A_bilstm/checkpoints/best.pth \
      --config     config/config.yaml \
      --batch      data/videos/new_samples/ \
      --output     results/batch_predictions.csv
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

# Import shared mediapipe objects and helpers from extract_keypoints.
# mp_pose, mp_hands, mp_drawing are the live module handles.
# mp_holistic is None (removed in mediapipe >= 0.10.13) — not used here.
from extract_keypoints import (
    mp_pose, mp_hands, mp_drawing,
    POSE_CONNECTIONS, HAND_CONNECTIONS,
    process_video,
    extract_frame_keypoints,
    normalize_sequence,
    FEAT_DIM,
)
from models import build_model
from utils  import get_device, load_label_map, set_seed, get_logger


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ckpt        = torch.load(checkpoint_path, map_location=device)
    model_type  = ckpt.get('model_type', 'bilstm')
    num_classes = ckpt.get('num_classes')

    if num_classes is None:
        raise ValueError("Checkpoint missing 'num_classes'. Re-train to regenerate.")

    model = build_model(model_type, cfg, num_classes).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, model_type, cfg


# ─── Sequence pre-processing ──────────────────────────────────────────────────

def preprocess_sequence(
    seq:         np.ndarray,   # (T, 75, 3)
    max_seq_len: int,
    model_type:  str,
) -> torch.Tensor:
    """Centre-crop or zero-pad to max_seq_len; reshape for model input."""
    T = seq.shape[0]
    if T > max_seq_len:
        start = (T - max_seq_len) // 2
        seq   = seq[start: start + max_seq_len]
        T     = max_seq_len
    if T < max_seq_len:
        pad = np.zeros((max_seq_len - T, 75, 3), dtype=np.float32)
        seq = np.concatenate([seq, pad], axis=0)

    if model_type == 'stgcn':
        # (T, 75, 3) → (1, 3, T, 75)
        return torch.from_numpy(seq).permute(2, 0, 1).unsqueeze(0)
    else:
        # (T, 75, 3) → (1, T, 225)
        return torch.from_numpy(seq.reshape(max_seq_len, -1)).unsqueeze(0)


def build_pad_mask(real_len: int, max_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, max_len, dtype=torch.bool, device=device)
    if real_len < max_len:
        mask[0, real_len:] = True
    return mask


def predict_tensor(
    model,
    tensor:      torch.Tensor,
    model_type:  str,
    real_len:    int,
    max_seq_len: int,
    device:      torch.device,
    top_k:       int = 5,
) -> List[Tuple[int, float]]:
    """One forward pass → list of (class_idx, probability) sorted by prob desc."""
    tensor = tensor.to(device)
    mask   = build_pad_mask(real_len, max_seq_len, device)
    length = torch.tensor([real_len], dtype=torch.long, device=device)

    with torch.no_grad():
        if model_type == 'bilstm':
            logits = model(tensor, lengths=length, mask=mask)
        elif model_type == 'transformer':
            logits = model(tensor, mask=mask)
        else:  # stgcn
            logits = model(tensor)

    probs              = F.softmax(logits, dim=1)[0]
    k                  = min(top_k, probs.numel())
    topk_probs, topk_idxs = probs.topk(k)
    return list(zip(topk_idxs.cpu().tolist(), topk_probs.cpu().tolist()))


# ─── Helper: open Pose + Hands detectors ─────────────────────────────────────

def _open_detectors(model_complexity: int = 1):
    """
    Return a context-managed pair of (pose, hands) detectors.
    Hands model_complexity is capped at 1 (mediapipe constraint).
    """
    return (
        mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
        mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=min(model_complexity, 1),
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ),
    )


# ─── Single-video mode ────────────────────────────────────────────────────────

def infer_video(
    video_path: str,
    model,
    model_type: str,
    cfg:        dict,
    idx2label:  dict,
    device:     torch.device,
    top_k:      int = 5,
) -> List[Tuple[str, float]]:
    max_seq_len = cfg['data']['max_seq_len']

    pose_det, hands_det = _open_detectors(model_complexity=2)
    try:
        result = process_video(
            video_path,
            detectors={'pose': pose_det, 'hands': hands_det},
            normalize=True,
        )
    finally:
        pose_det.close()
        hands_det.close()

    if result['keypoints'] is None:
        print(f"[ERROR] Could not extract keypoints from: {video_path}")
        return []

    seq      = result['keypoints']
    real_len = min(seq.shape[0], max_seq_len)
    tensor   = preprocess_sequence(seq, max_seq_len, model_type)
    preds    = predict_tensor(model, tensor, model_type, real_len, max_seq_len, device, top_k)

    print(f"\n{'─'*55}")
    print(f"  Video : {Path(video_path).name}")
    print(f"  Frames: {result['num_frames']}  FPS: {result['fps']:.1f}")
    print(f"  Hands : L={result['hands_present'][0]*100:.0f}%  "
          f"R={result['hands_present'][1]*100:.0f}%")
    print(f"  Top-{top_k} predictions:")
    for rank, (idx, prob) in enumerate(preds, start=1):
        label = idx2label.get(idx, str(idx))
        bar   = '█' * int(prob * 30)
        print(f"    {rank}. [{prob*100:5.1f}%] {bar:<30}  {label}")
    print(f"{'─'*55}\n")

    return [(idx2label.get(idx, str(idx)), prob) for idx, prob in preds]


# ─── Batch mode ───────────────────────────────────────────────────────────────

def infer_batch(
    video_dir:  str,
    model,
    model_type: str,
    cfg:        dict,
    idx2label:  dict,
    device:     torch.device,
    output_csv: str,
    top_k:      int = 1,
):
    video_dir  = Path(video_dir)
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm',
                  '.MP4', '.AVI', '.MOV', '.MKV'}
    videos     = sorted([p for p in video_dir.rglob('*') if p.suffix in video_exts])

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    rows        = []
    max_seq_len = cfg['data']['max_seq_len']

    pose_det, hands_det = _open_detectors(model_complexity=2)
    try:
        detectors = {'pose': pose_det, 'hands': hands_det}
        for vp in videos:
            result = process_video(str(vp), detectors, normalize=True)
            if result['keypoints'] is None:
                rows.append({'video': str(vp), 'pred_1': 'FAILED', 'conf_1': 0.0})
                print(f"  {vp.name:<40} → FAILED")
                continue

            seq      = result['keypoints']
            real_len = min(seq.shape[0], max_seq_len)
            tensor   = preprocess_sequence(seq, max_seq_len, model_type)
            preds    = predict_tensor(model, tensor, model_type, real_len,
                                      max_seq_len, device, top_k=top_k)
            row = {'video': str(vp)}
            for i, (idx, prob) in enumerate(preds, start=1):
                row[f'pred_{i}'] = idx2label.get(idx, str(idx))
                row[f'conf_{i}'] = round(prob, 4)
            rows.append(row)
            print(f"  {vp.name:<40} → {row.get('pred_1', '?')}")
    finally:
        pose_det.close()
        hands_det.close()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['video'] + [
        f'{t}_{i}' for i in range(1, top_k + 1) for t in ('pred', 'conf')
    ]
    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nBatch predictions saved → {output_csv}")


# ─── Webcam (live) mode ───────────────────────────────────────────────────────

def infer_webcam(
    cam_idx:    int,
    model,
    model_type: str,
    cfg:        dict,
    idx2label:  dict,
    device:     torch.device,
    buffer_sec: float = 2.5,
):
    """
    Accumulate `buffer_sec` seconds of frames → classify → overlay prediction.
    Press Q to quit, R to reset the keypoint buffer.
    """
    max_seq_len = cfg['data']['max_seq_len']
    cap         = cv2.VideoCapture(cam_idx)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    buf_frames  = int(fps * buffer_sec)

    kp_buffer = []
    last_pred = "Collecting frames..."
    last_conf = 0.0

    pose_det, hands_det = _open_detectors(model_complexity=1)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            p_res = pose_det.process(rgb)
            h_res = hands_det.process(rgb)

            # Accumulate keypoints
            kp = extract_frame_keypoints(p_res, h_res)
            kp_buffer.append(kp)

            # Draw landmarks
            if p_res.pose_landmarks:
                mp_drawing.draw_landmarks(frame, p_res.pose_landmarks, POSE_CONNECTIONS)
            if h_res.multi_hand_landmarks:
                for lm in h_res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, HAND_CONNECTIONS)

            # Classify when buffer is full
            if len(kp_buffer) >= buf_frames:
                seq      = normalize_sequence(np.stack(kp_buffer, axis=0))
                kp_buffer.clear()
                real_len = min(seq.shape[0], max_seq_len)
                tensor   = preprocess_sequence(seq, max_seq_len, model_type)
                preds    = predict_tensor(model, tensor, model_type, real_len,
                                          max_seq_len, device, top_k=1)
                if preds:
                    last_pred = idx2label.get(preds[0][0], '?')
                    last_conf = preds[0][1]

            # Overlay: black bar at bottom with prediction text
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, f"{last_pred}  ({last_conf*100:.1f}%)",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
            # Buffer fill progress bar
            fill = int(len(kp_buffer) / max(buf_frames, 1) * w)
            cv2.rectangle(frame, (0, h - 62), (fill, h - 58), (0, 200, 255), -1)

            cv2.imshow('MSL Recognition  [Q=quit  R=reset]', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                kp_buffer.clear()
                last_pred = "Buffer reset"
                last_conf = 0.0
    finally:
        pose_det.close()
        hands_det.close()
        cap.release()
        cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MSL Sign Language Inference')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--config',     default='config/config.yaml')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video',  help='Path to a single video file')
    group.add_argument('--webcam', type=int, metavar='CAM_IDX',
                       help='Webcam device index for live inference')
    group.add_argument('--batch',  help='Directory of videos for batch inference')

    parser.add_argument('--top_k',     type=int, default=5)
    parser.add_argument('--output',    default='results/batch_predictions.csv')
    parser.add_argument('--label_map', default=None)
    args = parser.parse_args()

    logger = get_logger('infer')
    device = get_device()

    model, model_type, cfg = load_model(args.checkpoint, args.config, device)
    logger.info(f"Loaded {model_type} from {args.checkpoint}")

    lm_path          = args.label_map or cfg['data']['label_map_file']
    label2idx, idx2label = load_label_map(lm_path)

    if args.video:
        infer_video(args.video, model, model_type, cfg, idx2label, device,
                    top_k=args.top_k)
    elif args.webcam is not None:
        infer_webcam(args.webcam, model, model_type, cfg, idx2label, device)
    elif args.batch:
        infer_batch(args.batch, model, model_type, cfg, idx2label,
                    device, output_csv=args.output, top_k=args.top_k)


if __name__ == '__main__':
    main()

