#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_onnx.py  –  Real-time MSL recognition on Windows using ONNX Runtime

Bugs fixed in this version
───────────────────────────
  [INFER-A] process_frame skipped frames without pose detection.
      Training always includes every frame (zeros for undetected joints).
      Fix: process_frame always returns the keypoint array, never None.

  [INFER-B] normalize_sequence used 2-D scale and left z unscaled.
      Fix: exact mirror of extract_keypoints.normalize_sequence() —
      full 3-D midpoint and 3-D Euclidean shoulder width, all axes.

  [INFER-C] *** ROOT CAUSE OF WEBCAM FAILURE ***
      Webcam classified every classify_every=15 frames = 0.5 seconds.
      Signs take 1–4 seconds; feeding 15 frames to a model trained on
      full-length signs produces garbage predictions.

      infer.py uses a TIME-BASED buffer: accumulate buffer_sec=2.5 sec
      of frames (~75 at 30 fps), then classify and clear.  This gives
      the model a realistic-duration sign sequence matching training.

      Fix: replaced the frame-count trigger with a time-based trigger
      using --buffer_sec (default 2.5), identical to infer.py.

  [INFER-D] np.int64 / int dict key mismatch.
      Fix: always cast to int() before idx2label lookup.

  [INFER-E] ONNXModel.predict returned a scalar instead of 1-D array.
      Fix: return output[0] — shape (num_classes,).

  [INFER-F] model_complexity=1 everywhere; training used complexity=2.
      Fix: complexity=2 for video (matching training), 1 for webcam
      (speed tradeoff, same as infer.py).

  [INFER-G] extractor.pose.model_complexity AttributeError on Windows.
      MediaPipe's Pose object does not expose model_complexity publicly.
      Fix: stored as self.model_complexity on KeypointExtractor.

New features
────────────
  • Video saving — two output files per inference:
      {name}_clean.mp4  —  label/confidence overlay only
      {name}_debug.mp4  —  label overlay + MediaPipe skeleton lines
    Works for both --video and --webcam modes.
    Saved to --output_dir (default: predictions_output/).

  • Webcam log — every classification event writes top-k results to the
    log file with timestamp and separator, matching the video log format.
"""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ── Constants (must match extract_keypoints.py exactly) ───────────────────────
POSE_N          = 33
HAND_N          = 21
TOTAL_JOINTS    = POSE_N + HAND_N * 2   # 75
COORDS          = 3
FEAT_DIM        = TOTAL_JOINTS * COORDS  # 225

POSE_LEFT_SHOULDER  = 11
POSE_RIGHT_SHOULDER = 12
LEFT_HAND_START     = POSE_N            # 33
RIGHT_HAND_START    = POSE_N + HAND_N   # 54


# ─── MediaPipe import ─────────────────────────────────────────────────────────

def _import_mp():
    for prefix in ["mediapipe.solutions", "mediapipe.python.solutions"]:
        try:
            pose  = importlib.import_module(f"{prefix}.pose")
            hands = importlib.import_module(f"{prefix}.hands")
            draw  = importlib.import_module(f"{prefix}.drawing_utils")
            return pose, hands, draw
        except (ImportError, ModuleNotFoundError):
            continue
    try:
        import mediapipe as _mp
        return _mp.solutions.pose, _mp.solutions.hands, _mp.solutions.drawing_utils
    except AttributeError:
        pass
    raise ImportError(
        "Cannot import MediaPipe solutions.\n"
        "Try: pip uninstall mediapipe -y && pip install mediapipe==0.10.18"
    )

mp_pose, mp_hands, mp_drawing = _import_mp()


# ─── Myanmar text rendering ──────────────────────────────────────────────────

class TextRenderer:
    _FONT_CANDIDATES = [
        "C:\\Windows\\Fonts\\Padauk.ttf",
        "C:\\Windows\\Fonts\\NotoSansMyanmar-Regular.ttf",
        "C:\\Windows\\Fonts\\NotoSansMyanmar.ttf",
        "C:\\Windows\\Fonts\\myanmar3.ttf",
        "/usr/share/fonts/truetype/padauk/Padauk-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMyanmar-Regular.ttf",
        "fonts\\Padauk.ttf",
        "fonts/NotoSansMyanmar-Regular.ttf",
    ]

    def __init__(self, font_size: int = 32):
        self.font = self._load_font(font_size)
        self.can_render_myanmar = (self.font is not None)
        if not self.can_render_myanmar:
            print()
            print("  ╔═══════════════════════════════════════════════════════════╗")
            print("  ║  INFO: No Myanmar shaping font found.                    ║")
            print("  ║  Displaying readable Class IDs instead of Myanmar text.  ║")
            print("  ╚═══════════════════════════════════════════════════════════╝")
            print()

    def _load_font(self, size):
        if not HAS_PIL:
            return None
        for path in self._FONT_CANDIDATES:
            if Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    continue
        return None

    def put_text(self, frame, text, position=(20, 20), color=(0, 255, 0)):
        if self.font and HAS_PIL:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            d   = ImageDraw.Draw(pil)
            r, g, b = color[2], color[1], color[0]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx or dy:
                        d.text((position[0]+dx, position[1]+dy), text,
                               font=self.font, fill=(0, 0, 0))
            d.text(position, text, font=self.font, fill=(r, g, b))
            return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2, cv2.LINE_AA)
        return frame


# ─── Keypoint extraction ──────────────────────────────────────────────────────

class KeypointExtractor:
    """
    Extracts (TOTAL_JOINTS, 3) keypoint arrays from BGR frames.

    Fix [INFER-G]: model_complexity is stored as self.model_complexity
    because mediapipe's Pose object does not expose it as a public attribute
    on all Windows builds (AttributeError otherwise).

    Fix [INFER-A]: process_frame always returns an array, never None.
    Matches extract_keypoints.py which includes every frame, using zeros
    for any joint that MediaPipe failed to detect.

    Hand convention (matches extract_keypoints.py):
      MediaPipe reports handedness from the camera perspective (mirrored).
      label='Right' → signer's LEFT  hand → slot LEFT_HAND_START  (33).
      label='Left'  → signer's RIGHT hand → slot RIGHT_HAND_START (54).
    """

    def __init__(self, model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence:  float = 0.5):
        # Fix [INFER-G]: store complexity ourselves — mediapipe's Pose object
        # does NOT expose a public .model_complexity on all Windows builds.
        self.model_complexity = model_complexity

        self.pose  = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=min(model_complexity, 1),  # Hands caps at 1
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._last_pose  = None
        self._last_hands = None

    def process_frame(self, frame: np.ndarray,
                      mirror: bool = False) -> np.ndarray:
        """
        Returns (TOTAL_JOINTS, 3) float32 for every frame.
        Zeros for any undetected joint/hand — never returns None.

        Args:
            frame:  BGR frame from cv2.VideoCapture.
            mirror: If True, flip the frame horizontally before processing.
                    Many laptop/webcam cameras output a mirrored (selfie-view)
                    image.  Training videos were likely recorded un-mirrored
                    (normal camera view).  Setting mirror=True corrects this so
                    the normalised coordinates match the training distribution.
                    The DISPLAYED frame is NOT flipped — only the frame sent to
                    MediaPipe is flipped, so landmark positions are in the
                    coordinate space of the un-mirrored view.
        """
        if mirror:
            frame = cv2.flip(frame, 1)   # horizontal flip only for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        p_res = self.pose.process(rgb)
        h_res = self.hands.process(rgb)
        self._last_pose  = p_res
        self._last_hands = h_res

        kp = np.zeros((TOTAL_JOINTS, COORDS), dtype=np.float32)

        if p_res.pose_landmarks:
            for i, lm in enumerate(p_res.pose_landmarks.landmark):
                kp[i] = [lm.x, lm.y, lm.z]

        if h_res.multi_hand_landmarks and h_res.multi_handedness:
            for lm_list, handedness in zip(
                h_res.multi_hand_landmarks, h_res.multi_handedness
            ):
                label  = handedness.classification[0].label
                offset = LEFT_HAND_START if label == 'Right' else RIGHT_HAND_START
                for i, lm in enumerate(lm_list.landmark):
                    kp[offset + i] = [lm.x, lm.y, lm.z]

        return kp  # always an array, never None

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw MediaPipe skeleton onto frame IN-PLACE and return it."""
        if self._last_pose and self._last_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, self._last_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        if self._last_hands and self._last_hands.multi_hand_landmarks:
            for lm in self._last_hands.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        return frame

    def close(self):
        self.pose.close()
        self.hands.close()


# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Fix [INFER-B]: exact mirror of extract_keypoints.normalize_sequence().
    Full 3-D shoulder-midpoint subtraction and 3-D shoulder-width scaling,
    applied to all three coordinate axes.
    """
    l_sh = seq[:, POSE_LEFT_SHOULDER,  :]   # (T, 3)
    r_sh = seq[:, POSE_RIGHT_SHOULDER, :]   # (T, 3)
    mid  = (l_sh + r_sh) / 2.0
    dist = np.linalg.norm(r_sh - l_sh, axis=1, keepdims=True)
    dist = np.where(dist < 1e-6, 1.0, dist)
    return ((seq - mid[:, None, :]) / dist[:, None, :]).astype(np.float32)


# ─── Preprocessor ─────────────────────────────────────────────────────────────

class Preprocessor:
    """
    Stack raw (75, 3) keypoint frames → normalize → crop/pad → model input.

    Processing order matches infer.py exactly:
      1. Stack   → (T, 75, 3)
      2. Normalize (3-D shoulder-centred, full 3-D scale)
      3. Centre-crop  if T > max_seq_len
      4. Zero-pad     if T < max_seq_len
      5. Build mask   (True = padded position)
      6. Reshape for model type
    """

    def __init__(self, max_seq_len: int = 200):
        self.max_seq_len = max_seq_len

    def __call__(self, kp_list: list, model_type: str,
                 zero_z: bool = False) -> dict:
        if not kp_list:
            return {}

        seq = np.stack(kp_list, axis=0).astype(np.float32)  # (T, 75, 3)
        seq = normalize_sequence(seq)

        # Zero out z-coordinates for webcam use.
        # Rationale from feature analysis:
        #   Training data: z_mean ≈ +0.03 (nearly zero, low variance)
        #   Webcam data:   z_mean ≈ -0.5 to -0.8 (large systematic offset)
        # The z offset comes from the signer being closer to the webcam and
        # different posture/depth than training studio recordings.
        # Since z contributes little discriminative information (std≈0.74 vs
        # x_std≈0.89, y_std≈1.94) and is systematically wrong on webcam,
        # zeroing it removes harmful noise without losing sign-relevant features.
        if zero_z:
            seq[:, :, 2] = 0.0
        T   = seq.shape[0]

        if T > self.max_seq_len:
            start = (T - self.max_seq_len) // 2
            seq   = seq[start: start + self.max_seq_len]
            T     = self.max_seq_len

        real_len = T

        if T < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - T, TOTAL_JOINTS, COORDS), dtype=np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        mask = np.zeros(self.max_seq_len, dtype=np.bool_)
        mask[real_len:] = True

        if model_type == "stgcn":
            kp_out = np.expand_dims(seq.transpose(2, 0, 1), axis=0).astype(np.float32)
            return {"keypoints": kp_out}
        else:
            kp_out = np.expand_dims(
                seq.reshape(self.max_seq_len, FEAT_DIM), axis=0
            ).astype(np.float32)
            return {"keypoints": kp_out, "mask": mask[np.newaxis]}


# ─── ONNX model wrapper ───────────────────────────────────────────────────────

class ONNXModel:
    def __init__(self, onnx_path: str, meta_path: str = None):
        available = ort.get_available_providers()
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred if p in available]

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[INFO] ONNX loaded: {Path(onnx_path).name}"
              f"  (provider: {self.session.get_providers()[0]})")

        self.meta = {}
        if meta_path and Path(meta_path).exists():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    self.meta = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not read metadata: {e}")
        else:
            print(f"[INFO] Metadata not found at '{meta_path}', using defaults.")

        self.model_type  = self.meta.get("model_type",  "transformer")
        self.max_seq_len = self.meta.get("max_seq_len", 200)
        self.num_classes = self.meta.get("num_classes", 558)

    def predict(self, inputs: dict) -> np.ndarray:
        """Fix [INFER-E]: return (num_classes,) not scalar."""
        return self.session.run(None, inputs)[0][0]

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max())
        return e / e.sum()


# ─── Label map ────────────────────────────────────────────────────────────────

def load_idx2label(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.get("idx2label", {}).items()}


def _get_label(idx2label: dict, class_id) -> str:
    """Fix [INFER-D]: always cast to int() before dict lookup."""
    return idx2label.get(int(class_id), f"[class {int(class_id)}]")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _overlay_text(renderer, frame, text, color):
    """Put prediction text at top-left of frame."""
    return renderer.put_text(frame, text, position=(10, 10), color=color)


def _pred_color(confidence: float) -> tuple:
    """Green ≥80%, yellow ≥50%, red otherwise."""
    if confidence >= 0.80: return (0, 200,   0)
    if confidence >= 0.50: return (0, 200, 200)
    return (0, 0, 255)


def _build_overlay_text(top_idx, probs, idx2label, renderer, top_k):
    """Single-line text for the video overlay (top-1 label + confidence)."""
    best = top_idx[0]
    conf = probs[int(best)]
    lbl  = _get_label(idx2label, best) if idx2label else f"Class {int(best)}"
    if renderer.can_render_myanmar:
        return f"{lbl}  ({conf*100:.1f}%)", conf
    return f"Class {int(best)}  ({conf*100:.1f}%)", conf


def _build_console_block(top_idx, probs, idx2label, model_type, source_name, top_k):
    """Multi-line console output block, same format as run_video."""
    sep   = "─" * 60
    lines = [f"Source : {source_name}", f"Model  : {model_type.upper()}", sep]
    for rank, idx in enumerate(top_idx, 1):
        lbl = _get_label(idx2label, idx) if idx2label else f"[class {int(idx)}]"
        lines.append(f"  Top-{rank}: {lbl}  ({probs[int(idx)]*100:.2f}%)")
    lines.append(sep)
    return "\n".join(lines)


def _build_log_block(top_idx, probs, idx2label, source_name, top_k):
    """Log block with timestamp and top-k results."""
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep   = "=" * 60
    lines = [sep, f"[{ts}]  {source_name}"]
    for rank, idx in enumerate(top_idx, 1):
        lbl = _get_label(idx2label, idx) if idx2label else "?"
        lbl = lbl.encode("utf-8", errors="replace").decode("utf-8")
        lines.append(f"  Top-{rank:>2}: Class {int(idx):>4d} | "
                     f"{lbl:<45s} | {probs[int(idx)]*100:6.2f}%")
    lines.append("")
    return "\n".join(lines) + "\n"


def _append_log(path, text):
    try:
        with open(path, "a", encoding="utf-8-sig") as f:
            f.write(text)
    except Exception as e:
        print(f"[WARNING] Could not write log: {e}")


def _make_video_writer(output_path: Path, fps: float, width: int, height: int):
    """Create a cv2.VideoWriter with MP4V codec."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _save_feature_log(kp_list, inputs, probs, top_idx, idx2label,
                      source_name, output_dir, mirror):
    """
    Save full diagnostic information for one classification event.

    Creates two files in output_dir/features/:
      {source_name}_raw_kp.npy      — raw (un-normalized) stacked keypoints (T,75,3)
      {source_name}_model_input.npy — the actual array fed to the ONNX model
      {source_name}_features.txt    — human-readable statistics for debugging

    The .txt file contains:
      - Shoulder positions and detection status (are they non-zero?)
      - Shoulder width (normalization scale — should be ~0.3-0.7 if detected)
      - Per-axis statistics of normalized features
      - Fraction of frames with active hand movement
      - Mirror setting used
      - Top-k classification results
      - Comparison hint vs expected training distribution
    """
    import os
    feat_dir = output_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Raw keypoints before normalization
    raw_kp = np.stack(kp_list, axis=0).astype(np.float32)   # (T, 75, 3)
    T      = raw_kp.shape[0]

    # Save .npy files
    np.save(str(feat_dir / f"{source_name}_raw_kp.npy"), raw_kp)

    # Also save the model input (already normalized, cropped, padded)
    model_input_key = "keypoints"
    if model_input_key in inputs:
        np.save(str(feat_dir / f"{source_name}_model_input.npy"),
                inputs[model_input_key])

    # --- Build text report ---
    L_SH, R_SH = 11, 12
    l_sh = raw_kp[:, L_SH, :]   # (T, 3)
    r_sh = raw_kp[:, R_SH, :]   # (T, 3)

    # How many frames had shoulders detected (non-zero)?
    l_detected = (np.abs(l_sh).sum(axis=1) > 1e-6).sum()
    r_detected = (np.abs(r_sh).sum(axis=1) > 1e-6).sum()

    # Shoulder width distribution (normalization scale)
    dist = np.linalg.norm(r_sh - l_sh, axis=1)  # (T,)
    dist_nonzero = dist[dist > 1e-6]

    # Hand activity: fraction of frames where hands are non-zero
    lh_block = raw_kp[:, 33:54, :]
    rh_block = raw_kp[:, 54:75, :]
    lh_active = (np.abs(lh_block).sum(axis=(1,2)) > 1e-6).sum()
    rh_active = (np.abs(rh_block).sum(axis=(1,2)) > 1e-6).sum()

    # Normalized keypoints for distribution check
    norm_kp = normalize_sequence(raw_kp)
    # For a correctly framed frontal signer:
    #   Left  shoulder (idx 11) normalized x should be ~ +0.4 to +0.6
    #   Right shoulder (idx 12) normalized x should be ~ -0.6 to -0.4
    l_sh_norm = norm_kp[:, L_SH, :]
    r_sh_norm = norm_kp[:, R_SH, :]

    lines = [
        "=" * 65,
        f"Feature Diagnostic Log — {source_name}",
        f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Mirror    : {'YES — webcam flipped before MediaPipe' if mirror else 'NO — raw frame used'}",
        "=" * 65,
        "",
        "── Sequence info ───────────────────────────────────────────",
        f"  Total frames         : {T}",
        f"  Left  shoulder detected: {l_detected}/{T} frames ({l_detected/T*100:.0f}%)",
        f"  Right shoulder detected: {r_detected}/{T} frames ({r_detected/T*100:.0f}%)",
        f"  Left  hand active    : {lh_active}/{T} frames ({lh_active/T*100:.0f}%)",
        f"  Right hand active    : {rh_active}/{T} frames ({rh_active/T*100:.0f}%)",
        "",
        "── Shoulder positions (RAW, image space [0,1]) ─────────────",
        f"  Left  shoulder x mean : {l_sh[:,0].mean():.4f}  std={l_sh[:,0].std():.4f}",
        f"  Right shoulder x mean : {r_sh[:,0].mean():.4f}  std={r_sh[:,0].std():.4f}",
        f"  Left  shoulder y mean : {l_sh[:,1].mean():.4f}",
        f"  Right shoulder y mean : {r_sh[:,1].mean():.4f}",
        "",
        "  EXPECTED for frontal un-mirrored signer:",
        "    Left  shoulder x ~ 0.55–0.65 (camera right = signer left)",
        "    Right shoulder x ~ 0.35–0.45 (camera left  = signer right)",
        "    Shoulder    y   ~ 0.45–0.55  (vertical CENTER of frame)",
        "  If x values are REVERSED → webcam is mirrored → add --mirror / remove --no_mirror",
        f"  {'!! WARNING: Shoulders too LOW in frame (y=' + f'{l_sh[:,1].mean():.2f}' + '). Move camera UP or sit further back.' if l_sh[:,1].mean() > 0.58 else ''}",
        f"  {'!! WARNING: Shoulders too HIGH in frame (y=' + f'{l_sh[:,1].mean():.2f}' + '). Move camera DOWN or closer.' if l_sh[:,1].mean() < 0.38 else ''}",
        "",
        "── Shoulder width (normalization scale) ────────────────────",
    ]
    if len(dist_nonzero) > 0:
        sw_warn = ""
        if dist_nonzero.mean() > 0.27:
            sw_warn = f"  !! WARNING: Too close to camera (width={dist_nonzero.mean():.3f} > 0.27). Move FURTHER BACK."
        elif dist_nonzero.mean() < 0.12:
            sw_warn = f"  !! WARNING: Too far from camera (width={dist_nonzero.mean():.3f} < 0.12). Move CLOSER."
        lines += [
            f"  Shoulder width mean : {dist_nonzero.mean():.4f}",
            f"  Shoulder width std  : {dist_nonzero.std():.4f}",
            f"  Shoulder width range: {dist_nonzero.min():.4f} – {dist_nonzero.max():.4f}",
            "  TARGET: 0.15–0.27 (training avg was 0.205; webcam was 0.313 = too close)",
            sw_warn if sw_warn else "  ✓ Shoulder width is in good range",
        ]
    else:
        lines.append("  !! Shoulder width is ZERO in ALL frames — shoulders NOT detected!")
        lines.append("     Normalization falls back to dist=1.0 → features NOT normalized correctly.")

    lines += [
        "",
        "── Normalized shoulder positions (after normalize_sequence) ─",
        f"  Left  shoulder norm x mean : {l_sh_norm[:,0].mean():.4f}",
        f"  Right shoulder norm x mean : {r_sh_norm[:,0].mean():.4f}",
        "  EXPECTED after correct normalization:",
        "    Left  shoulder norm x ~  +0.5  (signer's left = positive x)",
        "    Right shoulder norm x ~  -0.5  (signer's right = negative x)",
        "  If REVERSED (left~-0.5, right~+0.5) → mirror flip needed",
        "",
        "── Full keypoint distribution (normalized) ──────────────────",
        f"  All joints x: mean={norm_kp[:,:,0].mean():.4f}  std={norm_kp[:,:,0].std():.4f}"
        f"  min={norm_kp[:,:,0].min():.4f}  max={norm_kp[:,:,0].max():.4f}",
        f"  All joints y: mean={norm_kp[:,:,1].mean():.4f}  std={norm_kp[:,:,1].std():.4f}"
        f"  min={norm_kp[:,:,1].min():.4f}  max={norm_kp[:,:,1].max():.4f}",
        f"  All joints z: mean={norm_kp[:,:,2].mean():.4f}  std={norm_kp[:,:,2].std():.4f}"
        f"  min={norm_kp[:,:,2].min():.4f}  max={norm_kp[:,:,2].max():.4f}",
        "",
        "  Training reference (video idx20-30):",
        "    x mean=-0.44  y mean=+0.05  z mean=+0.03",
        f"  {'!! Z-AXIS OFFSET: z mean=' + f'{norm_kp[:,:,2].mean():.2f}' + ' (expected ~0). Use --zero_z to suppress depth noise.' if abs(norm_kp[:,:,2].mean()) > 0.2 else '  ✓ Z-axis looks reasonable'}",
        f"  {'!! Y-AXIS OFFSET: y mean=' + f'{norm_kp[:,:,1].mean():.2f}' + ' (expected ~0). Adjust vertical framing (see shoulder y tip above).' if abs(norm_kp[:,:,1].mean()) > 0.2 else '  ✓ Y-axis looks reasonable'}",
        f"  {'!! SEQUENCE LENGTH: only ' + str(T) + ' frames fed to model (training avg ~127). Increase --buf_frames or use complexity=1 for higher fps.' if T < 80 else '  ✓ Sequence length looks reasonable (' + str(T) + ' frames)'}",
        "",
        "── Classification results ───────────────────────────────────",
    ]
    for rank, idx in enumerate(top_idx, 1):
        lbl = _get_label(idx2label, idx) if idx2label else "?"
        lbl_safe = lbl.encode("utf-8", errors="replace").decode("utf-8")
        lines.append(f"  Top-{rank}: Class {int(idx):>4d} | {lbl_safe:<50s} | {probs[int(idx)]*100:.2f}%")

    lines += [
        "",
        "── Files saved ──────────────────────────────────────────────",
        f"  Raw keypoints : features/{source_name}_raw_kp.npy      shape=({T},75,3)",
        f"  Model input   : features/{source_name}_model_input.npy",
        "=" * 65,
        "",
    ]

    txt_path = feat_dir / f"{source_name}_features.txt"
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Feature log → {txt_path}")


# ─── Video processing ─────────────────────────────────────────────────────────

def run_video(model, prep, extractor, renderer, video_path,
              top_k=5, log_file=None, idx2label=None,
              output_dir=None, debug=False, mirror=False,
              save_features=False):
    """
    Process a single video file.  Saves two output videos:
      {stem}_clean.mp4  —  label overlay only
      {stem}_debug.mp4  —  label overlay + MediaPipe skeleton lines

    Matches infer.py / process_video() exactly:
      every frame included (zeros for undetected joints),
      normalize after stacking, centre-crop or zero-pad.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem         = Path(video_path).stem

    print(f"[INFO] {Path(video_path).name} — {total_frames} frames @ {fps_src:.1f} fps")
    print(f"[INFO] MediaPipe pose complexity: {extractor.model_complexity}"
          f"  (must match training; training default=2)")

    # --- Pass 1: extract all keypoints + collect frames for output video ---
    kp_list    = []
    raw_frames = []   # store for writing output videos after prediction
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        kp_list.append(extractor.process_frame(frame, mirror=mirror))
        raw_frames.append(frame.copy())
    cap.release()

    if not kp_list:
        print("[WARNING] No frames could be read from the video.")
        return

    T_raw = len(kp_list)
    T_use = min(T_raw, model.max_seq_len)
    print(f"[INFO] {T_raw} frames extracted → using {T_use} "
          f"({'centre-cropped' if T_raw > model.max_seq_len else 'zero-padded to'}"
          f" {model.max_seq_len})")

    # --- Inference ---
    inputs  = prep(kp_list, model.model_type)
    logits  = model.predict(inputs)
    probs   = ONNXModel.softmax(logits)
    top_idx = np.argsort(probs)[::-1][:top_k]

    # --- Console output ---
    print("\n" + _build_console_block(top_idx, probs, idx2label,
                                      model.model_type, Path(video_path).name, top_k))

    # --- Log file ---
    if log_file:
        _append_log(log_file,
                    _build_log_block(top_idx, probs, idx2label,
                                     Path(video_path).name, top_k))

    # --- Save keypoint feature log ---
    if save_features and output_dir:
        _save_feature_log(
            kp_list    = kp_list,
            inputs     = inputs,
            probs      = probs,
            top_idx    = top_idx,
            idx2label  = idx2label,
            source_name= Path(video_path).stem,
            output_dir = Path(output_dir),
            mirror     = mirror,
        )

    # --- Save output videos ---
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        overlay_text, conf = _build_overlay_text(top_idx, probs, idx2label, renderer, top_k)
        color              = _pred_color(conf)

        clean_path = out_dir / f"{stem}_clean.mp4"
        debug_path = out_dir / f"{stem}_debug.mp4"
        vw_clean   = _make_video_writer(clean_path, fps_src, w, h)
        vw_debug   = _make_video_writer(debug_path, fps_src, w, h)

        for i, raw in enumerate(raw_frames):
            # Clean: label overlay only
            clean_frame = renderer.put_text(raw.copy(), overlay_text,
                                            position=(10, 10), color=color)
            vw_clean.write(clean_frame)

            # Debug: re-run MediaPipe on this frame to get landmarks,
            # then draw skeleton + label
            debug_frame = raw.copy()
            extractor.process_frame(debug_frame)       # updates _last_pose/_last_hands
            extractor.draw_landmarks(debug_frame)       # draws skeleton in-place
            debug_frame = renderer.put_text(debug_frame, overlay_text,
                                            position=(10, 10), color=color)
            vw_debug.write(debug_frame)

        vw_clean.release()
        vw_debug.release()
        print(f"[INFO] Saved  clean  → {clean_path}")
        print(f"[INFO] Saved  debug  → {debug_path}")


# ─── Webcam processing ────────────────────────────────────────────────────────

def run_webcam(model, prep, extractor, renderer, idx2label,
               cam_idx=0, top_k=5, log_file=None,
               buffer_sec=2.5, buffer_frames=100, output_dir=None,
               mirror=True, save_features=False, zero_z=False):
    """
    Live webcam sign recognition.

    Fix [INFER-C]: time-based buffer (buffer_sec seconds), matching infer.py.
    Fix [MIRROR]:  mirror=True (default) flips frames before MediaPipe so
                   coordinates match the training data orientation.

    Keys:  Q = quit  |  C = clear buffer manually
    """
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam {cam_idx}")
        return

    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use a FIXED frame count for the buffer instead of time-based fps measurement.
    #
    # Why: When complexity=2 was used, actual processing fps dropped to ~9fps.
    # At 9fps, buffer_sec=2.5 gives only 22-36 frames, but the model was trained
    # on sequences averaging ~127 frames. Feeding 36 frames (18% of max_seq_len)
    # means 82% of the model input is zero-padding — the model has never seen
    # sequences this short at test time, so predictions are garbage.
    #
    # Fix: complexity=1 (~25fps) + fixed buf_frames=100 (from --buf_frames arg).
    # At 25fps: 100 frames = 4 seconds per sign = realistic signing duration.
    # This matches the training distribution (avg ~127 frames) much better.
    buf_frames = max(20, min(buffer_frames, model.max_seq_len))

    print(f"[INFO] Webcam  {w}x{h} @ {fps_meta:.1f} fps  (complexity={extractor.model_complexity})")
    print(f"[INFO] Buffer: {buf_frames} frames fixed  "
          f"(~{buf_frames/fps_meta:.1f}s at camera fps — fill buffer THEN sign)")
    print(f"[INFO] Mirror: {'YES' if mirror else 'NO'}")
    print(f"[INFO] Zero-Z: {'YES — z coordinates zeroed (reduces depth noise)' if zero_z else 'NO'}")
    print(f"[INFO] Positioning tip: stand so your SHOULDERS are at the VERTICAL CENTER")
    print(f"                        of the frame (not too high/low). Fill 40-60% of width.")
    print(f"[INFO] Q = quit | C = clear buffer manually")

    # ── Optional video writers ──
    ts_str     = datetime.now().strftime("%Y%m%d_%H%M%S")
    vw_clean   = vw_debug = None
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_path = out_dir / f"webcam_{ts_str}_clean.mp4"
        debug_path = out_dir / f"webcam_{ts_str}_debug.mp4"
        vw_clean   = _make_video_writer(clean_path, fps_meta, w, h)
        vw_debug   = _make_video_writer(debug_path, fps_meta, w, h)
        print(f"[INFO] Recording clean → {clean_path}")
        print(f"[INFO] Recording debug → {debug_path}")

    kp_buffer  = []
    last_text  = "Perform a sign — buffer filling..."
    last_color = (200, 200, 200)
    classify_count = 0
    total_frames_seen  = 0
    pose_detected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract keypoints — always append (Fix [INFER-A])
        kp = extractor.process_frame(frame, mirror=mirror)
        kp_buffer.append(kp)
        total_frames_seen += 1

        # Track whether pose was detected this frame (for status display)
        pose_this_frame = (
            extractor._last_pose is not None and
            extractor._last_pose.pose_landmarks is not None
        )
        if pose_this_frame:
            pose_detected_count += 1
        detect_rate = pose_detected_count / total_frames_seen * 100

        # Fix: classify when buffer reaches buf_frames, then clear
        if len(kp_buffer) >= buf_frames:
            inputs  = prep(list(kp_buffer), model.model_type, zero_z=zero_z)
            logits  = model.predict(inputs)
            probs   = ONNXModel.softmax(logits)
            top_idx = np.argsort(probs)[::-1][:top_k]

            overlay_text, conf = _build_overlay_text(
                top_idx, probs, idx2label, renderer, top_k
            )
            last_text  = overlay_text
            last_color = _pred_color(conf)
            classify_count += 1

            # Console output
            print(f"\n[Prediction #{classify_count}]")
            print(_build_console_block(top_idx, probs, idx2label,
                                       model.model_type, f"webcam frame", top_k))

            # Log file — top_k results
            if log_file:
                _append_log(log_file,
                            _build_log_block(top_idx, probs, idx2label,
                                             f"webcam #{classify_count}", top_k))

            # Feature log
            if save_features and output_dir:
                _save_feature_log(
                    kp_list    = list(kp_buffer),   # already cleared below, snapshot here
                    inputs     = inputs,
                    probs      = probs,
                    top_idx    = top_idx,
                    idx2label  = idx2label,
                    source_name= f"webcam_{ts_str}_#{classify_count:03d}",
                    output_dir = Path(output_dir),
                    mirror     = mirror,
                )

            kp_buffer.clear()   # start fresh for next sign

        # ── Build display frames ──
        # Progress bar towards buf_frames (not max_seq_len)
        fill   = int(len(kp_buffer) / buf_frames * w)
        pct    = len(kp_buffer) / buf_frames * 100

        # Pose detection status: green=detected, red=not detected
        pose_status_text  = f"Pose: {'OK' if pose_this_frame else 'NOT DETECTED'}"
        pose_status_color = (0, 200, 0) if pose_this_frame else (0, 0, 255)
        detect_rate_text  = f"Detect rate: {detect_rate:.0f}%"

        # Clean frame: label + progress bar + pose status
        clean_frame = renderer.put_text(frame.copy(), last_text,
                                        position=(10, 10), color=last_color)
        cv2.putText(clean_frame, pose_status_text,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_status_color, 2)
        cv2.putText(clean_frame, detect_rate_text,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.rectangle(clean_frame, (0, h - 12), (fill, h), (0, 180, 255), -1)
        cv2.putText(clean_frame, f"buf {len(kp_buffer)}/{buf_frames}  ({pct:.0f}%)",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Debug frame: skeleton + label + pose status + progress bar
        debug_frame = frame.copy()
        extractor.draw_landmarks(debug_frame)
        debug_frame = renderer.put_text(debug_frame, last_text,
                                        position=(10, 10), color=last_color)
        cv2.putText(debug_frame, pose_status_text,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_status_color, 2)
        cv2.putText(debug_frame, detect_rate_text,
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.rectangle(debug_frame, (0, h - 12), (fill, h), (0, 180, 255), -1)
        cv2.putText(debug_frame, f"buf {len(kp_buffer)}/{buf_frames}  ({pct:.0f}%)",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

        # Write to video files
        if vw_clean: vw_clean.write(clean_frame)
        if vw_debug: vw_debug.write(debug_frame)

        # Show the debug view on screen (most useful for real-time feedback)
        cv2.imshow("MSL Recognition  [Q=quit  C=clear]", debug_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            kp_buffer.clear()
            last_text  = "Buffer cleared — perform a sign"
            last_color = (200, 200, 200)
            print("[INFO] Buffer manually cleared.")

    cap.release()
    if vw_clean: vw_clean.release()
    if vw_debug: vw_debug.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Webcam session summary:")
    print(f"  Total frames     : {total_frames_seen}")
    print(f"  Pose detected    : {pose_detected_count}  ({pose_detected_count/max(total_frames_seen,1)*100:.0f}%)")
    print(f"  Classifications  : {classify_count}")
    if total_frames_seen > 0 and pose_detected_count / total_frames_seen < 0.3:
        print(f"  [WARNING] Low detection rate ({pose_detected_count/total_frames_seen*100:.0f}%).")
        print(f"            Tips: move closer, improve lighting, ensure upper body is visible.")
    if output_dir:
        print(f"  Saved  clean  → {clean_path}")
        print(f"  Saved  debug  → {debug_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MSL Sign Language Inference — ONNX Runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video with feature log for debugging
  python infer_onnx.py --onnx_model onnx_models\\bilstm_msl.onnx \\
      --video sample.mp4 --top_k 5 \\
      --output_dir predictions_output --log_file results.txt --save_features

  # Webcam — recommended starting command
  python infer_onnx.py --onnx_model onnx_models\\transformer_msl.onnx \\
      --webcam 0 --no_mirror --buf_frames 100 --zero_z --top_k 5 \\
      --output_dir predictions_output --log_file webcam_log.txt --save_features

  # Webcam — if no_mirror gives wrong results try with mirror
  python infer_onnx.py --onnx_model onnx_models\\transformer_msl.onnx \\
      --webcam 0 --buf_frames 100 --zero_z --top_k 5 \\
      --output_dir predictions_output --log_file webcam_log.txt --save_features

  POSITIONING TIPS (critical for correct webcam predictions):
  Feature analysis found these differences vs training data:
    Training shoulder y ~ 0.50 (frame center), width ~ 0.205
    Webcam   shoulder y ~ 0.63 (too low),      width ~ 0.313 (too close)
  1. Move camera UP or sit FURTHER BACK until shoulders are near frame center.
  2. Your upper body should fill about 40-60% of the frame width (move back).
  3. Perform the FULL sign within the buffer window (watch progress bar).
  4. Use --save_features and check *_features.txt to verify positioning.
     Good: shoulder y raw ~ 0.45-0.55, shoulder width ~ 0.15-0.25.
""")

    parser.add_argument("--onnx_model",   required=True,
                        help="Path to .onnx model file (NOT the .json metadata)")
    parser.add_argument("--metadata",     default=None,
                        help="Path to .json metadata (default: same stem as .onnx)")
    parser.add_argument("--label_map",    default=None,
                        help="Path to label_map.json (default: next to .onnx)")
    parser.add_argument("--output_dir",   default="predictions_output",
                        help="Directory for saved videos (default: predictions_output/)")
    parser.add_argument("--log_file",     default=None,
                        help="Append top-k results to this text file")
    parser.add_argument("--top_k",        type=int, default=5,
                        help="Number of top predictions to show/log (default: 5)")
    parser.add_argument("--font_size",    type=int, default=32,
                        help="Myanmar font size for overlay (default: 32)")
    parser.add_argument("--model_complexity", type=int, default=None,
                        help="MediaPipe pose complexity: 0/1/2. "
                             "Default: 2 for video (matches training), 1 for webcam.")

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video",  help="Path to a video file")
    grp.add_argument("--webcam", type=int, metavar="CAM_IDX",
                     help="Webcam device index (0 = default camera)")

    parser.add_argument("--buffer_sec", type=float, default=2.5,
                        help="[Webcam] Legacy: seconds-based buffer (overridden by --buf_frames)")
    parser.add_argument("--buf_frames", type=int, default=100,
                        help="[Webcam] Fixed number of frames to accumulate before classifying. "
                             "Default: 100 (matches training avg sequence length ~127 frames). "
                             "At complexity=1 (~25fps): 100 frames ≈ 4 seconds per sign. "
                             "Increase if signs are slow; decrease for faster response.")
    parser.add_argument("--zero_z", action="store_true",
                        help="[Webcam] Zero out the z (depth) coordinate before classification. "
                             "Feature analysis shows webcam z_mean ≈ -0.5 to -0.8 vs "
                             "training z_mean ≈ 0.03 — a large systematic offset caused by "
                             "the signer being closer to the webcam than in training recordings. "
                             "Zeroing z removes this noise. Recommended when webcam predictions "
                             "are wrong but x/y skeleton looks correct in debug video.")
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True,
                        help="[Webcam] Flip frame horizontally before MediaPipe (default: ON). "
                             "Most webcams output a mirrored (selfie) image; flipping corrects "
                             "the x-coordinates to match the training data orientation.")
    parser.add_argument("--no_mirror", dest="mirror", action="store_false",
                        help="[Webcam] Disable the horizontal flip. Use if your webcam "
                             "already outputs an un-mirrored view, or if --mirror makes "
                             "predictions worse.")
    parser.add_argument("--save_features", action="store_true",
                        help="Save keypoint feature diagnostics to output_dir/features/. "
                             "Each classification event produces: "
                             "*_raw_kp.npy (raw keypoints), "
                             "*_model_input.npy (model input), "
                             "*_features.txt (human-readable stats). "
                             "Use to compare video vs webcam feature distributions.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra diagnostic information")
    args = parser.parse_args()
    # ── Validate ONNX path ────────────────────────────────────────────────────
    model_p = Path(args.onnx_model)
    if model_p.suffix.lower() == ".json":
        sys.exit(
            f"[ERROR] --onnx_model must be a .onnx file, not a .json file.\n"
            f"        You passed : {model_p}\n"
            f"        Did you mean: {model_p.with_suffix('.onnx')}"
        )
    if not model_p.exists():
        sys.exit(f"[ERROR] ONNX model not found: {model_p}")

    meta_p = args.metadata or str(model_p.with_suffix(".json"))
    lmap_p = args.label_map or str(model_p.parent / "label_map.json")

    if not Path(lmap_p).exists():
        sys.exit(f"[ERROR] label_map.json not found: {lmap_p}\n"
                 f"        Copy it from your Linux server alongside the .onnx file.")

    # ── Load resources ────────────────────────────────────────────────────────
    idx2label = load_idx2label(lmap_p)
    print(f"[INFO] Loaded {len(idx2label)} labels from label map.")

    model    = ONNXModel(str(model_p), meta_p)
    prep     = Preprocessor(max_seq_len=model.max_seq_len)
    renderer = TextRenderer(font_size=args.font_size)

    if args.log_file:
        print(f"[INFO] Logging top-{args.top_k} results to: {args.log_file}")
    if args.output_dir:
        print(f"[INFO] Saving output videos to: {args.output_dir}/")

    # Complexity settings based on feature analysis:
    # - complexity=2 for video: matches training (extract_keypoints.py default=2)
    # - complexity=1 for webcam: much faster (~25fps vs ~9fps for complexity=2)
    #   Feature analysis showed the main webcam problems are sequence length
    #   and z-axis offset — NOT detection quality. Complexity=1 detects fine
    #   for a well-lit, close-up signer and gives 25fps → 100 frames in 4s.
    if args.model_complexity is not None:
        video_complexity  = args.model_complexity
        webcam_complexity = args.model_complexity
    else:
        video_complexity  = 2   # matches training
        webcam_complexity = 1   # faster fps -> more frames in buffer

    try:
        if args.video:
            extractor = KeypointExtractor(
                model_complexity=video_complexity,
                min_detection_confidence=0.5,   # matches training
                min_tracking_confidence=0.5,
            )
            try:
                run_video(model, prep, extractor, renderer, args.video,
                          top_k         = args.top_k,
                          log_file      = args.log_file,
                          idx2label     = idx2label,
                          output_dir    = args.output_dir,
                          debug         = args.debug,
                          mirror        = False,           # video files are not mirrored
                          save_features = args.save_features)
            finally:
                extractor.close()
        else:
            extractor = KeypointExtractor(
                model_complexity=webcam_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            try:
                run_webcam(model, prep, extractor, renderer, idx2label,
                           cam_idx        = args.webcam,
                           top_k          = args.top_k,
                           log_file       = args.log_file,
                           buffer_sec     = args.buffer_sec,
                           buffer_frames  = args.buf_frames,
                           output_dir     = args.output_dir,
                           mirror         = args.mirror,
                           save_features  = args.save_features,
                           zero_z         = args.zero_z)
            finally:
                extractor.close()
    finally:
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
