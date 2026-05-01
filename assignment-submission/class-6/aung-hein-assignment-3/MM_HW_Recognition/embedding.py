"""Stroke-level embedding utilities for Myanmar online handwriting.

This module is designed for fast, CPU-friendly stroke embeddings that are:
- dot-aware,
- line/position-aware using guide lines from previous preprocessing,
- suitable for prototypes, WFST-style decoding, retrieval, and statistical evaluation,
- lightweight enough for Raspberry Pi deployment.

Expected inputs
---------------
At minimum, a dataframe row should contain:
    - strokes OR cleaned_strokes: list of strokes, each [[x, y, t], ...]

If available from previous preprocessing, the module will also use:
    - guide_lines: {'top','mid_top','mid_bottom','bottom', ...}
    - is_dot: list[bool] per stroke
    - dot_role: list[str] per stroke
    - stroke_zone_pct: list[{'upper','middle','lower'}]
    - stroke_zone_coverage: list[{'upper','middle','lower'}]

If those columns are missing, the module falls back to simple internal estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, rcParams
from sklearn.decomposition import PCA
from sklearn.metrics import (
    roc_auc_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================


FEATURE_BLOCKS = {
    "geom":            slice(0, 31), 
    "pause":           slice(31, 36), 
    "turn":            slice(36, 40),
    "progression":     slice(40, 52),
    "zone_pct_cov":    slice(52, 58),
    "zone_active":     slice(58, 61),
    "line_rel":        slice(61, 67),
    "dot":             slice(67, 80),  
    "direction_hist":  slice(80, 88),
    "curvature_hist":  slice(88, 96),
    "projection_hist": slice(96, 112),
    "local_grid":      slice(112, None),
}


@dataclass
class StrokeEmbeddingConfig:
    # normalization for embedding input
    target_half_extent: float = 1.0
    keep_aspect: bool = True

    # shape histograms
    direction_bins: int = 8
    curvature_bins: int = 8

    # projection / occupancy
    projection_bins: int = 8
    local_grid: int = 6

    # feature toggles
    use_direction_hist: bool = True
    use_curvature_hist: bool = True
    use_projection_hist: bool = True
    use_local_grid: bool = True
    use_time_features: bool = True
    use_penup_features: bool = True
    use_pause_features: bool = True     
    use_line_position_features: bool = True
    use_dot_features: bool = True

    # evaluation / alignment thresholds (for interpreting line features)
    active_zone_pct_threshold: float = 0.20
    active_zone_coverage_threshold: float = 0.35

    # pause heuristic
    slow_pause_ratio_threshold: float = 1.75 

# ============================================================
# Basic helpers
# ============================================================

def _to_array(stroke: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Convert one stroke to numpy array.

    IMPORTANT:
    Use float64 so epoch timestamps keep sub-second precision.
    float32 is not sufficient for large epoch timestamps.
    """
    arr = np.asarray(stroke, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Each stroke must have shape [N,3] => [x,y,t]")
    return arr

def _shift_stroke_times_to_relative(
    strokes: Sequence[Sequence[Sequence[float]]]
) -> List[np.ndarray]:
    """
    Convert absolute timestamps to relative seconds from sample start.
    Keeps x,y unchanged.
    """
    if not strokes:
        return []

    arrays = [_to_array(s).copy() for s in strokes]
    t0 = min(arr[0, 2] for arr in arrays if len(arr) > 0)

    out = []
    for arr in arrays:
        arr = arr.copy()
        arr[:, 2] = arr[:, 2] - t0
        out.append(arr)

    return out



def _path_length(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    d = np.diff(xy, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())



def _sample_bbox(strokes: Sequence[Sequence[Sequence[float]]]) -> Dict[str, float]:
    if not strokes:
        return {"xmin": 0.0, "xmax": 1.0, "ymin": 0.0, "ymax": 1.0, "cx": 0.5, "cy": 0.5, "w": 1.0, "h": 1.0}
    pts = np.concatenate([_to_array(s)[:, :2] for s in strokes], axis=0)
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    return {
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "cx": float(pts[:, 0].mean()), "cy": float(pts[:, 1].mean()),
        "w": xmax - xmin, "h": ymax - ymin,
    }



def _stroke_stats(stroke: Sequence[Sequence[float]]) -> Dict[str, float]:
    arr = _to_array(stroke)
    xy = arr[:, :2]
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    return {
        "xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax,
        "cx": float(xy[:, 0].mean()), "cy": float(xy[:, 1].mean()),
        "w": xmax - xmin, "h": ymax - ymin,
        "len": _path_length(xy),
        "npts": int(len(arr)),
        "t_start": float(arr[0, 2]),
        "t_end": float(arr[-1, 2]),
    }



def _safe_norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n



def _safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a / max(abs(b), eps))


def _estimate_sample_time_scale(
    strokes: Sequence[Sequence[Sequence[float]]]
) -> float:
    """
    Estimate a robust time scale for one sample using median stroke duration.
    """
    durations = []
    for st in strokes:
        arr = _to_array(st)
        if len(arr) == 0:
            continue
        dt = float(arr[-1, 2] - arr[0, 2])
        if dt > 1e-6:
            durations.append(dt)

    if len(durations) == 0:
        return 1.0

    return float(np.median(durations))



def _apply_block_weights(parts: List[np.ndarray]) -> np.ndarray:
    """
    Balanced block weighting for cosine similarity.
    Order must match embed_stroke() assembly.
    """
    weights = [
        1.8,  # geom
        1.1,  # pause / jump features  
        1.2,  # turn stats
        1.5,  # progression samples
        1.2,  # zone pct + coverage
        1.0,  # active zone onehot
        1.2,  # line_rel
        2.5,  # dot features
        1.3,  # direction hist
        1.0,  # curvature hist
        0.8,  # projection hist
        0.6,  # local grid
    ]

    out = []
    for i, p in enumerate(parts):
        p = np.asarray(p, dtype=np.float32)

        # normalize each block individually before weighting
        n = np.linalg.norm(p)
        if n > 1e-8:
            p = p / n

        w = weights[i] if i < len(weights) else 1.0
        out.append(p * w)

    return np.concatenate(out).astype(np.float32)



def _angle_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros(2, dtype=np.float32)
    v = v / n
    return np.asarray([float(v[0]), float(v[1])], dtype=np.float32)


def _signed_turn_stats(xy: np.ndarray) -> np.ndarray:
    """
    Signed turning angle statistics.
    Helps distinguish clockwise / counterclockwise / mirrored-like strokes.
    """
    if len(xy) < 3:
        return np.zeros(4, dtype=np.float32)

    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]

    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-8) & (n2 > 1e-8)
    if not np.any(valid):
        return np.zeros(4, dtype=np.float32)

    v1 = v1[valid] / n1[valid][:, None]
    v2 = v2[valid] / n2[valid][:, None]

    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    ang = np.arctan2(cross, dot)  # signed turning angle in [-pi, pi]

    return np.asarray([
        float(np.mean(ang)),
        float(np.std(ang)),
        float(np.mean(np.abs(ang))),
        float(np.sum(ang)),
    ], dtype=np.float32)


def _resample_xy_by_arclength(xy: np.ndarray, n_points: int = 7) -> np.ndarray:
    """
    Resample stroke polyline to fixed number of points by arc length.
    Keeps order information.
    """
    if len(xy) == 0:
        return np.zeros((n_points, 2), dtype=np.float32)
    if len(xy) == 1:
        return np.repeat(xy.astype(np.float32), n_points, axis=0)

    d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = cum[-1]
    if total < 1e-8:
        return np.repeat(xy[:1].astype(np.float32), n_points, axis=0)

    targets = np.linspace(0.0, total, n_points)
    out = np.zeros((n_points, 2), dtype=np.float32)

    j = 0
    for i, t in enumerate(targets):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        c0, c1 = cum[j], cum[j + 1]
        if c1 - c0 < 1e-8:
            out[i] = xy[j]
        else:
            a = (t - c0) / (c1 - c0)
            out[i] = (1 - a) * xy[j] + a * xy[j + 1]
    return out

def _looks_like_dot(raw_stats: Dict[str, float], syllable_bbox: Dict[str, float]) -> bool:
    sw = max(syllable_bbox["w"], 1e-6)
    sh = max(syllable_bbox["h"], 1e-6)
    rel_w = raw_stats["w"] / sw
    rel_h = raw_stats["h"] / sh
    area = rel_w * rel_h
    return (
        raw_stats["npts"] <= 6
        or (rel_w < 0.12 and rel_h < 0.12)
        or area < 0.01
    )


def _centered_hist(hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=np.float32)
    if hist.size == 0:
        return hist
    hist = hist - hist.mean()
    n = np.linalg.norm(hist)
    if n > 1e-8:
        hist = hist / n
    return hist.astype(np.float32)


def _centered_two_part_hist(hist: np.ndarray, bins: int) -> np.ndarray:
    """
    For projection histogram [xh, yh], center x and y independently.
    """
    xh = _centered_hist(hist[:bins])
    yh = _centered_hist(hist[bins:])
    return np.concatenate([xh, yh]).astype(np.float32)



# ============================================================
# Basic normalization (simple center + scale)
# ============================================================

def normalize_strokes_basic(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    target_half_extent: float = 1.0,
    keep_aspect: bool = True,
) -> List[np.ndarray]:
    if not strokes:
        return []

    bbox = _sample_bbox(strokes)
    cx, cy = 0.5 * (bbox["xmin"] + bbox["xmax"]), 0.5 * (bbox["ymin"] + bbox["ymax"])

    if keep_aspect:
        s = (2.0 * target_half_extent) / max(bbox["w"], bbox["h"], 1e-6)
        sx = sy = s
    else:
        sx = (2.0 * target_half_extent) / max(bbox["w"], 1e-6)
        sy = (2.0 * target_half_extent) / max(bbox["h"], 1e-6)

    out = []
    for st in strokes:
        arr = _to_array(st).copy()
        arr[:, 0] = (arr[:, 0] - cx) * sx
        arr[:, 1] = (arr[:, 1] - cy) * sy
        out.append(arr)
    return out


# ============================================================
# Guide-line helpers (position-aware embedding)
# ============================================================

def _fallback_guide_lines(strokes: Sequence[Sequence[Sequence[float]]]) -> Dict[str, float]:
    """Fallback lines if previous preprocessing is not available.

    Creates 3 equal-height bands from sample bbox.
    """
    bbox = _sample_bbox(strokes)
    h = max(bbox["h"], 1e-6)
    top = bbox["ymin"]
    spacing = h / 3.0
    return {
        "top": float(top),
        "mid_top": float(top + spacing),
        "mid_bottom": float(top + 2 * spacing),
        "bottom": float(top + 3 * spacing),
        "spacing": float(spacing),
        "middle_center": float(top + 1.5 * spacing),
    }



def _zone_coverage_from_stats(stat: Dict[str, float], lines: Dict[str, float]) -> Dict[str, float]:
    """Coverage normalized by zone height.

    Example: if a stroke fully spans middle and lower zones,
    the result can be {'middle': 1.0, 'lower': 1.0}.
    """
    upper_a, upper_b = lines["top"], lines["mid_top"]
    mid_a, mid_b = lines["mid_top"], lines["mid_bottom"]
    low_a, low_b = lines["mid_bottom"], lines["bottom"]

    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    upper_h = max(upper_b - upper_a, 1e-6)
    mid_h = max(mid_b - mid_a, 1e-6)
    low_h = max(low_b - low_a, 1e-6)

    return {
        "upper": overlap(stat["ymin"], stat["ymax"], upper_a, upper_b) / upper_h,
        "middle": overlap(stat["ymin"], stat["ymax"], mid_a, mid_b) / mid_h,
        "lower": overlap(stat["ymin"], stat["ymax"], low_a, low_b) / low_h,
    }



def _zone_pct_from_stroke(stroke: Sequence[Sequence[float]], lines: Dict[str, float]) -> Dict[str, float]:
    arr = _to_array(stroke)
    xy = arr[:, :2]
    if len(xy) == 0:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}
    if len(xy) == 1:
        y = float(xy[0, 1])
        if y < lines["mid_top"]:
            return {"upper": 1.0, "middle": 0.0, "lower": 0.0}
        elif y <= lines["mid_bottom"]:
            return {"upper": 0.0, "middle": 1.0, "lower": 0.0}
        return {"upper": 0.0, "middle": 0.0, "lower": 1.0}

    d = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(d, axis=1)
    mids = 0.5 * (xy[:-1] + xy[1:])
    total = float(seg_len.sum()) + 1e-8
    u = m = l = 0.0
    for mid, w in zip(mids, seg_len):
        y = float(mid[1])
        if y < lines["mid_top"]:
            u += w
        elif y <= lines["mid_bottom"]:
            m += w
        else:
            l += w
    return {"upper": u / total, "middle": m / total, "lower": l / total}



def _zone_active_onehot(zone_pct: Dict[str, float], zone_cov: Dict[str, float], pct_thr: float, cov_thr: float) -> np.ndarray:
    vals = [
        float(zone_pct["upper"] >= pct_thr or zone_cov["upper"] >= cov_thr),
        float(zone_pct["middle"] >= pct_thr or zone_cov["middle"] >= cov_thr),
        float(zone_pct["lower"] >= pct_thr or zone_cov["lower"] >= cov_thr),
    ]
    return np.asarray(vals, dtype=np.float32)



def _dot_role_onehot(role: str) -> np.ndarray:
    roles = [
        "none",
        "double_dot_middle",
        "single_dot_upper",
        "single_dot_lower",
        "multi_dot_upper",
        "multi_dot_middle",
        "multi_dot_lower",
    ]
    out = np.zeros(len(roles), dtype=np.float32)
    if role in roles:
        out[roles.index(role)] = 1.0
    return out


# ============================================================
# Shape features
# ============================================================

def direction_histogram(stroke_arr: np.ndarray, bins: int = 8) -> np.ndarray:
    xy = stroke_arr[:, :2]
    if len(xy) < 2:
        return np.zeros(bins, dtype=np.float32)

    d = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(d, axis=1)
    valid = seg_len > 1e-8
    if not np.any(valid):
        return np.zeros(bins, dtype=np.float32)

    ang = np.arctan2(d[valid, 1], d[valid, 0])
    ang = (ang + np.pi) / (2 * np.pi)
    idx = np.minimum((ang * bins).astype(int), bins - 1)

    hist = np.zeros(bins, dtype=np.float32)
    for i, w in zip(idx, seg_len[valid]):
        hist[i] += float(w)
        
    hist = hist / max(hist.sum(), 1e-8)
    hist = 0.25 * np.roll(hist, -1) + 0.5 * hist + 0.25 * np.roll(hist, 1)
    hist = hist / max(hist.sum(), 1e-8)
    return _centered_hist(hist)



def curvature_histogram(stroke_arr: np.ndarray, bins: int = 8) -> np.ndarray:
    xy = stroke_arr[:, :2]
    if len(xy) < 3:
        return np.zeros(bins, dtype=np.float32)

    v1 = xy[1:-1] - xy[:-2]
    v2 = xy[2:] - xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    valid = (n1 > 1e-8) & (n2 > 1e-8)
    if not np.any(valid):
        return np.zeros(bins, dtype=np.float32)

    dot = np.sum(v1[valid] * v2[valid], axis=1) / (n1[valid] * n2[valid])
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(dot)
    idx = np.minimum((ang / np.pi * bins).astype(int), bins - 1)

    hist = np.zeros(bins, dtype=np.float32)
    for i in idx:
        hist[i] += 1.0

    hist = hist / max(hist.sum(), 1e-8)
    hist = 0.25 * np.roll(hist, -1) + 0.5 * hist + 0.25 * np.roll(hist, 1)
    hist = hist / max(hist.sum(), 1e-8)
    return _centered_hist(hist)



def projection_histogram(stroke_arr: np.ndarray, bins: int = 8) -> np.ndarray:
    xy = stroke_arr[:, :2]
    if len(xy) < 2:
        return np.zeros(2 * bins, dtype=np.float32)

    mids = 0.5 * (xy[:-1] + xy[1:])
    d = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(d, axis=1)

    x = np.clip((mids[:, 0] + 1.0) / 2.0, 0.0, 0.999999)
    y = np.clip((mids[:, 1] + 1.0) / 2.0, 0.0, 0.999999)
    xi = np.minimum((x * bins).astype(int), bins - 1)
    yi = np.minimum((y * bins).astype(int), bins - 1)

    xh = np.zeros(bins, dtype=np.float32)
    yh = np.zeros(bins, dtype=np.float32)
    for i, w in zip(xi, seg_len):
        xh[i] += float(w)
    for i, w in zip(yi, seg_len):
        yh[i] += float(w)

    xh /= max(xh.sum(), 1e-8)
    yh /= max(yh.sum(), 1e-8)
    return _centered_two_part_hist(np.concatenate([xh, yh]).astype(np.float32), bins)



def local_occupancy_grid(stroke_arr: np.ndarray, grid: int = 6) -> np.ndarray:
    occ = np.zeros((grid, grid), dtype=np.float32)
    xy = stroke_arr[:, :2]
    if len(xy) < 2:
        return occ.reshape(-1)

    # local normalize around stroke itself into [-1,1]
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    s = 2.0 / max(xmax - xmin, ymax - ymin, 1e-6)
    norm = xy.copy()
    norm[:, 0] = (norm[:, 0] - cx) * s
    norm[:, 1] = (norm[:, 1] - cy) * s

    for p0, p1 in zip(norm[:-1], norm[1:]):
        dist = float(np.linalg.norm(p1 - p0))
        n = max(2, int(np.ceil(dist * grid * 1.5)))
        ts = np.linspace(0.0, 1.0, n)
        pts = (1 - ts)[:, None] * p0[None, :] + ts[:, None] * p1[None, :]
        xs = np.clip((pts[:, 0] + 1.0) / 2.0, 0.0, 0.999999)
        ys = np.clip((pts[:, 1] + 1.0) / 2.0, 0.0, 0.999999)
        xi = np.minimum((xs * grid).astype(int), grid - 1)
        yi = np.minimum((ys * grid).astype(int), grid - 1)
        for xx, yy in zip(xi, yi):
            occ[yy, xx] += 1.0

    s_occ = occ.sum()
    if s_occ > 1e-8:
        occ /= s_occ
        
    flat = occ.reshape(-1).astype(np.float32)
    flat = flat - flat.mean()
    n = np.linalg.norm(flat)
    if n > 1e-8:
        flat = flat / n
    return flat.astype(np.float32)


# ============================================================
# Stroke embedding
# ============================================================

def embed_stroke(
    raw_stroke: Sequence[Sequence[float]],
    norm_stroke: np.ndarray,
    *,
    syllable_bbox: Dict[str, float],
    raw_syllable_bbox: Dict[str, float],
    guide_lines: Dict[str, float],
    stroke_index: int,
    num_strokes: int,
    prev_norm_stroke: Optional[np.ndarray] = None,
    prev_raw_stroke: Optional[Sequence[Sequence[float]]] = None,   # ✅ NEW
    sample_time_scale: float = 1.0,                                # ✅ NEW
    is_dot: bool = False,
    dot_role: str = "none",
    stroke_zone_pct: Optional[Dict[str, float]] = None,
    stroke_zone_coverage: Optional[Dict[str, float]] = None,
    config: StrokeEmbeddingConfig | None = None,
) -> np.ndarray:
    cfg = config or StrokeEmbeddingConfig()

    raw_arr = _to_array(raw_stroke)         # float64 for correct timing
    raw_xy = raw_arr[:, :2]
    xy = norm_stroke[:, :2]
    raw_stats = _stroke_stats(raw_stroke)

    # ---------- geometry in normalized space ----------
    xmin, xmax = float(xy[:, 0].min()), float(xy[:, 0].max())
    ymin, ymax = float(xy[:, 1].min()), float(xy[:, 1].max())
    w = xmax - xmin
    h = ymax - ymin
    cx = float(xy[:, 0].mean())
    cy = float(xy[:, 1].mean())
    plen = _path_length(xy)
    start_end = float(np.linalg.norm(xy[-1] - xy[0])) if len(xy) > 0 else 0.0

    if len(xy) >= 2:
        start_dir = _safe_norm(xy[1] - xy[0])
        end_dir = _safe_norm(xy[-1] - xy[-2])
    else:
        start_dir = np.zeros(2, dtype=np.float32)
        end_dir = np.zeros(2, dtype=np.float32)

    disp = xy[-1] - xy[0] if len(xy) >= 2 else np.zeros(2, dtype=np.float32)
    disp_dir = _angle_vec(disp)

    # ✅ signed displacement in normalized space
    disp_dx = float(disp[0]) if len(xy) >= 2 else 0.0
    disp_dy = float(disp[1]) if len(xy) >= 2 else 0.0

    centroid = np.asarray([cx, cy], dtype=np.float32)
    c_to_start = _angle_vec(xy[0] - centroid) if len(xy) > 0 else np.zeros(2, dtype=np.float32)
    c_to_end = _angle_vec(xy[-1] - centroid) if len(xy) > 0 else np.zeros(2, dtype=np.float32)

    straightness = _safe_div(start_end, plen)
    aspect_log = float(np.log1p(w) - np.log1p(h))

    turn_stats = _signed_turn_stats(xy)

    resampled = _resample_xy_by_arclength(xy, n_points=7)
    progression = resampled[1:-1].reshape(-1).astype(np.float32)

    # ---------- relative position inside syllable ----------
    sw = max(raw_syllable_bbox["w"], 1e-6)
    sh = max(raw_syllable_bbox["h"], 1e-6)

    total_len = max(
        sum(_path_length(_to_array(s)[:, :2]) for s in syllable_bbox.get("raw_strokes", []))
        if "raw_strokes" in syllable_bbox else raw_stats["len"],
        1e-6
    )

    rel_cx = _safe_div(raw_stats["cx"] - raw_syllable_bbox["cx"], sw)
    rel_cy = _safe_div(raw_stats["cy"] - raw_syllable_bbox["cy"], sh)
    rel_w = _safe_div(raw_stats["w"], sw)
    rel_h = _safe_div(raw_stats["h"], sh)
    rel_len = _safe_div(raw_stats["len"], total_len)

    # ✅ signed displacement in raw space, normalized by syllable bbox
    raw_disp_dx = float(raw_xy[-1, 0] - raw_xy[0, 0]) if len(raw_xy) >= 2 else 0.0
    raw_disp_dy = float(raw_xy[-1, 1] - raw_xy[0, 1]) if len(raw_xy) >= 2 else 0.0
    raw_disp_dx_rel = _safe_div(raw_disp_dx, sw)
    raw_disp_dy_rel = _safe_div(raw_disp_dy, sh)

    # ---------- pen-up / timing ----------
    penup = 0.0
    if cfg.use_penup_features and prev_norm_stroke is not None and len(prev_norm_stroke) > 0 and len(norm_stroke) > 0:
        penup = float(np.linalg.norm(norm_stroke[0, :2] - prev_norm_stroke[-1, :2]))

    duration = max(raw_stats["t_end"] - raw_stats["t_start"], 0.0)
    speed = _safe_div(raw_stats["len"], duration, eps=1e-6) if cfg.use_time_features else 0.0

    # ---------- pause / inter-stroke transition ----------
    jump_dx = 0.0
    jump_dy = 0.0
    pause_dt = 0.0
    pause_ratio = 0.0
    slow_pause_flag = 0.0

    if prev_norm_stroke is not None and len(prev_norm_stroke) > 0 and len(norm_stroke) > 0:
        jump_vec = norm_stroke[0, :2] - prev_norm_stroke[-1, :2]
        jump_dx = float(jump_vec[0])
        jump_dy = float(jump_vec[1])

    if prev_raw_stroke is not None:
        prev_raw_arr = _to_array(prev_raw_stroke)
        pause_dt = max(float(raw_arr[0, 2] - prev_raw_arr[-1, 2]), 0.0)
        pause_ratio = _safe_div(pause_dt, sample_time_scale, eps=1e-6)
        slow_pause_flag = float(pause_ratio >= cfg.slow_pause_ratio_threshold)

    # ---------- line-aware / position-aware ----------
    if stroke_zone_pct is None:
        stroke_zone_pct = _zone_pct_from_stroke(raw_stroke, guide_lines)
    if stroke_zone_coverage is None:
        stroke_zone_coverage = _zone_coverage_from_stats(raw_stats, guide_lines)

    active_onehot = _zone_active_onehot(
        stroke_zone_pct,
        stroke_zone_coverage,
        pct_thr=cfg.active_zone_pct_threshold,
        cov_thr=cfg.active_zone_coverage_threshold,
    )

    spacing = max(float(guide_lines.get("spacing", guide_lines["mid_bottom"] - guide_lines["mid_top"])), 1e-6)
    line_rel = np.array([
        _safe_div(raw_stats["cy"] - guide_lines["top"], spacing),
        _safe_div(raw_stats["cy"] - guide_lines["mid_top"], spacing),
        _safe_div(raw_stats["cy"] - guide_lines["mid_bottom"], spacing),
        _safe_div(raw_stats["cy"] - guide_lines["bottom"], spacing),
        _safe_div(raw_stats["ymin"] - guide_lines["mid_top"], spacing),
        _safe_div(raw_stats["ymax"] - guide_lines["mid_bottom"], spacing),
    ], dtype=np.float32)

    # ---------- dot-aware ----------
    is_dot_eff = bool(is_dot) or _looks_like_dot(raw_stats, raw_syllable_bbox)

    dot_feats = np.array([float(is_dot_eff)], dtype=np.float32)
    if cfg.use_dot_features:
        compactness = _safe_div(raw_stats["len"], np.sqrt(max(raw_stats["w"] * raw_stats["h"], 1e-8)))
        aspect = _safe_div(raw_stats["w"], raw_stats["h"] + 1e-8)
        dot_feats = np.concatenate([
            dot_feats,
            _dot_role_onehot(dot_role),
            np.array([
                float(raw_stats["w"]),
                float(raw_stats["h"]),
                _safe_div(raw_stats["w"], sw),
                _safe_div(raw_stats["h"], sh),
                np.log1p(compactness),
                np.log1p(aspect),
                rel_cx,
                rel_cy,
                line_rel[0], line_rel[1], line_rel[2], line_rel[3],
            ], dtype=np.float32),
        ]).astype(np.float32)

    # ---------- assemble ----------
    parts: List[np.ndarray] = []

    geom = np.array([
        cx, cy,
        np.log1p(w), np.log1p(h),
        np.log1p(plen),
        straightness,
        rel_cx, rel_cy,
        np.log1p(rel_w), np.log1p(rel_h),
        np.log1p(rel_len),

        start_dir[0], start_dir[1],
        end_dir[0], end_dir[1],
        disp_dir[0], disp_dir[1],

        # ✅ NEW signed displacement features
        disp_dx, disp_dy,
        raw_disp_dx_rel, raw_disp_dy_rel,

        c_to_start[0], c_to_start[1],
        c_to_end[0], c_to_end[1],

        aspect_log,
        np.log1p(penup),
        np.log1p(duration),
        np.log1p(speed),

        stroke_index / max(num_strokes - 1, 1),
        float(num_strokes),
    ], dtype=np.float32)

    parts.append(geom)

    # ✅ NEW pause block
    if cfg.use_pause_features:
        pause_feats = np.array([
            jump_dx,
            jump_dy,
            np.log1p(pause_dt),
            np.log1p(pause_ratio),
            slow_pause_flag,
        ], dtype=np.float32)
        parts.append(pause_feats)

    parts.append(turn_stats)
    parts.append(progression)

    if cfg.use_line_position_features:
        parts.append(np.array([
            stroke_zone_pct["upper"], stroke_zone_pct["middle"], stroke_zone_pct["lower"],
            stroke_zone_coverage["upper"], stroke_zone_coverage["middle"], stroke_zone_coverage["lower"],
        ], dtype=np.float32))
        parts.append(active_onehot)
        parts.append(line_rel)

    if cfg.use_dot_features:
        parts.append(dot_feats)

    if cfg.use_direction_hist:
        dh = direction_histogram(norm_stroke, bins=cfg.direction_bins)
        if is_dot_eff:
            dh = np.zeros_like(dh)
        parts.append(dh)

    if cfg.use_curvature_hist:
        ch = curvature_histogram(norm_stroke, bins=cfg.curvature_bins)
        if is_dot_eff:
            ch = np.zeros_like(ch)
        parts.append(ch)

    if cfg.use_projection_hist:
        ph = projection_histogram(norm_stroke, bins=cfg.projection_bins)
        parts.append(ph)

    if cfg.use_local_grid:
        lg = local_occupancy_grid(norm_stroke, grid=cfg.local_grid)
        if is_dot_eff:
            lg = np.zeros_like(lg)
        parts.append(lg)

    emb = _apply_block_weights(parts)

    norm = np.linalg.norm(emb)
    if norm > 1e-8:
        emb = emb / norm

    return emb.astype(np.float32)



# ============================================================
# Stroke embedding from syllable / row
# ============================================================
def embed_syllable_strokes(
    strokes: Sequence[Sequence[Sequence[float]]],
    *,
    guide_lines: Optional[Dict[str, float]] = None,
    is_dot_list: Optional[Sequence[bool]] = None,
    dot_role_list: Optional[Sequence[str]] = None,
    stroke_zone_pct_list: Optional[Sequence[Dict[str, float]]] = None,
    stroke_zone_coverage_list: Optional[Sequence[Dict[str, float]]] = None,
    config: StrokeEmbeddingConfig | None = None,
) -> List[np.ndarray]:
    cfg = config or StrokeEmbeddingConfig()
    if not strokes:
        return []

    # ✅ Convert times to relative seconds first
    raw_time_strokes = _shift_stroke_times_to_relative(strokes)

    # bbox still comes from original x,y coordinates
    raw_bbox = _sample_bbox(strokes)
    raw_bbox["raw_strokes"] = strokes

    lines = guide_lines if guide_lines is not None else _fallback_guide_lines(strokes)
    norm_strokes = normalize_strokes_basic(
        strokes,
        target_half_extent=cfg.target_half_extent,
        keep_aspect=cfg.keep_aspect,
    )

    n = len(strokes)
    if is_dot_list is None:
        is_dot_list = [False] * n
    if dot_role_list is None:
        dot_role_list = ["none"] * n
    if stroke_zone_pct_list is None:
        stroke_zone_pct_list = [None] * n
    if stroke_zone_coverage_list is None:
        stroke_zone_coverage_list = [None] * n

    sample_time_scale = _estimate_sample_time_scale(raw_time_strokes)

    embs = []
    for i in range(n):
        prev = norm_strokes[i - 1] if i > 0 else None
        prev_raw = raw_time_strokes[i - 1] if i > 0 else None

        embs.append(
            embed_stroke(
                raw_time_strokes[i],          
                norm_strokes[i],
                syllable_bbox=raw_bbox,
                raw_syllable_bbox=raw_bbox,
                guide_lines=lines,
                stroke_index=i,
                num_strokes=n,
                prev_norm_stroke=prev,
                prev_raw_stroke=prev_raw,     
                sample_time_scale=sample_time_scale,
                is_dot=bool(is_dot_list[i]),
                dot_role=str(dot_role_list[i]),
                stroke_zone_pct=stroke_zone_pct_list[i],
                stroke_zone_coverage=stroke_zone_coverage_list[i],
                config=cfg,
            )
        )
    return embs


def embed_stroke_dataframe(
    df: pd.DataFrame,
    *,
    strokes_col: str = "cleaned_strokes",
    fallback_strokes_col: str = "strokes",
    label_col: str = "id",
    guide_lines_col: str = "guide_lines",
    dots_col: str = "is_dot",
    dot_role_col: str = "dot_role",
    zone_pct_col: str = "stroke_zone_pct",
    zone_coverage_col: str = "stroke_zone_coverage",
    config: StrokeEmbeddingConfig | None = None,
) -> pd.DataFrame:
    """
    Explode one-row-per-syllable dataframe into one-row-per-stroke embedding dataframe.

    Notes
    -----
    - `label_col` is typically syllable/class id, not stroke id.
    - This function therefore creates multiple labels:
        * syllable_id_label   : numeric/class id from `label_col`
        * syllable_label      : human-readable syllable string if available
        * stroke_order_label  : <syllable>__s<stroke_index>
        * stroke_struct_label : <syllable>__<dot_role or s<stroke_index>>
        * embedding_label     : defaults to stroke_order_label for stroke-level evaluation

    This is important because evaluating stroke embeddings using only syllable id
    can incorrectly treat different stroke roles inside the same syllable as "same class".
    """
    cfg = config or StrokeEmbeddingConfig()
    rows: List[Dict[str, object]] = []

    def _is_list_like(v) -> bool:
        return isinstance(v, (list, tuple))

    def _pad_list(vals, n, default):
        if not _is_list_like(vals):
            return [default] * n
        vals = list(vals)
        if len(vals) < n:
            vals = vals + [default] * (n - len(vals))
        elif len(vals) > n:
            vals = vals[:n]
        return vals

    def _safe_row_get(row: pd.Series, col: str, default=None):
        return row[col] if col in row.index else default

    for sample_index, row in df.iterrows():
        # ------------------------------------------------------------
        # Select strokes
        # ------------------------------------------------------------
        raw_strokes = _safe_row_get(row, strokes_col, None)
        if not _is_list_like(raw_strokes):
            raw_strokes = _safe_row_get(row, fallback_strokes_col, None)

        if not _is_list_like(raw_strokes) or len(raw_strokes) == 0:
            continue

        strokes = raw_strokes
        n_strokes = len(strokes)

        # ------------------------------------------------------------
        # Optional per-syllable / per-stroke metadata
        # ------------------------------------------------------------
        guide_lines = _safe_row_get(row, guide_lines_col, None)
        if not isinstance(guide_lines, dict):
            guide_lines = _fallback_guide_lines(strokes)

        is_dot_list = _pad_list(_safe_row_get(row, dots_col, None), n_strokes, False)
        dot_role_list = _pad_list(_safe_row_get(row, dot_role_col, None), n_strokes, "none")
        zone_pct_list = _pad_list(_safe_row_get(row, zone_pct_col, None), n_strokes, None)
        zone_cov_list = _pad_list(_safe_row_get(row, zone_coverage_col, None), n_strokes, None)

        # Additional optional debug fields if present in dataframe
        major_zone_list = _pad_list(_safe_row_get(row, "stroke_major_zone", None), n_strokes, None)
        active_zones_list = _pad_list(_safe_row_get(row, "stroke_active_zones", None), n_strokes, None)
        single_position_list = _pad_list(_safe_row_get(row, "stroke_single_position", None), n_strokes, None)

        # ------------------------------------------------------------
        # Base labels
        # ------------------------------------------------------------
        base_id_label = _safe_row_get(row, label_col, None)
        base_syllable = _safe_row_get(row, "syllable", None)

        # Prefer readable syllable text if available, else fall back to id
        if base_syllable is None:
            base_syllable = str(base_id_label) if base_id_label is not None else "unknown"

        # ------------------------------------------------------------
        # Compute embeddings
        # ------------------------------------------------------------
        embs = embed_syllable_strokes(
            strokes,
            guide_lines=guide_lines,
            is_dot_list=is_dot_list,
            dot_role_list=dot_role_list,
            stroke_zone_pct_list=zone_pct_list,
            stroke_zone_coverage_list=zone_cov_list,
            config=cfg,
        )

        # ------------------------------------------------------------
        # Explode into stroke rows
        # ------------------------------------------------------------
        for stroke_index, emb in enumerate(embs):
            r = dict(row)

            stroke_is_dot = bool(is_dot_list[stroke_index])
            stroke_dot_role = str(dot_role_list[stroke_index])

            # Stroke labels:
            # - order label is best default for stroke-level evaluation
            # - struct label is useful for debugging / dot-aware analysis
            stroke_order_label = f"{base_syllable}__s{stroke_index}"
            stroke_struct_suffix = stroke_dot_role if stroke_is_dot and stroke_dot_role != "none" else f"s{stroke_index}"
            stroke_struct_label = f"{base_syllable}__{stroke_struct_suffix}"

            # Optional quick geometry metadata for debug
            try:
                arr = _to_array(strokes[stroke_index])
                xy = arr[:, :2]
                if len(xy) >= 2:
                    disp = xy[-1] - xy[0]
                    disp_angle = float(np.arctan2(disp[1], disp[0])) if np.linalg.norm(disp) > 1e-8 else 0.0
                else:
                    disp_angle = 0.0
                stroke_n_points = int(len(arr))
            except Exception:
                disp_angle = 0.0
                stroke_n_points = 0

            r["stroke_index"] = stroke_index
            r["num_strokes"] = n_strokes

            r["embedding"] = emb
            r["embedding_dim"] = int(len(emb))

            # ----- labels -----
            r["syllable_id_label"] = base_id_label
            r["syllable_label"] = base_syllable
            r["stroke_order_label"] = stroke_order_label
            r["stroke_struct_label"] = stroke_struct_label

            # Default label for stroke-level embedding evaluation
            r["embedding_label"] = stroke_order_label

            # ----- explicit stroke-level metadata -----
            r["stroke_is_dot"] = stroke_is_dot
            r["stroke_dot_role"] = stroke_dot_role
            r["stroke_major_zone"] = major_zone_list[stroke_index]
            r["stroke_active_zones"] = active_zones_list[stroke_index]
            r["stroke_single_position"] = single_position_list[stroke_index]

            # ----- debug geometry -----
            r["stroke_num_points"] = stroke_n_points
            r["stroke_disp_angle"] = disp_angle

            rows.append(r)

    return pd.DataFrame(rows)


def stack_embeddings(emb_df: pd.DataFrame, emb_col: str = "embedding") -> np.ndarray:
    return np.vstack(emb_df[emb_col].values).astype(np.float32)


# ============================================================
# Optional compression / scaling
# ============================================================

def fit_pca_embedder(X_train: np.ndarray, n_components: int = 32) -> Dict[str, object]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train)
    n_components = min(n_components, Xs.shape[1], max(1, Xs.shape[0]))
    pca = PCA(n_components=n_components, random_state=42)
    Xp = pca.fit_transform(Xs)
    return {"scaler": scaler, "pca": pca, "output_dim": Xp.shape[1]}



def transform_with_embedder(X: np.ndarray, model: Dict[str, object]) -> np.ndarray:
    scaler = model["scaler"]
    pca = model["pca"]
    return pca.transform(scaler.transform(X)).astype(np.float32)


# ============================================================
# Plot helpers
# ============================================================

def _plot_single_stroke(ax, stroke: Sequence[Sequence[float]], title: str = "", invert_y: bool = True):
    arr = _to_array(stroke)
    xy = arr[:, :2]
    ax.set_title(title)
    if len(xy) == 1:
        ax.scatter(xy[:, 0], xy[:, 1], s=20)
    else:
        ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.6)
        ax.scatter(xy[:, 0], xy[:, 1], s=6)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    if invert_y:
        ax.invert_yaxis()


# ============================================================
# Similarity / retrieval
# ============================================================

def cosine_similarity_matrix(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    return cosine_similarity(X, X if Y is None else Y)



def plot_query_neighbors(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    query_idx: int,
    top_k: int = 5,
    strokes_col: str = "cleaned_strokes",
    fallback_strokes_col: str = "strokes",
    stroke_index_col: str = "stroke_index",
    label_col: str = "embedding_label",
    title_col: Optional[str] = None,
):
    """Plot nearest stroke neighbors for one query stroke."""
    sims = cosine_similarity(X[query_idx:query_idx + 1], X).ravel()
    order = np.argsort(-sims)
    order = [i for i in order if i != query_idx][:top_k]

    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 4))

    def _get_stroke(i: int):
        row = df.iloc[i]
        strokes = row[strokes_col] if strokes_col in df.columns and isinstance(row.get(strokes_col, None), (list, tuple)) else row[fallback_strokes_col]
        sidx = int(row[stroke_index_col])
        return strokes[sidx]

    def _title_for(i: int, prefix: str = "") -> str:
        row = df.iloc[i]
        parts = [prefix]
        if label_col in df.columns:
            parts.append(f"label={row[label_col]}")
        parts.append(f"stroke={row[stroke_index_col]}")
        if title_col is not None and title_col in df.columns:
            parts.append(str(row[title_col]))
        return "\n".join([p for p in parts if p])

    _plot_single_stroke(axes[0], _get_stroke(query_idx), title=_title_for(query_idx, prefix="Query"))
    for ax, idx in zip(axes[1:], order):
        _plot_single_stroke(ax, _get_stroke(idx), title=_title_for(idx, prefix=f"sim={sims[idx]:.3f}"))

    plt.tight_layout()
    plt.show()
    return fig, axes


# ============================================================
# Evaluation: prototypes, kNN, retrieval
# ============================================================

def train_class_prototypes(X_train: np.ndarray, y_train: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    y_train = np.asarray(y_train)
    classes = np.unique(y_train)
    protos = []
    for c in classes:
        protos.append(X_train[y_train == c].mean(axis=0))
    return np.vstack(protos).astype(np.float32), classes



def predict_topk_prototype(X: np.ndarray, prototypes: np.ndarray, proto_labels: np.ndarray, k: int = 5) -> List[List]:
    S = cosine_similarity(X, prototypes)
    order = np.argsort(-S, axis=1)[:, :k]
    return [[proto_labels[j] for j in row] for row in order]



def evaluate_prototype_classifier(
    X_train: np.ndarray,
    y_train: Sequence,
    X_test: np.ndarray,
    y_test: Sequence,
    *,
    ks: Sequence[int] = (1, 5),
) -> pd.DataFrame:
    protos, proto_labels = train_class_prototypes(X_train, y_train)
    y_test = np.asarray(y_test)

    results = []
    for k in ks:
        preds_topk = predict_topk_prototype(X_test, protos, proto_labels, k=k)
        hit = np.array([yt in pred for yt, pred in zip(y_test, preds_topk)], dtype=np.float32)
        results.append({"metric": f"top{k}_prototype_acc", "value": float(hit.mean())})

    top1 = predict_topk_prototype(X_test, protos, proto_labels, k=1)
    y_pred = np.array([p[0] for p in top1])
    acc = float((y_pred == y_test).mean())
    results.append({"metric": "top1_prototype_acc", "value": acc})
    return pd.DataFrame(results)



def evaluate_knn_classifier(
    X_train: np.ndarray,
    y_train: Sequence,
    X_test: np.ndarray,
    y_test: Sequence,
    *,
    n_neighbors: int = 5,
    ks: Sequence[int] = (1, 5),
) -> pd.DataFrame:
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    nn = NearestNeighbors(n_neighbors=max(max(ks), n_neighbors), metric="cosine")
    nn.fit(X_train)
    _, idx = nn.kneighbors(X_test)
    neigh_labels = y_train[idx]

    results = []
    for k in ks:
        topk = neigh_labels[:, :k]
        hit = np.array([yt in row for yt, row in zip(y_test, topk)], dtype=np.float32)
        results.append({"metric": f"top{k}_knn_acc", "value": float(hit.mean())})
    return pd.DataFrame(results)



def evaluate_retrieval(
    X: np.ndarray,
    y: Sequence,
    *,
    ks: Sequence[int] = (1, 5, 10),
    exclude_self: bool = True,
) -> pd.DataFrame:
    y = np.asarray(y)
    S = cosine_similarity(X)
    n = len(y)

    if exclude_self:
        np.fill_diagonal(S, -np.inf)

    order = np.argsort(-S, axis=1)
    results = []

    for k in ks:
        hits = []
        for i in range(n):
            topk = order[i, :k]
            rel = (y[topk] == y[i]).astype(np.float32)
            hits.append(float(rel.mean()))
        results.append({"metric": f"precision@{k}", "value": float(np.mean(hits))})

    ranks = []
    for i in range(n):
        same = (y[order[i]] == y[i])
        where = np.where(same)[0]
        if len(where) > 0:
            ranks.append(int(where[0] + 1))
    if ranks:
        results.append({"metric": "mean_first_correct_rank", "value": float(np.mean(ranks))})
    return pd.DataFrame(results)


# ============================================================
# Statistical evaluation (without heatmap)
# ============================================================

def _sample_pair_indices_stratified(
    y: Sequence,
    *,
    max_same_pairs: int = 5000,
    max_diff_pairs: int = 5000,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    label_to_idx = {}
    for i, lab in enumerate(y):
        label_to_idx.setdefault(lab, []).append(i)

    # -------------------------
    # Same-label pairs
    # -------------------------
    same_pairs = []
    for lab, idxs in label_to_idx.items():
        if len(idxs) < 2:
            continue
        local = []
        for a_i in range(len(idxs)):
            for b_i in range(a_i + 1, len(idxs)):
                local.append((idxs[a_i], idxs[b_i]))

        same_pairs.extend(local)

    if len(same_pairs) > max_same_pairs:
        take = rng.choice(len(same_pairs), size=max_same_pairs, replace=False)
        same_pairs = [same_pairs[i] for i in take]

    # -------------------------
    # Different-label pairs
    # -------------------------
    unique_labels = list(label_to_idx.keys())
    diff_pairs = set()

    while len(diff_pairs) < max_diff_pairs:
        la, lb = rng.choice(unique_labels, size=2, replace=False)
        ia = int(rng.choice(label_to_idx[la]))
        ib = int(rng.choice(label_to_idx[lb]))
        a, b = (ia, ib) if ia < ib else (ib, ia)
        diff_pairs.add((a, b))

    diff_pairs = list(diff_pairs)

    return np.asarray(same_pairs + diff_pairs, dtype=np.int32)



def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    if pooled < 1e-8:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)



def pairwise_similarity_stats(
    X: np.ndarray,
    y: Sequence,
    *,
    max_pairs: int = 50000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    y = np.asarray(y)

    pairs = _sample_pair_indices_stratified(
        y,
        max_same_pairs=max_pairs // 2,
        max_diff_pairs=max_pairs // 2,
        random_state=random_state,
    )

    sims = np.sum(X[pairs[:, 0]] * X[pairs[:, 1]], axis=1) / (
        np.linalg.norm(X[pairs[:, 0]], axis=1) * np.linalg.norm(X[pairs[:, 1]], axis=1) + 1e-8
    )
    same = y[pairs[:, 0]] == y[pairs[:, 1]]
    same_sims = sims[same]
    diff_sims = sims[~same]

    labels = np.concatenate([np.ones(len(same_sims)), np.zeros(len(diff_sims))]).astype(np.int32)
    scores = np.concatenate([same_sims, diff_sims]) if len(diff_sims) > 0 else same_sims.copy()
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else float("nan")

    summary = pd.DataFrame([
        {"metric": "same_mean", "value": float(same_sims.mean()) if len(same_sims) else float("nan")},
        {"metric": "same_std", "value": float(same_sims.std(ddof=1)) if len(same_sims) > 1 else float("nan")},
        {"metric": "diff_mean", "value": float(diff_sims.mean()) if len(diff_sims) else float("nan")},
        {"metric": "diff_std", "value": float(diff_sims.std(ddof=1)) if len(diff_sims) > 1 else float("nan")},
        {"metric": "similarity_gap", "value": float(same_sims.mean() - diff_sims.mean()) if len(same_sims) and len(diff_sims) else float("nan")},
        {"metric": "cohens_d", "value": _cohens_d(same_sims, diff_sims)},
        {"metric": "same_vs_diff_auc", "value": float(auc)},
        {"metric": "num_same_pairs", "value": int(len(same_sims))},
        {"metric": "num_diff_pairs", "value": int(len(diff_sims))},
    ])
    return summary, {"same_sims": same_sims, "diff_sims": diff_sims}



def bootstrap_similarity_stats(
    same_sims: np.ndarray,
    diff_sims: np.ndarray,
    *,
    n_boot: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    same_sims = np.asarray(same_sims, dtype=np.float32)
    diff_sims = np.asarray(diff_sims, dtype=np.float32)

    if len(same_sims) == 0 or len(diff_sims) == 0:
        return pd.DataFrame(columns=["metric", "mean", "ci_low", "ci_high"])

    same_means, diff_means, gaps, aucs = [], [], [], []
    labels = np.concatenate([np.ones(len(same_sims)), np.zeros(len(diff_sims))]).astype(np.int32)

    for _ in range(n_boot):
        s = same_sims[rng.integers(0, len(same_sims), len(same_sims))]
        d = diff_sims[rng.integers(0, len(diff_sims), len(diff_sims))]
        same_means.append(float(s.mean()))
        diff_means.append(float(d.mean()))
        gaps.append(float(s.mean() - d.mean()))
        scores = np.concatenate([s, d])
        aucs.append(float(roc_auc_score(labels, scores)))

    def _row(name: str, vals: Sequence[float]) -> Dict[str, float]:
        vals = np.asarray(vals, dtype=np.float32)
        return {
            "metric": name,
            "mean": float(vals.mean()),
            "ci_low": float(np.quantile(vals, 0.025)),
            "ci_high": float(np.quantile(vals, 0.975)),
        }

    return pd.DataFrame([
        _row("same_mean", same_means),
        _row("diff_mean", diff_means),
        _row("similarity_gap", gaps),
        _row("same_vs_diff_auc", aucs),
    ])



def evaluate_embedding_statistics(
    X: np.ndarray,
    y: Sequence,
    *,
    max_pairs: int = 50000,
    n_boot: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    summary, raw = pairwise_similarity_stats(X, y, max_pairs=max_pairs, random_state=random_state)
    boot = bootstrap_similarity_stats(raw["same_sims"], raw["diff_sims"], n_boot=n_boot, random_state=random_state)
    if len(boot) > 0:
        boot_long = []
        for _, row in boot.iterrows():
            boot_long.append({"metric": row["metric"] + "_mean", "value": row["mean"]})
            boot_long.append({"metric": row["metric"] + "_ci_low", "value": row["ci_low"]})
            boot_long.append({"metric": row["metric"] + "_ci_high", "value": row["ci_high"]})
        boot_df = pd.DataFrame(boot_long)
        return pd.concat([summary, boot_df], ignore_index=True)
    return summary



def evaluate_cluster_indices(X: np.ndarray, y: Sequence) -> pd.DataFrame:
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2:
        return pd.DataFrame([
            {"metric": "silhouette", "value": float("nan")},
            {"metric": "calinski_harabasz", "value": float("nan")},
            {"metric": "davies_bouldin", "value": float("nan")},
        ])

    results = [
        {"metric": "silhouette", "value": float(silhouette_score(X, y, metric="cosine"))},
        {"metric": "calinski_harabasz", "value": float(calinski_harabasz_score(X, y))},
        {"metric": "davies_bouldin", "value": float(davies_bouldin_score(X, y))},
    ]
    return pd.DataFrame(results)



def plot_similarity_distributions(
    X: np.ndarray,
    y: Sequence,
    *,
    max_pairs: int = 50000,
    bins: int = 50,
    random_state: int = 42,
):
    summary, raw = pairwise_similarity_stats(X, y, max_pairs=max_pairs, random_state=random_state)
    same_sims = raw["same_sims"]
    diff_sims = raw["diff_sims"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diff_sims, bins=bins, alpha=0.6, label="different class", density=True)
    ax.hist(same_sims, bins=bins, alpha=0.6, label="same class", density=True)
    ax.set_title("Stroke embedding similarity distributions")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax, summary


# ============================================================
# Debug / inspection helpers
# ============================================================

def describe_embedding_config(config: StrokeEmbeddingConfig | None = None) -> Dict[str, object]:
    cfg = config or StrokeEmbeddingConfig()
    return asdict(cfg)



def debug_retrieval_confusions(
    emb_df: pd.DataFrame,
    X: np.ndarray,
    *,
    label_col: str = "embedding_label",
    syllable_col: str = "syllable_label",
    top_k: int = 10,
) -> pd.DataFrame:
    y = emb_df[label_col].astype(str).values
    syll = emb_df[syllable_col].astype(str).values if syllable_col in emb_df.columns else y

    S = cosine_similarity(X)
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)

    rows = []
    for i in range(len(X)):
        q = emb_df.iloc[i]

        # first wrong neighbor
        wrong_j = None
        for j in order[i]:
            if y[j] != y[i]:
                wrong_j = j
                break

        # first correct neighbor
        correct_j = None
        for j in order[i]:
            if y[j] == y[i]:
                correct_j = j
                break

        if wrong_j is not None:
            n = emb_df.iloc[wrong_j]
            rows.append({
                "query_idx": i,
                "pair_type": "wrong_top1",
                "query_label": y[i],
                "neighbor_label": y[wrong_j],
                "query_syllable": syll[i],
                "neighbor_syllable": syll[wrong_j],
                "same_syllable": syll[i] == syll[wrong_j],
                "query_stroke_index": q.get("stroke_index", None),
                "neighbor_stroke_index": n.get("stroke_index", None),
                "same_stroke_index": q.get("stroke_index", None) == n.get("stroke_index", None),
                "query_is_dot": q.get("stroke_is_dot", None),
                "neighbor_is_dot": n.get("stroke_is_dot", None),
                "same_dot_flag": q.get("stroke_is_dot", None) == n.get("stroke_is_dot", None),
                "query_dot_role": q.get("stroke_dot_role", None),
                "neighbor_dot_role": n.get("stroke_dot_role", None),
                "query_zone": q.get("stroke_major_zone", None),
                "neighbor_zone": n.get("stroke_major_zone", None),
                "same_zone": q.get("stroke_major_zone", None) == n.get("stroke_major_zone", None),
                "similarity": float(S[i, wrong_j]),
            })

        if correct_j is not None:
            c = emb_df.iloc[correct_j]
            rows.append({
                "query_idx": i,
                "pair_type": "best_correct",
                "query_label": y[i],
                "neighbor_label": y[correct_j],
                "query_syllable": syll[i],
                "neighbor_syllable": syll[correct_j],
                "same_syllable": syll[i] == syll[correct_j],
                "query_stroke_index": q.get("stroke_index", None),
                "neighbor_stroke_index": c.get("stroke_index", None),
                "same_stroke_index": q.get("stroke_index", None) == c.get("stroke_index", None),
                "query_is_dot": q.get("stroke_is_dot", None),
                "neighbor_is_dot": c.get("stroke_is_dot", None),
                "same_dot_flag": q.get("stroke_is_dot", None) == c.get("stroke_is_dot", None),
                "query_dot_role": q.get("stroke_dot_role", None),
                "neighbor_dot_role": c.get("stroke_dot_role", None),
                "query_zone": q.get("stroke_major_zone", None),
                "neighbor_zone": c.get("stroke_major_zone", None),
                "same_zone": q.get("stroke_major_zone", None) == c.get("stroke_major_zone", None),
                "similarity": float(S[i, correct_j]),
            })

    return pd.DataFrame(rows)

def debug_sample_time_features(
    strokes: Sequence[Sequence[Sequence[float]]],
):
    """
    Print raw and relative timing per stroke for quick debugging.
    """
    raw = [_to_array(s) for s in strokes]
    rel = _shift_stroke_times_to_relative(strokes)

    print("=== RAW TIMES ===")
    for i, arr in enumerate(raw):
        print(
            i,
            "start=", float(arr[0, 2]),
            "end=", float(arr[-1, 2]),
            "dur=", float(arr[-1, 2] - arr[0, 2]),
        )

    print("\n=== RELATIVE TIMES ===")
    for i, arr in enumerate(rel):
        print(
            i,
            "start=", float(arr[0, 2]),
            "end=", float(arr[-1, 2]),
            "dur=", float(arr[-1, 2] - arr[0, 2]),
        )

    print("\n=== PAUSES ===")
    for i in range(1, len(rel)):
        pause = float(rel[i][0, 2] - rel[i - 1][-1, 2])
        print(i - 1, "->", i, "pause=", pause)


__all__ = [
    "StrokeEmbeddingConfig",
    "configure_myanmar_font",
    "normalize_strokes_basic",
    "direction_histogram",
    "curvature_histogram",
    "projection_histogram",
    "local_occupancy_grid",
    "embed_stroke",
    "embed_syllable_strokes",
    "embed_stroke_dataframe",
    "stack_embeddings",
    "fit_pca_embedder",
    "transform_with_embedder",
    "cosine_similarity_matrix",
    "plot_query_neighbors",
    "train_class_prototypes",
    "predict_topk_prototype",
    "evaluate_prototype_classifier",
    "evaluate_knn_classifier",
    "evaluate_retrieval",
    "pairwise_similarity_stats",
    "bootstrap_similarity_stats",
    "evaluate_embedding_statistics",
    "evaluate_cluster_indices",
    "plot_similarity_distributions",
    "describe_embedding_config",
]
