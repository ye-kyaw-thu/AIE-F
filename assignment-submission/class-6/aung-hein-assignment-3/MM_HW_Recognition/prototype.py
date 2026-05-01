"""Prototype-based syllable matching for Myanmar online handwriting.

This module operates on stroke embeddings produced by embedding.py,
especially the dataframe returned by `embed_stroke_dataframe(...)`.

Main idea
---------
Train one or more prototype embeddings per (syllable, stroke_index), then score
a whole query syllable against candidate syllable prototype sequences.

This module supports TWO roles:

1) Prototype classifier / scorer
   - direct syllable prototype scoring
   - top-k prediction
   - evaluation

2) Recall-safe syllable gating (recommended for Segmental HMM)
   - returns a candidate syllable set by UNION of:
       * shape-based prototype recall
       * stroke-count compatibility
       * char-length compatibility
   - designed for HIGH RECALL, not final ranking

Designed for:
- Raspberry Pi / CPU-only deployment
- prototype classification
- recall-safe candidate generation
- Segmental HMM shortlist generation
- interpretable debugging
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from collections import Counter
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class PrototypeConfig:
    # prototype construction
    prototype_mode: str = "mean"   # "mean" or "medoid"
    normalize_prototypes: bool = True

    # optional multi-prototype support per stroke index
    max_subprototypes_per_stroke: int = 1
    min_examples_for_subprototype_split: int = 6
    subprototype_refine_iters: int = 2

    # scoring
    average_similarity: bool = True
    missing_stroke_penalty: float = 0.45
    extra_stroke_penalty: float = 0.45
    zone_mismatch_penalty: float = 0.08
    dot_mismatch_penalty: float = 0.18
    dot_role_mismatch_penalty: float = 0.10

    # candidate pruning inside prototype scorer
    max_num_strokes_diff: Optional[int] = 2

    # score->cost mapping support
    score_temperature: float = 1.0
    cost_eps: float = 1e-8


# ============================================================
# Basic helpers
# ============================================================

def _ensure_char_array(
    row: pd.Series,
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
) -> List[str]:
    if char_array_col in row.index and isinstance(row.get(char_array_col, None), (list, tuple)):
        return [str(x) for x in row[char_array_col]]
    return list(str(row[syllable_col]))


def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _stack_rows(embs: Sequence[np.ndarray]) -> np.ndarray:
    if len(embs) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack([np.asarray(e, dtype=np.float32) for e in embs]).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_cosine_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-8)
    return (Xn @ Xn.T).astype(np.float32)


def _mode_value(vals: Sequence, default=None):
    vals = [v for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return default
    return Counter(vals).most_common(1)[0][0]


def _safe_bool_mode(vals: Sequence[bool], default: bool = False) -> bool:
    vals = [bool(v) for v in vals if pd.notna(v)]
    if len(vals) == 0:
        return default
    c = Counter(vals)
    return bool(c.most_common(1)[0][0])


# ============================================================
# Prototype construction helpers
# ============================================================

def _prototype_from_rows(
    X: np.ndarray,
    *,
    mode: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    """Build one prototype vector from a set of embeddings."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(X) == 1:
        proto = X[0]
        return _l2_normalize(proto) if normalize else proto.astype(np.float32)

    if mode == "mean":
        proto = X.mean(axis=0)
    elif mode == "medoid":
        S = _pairwise_cosine_matrix(X)
        idx = int(np.argmax(S.sum(axis=1)))
        proto = X[idx]
    else:
        raise ValueError(f"Unknown prototype_mode={mode!r}. Use 'mean' or 'medoid'.")

    proto = proto.astype(np.float32)
    return _l2_normalize(proto) if normalize else proto


def _select_subprototype_indices(
    X: np.ndarray,
    *,
    k: int,
    refine_iters: int = 2,
) -> List[int]:
    """Greedy farthest-point + medoid refinement on cosine space."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n == 0:
        return []
    if k <= 1 or n == 1:
        return [0]

    S = _pairwise_cosine_matrix(X)

    # first center = medoid
    centers = [int(np.argmax(S.sum(axis=1)))]

    # greedy farthest-point
    while len(centers) < min(k, n):
        best_idx = None
        best_min_dist = -np.inf
        for i in range(n):
            if i in centers:
                continue
            sim_to_centers = np.max(S[i, centers])
            min_dist = 1.0 - sim_to_centers
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = i
        if best_idx is None:
            break
        centers.append(int(best_idx))

    # small refinement
    centers = list(dict.fromkeys(centers))
    for _ in range(max(0, refine_iters)):
        if len(centers) <= 1:
            break
        center_sims = S[:, centers]
        assign = np.argmax(center_sims, axis=1)

        new_centers = []
        for c in range(len(centers)):
            idxs = np.where(assign == c)[0]
            if len(idxs) == 0:
                new_centers.append(centers[c])
                continue
            local_S = S[np.ix_(idxs, idxs)]
            local_best = idxs[int(np.argmax(local_S.sum(axis=1)))]
            new_centers.append(int(local_best))

        centers = list(dict.fromkeys(new_centers))

    return centers


def _build_subprototypes(
    X: np.ndarray,
    *,
    cfg: PrototypeConfig,
) -> List[np.ndarray]:
    """Return one or more subprototype vectors for a stroke class."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    if n == 0:
        return []

    k = int(cfg.max_subprototypes_per_stroke)
    if k <= 1 or n < max(cfg.min_examples_for_subprototype_split, 2):
        return [_prototype_from_rows(X, mode=cfg.prototype_mode, normalize=cfg.normalize_prototypes)]

    k = min(k, n)
    center_idxs = _select_subprototype_indices(
        X,
        k=k,
        refine_iters=cfg.subprototype_refine_iters,
    )

    # final assignment
    S = _pairwise_cosine_matrix(X)
    center_sims = S[:, center_idxs]
    assign = np.argmax(center_sims, axis=1)

    protos: List[np.ndarray] = []
    for c in range(len(center_idxs)):
        idxs = np.where(assign == c)[0]
        if len(idxs) == 0:
            continue
        proto = _prototype_from_rows(
            X[idxs],
            mode=cfg.prototype_mode,
            normalize=cfg.normalize_prototypes,
        )
        protos.append(proto)

    if len(protos) == 0:
        protos.append(_prototype_from_rows(X, mode=cfg.prototype_mode, normalize=cfg.normalize_prototypes))
    return protos


# ============================================================
# Training
# ============================================================

def train_syllable_prototype_bank(
    emb_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    emb_col: str = "embedding",
    num_strokes_col: str = "num_strokes",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    config: PrototypeConfig | None = None,
) -> Dict[str, object]:
    """
    Train a syllable prototype bank from stroke embeddings.

    Output structure
    ----------------
    bank = {
        "config": ...,
        "embedding_dim": ...,
        "syllables": {
            syllable_label: {
                "syllable_label": ...,
                "expected_num_strokes": ...,
                "char_len": ...,
                "num_samples": ...,
                "stroke_prototypes": {
                    stroke_index: {
                        "stroke_index": ...,
                        "prototypes": [emb1, emb2, ...],
                        "n_examples": ...,
                        "major_zone": ...,
                        "is_dot": ...,
                        "dot_role": ...,
                    },
                    ...
                }
            }
        }
    }
    """
    cfg = config or PrototypeConfig()
    if len(emb_df) == 0:
        raise ValueError("emb_df is empty")

    required = [syllable_col, sample_index_col, stroke_index_col, emb_col]
    missing = [c for c in required if c not in emb_df.columns]
    if missing:
        raise ValueError(f"emb_df is missing required columns: {missing}")

    bank: Dict[str, object] = {
        "config": asdict(cfg),
        "embedding_dim": int(len(np.asarray(emb_df.iloc[0][emb_col], dtype=np.float32))),
        "syllables": {},
    }

    for syllable_label, syll_df in emb_df.groupby(syllable_col, sort=False):
        syll_df = syll_df.copy()

        # expected number of strokes
        if num_strokes_col in syll_df.columns:
            expected_num_strokes = int(
                _mode_value(
                    list(syll_df[num_strokes_col].values),
                    default=int(syll_df[stroke_index_col].max()) + 1,
                )
            )
        else:
            expected_num_strokes = int(syll_df[stroke_index_col].max()) + 1

        sample_count = int(syll_df[sample_index_col].nunique())

        # char length metadata for recall-safe gating
        first_row = syll_df.iloc[0]
        if "char_array" in syll_df.columns:
            char_len = len(
                _ensure_char_array(
                    first_row,
                    syllable_col=syllable_col,
                    char_array_col="char_array",
                )
            )
        else:
            char_len = len(str(syllable_label))

        syll_entry = {
            "syllable_label": str(syllable_label),
            "expected_num_strokes": expected_num_strokes,
            "char_len": int(char_len),
            "num_samples": sample_count,
            "stroke_prototypes": {},
        }

        for stroke_index, grp in syll_df.groupby(stroke_index_col, sort=True):
            X = _stack_rows(grp[emb_col].tolist())
            protos = _build_subprototypes(X, cfg=cfg)

            entry = {
                "stroke_index": int(stroke_index),
                "prototypes": [np.asarray(p, dtype=np.float32) for p in protos],
                "n_examples": int(len(grp)),
                "major_zone": _mode_value(grp[zone_col].tolist(), default=None) if zone_col in grp.columns else None,
                "is_dot": _safe_bool_mode(grp[is_dot_col].tolist(), default=False) if is_dot_col in grp.columns else False,
                "dot_role": _mode_value(grp[dot_role_col].tolist(), default="none") if dot_role_col in grp.columns else "none",
            }

            syll_entry["stroke_prototypes"][int(stroke_index)] = entry

        bank["syllables"][str(syllable_label)] = syll_entry

    return bank


# ============================================================
# Query preparation
# ============================================================

def group_query_samples(
    emb_df: pd.DataFrame,
    *,
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
) -> Dict[object, pd.DataFrame]:
    """Return {sample_index: sorted_stroke_df} for query/evaluation."""
    out: Dict[object, pd.DataFrame] = {}
    for sample_index, grp in emb_df.groupby(sample_index_col, sort=False):
        out[sample_index] = grp.sort_values(stroke_index_col).reset_index(drop=True)
    return out


def _query_row_map(
    query_stroke_df: pd.DataFrame,
    *,
    stroke_index_col: str = "stroke_index",
) -> Dict[int, pd.Series]:
    rowmap: Dict[int, pd.Series] = {}
    for _, row in query_stroke_df.iterrows():
        rowmap[int(row[stroke_index_col])] = row
    return rowmap


# ============================================================
# Scoring
# ============================================================

def _stroke_meta_penalty(
    query_row: pd.Series,
    proto_info: Dict[str, object],
    *,
    zone_col: str,
    is_dot_col: str,
    dot_role_col: str,
    cfg: PrototypeConfig,
) -> float:
    penalty = 0.0

    q_zone = query_row.get(zone_col, None)
    p_zone = proto_info.get("major_zone", None)
    if q_zone is not None and p_zone is not None and q_zone != p_zone:
        penalty += float(cfg.zone_mismatch_penalty)

    q_is_dot = bool(query_row.get(is_dot_col, False))
    p_is_dot = bool(proto_info.get("is_dot", False))
    if q_is_dot != p_is_dot:
        penalty += float(cfg.dot_mismatch_penalty)

    if q_is_dot and p_is_dot:
        q_dot_role = str(query_row.get(dot_role_col, "none"))
        p_dot_role = str(proto_info.get("dot_role", "none"))
        if q_dot_role != p_dot_role:
            penalty += float(cfg.dot_role_mismatch_penalty)

    return penalty


def _best_subprototype_similarity(query_emb: np.ndarray, proto_embs: Sequence[np.ndarray]) -> float:
    if len(proto_embs) == 0:
        return -1.0
    sims = [_cosine(query_emb, p) for p in proto_embs]
    return float(max(sims))


def score_query_against_syllable(
    query_stroke_df: pd.DataFrame,
    bank: Dict[str, object],
    syllable_label: str,
    *,
    emb_col: str = "embedding",
    stroke_index_col: str = "stroke_index",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    config: PrototypeConfig | None = None,
    return_details: bool = True,
) -> Dict[str, object]:
    """
    Score one query syllable (stroke dataframe for one sample) against one syllable prototype.
    """
    cfg = config or PrototypeConfig()

    syll_key = str(syllable_label)
    if "syllables" not in bank or syll_key not in bank["syllables"]:
        raise KeyError(f"Unknown syllable_label={syll_key!r} in prototype bank")

    proto_syll = bank["syllables"][syll_key]
    proto_strokes = proto_syll["stroke_prototypes"]
    expected_num_strokes = int(proto_syll.get("expected_num_strokes", len(proto_strokes)))

    query_rows = _query_row_map(query_stroke_df, stroke_index_col=stroke_index_col)
    query_indices = set(query_rows.keys())
    proto_indices = set(int(k) for k in proto_strokes.keys())

    if cfg.max_num_strokes_diff is not None:
        if abs(len(query_indices) - expected_num_strokes) > int(cfg.max_num_strokes_diff):
            return {
                "syllable_label": syll_key,
                "score": -999.0,
                "matched_similarity_mean": -1.0,
                "matched_similarity_sum": -1.0,
                "penalty_total": 999.0,
                "details": [],
            }

    all_indices = sorted(query_indices | proto_indices)

    matched_sims: List[float] = []
    penalty_total = 0.0
    details: List[Dict[str, object]] = []

    for sidx in all_indices:
        q_present = sidx in query_rows
        p_present = sidx in proto_strokes

        if q_present and p_present:
            qrow = query_rows[sidx]
            qemb = np.asarray(qrow[emb_col], dtype=np.float32)

            pinfo = proto_strokes[sidx]
            proto_embs = pinfo["prototypes"]
            sim = _best_subprototype_similarity(qemb, proto_embs)

            meta_pen = _stroke_meta_penalty(
                qrow,
                pinfo,
                zone_col=zone_col,
                is_dot_col=is_dot_col,
                dot_role_col=dot_role_col,
                cfg=cfg,
            )

            sim_adj = sim - meta_pen
            matched_sims.append(sim_adj)
            penalty_total += meta_pen

            if return_details:
                details.append({
                    "stroke_index": int(sidx),
                    "status": "matched",
                    "raw_similarity": float(sim),
                    "adjusted_similarity": float(sim_adj),
                    "penalty": float(meta_pen),
                    "query_zone": qrow.get(zone_col, None),
                    "proto_zone": pinfo.get("major_zone", None),
                    "query_is_dot": bool(qrow.get(is_dot_col, False)),
                    "proto_is_dot": bool(pinfo.get("is_dot", False)),
                    "query_dot_role": qrow.get(dot_role_col, None),
                    "proto_dot_role": pinfo.get("dot_role", None),
                    "num_subprototypes": len(proto_embs),
                })

        elif q_present and not p_present:
            penalty_total += float(cfg.extra_stroke_penalty)
            if return_details:
                details.append({
                    "stroke_index": int(sidx),
                    "status": "extra_query_stroke",
                    "raw_similarity": np.nan,
                    "adjusted_similarity": np.nan,
                    "penalty": float(cfg.extra_stroke_penalty),
                })

        elif p_present and not q_present:
            penalty_total += float(cfg.missing_stroke_penalty)
            if return_details:
                details.append({
                    "stroke_index": int(sidx),
                    "status": "missing_query_stroke",
                    "raw_similarity": np.nan,
                    "adjusted_similarity": np.nan,
                    "penalty": float(cfg.missing_stroke_penalty),
                })

    if len(matched_sims) == 0:
        base_score = -penalty_total
        matched_mean = -1.0
        matched_sum = 0.0
    else:
        matched_sum = float(np.sum(matched_sims))
        matched_mean = float(np.mean(matched_sims))
        base_score = matched_mean if cfg.average_similarity else matched_sum
        base_score -= penalty_total

    return {
        "syllable_label": syll_key,
        "score": float(base_score),
        "matched_similarity_mean": float(matched_mean),
        "matched_similarity_sum": float(matched_sum),
        "penalty_total": float(penalty_total),
        "details": details if return_details else None,
    }


def score_query_against_all_syllables(
    query_stroke_df: pd.DataFrame,
    bank: Dict[str, object],
    *,
    config: PrototypeConfig | None = None,
    emb_col: str = "embedding",
    stroke_index_col: str = "stroke_index",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    candidates: Optional[Sequence[str]] = None,
    return_details: bool = False,
) -> pd.DataFrame:
    """Score one query sample against all (or candidate) syllable prototypes."""
    cfg = config or PrototypeConfig()
    syllables = list(bank["syllables"].keys()) if candidates is None else [str(x) for x in candidates]

    rows = []
    for syllable_label in syllables:
        out = score_query_against_syllable(
            query_stroke_df,
            bank,
            syllable_label,
            emb_col=emb_col,
            stroke_index_col=stroke_index_col,
            zone_col=zone_col,
            is_dot_col=is_dot_col,
            dot_role_col=dot_role_col,
            config=cfg,
            return_details=return_details,
        )
        rows.append(out)

    score_df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return score_df


def predict_topk_syllable_prototype(
    query_stroke_df: pd.DataFrame,
    bank: Dict[str, object],
    *,
    k: int = 5,
    config: PrototypeConfig | None = None,
    emb_col: str = "embedding",
    stroke_index_col: str = "stroke_index",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    candidates: Optional[Sequence[str]] = None,
) -> List[Tuple[str, float]]:
    """Return top-k (syllable_label, score)."""
    score_df = score_query_against_all_syllables(
        query_stroke_df,
        bank,
        config=config,
        emb_col=emb_col,
        stroke_index_col=stroke_index_col,
        zone_col=zone_col,
        is_dot_col=is_dot_col,
        dot_role_col=dot_role_col,
        candidates=candidates,
        return_details=False,
    )
    top = score_df.head(k)
    return list(zip(top["syllable_label"].astype(str).tolist(), top["score"].astype(float).tolist()))


# ============================================================
# Batch prediction / evaluation (classifier-style baseline)
# ============================================================

def batch_predict_syllable_prototype(
    query_emb_df: pd.DataFrame,
    bank: Dict[str, object],
    *,
    sample_index_col: str = "sample_index",
    true_label_col: str = "syllable",
    emb_col: str = "embedding",
    stroke_index_col: str = "stroke_index",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    k: int = 5,
    config: PrototypeConfig | None = None,
    candidates_per_query: Optional[Dict[object, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Predict top-k syllables for each query sample.
    """
    cfg = config or PrototypeConfig()
    groups = group_query_samples(
        query_emb_df,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
    )

    rows = []
    for sample_index, qdf in groups.items():
        true_label = qdf.iloc[0][true_label_col] if true_label_col in qdf.columns else None
        candidates = None
        if candidates_per_query is not None and sample_index in candidates_per_query:
            candidates = candidates_per_query[sample_index]

        topk = predict_topk_syllable_prototype(
            qdf,
            bank,
            k=k,
            config=cfg,
            emb_col=emb_col,
            stroke_index_col=stroke_index_col,
            zone_col=zone_col,
            is_dot_col=is_dot_col,
            dot_role_col=dot_role_col,
            candidates=candidates,
        )

        pred_labels = [str(lab) for lab, _ in topk]
        pred_scores = [float(sc) for _, sc in topk]

        rows.append({
            "sample_index": sample_index,
            "true_label": true_label,
            "pred_top1": pred_labels[0] if len(pred_labels) else None,
            "pred_topk": pred_labels,
            "top1_score": pred_scores[0] if len(pred_scores) else np.nan,
            "topk_scores": pred_scores,
            "num_query_strokes": int(qdf[stroke_index_col].nunique()),
        })

    return pd.DataFrame(rows)


def evaluate_syllable_prototypes(
    query_emb_df: pd.DataFrame,
    bank: Dict[str, object],
    *,
    sample_index_col: str = "sample_index",
    true_label_col: str = "syllable",
    emb_col: str = "embedding",
    stroke_index_col: str = "stroke_index",
    zone_col: str = "stroke_major_zone",
    is_dot_col: str = "stroke_is_dot",
    dot_role_col: str = "stroke_dot_role",
    ks: Sequence[int] = (1, 5),
    config: PrototypeConfig | None = None,
    candidates_per_query: Optional[Dict[object, Sequence[str]]] = None,
) -> pd.DataFrame:
    """
    Evaluate syllable prototype classifier.

    Important
    ---------
    Use a train/test split externally. If you train the bank on the same samples
    used for query evaluation, results will be optimistic.
    """
    cfg = config or PrototypeConfig()
    max_k = int(max(ks))
    pred_df = batch_predict_syllable_prototype(
        query_emb_df,
        bank,
        sample_index_col=sample_index_col,
        true_label_col=true_label_col,
        emb_col=emb_col,
        stroke_index_col=stroke_index_col,
        zone_col=zone_col,
        is_dot_col=is_dot_col,
        dot_role_col=dot_role_col,
        k=max_k,
        config=cfg,
        candidates_per_query=candidates_per_query,
    )

    y_true = pred_df["true_label"].astype(str).values

    results = []
    seen_top1 = False
    for k in ks:
        topk = pred_df["pred_topk"].tolist()
        hit = np.array([
            (str(y_true[i]) in [str(x) for x in topk[i][:k]])
            for i in range(len(y_true))
        ], dtype=np.float32)
        results.append({"metric": f"top{k}_prototype_acc", "value": float(hit.mean())})
        if int(k) == 1:
            seen_top1 = True

    if not seen_top1:
        top1 = pred_df["pred_top1"].astype(str).values
        acc = float((top1 == y_true).mean()) if len(y_true) > 0 else float("nan")
        results.append({"metric": "top1_prototype_acc", "value": acc})

    # mean rank of correct label
    ranks = []
    for i in range(len(pred_df)):
        preds = [str(x) for x in pred_df.iloc[i]["pred_topk"]]
        target = str(pred_df.iloc[i]["true_label"])
        if target in preds:
            ranks.append(int(preds.index(target) + 1))
    if len(ranks) > 0:
        results.append({"metric": "mean_correct_rank_in_topk", "value": float(np.mean(ranks))})

    return pd.DataFrame(results)


# ============================================================
# Recall-safe syllable gating (recommended for Segmental HMM)
# ============================================================
def recall_syllable_candidates(
    query_stroke_df: pd.DataFrame,
    prototype_bank: Dict[str, object],
    *,
    max_segment_len: int = 3,
    shape_top_k: int = 25,          # Tier‑1 shape shortlist
    shape_rescue_k: int = 50,       # Tier‑2 rescue
    stroke_margin: int = 1,
    char_margin: int = 0,
    max_total_candidates: int = 120,
    prototype_config: PrototypeConfig | None = None,
) -> List[str]:
    """
    FINAL recall-safe syllable gating.

    Design:
      Tier‑1: strict shortlist (shape + structure)
      Tier‑2: bounded shape-only rescue
      Final: capped union

    Guarantees:
      - high recall
      - bounded candidate set
      - fast enough for Segmental HMM / Pi
    """

    # ------------------------------------------------------------
    # Compute prototype scores ONCE
    # ------------------------------------------------------------
    score_df = score_query_against_all_syllables(
        query_stroke_df,
        prototype_bank,
        config=prototype_config,
        return_details=False,
    ).copy()

    score_df["syllable_label"] = score_df["syllable_label"].astype(str)
    score_df = score_df.sort_values("score", ascending=False).reset_index(drop=True)

    if len(score_df) == 0:
        return []

    # ------------------------------------------------------------
    # Query statistics
    # ------------------------------------------------------------
    N = int(query_stroke_df["stroke_index"].nunique())
    min_chars = int(np.ceil(N / max(int(max_segment_len), 1)))
    max_chars = int(N + int(char_margin))

    # ------------------------------------------------------------
    # Tier‑1A: shape shortlist
    # ------------------------------------------------------------
    shape_candidates = score_df.head(int(shape_top_k))[
        "syllable_label"
    ].tolist()

    # ------------------------------------------------------------
    # Tier‑1B: structural shortlist (ranked & bounded)
    # ------------------------------------------------------------
    structural_rows = []

    for _, row in score_df.iterrows():
        syll = row["syllable_label"]
        info = prototype_bank["syllables"].get(syll, {})
        exp_n = info.get("expected_num_strokes", None)
        clen = info.get("char_len", None)

        if exp_n is None or clen is None:
            continue

        stroke_diff = abs(int(exp_n) - N)
        if stroke_diff > int(stroke_margin):
            continue

        if not (min_chars <= int(clen) <= max_chars):
            continue

        structural_rows.append({
            "syllable_label": syll,
            "score": float(row["score"]),
            "stroke_diff": stroke_diff,
            "char_diff": abs(int(clen) - min_chars),
        })

    if structural_rows:
        structural_df = pd.DataFrame(structural_rows)
        structural_df = structural_df.sort_values(
            ["stroke_diff", "char_diff", "score"],
            ascending=[True, True, False],
        )
        structural_candidates = structural_df.head(50)[
            "syllable_label"
        ].tolist()
    else:
        structural_candidates = []

    # ------------------------------------------------------------
    # Tier‑1 union
    # ------------------------------------------------------------
    tier1 = list(dict.fromkeys(shape_candidates + structural_candidates))

    # ------------------------------------------------------------
    # Tier‑2: shape-only rescue (recall safety)
    # ------------------------------------------------------------
    rescue_candidates = score_df.head(int(shape_rescue_k))[
        "syllable_label"
    ].tolist()

    final = list(dict.fromkeys(tier1 + rescue_candidates))

    # ------------------------------------------------------------
    # Final hard cap (rank by prototype score)
    # ------------------------------------------------------------
    if len(final) > int(max_total_candidates):
        final_df = score_df[score_df["syllable_label"].isin(final)]
        final = final_df.head(int(max_total_candidates))[
            "syllable_label"
        ].tolist()

    return final


def batch_recall_syllable_candidates(
    query_emb_df: pd.DataFrame,
    prototype_bank: Dict[str, object],
    *,
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    max_segment_len: int = 3,
    shape_top_k: int = 50,
    stroke_margin: int = 1,
    char_margin: int = 1,
    prototype_config: PrototypeConfig | None = None,
) -> Dict[object, List[str]]:
    """
    Build recall-safe candidate syllable sets for all query samples.
    """
    groups = group_query_samples(
        query_emb_df,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
    )

    out: Dict[object, List[str]] = {}
    for sample_index, qdf in groups.items():
        out[sample_index] = recall_syllable_candidates(
            qdf,
            prototype_bank,
            max_segment_len=max_segment_len,
            shape_top_k=shape_top_k,
            stroke_margin=stroke_margin,
            char_margin=char_margin,
            prototype_config=prototype_config,
        )
    return out


def evaluate_recall_candidates(
    query_emb_df: pd.DataFrame,
    prototype_bank: Dict[str, object],
    *,
    true_label_col: str = "syllable",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    max_segment_len: int = 3,
    shape_top_k: int = 50,
    stroke_margin: int = 1,
    char_margin: int = 1,
    prototype_config: PrototypeConfig | None = None,
) -> pd.DataFrame:
    """
    Evaluate recall-safe candidate generation.

    This is the correct metric for using prototype.py as a recall gate:
        Is the true syllable included in the candidate set?
    """
    shortlist = batch_recall_syllable_candidates(
        query_emb_df,
        prototype_bank,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
        max_segment_len=max_segment_len,
        shape_top_k=shape_top_k,
        stroke_margin=stroke_margin,
        char_margin=char_margin,
        prototype_config=prototype_config,
    )

    groups = group_query_samples(
        query_emb_df,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
    )

    hits = []
    sizes = []

    for sample_index, qdf in groups.items():
        true_label = str(qdf.iloc[0][true_label_col])
        cands = shortlist[sample_index]
        sizes.append(len(cands))
        hits.append(float(true_label in cands))

    return pd.DataFrame([
        {"metric": "candidate_recall", "value": float(np.mean(hits))},
        {"metric": "mean_candidate_set_size", "value": float(np.mean(sizes))},
        {"metric": "max_candidate_set_size", "value": float(np.max(sizes)) if len(sizes) else 0.0},
        {"metric": "min_candidate_set_size", "value": float(np.min(sizes)) if len(sizes) else 0.0},
    ])


# ============================================================
# Score -> cost conversion
# ============================================================

def scores_to_log_costs(
    score_df: pd.DataFrame,
    *,
    score_col: str = "score",
    label_col: str = "syllable_label",
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Convert prototype scores to normalized negative log costs.
    """
    if len(score_df) == 0:
        return pd.DataFrame(columns=[label_col, score_col, "prob", "cost"])

    scores = score_df[score_col].astype(float).values / max(float(temperature), eps)
    scores = scores - np.max(scores)
    ex = np.exp(scores)
    probs = ex / max(float(np.sum(ex)), eps)
    costs = -np.log(np.maximum(probs, eps))

    out = score_df.copy()
    out["prob"] = probs.astype(np.float32)
    out["cost"] = costs.astype(np.float32)
    return out


# ============================================================
# Inspection / export helpers
# ============================================================

def prototype_bank_to_dataframe(bank: Dict[str, object]) -> pd.DataFrame:
    """Flatten prototype bank into a debug dataframe."""
    rows = []

    for syllable_label, syll_info in bank.get("syllables", {}).items():
        expected_num_strokes = syll_info.get("expected_num_strokes", None)
        num_samples = syll_info.get("num_samples", None)
        char_len = syll_info.get("char_len", None)

        for stroke_index, stroke_info in syll_info.get("stroke_prototypes", {}).items():
            protos = stroke_info.get("prototypes", [])
            for proto_rank, proto_emb in enumerate(protos):
                rows.append({
                    "syllable_label": syllable_label,
                    "expected_num_strokes": expected_num_strokes,
                    "char_len": char_len,
                    "num_samples": num_samples,
                    "stroke_index": int(stroke_index),
                    "proto_rank": int(proto_rank),
                    "n_examples": int(stroke_info.get("n_examples", 0)),
                    "major_zone": stroke_info.get("major_zone", None),
                    "is_dot": bool(stroke_info.get("is_dot", False)),
                    "dot_role": stroke_info.get("dot_role", "none"),
                    "embedding_dim": int(len(proto_emb)),
                    "embedding": np.asarray(proto_emb, dtype=np.float32),
                })

    return pd.DataFrame(rows)


def describe_prototype_config(config: PrototypeConfig | None = None) -> Dict[str, object]:
    cfg = config or PrototypeConfig()
    return asdict(cfg)


__all__ = [
    "PrototypeConfig",
    "train_syllable_prototype_bank",
    "group_query_samples",
    "score_query_against_syllable",
    "score_query_against_all_syllables",
    "predict_topk_syllable_prototype",
    "batch_predict_syllable_prototype",
    "evaluate_syllable_prototypes",
    "recall_syllable_candidates",
    "batch_recall_syllable_candidates",
    "evaluate_recall_candidates",
    "scores_to_log_costs",
    "prototype_bank_to_dataframe",
    "describe_prototype_config",
]