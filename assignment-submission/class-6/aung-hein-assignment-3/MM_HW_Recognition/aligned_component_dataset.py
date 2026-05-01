# aligned_component_dataset.py
# ============================================================
# Build aligned component-level dataset using Segmental HMM
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Sequence
import numpy as np
import pandas as pd

from segmental_hmm import (
    build_token_lattice_from_emb_df,
    score_char_array_with_segmental_hmm,
)

from writing_units import infer_unicode_myanmar_role


# ============================================================
# Configuration
# ============================================================

@dataclass
class AlignedComponentDatasetConfig:
    segment_embedding_mode: str = "mean"   # mean | max | mean_max
    keep_zero_length_segments: bool = True


# ============================================================
# Helpers
# ============================================================

def infer_position_tag(n: int, i: int) -> str:
    if n <= 1:
        return "SINGLE"
    if i == 0:
        return "START"
    if i == n - 1:
        return "END"
    return "MID"


def aggregate_embeddings(X: np.ndarray, mode: str) -> np.ndarray:
    if len(X) == 0:
        return np.zeros((X.shape[1],), dtype=np.float32)

    if mode == "mean":
        return X.mean(axis=0)
    if mode == "max":
        return X.max(axis=0)
    if mode == "mean_max":
        return np.concatenate([X.mean(axis=0), X.max(axis=0)], axis=0)

    raise ValueError(f"Unknown segment_embedding_mode={mode}")


# ============================================================
# Main builder
# ============================================================

def build_aligned_component_dataset(
    emb_df: pd.DataFrame,
    seg_model: Dict[str, object],
    *,
    config: AlignedComponentDatasetConfig | None = None,
    emb_col: str = "embedding",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    char_array_col: str = "char_array",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build component-level dataset from Viterbi alignment.
    """
    cfg = config or AlignedComponentDatasetConfig()

    items = build_token_lattice_from_emb_df(
        emb_df,
        seg_model["tokenizer"],
        emb_col=emb_col,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
        char_array_col=char_array_col,
    )

    grouped = {
        k: v.sort_values(stroke_index_col).reset_index(drop=True)
        for k, v in emb_df.groupby(sample_index_col)
    }

    rows: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for sample_index, item in items.items():
        token_lattice = item["token_lattice"]
        target_array = item["char_array"]
        stroke_df = grouped[sample_index]

        score, segments = score_char_array_with_segmental_hmm(
            token_lattice,
            target_array,
            seg_model,
        )

        ok = np.isfinite(score) and len(segments) == len(target_array)

        summary.append({
            "sample_index": sample_index,
            "num_targets": len(target_array),
            "num_strokes": len(token_lattice),
            "alignment_score": score,
            "aligned_ok": ok,
        })

        if not ok:
            continue

        for i, (sym, (s, e)) in enumerate(zip(target_array, segments)):
            seg_len = e - s
            if seg_len == 0 and not cfg.keep_zero_length_segments:
                continue

            role = infer_unicode_myanmar_role(sym)
            pos = infer_position_tag(len(target_array), i)

            seg_df = stroke_df.iloc[s:e]
            if len(seg_df) > 0:
                X = np.vstack(seg_df[emb_col].values)
                emb = aggregate_embeddings(X, cfg.segment_embedding_mode)
            else:
                emb = np.zeros_like(stroke_df.iloc[0][emb_col])

            rows.append({
                "sample_index": sample_index,
                "target_symbol": sym,
                "target_pos": i,
                "role": role,
                "position_tag": pos,
                "segment_len_strokes": seg_len,
                "segment_start_stroke": s,
                "segment_end_stroke": e,
                "segment_embedding": emb,
            })

    return pd.DataFrame(rows), pd.DataFrame(summary)


def evaluate_alignment_coverage(summary_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"metric": "num_samples", "value": len(summary_df)},
        {"metric": "alignment_success_rate", "value": summary_df["aligned_ok"].mean()},
        {"metric": "num_failed_alignments", "value": (~summary_df["aligned_ok"]).sum()},
    ])


__all__ = [
    "AlignedComponentDatasetConfig",
    "build_aligned_component_dataset",
    "evaluate_alignment_coverage",
]