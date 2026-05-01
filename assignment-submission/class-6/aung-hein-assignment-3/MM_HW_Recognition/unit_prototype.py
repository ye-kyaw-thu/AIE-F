# unit_prototype.py
# ============================================================
# Baseline unit prototype training for Myanmar handwriting
#
# This module:
#   1. Builds a baseline-unit dataset from dbg_component_df
#   2. Merges adjacent HMM-aligned symbols into handwriting-friendly units
#   3. Trains prototype vectors for those baseline units
#   4. Optionally adds a soft role-grammar score on top of local prototype score
#
# Input schema expected from dbg_component_df:
#   - sample_index
#   - target_symbol
#   - target_pos
#   - role
#   - position_tag
#   - segment_len_strokes
#   - segment_start_stroke
#   - segment_end_stroke
#   - segment_embedding
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Sequence
from collections import Counter, defaultdict
import math

import numpy as np
import pandas as pd
import unicodedata as ud


# ============================================================
# Configuration
# ============================================================

@dataclass
class BaselineUnitRuleConfig:
    # ------------------------------------------------
    # Optional v2 safety rule
    # ------------------------------------------------
    # If you already merge ေ + BASE upstream in align_unit_array,
    # keep this False to avoid double-merging.
    merge_e_base: bool = False      # ေ + BASE -> ေBASE

    # ------------------------------------------------
    # Vowel bundle rules (generic, reusable)
    # ------------------------------------------------
    merge_e_aa: bool = True         # ေ + ာ -> ော
    merge_e_tall_aa: bool = True    # ေ + ါ -> ေါ

    # Optional later if needed:
    merge_i_u: bool = False         # ိ + ု -> ို-like bundle

    # ------------------------------------------------
    # Final / asat bundles
    # ------------------------------------------------
    merge_base_asat: bool = True    # generic BASE + ်
    merge_tall_aa_asat: bool = True # ာ + ်
    merge_nga_asat: bool = True     # င + ်

    # ------------------------------------------------
    # Stacked forms
    # ------------------------------------------------
    merge_virama_next_base: bool = True

    # ------------------------------------------------
    # Medial bundles
    # ------------------------------------------------
    merge_medial_ya_ha: bool = True    # ျ + ှ
    merge_medial_pairs: bool = True    # ျ+ွ, ြ+ွ, ြ+ှ
    merge_medial_triplets: bool = False

    # ------------------------------------------------
    # Special symbols
    # ------------------------------------------------
    preserve_special_symbols: bool = True

    # ------------------------------------------------
    # Auto-discovery fallback
    # ------------------------------------------------
    use_auto_pair_merges: bool = True
    min_auto_pair_count: int = 20
    min_auto_zero_rate: float = 0.25

    # ------------------------------------------------
    # Embedding aggregation
    # ------------------------------------------------
    embedding_agg_mode: str = "mean"   # mean | max | mean_max


@dataclass
class BaselineUnitSeqConfig:
    """
    Deterministic baseline-unit conversion from canonical char_array.
    This should match the same logic as your baseline unitization rules.
    """
    merge_e_aa: bool = True         # ေ + ာ -> ော
    merge_e_tall_aa: bool = True    # ေ + ါ -> ေါ
    merge_i_u: bool = False         # optional later

    merge_base_asat: bool = True    # BASE + ် -> merged final
    merge_tall_aa_asat: bool = True # ာ + ်
    merge_nga_asat: bool = True     # င + ်
    merge_virama_next_base: bool = True

    merge_medial_ya_ha: bool = True
    merge_medial_pairs: bool = True
    merge_medial_triplets: bool = False




@dataclass
class UnitPrototypeConfig:
    prototype_mode: str = "mean"      # mean | medoid
    normalize: bool = True
    max_prototypes_per_unit: int = 1
    min_examples_for_split: int = 10
    average_similarity: bool = True
    missing_unit_penalty: float = 0.40
    eps: float = 1e-8


@dataclass
class UnitGrammarConfig:
    """
    Soft grammar over unit roles.
    This is optional and can be turned on/off independently.
    """
    enable_grammar: bool = True
    lambda_grammar: float = 0.20
    smoothing: float = 1.0


# ============================================================
# Helpers
# ============================================================

SPECIAL_SINGLE_UNITS = {"၏", "။", "၊", "၌", "၍"}

MY_ASAT = "်"
MY_VIRAMA = "္"
MY_E_VOWEL = "ေ"
MY_AA = "ာ"
MY_TALL_AA = "ါ"

MY_MEDIAL_YA = "ျ"
MY_MEDIAL_RA = "ြ"
MY_MEDIAL_WA = "ွ"
MY_MEDIAL_HA = "ှ"


MY_E = "ေ"

MY_MARKS = {
    "ေ", "ာ", "ါ", "ိ", "ီ", "ု", "ူ", "ဲ", "ံ", "့", "း",
    "ျ", "ြ", "ွ", "ှ", "်", "္",
}

def is_base_like(ch: str) -> bool:
    """
    Very simple heuristic:
    any Myanmar char not in the common mark set is treated as base-like.
    """
    if ch in MY_MARKS:
        return False

    try:
        name = ud.name(ch, "")
    except TypeError:
        return False

    return "MYANMAR" in name

def _l2_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return (v / n).astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.maximum(norms, 1e-8)
    return Xn @ Xn.T


def _prototype_from_rows(
    X: np.ndarray,
    *,
    mode: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    if len(X) == 1:
        p = X[0]
    elif mode == "mean":
        p = X.mean(axis=0)
    elif mode == "medoid":
        S = _pairwise_cosine(X)
        idx = int(np.argmax(S.sum(axis=1)))
        p = X[idx]
    else:
        raise ValueError(f"Unknown prototype_mode={mode}")

    p = p.astype(np.float32)
    return _l2_normalize(p) if normalize else p


def aggregate_segment_embedding(X: np.ndarray, mode: str = "mean") -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return np.zeros((X.shape[1],), dtype=np.float32)

    if mode == "mean":
        return X.mean(axis=0)

    if mode == "max":
        return X.max(axis=0)

    if mode == "mean_max":
        return np.concatenate(
            [X.mean(axis=0), X.max(axis=0)],
            axis=0,
        ).astype(np.float32)

    raise ValueError(f"Unknown segment aggregation mode: {mode}")


# ============================================================
# 1) syllable -> baseline-unit sequence
# ============================================================

def char_array_to_units(
    chars: Iterable[str],
    cfg: Optional[BaselineUnitSeqConfig] = None,
) -> Tuple[str, ...]:
    """
    Convert canonical Unicode char_array -> baseline units.

    This should mirror the rule choices that worked best in your experiments:
      - keep BASE separate
      - merge generic vowel bundles like ော / ေါ
      - merge final-asat bundles like င် / န် / က် / ...
      - merge stacked forms like ္က / ္ဓ
      - merge common medial bundles like ျှ / ျွ / ြွ / ြှ
    """
    cfg = cfg or BaselineUnitSeqConfig()
    chars = [str(x) for x in chars]

    units: List[str] = []
    i = 0
    n = len(chars)

    while i < n:
        # --------------------------------------------
        # Longest-match rules first
        # --------------------------------------------

        # ေ + ာ -> ော
        if (
            cfg.merge_e_aa
            and i + 1 < n
            and chars[i] == MY_E
            and chars[i + 1] == MY_AA
        ):
            units.append("ော")
            i += 2
            continue

        # ေ + ါ -> ေါ
        if (
            cfg.merge_e_tall_aa
            and i + 1 < n
            and chars[i] == MY_E
            and chars[i + 1] == MY_TALL_AA
        ):
            units.append("ေါ")
            i += 2
            continue

        # Optional: ိ + ု
        if (
            cfg.merge_i_u
            and i + 1 < n
            and chars[i] == "ိ"
            and chars[i + 1] == "ု"
        ):
            units.append("ို")
            i += 2
            continue

        # ျ + ွ + ှ   or   ြ + ွ + ှ
        if (
            cfg.merge_medial_triplets
            and i + 2 < n
            and chars[i] in {MY_MEDIAL_YA, MY_MEDIAL_RA}
            and chars[i + 1] == MY_MEDIAL_WA
            and chars[i + 2] == MY_MEDIAL_HA
        ):
            units.append(chars[i] + chars[i + 1] + chars[i + 2])
            i += 3
            continue

        # --------------------------------------------
        # Two-symbol rules
        # --------------------------------------------
        if i + 1 < n:
            a = chars[i]
            b = chars[i + 1]

            # generic BASE + ်
            if cfg.merge_base_asat and is_base_like(a) and b == MY_ASAT:
                units.append(a + b)
                i += 2
                continue

            # င + ် -> င်
            if cfg.merge_nga_asat and a == "င" and b == MY_ASAT:
                units.append("င်")
                i += 2
                continue

            # ာ + ်
            if cfg.merge_tall_aa_asat and a == MY_AA and b == MY_ASAT:
                units.append(a + b)
                i += 2
                continue

            # Virama + next BASE
            if cfg.merge_virama_next_base and a == MY_VIRAMA and is_base_like(b):
                units.append(a + b)
                i += 2
                continue

            # ျ + ှ
            if cfg.merge_medial_ya_ha and a == MY_MEDIAL_YA and b == MY_MEDIAL_HA:
                units.append(a + b)
                i += 2
                continue

            # other medial pairs
            if cfg.merge_medial_pairs and (a, b) in {
                (MY_MEDIAL_YA, MY_MEDIAL_WA),
                (MY_MEDIAL_RA, MY_MEDIAL_WA),
                (MY_MEDIAL_RA, MY_MEDIAL_HA),
            }:
                units.append(a + b)
                i += 2
                continue

        # default: keep single char
        units.append(chars[i])
        i += 1

    return tuple(units)


def build_syllable_to_units(
    syllable_df: pd.DataFrame,
    cfg: Optional[BaselineUnitSeqConfig] = None,
) -> Dict[str, Tuple[str, ...]]:
    """
    Build:
        syllable -> baseline unit sequence

    from canonical Unicode char_array.
    """
    required = ["syllable", "char_array"]
    missing = [c for c in required if c not in syllable_df.columns]
    if missing:
        raise ValueError(f"syllable_df missing required columns: {missing}")

    out: Dict[str, Tuple[str, ...]] = {}
    for _, row in syllable_df.iterrows():
        syll = str(row["syllable"])
        out[syll] = char_array_to_units(row["char_array"], cfg=cfg)
    return out



# ============================================================
# Auto pair discovery from dbg_component_df
# ============================================================

def discover_auto_pair_merges(
    component_df: pd.DataFrame,
    min_pair_count: int = 20,
    min_zero_rate: float = 0.25,
) -> Dict[Tuple[str, str], int]:
    """
    Discover adjacent pairs worth merging.

    Returns:
        {(a,b): count, ...}
    """
    zero_rate = (
        component_df
        .groupby("target_symbol")["segment_len_strokes"]
        .apply(lambda x: float((np.asarray(x) == 0).mean()))
        .to_dict()
    )

    pair_counter = Counter()

    for _, grp in component_df.groupby("sample_index"):
        grp = grp.sort_values("target_pos").reset_index(drop=True)

        for i in range(len(grp) - 1):
            a = grp.iloc[i]
            b = grp.iloc[i + 1]

            if (
                int(b["segment_len_strokes"]) == 0 or
                (
                    int(a["segment_start_stroke"]) == int(b["segment_start_stroke"])
                    and int(a["segment_end_stroke"]) == int(b["segment_end_stroke"])
                )
            ):
                pair_counter[(str(a["target_symbol"]), str(b["target_symbol"]))] += 1

    out = {}
    for pair, cnt in pair_counter.items():
        b = pair[1]
        if cnt >= min_pair_count and zero_rate.get(b, 0.0) >= min_zero_rate:
            out[pair] = cnt

    return out


# ============================================================
# Baseline-unit dataset builder
# ============================================================

def _merge_label(symbols: List[str]) -> str:
    return "".join(symbols)


def _merge_role(symbols: List[str], roles: List[str]) -> str:
    """
    Coarse merged role label.
    """
    if len(symbols) == 1:
        return roles[0]

    if symbols[0] == MY_VIRAMA:
        return "STACKED_BASE"

    if symbols[-1] == MY_ASAT:
        return "FINAL_ASAT"

    # v2 safety case: ေBASE
    if len(symbols) == 2 and symbols[0] == MY_E_VOWEL and roles[1] == "BASE":
        return "E_BASE"

    # generic vowel bundles like ော / ေါ / ို
    if (
        symbols[0] == MY_E_VOWEL
        or MY_AA in symbols
        or MY_TALL_AA in symbols
    ):
        return "VOWEL_BUNDLE"

    if all(r == "MEDIAL" for r in roles):
        return "MEDIAL_BUNDLE"

    return "COMPOSITE"


def _merge_rows(rows: List[pd.Series], cfg: BaselineUnitRuleConfig) -> Dict[str, object]:
    """
    Merge one or more component rows into a baseline unit row.
    """
    symbols = [str(r["target_symbol"]) for r in rows]
    roles = [str(r["role"]) for r in rows]
    pos = [int(r["target_pos"]) for r in rows]

    # use only visible segment embeddings for aggregation
    visible_rows = [r for r in rows if int(r["segment_len_strokes"]) > 0]
    if len(visible_rows) > 0:
        X = np.vstack([np.asarray(r["segment_embedding"], dtype=np.float32) for r in visible_rows])
        emb = aggregate_segment_embedding(X, cfg.embedding_agg_mode)

        start = min(int(r["segment_start_stroke"]) for r in visible_rows)
        end = max(int(r["segment_end_stroke"]) for r in visible_rows)
        seg_len = int(sum(int(r["segment_len_strokes"]) for r in visible_rows))
    else:
        emb_dim = len(np.asarray(rows[0]["segment_embedding"], dtype=np.float32))
        if cfg.embedding_agg_mode == "mean_max":
            emb = np.zeros((2 * emb_dim,), dtype=np.float32)
        else:
            emb = np.zeros((emb_dim,), dtype=np.float32)
        start = int(rows[0]["segment_start_stroke"])
        end = int(rows[0]["segment_end_stroke"])
        seg_len = 0

    return {
        "sample_index": rows[0]["sample_index"],
        "baseline_unit": _merge_label(symbols),
        "baseline_role": _merge_role(symbols, roles),
        "source_symbols": symbols,
        "source_roles": roles,
        "source_positions": pos,
        "target_pos": pos[0],
        "segment_start_stroke": start,
        "segment_end_stroke": end,
        "segment_len_strokes": seg_len,
        "segment_embedding": emb,
    }


def build_baseline_unit_dataset(
    component_df: pd.DataFrame,
    config: BaselineUnitRuleConfig | None = None,
) -> pd.DataFrame:
    """
    Convert dbg_component_df into a baseline-unit dataset.

    Rules are applied left-to-right per sample, with longest-match priority.
    """
    cfg = config or BaselineUnitRuleConfig()

    auto_pairs = {}
    if cfg.use_auto_pair_merges:
        auto_pairs = discover_auto_pair_merges(
            component_df,
            min_pair_count=cfg.min_auto_pair_count,
            min_zero_rate=cfg.min_auto_zero_rate,
        )

    out_rows = []

    for sample_index, grp in component_df.groupby("sample_index"):
        grp = grp.sort_values("target_pos").reset_index(drop=True)

        i = 0
        n = len(grp)

        while i < n:
            row = grp.iloc[i]
            sym = str(row["target_symbol"])
            role = str(row["role"])

            # ------------------------------------------------
            # Special single units
            # ------------------------------------------------
            if cfg.preserve_special_symbols and sym in SPECIAL_SINGLE_UNITS:
                out_rows.append(_merge_rows([row], cfg))
                i += 1
                continue

            # ------------------------------------------------
            # Longest-match rules first
            # ------------------------------------------------

            # Optional v2 safety: ေ + BASE -> ေBASE
            if (
                cfg.merge_e_base
                and i + 1 < n
                and str(grp.iloc[i]["target_symbol"]) == MY_E_VOWEL
                and str(grp.iloc[i + 1]["role"]) == "BASE"
            ):
                out_rows.append(_merge_rows([grp.iloc[i], grp.iloc[i + 1]], cfg))
                i += 2
                continue

            # ေ + ာ -> ော
            if (
                cfg.merge_e_aa
                and i + 1 < n
                and str(grp.iloc[i]["target_symbol"]) == MY_E_VOWEL
                and str(grp.iloc[i + 1]["target_symbol"]) == MY_AA
            ):
                out_rows.append(_merge_rows([grp.iloc[i], grp.iloc[i + 1]], cfg))
                i += 2
                continue

            # ေ + ါ -> ေါ
            if (
                cfg.merge_e_tall_aa
                and i + 1 < n
                and str(grp.iloc[i]["target_symbol"]) == MY_E_VOWEL
                and str(grp.iloc[i + 1]["target_symbol"]) == MY_TALL_AA
            ):
                out_rows.append(_merge_rows([grp.iloc[i], grp.iloc[i + 1]], cfg))
                i += 2
                continue

            # Optional: ိ + ု -> vowel bundle
            if (
                cfg.merge_i_u
                and i + 1 < n
                and str(grp.iloc[i]["target_symbol"]) == "ိ"
                and str(grp.iloc[i + 1]["target_symbol"]) == "ု"
            ):
                out_rows.append(_merge_rows([grp.iloc[i], grp.iloc[i + 1]], cfg))
                i += 2
                continue

            # ျ + ွ + ှ   or   ြ + ွ + ှ
            if (
                cfg.merge_medial_triplets
                and i + 2 < n
            ):
                s0 = str(grp.iloc[i]["target_symbol"])
                s1 = str(grp.iloc[i + 1]["target_symbol"])
                s2 = str(grp.iloc[i + 2]["target_symbol"])
                if (s0 in {MY_MEDIAL_YA, MY_MEDIAL_RA}) and s1 == MY_MEDIAL_WA and s2 == MY_MEDIAL_HA:
                    out_rows.append(_merge_rows([grp.iloc[i], grp.iloc[i + 1], grp.iloc[i + 2]], cfg))
                    i += 3
                    continue

            # ------------------------------------------------
            # Two-symbol rules
            # ------------------------------------------------

            if i + 1 < n:
                next_row = grp.iloc[i + 1]
                next_sym = str(next_row["target_symbol"])

                # generic BASE + ်
                if cfg.merge_base_asat and role == "BASE" and next_sym == MY_ASAT:
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

                # င + ်
                if cfg.merge_nga_asat and sym == "င" and next_sym == MY_ASAT:
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

                # ာ + ်
                if cfg.merge_tall_aa_asat and sym == MY_AA and next_sym == MY_ASAT:
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

                # Virama + next BASE
                if cfg.merge_virama_next_base and sym == MY_VIRAMA and str(next_row["role"]) == "BASE":
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

                # ျ + ှ
                if cfg.merge_medial_ya_ha and sym == MY_MEDIAL_YA and next_sym == MY_MEDIAL_HA:
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

                # Other medial pairs
                if cfg.merge_medial_pairs:
                    medial_pairs = {
                        (MY_MEDIAL_YA, MY_MEDIAL_WA),
                        (MY_MEDIAL_RA, MY_MEDIAL_WA),
                        (MY_MEDIAL_RA, MY_MEDIAL_HA),
                    }
                    if (sym, next_sym) in medial_pairs:
                        out_rows.append(_merge_rows([row, next_row], cfg))
                        i += 2
                        continue

                # Auto-discovered merge pairs
                if (sym, next_sym) in auto_pairs:
                    out_rows.append(_merge_rows([row, next_row], cfg))
                    i += 2
                    continue

            # ------------------------------------------------
            # default: keep single symbol
            # ------------------------------------------------
            out_rows.append(_merge_rows([row], cfg))
            i += 1

    return pd.DataFrame(out_rows)


# ============================================================
# Prototype training
# ============================================================

def train_unit_prototype_bank(
    unit_df: pd.DataFrame,
    *,
    unit_col: str = "baseline_unit",
    emb_col: str = "segment_embedding",
    config: UnitPrototypeConfig | None = None,
) -> Dict[str, object]:
    """
    Train prototype bank from baseline-unit dataset.

    Supports up to max_prototypes_per_unit prototypes for frequent units.
    """
    cfg = config or UnitPrototypeConfig()

    if len(unit_df) == 0:
        raise ValueError("unit_df is empty")

    if unit_col not in unit_df.columns:
        raise KeyError(f"unit_col '{unit_col}' not found in unit_df")

    if emb_col not in unit_df.columns:
        raise KeyError(f"emb_col '{emb_col}' not found in unit_df")

    bank: Dict[str, object] = {
        "config": asdict(cfg),
        "embedding_dim": int(len(unit_df.iloc[0][emb_col])),
        "units": {},
    }

    for unit, grp in unit_df.groupby(unit_col, sort=False):
        X = np.vstack(grp[emb_col].values).astype(np.float32)
        n = len(X)

        # normalize for stable cosine-based splitting
        if cfg.normalize:
            Xn = np.vstack([_l2_normalize(x, cfg.eps) for x in X])
        else:
            Xn = X.copy()

        # --------------------------------------------------
        # Single prototype for rare units
        # --------------------------------------------------
        if (
            cfg.max_prototypes_per_unit <= 1
            or n < cfg.min_examples_for_split
        ):
            proto = _prototype_from_rows(
                X,
                mode=cfg.prototype_mode,
                normalize=cfg.normalize,
            )
            protos = [proto]

        # --------------------------------------------------
        # Multi-prototype for frequent units
        # --------------------------------------------------
        else:
            k = min(cfg.max_prototypes_per_unit, n)

            # Greedy farthest-point initialization
            centers = [0]
            sims = _pairwise_cosine(Xn)

            while len(centers) < k:
                best_i = None
                best_dist = -1.0

                for i in range(n):
                    if i in centers:
                        continue
                    d = min(1.0 - sims[i, c] for c in centers)
                    if d > best_dist:
                        best_dist = d
                        best_i = i

                if best_i is None:
                    break
                centers.append(best_i)

            # assign each example to nearest center
            assignments = []
            for i in range(n):
                best_c = max(centers, key=lambda c: sims[i, c])
                assignments.append(best_c)

            # build one prototype per cluster
            protos = []
            for c in centers:
                idx = [i for i, a in enumerate(assignments) if a == c]
                if len(idx) == 0:
                    continue
                proto = _prototype_from_rows(
                    X[idx],
                    mode=cfg.prototype_mode,
                    normalize=cfg.normalize,
                )
                protos.append(proto)

            if len(protos) == 0:
                proto = _prototype_from_rows(
                    X,
                    mode=cfg.prototype_mode,
                    normalize=cfg.normalize,
                )
                protos = [proto]

        bank["units"][str(unit)] = {
            "unit": str(unit),
            "prototypes": protos,
            "n_examples": int(n),
        }

    return bank


# ============================================================
# Local scoring
# ============================================================

def _best_proto_similarity(emb: np.ndarray, protos: List[np.ndarray]) -> float:
    if len(protos) == 0:
        return 0.0
    return max(_cosine(emb, p) for p in protos)


def score_unit_sequence(
    unit_embeddings: List[np.ndarray],
    unit_labels: List[str],
    unit_bank: Dict[str, object],
    *,
    config: UnitPrototypeConfig | None = None,
) -> float:
    """
    Score a sequence of baseline units using prototype similarities.
    This is the original local scorer (kept unchanged).
    """
    cfg = config or UnitPrototypeConfig()

    sims = []
    penalty = 0.0

    for emb, unit in zip(unit_embeddings, unit_labels):
        info = unit_bank["units"].get(str(unit), None)
        if info is None:
            penalty += cfg.missing_unit_penalty
            continue

        s = _best_proto_similarity(np.asarray(emb, dtype=np.float32), info["prototypes"])
        sims.append(s)

    if len(sims) == 0:
        return -penalty

    base = np.mean(sims) if cfg.average_similarity else float(np.sum(sims))
    return float(base - penalty)


# ============================================================
# Soft grammar (order) scoring
# ============================================================

def build_unit_to_role_map(
    unit_df: pd.DataFrame,
    *,
    unit_col: str = "baseline_unit",
    role_col: str = "baseline_role",
) -> Dict[str, str]:
    """
    baseline_unit -> most common baseline_role
    """
    tmp = (
        unit_df.groupby([unit_col, role_col])
        .size()
        .reset_index(name="count")
        .sort_values([unit_col, "count"], ascending=[True, False])
    )

    return (
        tmp.drop_duplicates(unit_col)
        .set_index(unit_col)[role_col]
        .to_dict()
    )


def build_role_bigram_logprob(
    train_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    syllable_to_units: Dict[str, Tuple[str, ...]],
    unit_to_role: Dict[str, str],
    smoothing: float = 1.0,
) -> Dict[Tuple[str, str], float]:
    """
    Estimate log P(role_t | role_{t-1}) from training syllables.
    """
    bigram_counts = Counter()
    unigram_counts = Counter()
    role_vocab = set()

    for syll in train_df[syllable_col].astype(str).tolist():
        unit_seq = syllable_to_units.get(syll, tuple())
        roles = [unit_to_role.get(u, "OTHER") for u in unit_seq]

        for r in roles:
            role_vocab.add(r)

        for i in range(len(roles) - 1):
            r1, r2 = roles[i], roles[i + 1]
            bigram_counts[(r1, r2)] += 1
            unigram_counts[r1] += 1

    role_vocab = sorted(role_vocab)
    V = max(1, len(role_vocab))

    logprob = {}
    for r1 in role_vocab:
        denom = unigram_counts[r1] + smoothing * V
        for r2 in role_vocab:
            num = bigram_counts[(r1, r2)] + smoothing
            logprob[(r1, r2)] = math.log(num / denom)

    return logprob


def score_unit_grammar(
    unit_seq: Sequence[str],
    *,
    unit_to_role: Dict[str, str],
    role_bigram_logprob: Dict[Tuple[str, str], float],
) -> float:
    """
    Sum role-bigram log-probabilities for the unit sequence.
    Higher (less negative) is better.
    """
    roles = [unit_to_role.get(u, "OTHER") for u in unit_seq]

    if len(roles) <= 1:
        return 0.0

    score = 0.0
    for i in range(len(roles) - 1):
        score += role_bigram_logprob.get((roles[i], roles[i + 1]), -10.0)
    return score


def score_unit_sequence_with_grammar(
    unit_embeddings: List[np.ndarray],
    unit_labels: List[str],
    unit_bank: Dict[str, object],
    *,
    proto_config: UnitPrototypeConfig | None = None,
    grammar_cfg: UnitGrammarConfig | None = None,
    unit_to_role: Optional[Dict[str, str]] = None,
    role_bigram_logprob: Optional[Dict[Tuple[str, str], float]] = None,
) -> float:
    """
    Combined unit score:
      local prototype score + lambda * soft grammar score

    If grammar is disabled or missing, this falls back to the local score.
    """
    proto_config = proto_config or UnitPrototypeConfig()
    grammar_cfg = grammar_cfg or UnitGrammarConfig(enable_grammar=False)

    proto_score = score_unit_sequence(
        unit_embeddings=unit_embeddings,
        unit_labels=unit_labels,
        unit_bank=unit_bank,
        config=proto_config,
    )

    if not grammar_cfg.enable_grammar:
        return proto_score

    if unit_to_role is None or role_bigram_logprob is None:
        return proto_score

    grammar_score = score_unit_grammar(
        unit_seq=unit_labels,
        unit_to_role=unit_to_role,
        role_bigram_logprob=role_bigram_logprob,
    )

    return proto_score + grammar_cfg.lambda_grammar * grammar_score


# ============================================================
# Inspection helpers
# ============================================================

def summarize_baseline_units(unit_df: pd.DataFrame, unit_col: str = "baseline_unit") -> pd.DataFrame:
    cnt = unit_df[unit_col].astype(str).value_counts()
    return cnt.rename_axis(unit_col).reset_index(name="count")


__all__ = [
    # configs
    "BaselineUnitRuleConfig",
    "UnitPrototypeConfig",
    "UnitGrammarConfig",

    # data building
    "discover_auto_pair_merges",
    "build_baseline_unit_dataset",
    "summarize_baseline_units",

    # prototype bank
    "train_unit_prototype_bank",

    # local scoring
    "score_unit_sequence",

    # soft grammar
    "build_unit_to_role_map",
    "build_role_bigram_logprob",
    "score_unit_grammar",
    "score_unit_sequence_with_grammar",
]