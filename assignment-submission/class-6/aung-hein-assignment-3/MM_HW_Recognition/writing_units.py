# writing_units.py
# ============================================================
# Myanmar handwriting-friendly unit construction using Unicode
# Canonical units + v2 alignment-only helpers
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Sequence, Optional
import unicodedata as ud
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class WritingUnitConfig:
    """
    Canonical (Unicode-order) unitization config.
    """
    merge_base_asat: bool = True
    merge_virama_next_base: bool = True
    keep_unit_meta: bool = True
    annotate_role_in_unit_string: bool = False


@dataclass
class AlignUnitConfig:
    """
    v2 alignment-order unitization config.
    ONLY for alignment / plotting / HMM training in v2.
    """
    merge_e_base: bool = True
    merge_base_asat: bool = True
    merge_virama_next_base: bool = True
    keep_unit_meta: bool = True


# ============================================================
# Manual symbol sets (deterministic alignment logic)
# ============================================================

MY_E = "ေ"
MY_AA = "ာ"
MY_TALL_AA = "ါ"
MY_ASAT = "်"
MY_VIRAMA = "္"

MY_VOWEL_SIGNS = {
    "ေ", "ာ", "ါ", "ိ", "ီ", "ု", "ူ", "ဲ", "ံ",
}

MY_MEDIALS = {
    "ျ", "ြ", "ွ", "ှ",
}

MY_TONES = {
    "့", "း",
}

MY_MARKS = (
    MY_VOWEL_SIGNS
    | MY_MEDIALS
    | MY_TONES
    | {MY_ASAT, MY_VIRAMA}
)


def is_base_like(ch: str) -> bool:
    """
    Deterministic alignment-only notion of a base-like symbol.
    """
    return ch not in MY_MARKS


# ============================================================
# Unicode-based role inference (semantic, not ordering)
# ============================================================

def infer_unicode_myanmar_role(ch: str) -> str:
    """
    Infer a coarse Myanmar role from Unicode metadata.

    SAFE VERSION:
    - Handles single characters
    - Handles multi-character unit strings by inspecting components
    """
    # If this is a unit (multi-char), infer from its components
    if isinstance(ch, str) and len(ch) > 1:
        roles = [infer_unicode_myanmar_role(c) for c in ch]
        for r in roles:
            if r != "BASE":
                return r
        return "BASE"

    try:
        name = ud.name(ch, "")
        cat = ud.category(ch)
    except TypeError:
        return "BASE"

    if "MYANMAR" not in name:
        return "OTHER"

    if "MEDIAL" in name:
        return "MEDIAL"
    if "VOWEL SIGN" in name:
        return "VOWEL"
    if "ASAT" in name:
        return "ASAT"
    if "VIRAMA" in name:
        return "VIRAMA"
    if "DOT BELOW" in name or "VISARGA" in name:
        return "TONE"
    if "ANUSVARA" in name:
        return "VOWEL"
    if cat.startswith("M"):
        return "OTHER_MARK"

    return "BASE"


def infer_unit_role_from_chars(chars: Sequence[str]) -> str:
    """
    Infer a semantic role for a merged unit.
    """
    if len(chars) == 1:
        return infer_unicode_myanmar_role(chars[0])

    roles = [infer_unicode_myanmar_role(c) for c in chars]

    if roles == ["BASE", "ASAT"]:
        return "FINAL_ASAT"

    if roles == ["VIRAMA", "BASE"]:
        return "STACKED_BASE"

    # v2 alignment special case
    if len(chars) == 2 and chars[0] == MY_E and is_base_like(chars[1]):
        return "E_BASE"

    if len(set(roles)) == 1:
        return roles[0]

    return "COMPOSITE"


# ============================================================
# Canonical (Unicode-order) unit conversion
# ============================================================

def char_array_to_unit_info(
    char_array: Sequence[str],
    config: WritingUnitConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Convert canonical Unicode-order char_array to units.
    """
    cfg = config or WritingUnitConfig()
    chars = [str(c) for c in char_array]
    units: List[Dict[str, Any]] = []

    i = 0
    while i < len(chars):
        ch = chars[i]
        role = infer_unicode_myanmar_role(ch)

        # VIRAMA + BASE
        if (
            cfg.merge_virama_next_base
            and role == "VIRAMA"
            and i + 1 < len(chars)
            and infer_unicode_myanmar_role(chars[i + 1]) == "BASE"
        ):
            src = [chars[i], chars[i + 1]]
            units.append({
                "unit_text": "".join(src),
                "unit_role": infer_unit_role_from_chars(src),
                "source_chars": src,
                "source_indices": [i, i + 1],
            })
            i += 2
            continue

        # BASE + ASAT
        if (
            cfg.merge_base_asat
            and role == "BASE"
            and i + 1 < len(chars)
            and infer_unicode_myanmar_role(chars[i + 1]) == "ASAT"
        ):
            src = [chars[i], chars[i + 1]]
            units.append({
                "unit_text": "".join(src),
                "unit_role": infer_unit_role_from_chars(src),
                "source_chars": src,
                "source_indices": [i, i + 1],
            })
            i += 2
            continue

        # default single-char unit
        units.append({
            "unit_text": ch,
            "unit_role": role,
            "source_chars": [ch],
            "source_indices": [i],
        })
        i += 1

    return units


def char_array_to_unit_array(
    char_array: Sequence[str],
    config: WritingUnitConfig | None = None,
) -> List[str]:
    cfg = config or WritingUnitConfig()
    info = char_array_to_unit_info(char_array, cfg)
    if cfg.annotate_role_in_unit_string:
        return [f"{u['unit_text']}|{u['unit_role']}" for u in info]
    return [u["unit_text"] for u in info]


# ============================================================
# v2 alignment-order helpers
# ============================================================

def canonical_to_align_char_array(chars: Sequence[str]) -> List[str]:
    """
    Canonical Unicode order -> handwriting-friendly alignment order.

    Rule:
      BASE + ေ (+ optional ာ/ါ)  ->  ေ + BASE (+ optional ာ/ါ)

    v2 ONLY. Do not use for labels or evaluation.
    """
    chars = [str(c) for c in chars]
    out: List[str] = []
    i = 0

    while i < len(chars):
        # BASE + ေ + ာ
        if (
            i + 2 < len(chars)
            and is_base_like(chars[i])
            and chars[i + 1] == MY_E
            and chars[i + 2] == MY_AA
        ):
            out.extend([MY_E, chars[i], MY_AA])
            i += 3
            continue

        # BASE + ေ + ါ
        if (
            i + 2 < len(chars)
            and is_base_like(chars[i])
            and chars[i + 1] == MY_E
            and chars[i + 2] == MY_TALL_AA
        ):
            out.extend([MY_E, chars[i], MY_TALL_AA])
            i += 3
            continue

        # BASE + ေ
        if (
            i + 1 < len(chars)
            and is_base_like(chars[i])
            and chars[i + 1] == MY_E
        ):
            out.extend([MY_E, chars[i]])
            i += 2
            continue

        out.append(chars[i])
        i += 1

    return out


def align_char_array_to_unit_info(
    align_chars: Sequence[str],
    config: AlignUnitConfig | None = None,
) -> List[Dict[str, Any]]:
    """
    Convert alignment-order char sequence to alignment-order units.

    Main v2 rule:
      ေ + BASE -> ေBASE
    """
    cfg = config or AlignUnitConfig()
    chars = [str(c) for c in align_chars]
    units: List[Dict[str, Any]] = []

    i = 0
    while i < len(chars):
        # ေ + BASE
        if (
            cfg.merge_e_base
            and i + 1 < len(chars)
            and chars[i] == MY_E
            and is_base_like(chars[i + 1])
        ):
            src = [chars[i], chars[i + 1]]
            units.append({
                "unit_text": "".join(src),
                "unit_role": infer_unit_role_from_chars(src),
                "source_chars": src,
                "source_indices": [i, i + 1],
            })
            i += 2
            continue

        # VIRAMA + BASE
        if (
            cfg.merge_virama_next_base
            and i + 1 < len(chars)
            and chars[i] == MY_VIRAMA
            and is_base_like(chars[i + 1])
        ):
            src = [chars[i], chars[i + 1]]
            units.append({
                "unit_text": "".join(src),
                "unit_role": infer_unit_role_from_chars(src),
                "source_chars": src,
                "source_indices": [i, i + 1],
            })
            i += 2
            continue

        # BASE + ASAT
        if (
            cfg.merge_base_asat
            and i + 1 < len(chars)
            and is_base_like(chars[i])
            and chars[i + 1] == MY_ASAT
        ):
            src = [chars[i], chars[i + 1]]
            units.append({
                "unit_text": "".join(src),
                "unit_role": infer_unit_role_from_chars(src),
                "source_chars": src,
                "source_indices": [i, i + 1],
            })
            i += 2
            continue

        units.append({
            "unit_text": chars[i],
            "unit_role": infer_unicode_myanmar_role(chars[i]),
            "source_chars": [chars[i]],
            "source_indices": [i],
        })
        i += 1

    return units


def align_char_array_to_unit_array(
    align_chars: Sequence[str],
    config: AlignUnitConfig | None = None,
) -> List[str]:
    info = align_char_array_to_unit_info(align_chars, config)
    return [u["unit_text"] for u in info]


# ============================================================
# DataFrame integration
# ============================================================

def add_unit_array_columns(
    df: pd.DataFrame,
    *,
    char_array_col: str = "char_array",
    syllable_col: str = "syllable",
    config: WritingUnitConfig | None = None,
) -> pd.DataFrame:
    """
    Add canonical unit_array, unit_roles, unit_meta, unit_string.
    """
    cfg = config or WritingUnitConfig()
    out = df.copy()

    unit_arrays, unit_roles, unit_meta, unit_strings = [], [], [], []

    for _, row in out.iterrows():
        chars = (
            row[char_array_col]
            if isinstance(row.get(char_array_col), (list, tuple))
            else list(str(row[syllable_col]))
        )

        info = char_array_to_unit_info(chars, cfg)
        unit_arrays.append([u["unit_text"] for u in info])
        unit_roles.append([u["unit_role"] for u in info])
        unit_meta.append(info if cfg.keep_unit_meta else None)
        unit_strings.append("".join(u["unit_text"] for u in info))

    out["unit_array"] = unit_arrays
    out["unit_roles"] = unit_roles
    out["unit_meta"] = unit_meta
    out["unit_string"] = unit_strings
    return out


def add_align_array_columns(
    df: pd.DataFrame,
    *,
    char_array_col: str = "char_array",
    syllable_col: str = "syllable",
    config: AlignUnitConfig | None = None,
) -> pd.DataFrame:
    """
    Add v2 alignment-only columns:
      - align_char_array
      - align_unit_array
      - align_unit_roles
      - align_unit_meta
      - align_unit_string
    """
    cfg = config or AlignUnitConfig()
    out = df.copy()

    align_chars_all, align_units_all = [], []
    align_roles_all, align_meta_all, align_unit_strings = [], [], []

    for _, row in out.iterrows():
        chars = (
            row[char_array_col]
            if isinstance(row.get(char_array_col), (list, tuple))
            else list(str(row[syllable_col]))
        )

        align_chars = canonical_to_align_char_array(chars)
        info = align_char_array_to_unit_info(align_chars, cfg)

        align_chars_all.append(align_chars)
        align_units_all.append([u["unit_text"] for u in info])
        align_roles_all.append([u["unit_role"] for u in info])
        align_meta_all.append(info if cfg.keep_unit_meta else None)
        align_unit_strings.append("".join(u["unit_text"] for u in info))

    out["align_char_array"] = align_chars_all
    out["align_unit_array"] = align_units_all
    out["align_unit_roles"] = align_roles_all
    out["align_unit_meta"] = align_meta_all
    out["align_unit_string"] = align_unit_strings
    return out


def summarize_unit_conversion(df: pd.DataFrame) -> pd.DataFrame:
    raw_len = df["char_array"].apply(len)
    unit_len = df["unit_array"].apply(len)

    return pd.DataFrame([
        {"metric": "num_samples", "value": len(df)},
        {"metric": "mean_raw_len", "value": raw_len.mean()},
        {"metric": "mean_unit_len", "value": unit_len.mean()},
        {"metric": "mean_reduction", "value": (raw_len - unit_len).mean()},
        {"metric": "pct_changed", "value": (raw_len != unit_len).mean()},
    ])


def summarize_align_conversion(df: pd.DataFrame) -> pd.DataFrame:
    raw_len = df["char_array"].apply(len)
    align_char_len = df["align_char_array"].apply(len)
    align_unit_len = df["align_unit_array"].apply(len)

    return pd.DataFrame([
        {"metric": "num_samples", "value": len(df)},
        {"metric": "mean_raw_char_len", "value": raw_len.mean()},
        {"metric": "mean_align_char_len", "value": align_char_len.mean()},
        {"metric": "mean_align_unit_len", "value": align_unit_len.mean()},
        {"metric": "pct_align_char_changed", "value": (df["char_array"] != df["align_char_array"]).mean()},
        {"metric": "pct_align_unit_changed", "value": (raw_len != align_unit_len).mean()},
    ])


__all__ = [
    # configs
    "WritingUnitConfig",
    "AlignUnitConfig",

    # role helpers
    "infer_unicode_myanmar_role",
    "infer_unit_role_from_chars",
    "is_base_like",

    # canonical unitization
    "char_array_to_unit_info",
    "char_array_to_unit_array",
    "add_unit_array_columns",
    "summarize_unit_conversion",

    # v2 alignment-only helpers
    "canonical_to_align_char_array",
    "align_char_array_to_unit_info",
    "align_char_array_to_unit_array",
    "add_align_array_columns",
    "summarize_align_conversion",
]