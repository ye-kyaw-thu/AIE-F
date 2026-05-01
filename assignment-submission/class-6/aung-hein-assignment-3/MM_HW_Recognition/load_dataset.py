
"""Utilities for loading Myanmar syllable metadata and online handwriting stroke files.

This module is designed to be imported from notebooks or training scripts.
"""


from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_PAD_TOKEN = "<PAD>"


def get_syllable_df(path: str | os.PathLike, pad_token: str = DEFAULT_PAD_TOKEN) -> tuple[pd.DataFrame, dict, dict]:
    """Load syllables and build character-level padded/encoded columns.

    Parameters
    ----------
    path:
        Path to a text/csv file containing one syllable per line.
    pad_token:
        Special token used for padding shorter syllables.

    Returns
    -------
    syllable_df, char2idx, idx2char

    Notes
    -----
    The returned DataFrame contains:
      - syllable
      - id                  (1-based index matching your filename convention)
      - num_char
      - char_array
      - padded_chars
      - encoded
      - mask
    """
    syllable_path = Path(path)
    if not syllable_path.exists():
        raise FileNotFoundError(f"Syllable file not found: {syllable_path}")

    syllable_df = pd.read_csv(syllable_path, header=None, names=["syllable"], encoding="utf-8")
    syllable_df["syllable"] = syllable_df["syllable"].fillna("").astype(str)
    syllable_df["id"] = syllable_df.index + 1

    syllable_df["num_char"] = syllable_df["syllable"].apply(len)
    syllable_df["char_array"] = syllable_df["syllable"].apply(list)

    max_len = int(syllable_df["num_char"].max()) if len(syllable_df) > 0 else 0

    # Build vocabulary from all characters plus padding token.
    all_chars = set()
    for chars in syllable_df["char_array"]:
        all_chars.update(chars)
    all_chars.add(pad_token)

    char2idx = {c: i for i, c in enumerate(sorted(all_chars))}
    idx2char = {i: c for c, i in char2idx.items()}
    pad_idx = char2idx[pad_token]

    def pad_sequence(chars: Sequence[str]) -> list[str]:
        return list(chars) + [pad_token] * (max_len - len(chars))

    def encode_sequence(chars: Sequence[str]) -> list[int]:
        return [char2idx[c] for c in chars]

    def create_mask(encoded_seq: Sequence[int]) -> list[int]:
        return [1 if x != pad_idx else 0 for x in encoded_seq]

    syllable_df["padded_chars"] = syllable_df["char_array"].apply(pad_sequence)
    syllable_df["encoded"] = syllable_df["padded_chars"].apply(encode_sequence)
    syllable_df["mask"] = syllable_df["encoded"].apply(create_mask)

    return syllable_df, char2idx, idx2char


def parse_stroke_file(filepath: str | os.PathLike) -> list[list[list[float]]]:
    """Parse a stroke file with the format:

    STROKE
    x y t
    x y t
    STROKE
    x y t
    ...

    Returns
    -------
    List of strokes, where each stroke is a list of [x, y, t] points.
    """
    stroke_path = Path(filepath)
    if not stroke_path.exists():
        raise FileNotFoundError(f"Stroke file not found: {stroke_path}")

    strokes: list[list[list[float]]] = []
    current_stroke: list[list[float]] = []

    with open(stroke_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if line.upper().startswith("STROKE"):
                if current_stroke:
                    strokes.append(current_stroke)
                    current_stroke = []
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid point format in {stroke_path} at line {line_no}: {line!r}. "
                    "Expected exactly 3 values: x y t"
                )

            x_str, y_str, t_str = parts
            try:
                current_stroke.append([float(x_str), float(y_str), float(t_str)])
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric point in {stroke_path} at line {line_no}: {line!r}"
                ) from exc

    if current_stroke:
        strokes.append(current_stroke)

    return strokes


def _parse_filename(stem: str) -> tuple[int, int]:
    """Parse filenames formatted like '<id>-<sample_id>.txt'."""
    try:
        index_str, sample_id_str = stem.split("-", 1)
        return int(index_str), int(sample_id_str)
    except Exception as exc:
        raise ValueError(
            f"Filename {stem!r} does not match expected pattern '<id>-<sample_id>.txt'"
        ) from exc



def load_all_strokes(data_dir: str | os.PathLike, *, skip_invalid: bool = True) -> pd.DataFrame:
    """Load all stroke files from a directory into a DataFrame.

    Parameters
    ----------
    data_dir:
        Directory containing text files named like '<id>-<sample_id>.txt'.
    skip_invalid:
        If True, skip files that fail filename parsing or content parsing.
        If False, raise the first exception encountered.

    Returns
    -------
    pd.DataFrame
        Columns:
          - id
          - sample_id
          - strokes
          - num_strokes
          - max_stroke_len
          - total_points
          - source_file
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    if not data_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_path}")

    records: list[dict] = []

    for file_path in sorted(data_path.glob("*.txt")):
        try:
            index, sample_id = _parse_filename(file_path.stem)
            strokes = parse_stroke_file(file_path)
            stroke_lengths = [len(stroke) for stroke in strokes]

            records.append({
                "id": index,
                "sample_id": sample_id,
                "strokes": strokes,
                "num_strokes": len(strokes),
                "max_stroke_len": max(stroke_lengths) if stroke_lengths else 0,
                "total_points": sum(stroke_lengths),
                "source_file": str(file_path),
            })
        except Exception:
            if skip_invalid:
                continue
            raise

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["id", "sample_id"]).reset_index(drop=True)
    return df



def build_train_df(
    syllable_path: str | os.PathLike,
    data_dir: str | os.PathLike,
    *,
    pad_token: str = DEFAULT_PAD_TOKEN,
    skip_invalid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    """Convenience function to build a merged training DataFrame.

    Returns
    -------
    train_df, syllable_df, stroke_df, char2idx, idx2char
    """
    syllable_df, char2idx, idx2char = get_syllable_df(syllable_path, pad_token=pad_token)
    stroke_df = load_all_strokes(data_dir, skip_invalid=skip_invalid)
    train_df = stroke_df.merge(syllable_df, on="id", how="left", validate="many_to_one")
    return train_df, syllable_df, stroke_df, char2idx, idx2char


__all__ = [
    "DEFAULT_PAD_TOKEN",
    "get_syllable_df",
    "parse_stroke_file",
    "load_all_strokes",
    "build_train_df",
]