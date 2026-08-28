#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
myanmar_asr.py  —  Myanmar (Burmese) ASR Tutorial
==================================================
Dataset  : SLR80 Crowdsourced Burmese Speech  https://www.openslr.org/80/
Model    : HuggingFace Wav2Vec2ForCTC  (CTC fine-tuning)
Tokeniser: Custom Myanmar CHARACTER vocabulary  (~64 tokens)
Metrics  : WER  +  Myanmar Character Error Rate (CER)

Why character-level is the right choice for Myanmar CTC ASR
-----------------------------------------------------------
• Myanmar is an abugida script; each Unicode code point is a meaningful unit
  (consonant, vowel diacritic, tone mark, stacking symbol).
• Character vocabulary is small (~64), making CTC alignment easy to learn.
• No OOV problem: every Myanmar character seen at test time is in the vocab.
• CER is more linguistically meaningful than WER for Myanmar because word
  boundaries are written inconsistently (phrases, not strict words).
• With only ~2.5 h of data (SLR80), character-level CTC is the only reliable
  approach. Syllable or word-level units need ≥10 h data.

Why a CUSTOM tokeniser is mandatory
------------------------------------
facebook/wav2vec2-base ships with an English-only tokeniser (A-Z).
Myanmar Unicode (U+1000–U+109F) is absent. Feeding Myanmar text to the
stock processor causes every label encoding to raise an exception, the
dataset generator silently skips every sample, and training sees an empty
dataset — causing:  "ValueError: Instruction train corresponds to no data"

Correct workflow
----------------
  1. Collect every unique character from training transcripts
  2. Build myanmar_vocab.json  { "[PAD]":0, "[UNK]":1, "|":2, "က":3, … }
  3. Wav2Vec2CTCTokenizer  from that vocab file
  4. Wav2Vec2FeatureExtractor  from the pretrained hub checkpoint
  5. Wav2Vec2Processor  = tokeniser + feature extractor
  6. Wav2Vec2ForCTC     with ignore_mismatched_sizes=True  (re-init lm_head)

Pipeline stages
---------------
  prep  → parse TSV, split train/dev/test, build char vocab, save manifests
  train → build custom processor, fine-tune model, save best checkpoint
  eval  → greedy CTC decode test set, compute WER + CER, save report
  all   → prep → train → eval  (default)

Quick start
-----------
  # Full pipeline (GPU recommended)
  python myanmar_asr.py --stage all \\
         --data_dir ./data/slr80 --output_dir ./asr_output \\
         --epochs 20 --lr 1e-4 --batch_size 8 --grad_accum 2 --fp16

  # Step by step
  python myanmar_asr.py --stage prep --data_dir ./data/slr80
  python myanmar_asr.py --stage train --epochs 20 --lr 1e-4 --fp16
  python myanmar_asr.py --stage eval

  # Quick demo  (500 samples, 5 epochs — ~5 min on GPU)
  python myanmar_asr.py --stage all --max_samples 500 --epochs 5

  python myanmar_asr.py --help

UPDATED FOR AIEF CLASS: Ye Kyaw Thu, LU Lab., Myanmar
DATE: 11 June 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_file: Optional[str] = None,
                  verbose: bool = False) -> logging.Logger:
    level   = logging.DEBUG if verbose else logging.INFO
    fmt     = "%(asctime)s  [%(levelname)-8s]  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    return logging.getLogger("asr")


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency check
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED = {
    "torch":        "pip install torch torchaudio",
    "transformers": "pip install transformers",
    "datasets":     "pip install datasets",
    "soundfile":    "pip install soundfile",
    "librosa":      "pip install librosa",
    "matplotlib":   "pip install matplotlib",
}

def check_dependencies(logger: logging.Logger) -> bool:
    ok = True
    for pkg, hint in REQUIRED.items():
        try:
            __import__(pkg)
            logger.info(f"  ✓ {pkg}")
        except ImportError:
            logger.error(f"  ✗ {pkg}  →  {hint}")
            ok = False
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# Myanmar CER  (character-level edit distance)
# ═══════════════════════════════════════════════════════════════════════════════

def _edit_distance(ref: list, hyp: list) -> Tuple[int, int, int, int]:
    """Levenshtein DP — returns (total_edits, substitutions, deletions, insertions)."""
    m, n = len(ref), len(hyp)
    dp   = [(j, 0, 0, j) for j in range(n + 1)]   # (cost, S, D, I)
    for i in range(1, m + 1):
        new = [(i, 0, i, 0)] + [(0, 0, 0, 0)] * n
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                new[j] = dp[j-1]
            else:
                sub = (dp[j-1][0]+1, dp[j-1][1]+1, dp[j-1][2],   dp[j-1][3])
                dlt = (dp[j][0]+1,   dp[j][1],      dp[j][2]+1,   dp[j][3])
                ins = (new[j-1][0]+1,new[j-1][1],   new[j-1][2],  new[j-1][3]+1)
                new[j] = min(sub, dlt, ins, key=lambda x: x[0])
        dp = new
    return dp[n]


def compute_cer(references: List[str], hypotheses: List[str]) -> Dict:
    """Character Error Rate = (S + D + I) / N  where N = total reference chars."""
    total = S = D = I = edits = 0
    for ref, hyp in zip(references, hypotheses):
        rc, hc = list(ref.strip()), list(hyp.strip())
        total += len(rc)
        cost, s, d, ins = _edit_distance(rc, hc)
        edits += cost; S += s; D += d; I += ins
    cer = edits / total if total > 0 else 0.0
    return {
        "cer": round(cer, 4), "cer_pct": round(cer * 100, 2),
        "total_chars": total,
        "substitutions": S, "deletions": D, "insertions": I,
    }


def compute_wer(references: List[str], hypotheses: List[str]) -> Dict:
    """Word Error Rate on space-tokenised words."""
    total = edits = 0
    for ref, hyp in zip(references, hypotheses):
        rw, hw = ref.strip().split(), hyp.strip().split()
        total += len(rw)
        edits += _edit_distance(rw, hw)[0]
    wer = edits / total if total > 0 else 0.0
    return {"wer": round(wer, 4), "wer_pct": round(wer * 100, 2),
            "total_words": total}


# ═══════════════════════════════════════════════════════════════════════════════
# SLR80 data-layout helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_wav(data_dir: Path, fid: str) -> Optional[Path]:
    """
    SLR80 may be extracted as:
      (a) <data_dir>/wavs/<fid>.wav   — zip default
      (b) <data_dir>/<fid>.wav        — flat extraction (common on Linux)
    """
    for c in [data_dir / "wavs" / f"{fid}.wav",
              data_dir / f"{fid}.wav"]:
        if c.exists():
            return c
    return None


def _parse_tsv_line(raw: str) -> Optional[Tuple[str, str]]:
    """
    Parse one SLR80 TSV line robustly.
    Handles real TAB and 2+-space separators (editors sometimes convert tabs).
    Returns (file_id, transcript) or None.
    """
    raw = raw.strip()
    if not raw:
        return None
    if "\t" in raw:
        parts = raw.split("\t", 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return parts[0].strip(), parts[1].strip()
    parts = re.split(r"\s{2,}", raw, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[0].strip(), parts[1].strip()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Myanmar text normalisation
# ═══════════════════════════════════════════════════════════════════════════════

# Zero-width and invisible Unicode characters to strip
_STRIP_CHARS = frozenset("\u200b\u200c\u200d\ufeff\xa0")


def normalize_transcript(text: str) -> str:
    """NFC-normalise and strip invisible Unicode characters."""
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if c not in _STRIP_CHARS)


# ═══════════════════════════════════════════════════════════════════════════════
# Myanmar character vocabulary builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_myanmar_vocab(texts: List[str]) -> Dict[str, int]:
    """
    Build a Wav2Vec2CTCTokenizer-compatible character vocabulary.

    Special tokens follow the Wav2Vec2 convention:
      [PAD] = 0   CTC blank token  (MUST be 0 for HuggingFace CTC)
      [UNK] = 1   unknown character
      |     = 2   word-boundary token (replaces space in transcripts)

    All unique characters from the training set are appended in sorted order
    so the mapping is deterministic across runs.
    """
    chars: set = set()
    for text in texts:
        for c in normalize_transcript(text):
            if c != " ":          # spaces become "|" word-boundary
                chars.add(c)
    chars.discard("|")             # "|" is reserved as word-boundary

    vocab: Dict[str, int] = {"[PAD]": 0, "[UNK]": 1, "|": 2}
    for c in sorted(chars):
        vocab[c] = len(vocab)
    return vocab


# ═══════════════════════════════════════════════════════════════════════════════
# Data preparation stage
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    data_dir:    str,
    tsv_file:    str,
    output_dir:  str,
    train_ratio: float,
    dev_ratio:   float,
    seed:        int,
    max_samples: int,
    logger:      logging.Logger,
) -> Dict[str, List[Dict]]:
    """
    Parse SLR80 TSV  →  split train/dev/test  →  build char vocab  →  save.

    Output files written to output_dir:
      train.json / dev.json / test.json  — {id, wav, text} manifests
      myanmar_vocab.json                 — character vocabulary
    """
    random.seed(seed)

    data_path = Path(data_dir)
    tsv_path  = data_path / tsv_file
    if not tsv_path.exists():
        logger.error(f"TSV not found: {tsv_path}")
        logger.info("  Download SLR80: https://www.openslr.org/80/")
        sys.exit(1)

    # Report audio layout
    wavs_sub = data_path / "wavs"
    if wavs_sub.exists() and any(wavs_sub.glob("*.wav")):
        logger.info(f"Audio found in: {wavs_sub}/")
    else:
        n_flat = sum(1 for _ in data_path.glob("*.wav"))
        if n_flat > 0:
            logger.info(f"Audio found flat in: {data_path}/  ({n_flat} wav files)")
        else:
            logger.warning(
                f"No .wav files found under {data_path}!\n"
                "  Unzip with:  cd data/slr80 && unzip my_mm_female.zip"
            )

    samples: List[Dict] = []
    n_miss = n_bad = 0
    with open(tsv_path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            parsed = _parse_tsv_line(raw)
            if parsed is None:
                n_bad += 1
                continue
            fid, text = parsed
            wav = _find_wav(data_path, fid)
            if wav is None:
                n_miss += 1
                if n_miss <= 3:
                    logger.debug(f"  Missing: {fid}")
                continue
            samples.append({
                "id":   fid,
                "wav":  str(wav),
                "text": normalize_transcript(text),
            })

    logger.info(
        f"TSV parse: {len(samples)} valid rows  "
        f"({n_miss} missing audio,  {n_bad} malformed lines)"
    )
    if len(samples) == 0:
        logger.error(
            "No valid samples found!\n"
            "  • ls data/slr80/*.wav  or  ls data/slr80/wavs/*.wav\n"
            "  • head data/slr80/line_index_female.tsv"
        )
        sys.exit(1)

    if max_samples > 0 and len(samples) > max_samples:
        random.shuffle(samples)
        samples = samples[:max_samples]
        logger.info(f"  Capped at {max_samples} samples (--max_samples)")

    random.shuffle(samples)
    n       = len(samples)
    n_train = max(1, int(n * train_ratio))
    n_dev   = max(1, min(int(n * dev_ratio), n - n_train - 1))

    splits = {
        "train": samples[:n_train],
        "dev":   samples[n_train : n_train + n_dev],
        "test":  samples[n_train + n_dev :],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        p = out / f"{split}.json"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        logger.info(f"  {split:5s}: {len(data):5d} utterances  →  {p}")

    # Build vocabulary from training set only (no test-set leakage)
    vocab = build_myanmar_vocab([s["text"] for s in splits["train"]])
    vocab_path = out / "myanmar_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    logger.info(
        f"Character vocab: {len(vocab)} tokens  →  {vocab_path}\n"
        f"  ([PAD]=0, [UNK]=1, |=word-boundary, +{len(vocab)-3} unique chars)"
    )

    logger.info("Data preparation complete.")
    return splits


def load_split(output_dir: str, split: str) -> List[Dict]:
    path = Path(output_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            f"  Run:  python myanmar_asr.py --stage prep "
            f"--data_dir ./data/slr80 --output_dir {output_dir}"
        )
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if len(data) == 0:
        raise ValueError(
            f"{path} is empty (0 samples)!\n"
            "  Re-run --stage prep and confirm .wav files are present."
        )
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# Custom Myanmar processor  (tokeniser + feature extractor)
# ═══════════════════════════════════════════════════════════════════════════════

def build_myanmar_processor(output_dir: str, model_name: str,
                             logger: logging.Logger):
    """
    Build Wav2Vec2Processor with:
      • Wav2Vec2CTCTokenizer  — custom Myanmar character vocabulary
      • Wav2Vec2FeatureExtractor — loaded from the pretrained model hub

    The pretrained English tokeniser is discarded; only the CNN feature
    extractor weights and transformer weights are reused.
    Saved to <output_dir>/myanmar_processor/ for reloading at eval time.
    """
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )

    vocab_path = Path(output_dir) / "myanmar_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"{vocab_path} not found.  Run --stage prep first."
        )

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token           = "[UNK]",
        pad_token           = "[PAD]",
        word_delimiter_token= "|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name,
        feature_size         = 1,
        sampling_rate        = 16_000,
        padding_value        = 0.0,
        do_normalize         = True,
        return_attention_mask= True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor = feature_extractor,
        tokenizer         = tokenizer,
    )

    proc_path = Path(output_dir) / "myanmar_processor"
    proc_path.mkdir(exist_ok=True)
    processor.save_pretrained(str(proc_path))

    logger.info(f"Myanmar processor saved  →  {proc_path}")
    logger.info(f"  Tokeniser vocab size: {tokenizer.vocab_size}")
    return processor


# ═══════════════════════════════════════════════════════════════════════════════
# Audio loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(wav_path: str, target_sr: int = 16_000) -> np.ndarray:
    """Load WAV as float32 mono, resampling to target_sr if needed."""
    import soundfile as sf
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            n_out   = int(len(audio) * target_sr / sr)
            indices = np.linspace(0, len(audio)-1, n_out).astype(np.int32)
            audio   = audio[indices]
    return audio


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace dataset builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_hf_dataset(samples: List[Dict], processor,
                     max_duration: float, logger: logging.Logger):
    """
    Build a HuggingFace Dataset from sample manifests.

    Label encoding
    --------------
    Wav2Vec2CTCTokenizer tokenises CHARACTER BY CHARACTER.
    For Myanmar (character-level vocab) this is exactly what we want:
    each Unicode code point maps directly to its vocab ID.
    Spaces in the transcript are mapped to the word-boundary token "|" (id=2).

    We call processor.tokenizer() DIRECTLY (not as_target_processor, which
    was deprecated and removed in transformers ≥ 4.27 and fails silently).

    We build the dataset EAGERLY (from_dict, not from_generator) to avoid
    the Arrow cache replaying a stale empty result from a previous broken run.
    """
    from datasets import Dataset

    rows_input_values: List = []
    rows_labels:       List = []
    rows_text:         List = []
    rows_id:           List = []

    n_ok = n_dur = n_err = 0

    for idx, s in enumerate(samples):
        # ── 1. Load and validate audio ─────────────────────────────────────
        try:
            audio = load_audio(s["wav"], target_sr=16_000)
        except Exception as e:
            logger.debug(f"  Audio load failed {s['id']}: {e}")
            n_err += 1
            continue

        dur = len(audio) / 16_000
        if dur > max_duration:
            n_dur += 1
            continue

        # ── 2. Extract acoustic features ───────────────────────────────────
        try:
            inp = processor.feature_extractor(
                audio,
                sampling_rate  = 16_000,
                return_tensors = "np",
                padding        = False,
            )
            input_values = inp.input_values[0].tolist()
        except Exception as e:
            logger.debug(f"  Feature extraction failed {s['id']}: {e}")
            n_err += 1
            continue

        # ── 3. Encode text labels — direct tokeniser call ──────────────────
        # processor.tokenizer() handles space → "|" internally.
        # For character-level vocab every Myanmar char is a single token,
        # so no pre-segmentation is needed.
        try:
            enc       = processor.tokenizer(
                s["text"],
                return_tensors = None,
                padding        = False,
                truncation     = False,
            )
            label_ids = enc["input_ids"]
            if not label_ids:
                logger.debug(f"  Empty label for {s['id']}, skipping")
                n_err += 1
                continue
        except Exception as e:
            logger.debug(f"  Tokenisation failed {s['id']}: {e}")
            n_err += 1
            continue

        rows_input_values.append(input_values)
        rows_labels.append([int(x) for x in label_ids])
        rows_text.append(s["text"])
        rows_id.append(s["id"])
        n_ok += 1

        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx+1}/{len(samples)} ...")

    logger.info(
        f"  Feature extraction: {n_ok} OK,  "
        f"{n_dur} too-long (>{max_duration}s),  {n_err} errors"
    )

    if n_ok == 0:
        raise RuntimeError(
            "Dataset is empty after feature extraction — 0 samples processed!\n"
            "  Possible causes:\n"
            "  1. soundfile / librosa not installed or audio corrupted\n"
            "  2. myanmar_vocab.json missing — re-run --stage prep\n"
            "  3. WAV files not 16kHz PCM — check: soxi data/slr80/*.wav\n"
            "  Run with --verbose to see per-file errors."
        )

    return Dataset.from_dict({
        "input_values": rows_input_values,
        "labels":       rows_labels,
        "text":         rows_text,
        "id":           rows_id,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CTC data collator
# ═══════════════════════════════════════════════════════════════════════════════

def make_ctc_collator(processor):
    """
    Pad variable-length input_values and labels within each batch.
    Labels are padded with -100 so CTC loss ignores padding positions.
    Pads audio manually (more robust across transformers versions than
    processor.pad() whose signature changed in 5.x).
    """
    import torch

    def collate(features: List[Dict]) -> Dict:
        # ── Pad audio ──────────────────────────────────────────────────────
        input_values = [
            torch.tensor(f["input_values"], dtype=torch.float32)
            for f in features
        ]
        max_audio_len  = max(v.size(0) for v in input_values)
        padded_audio   = torch.zeros(len(input_values), max_audio_len)
        attention_mask = torch.zeros(len(input_values), max_audio_len,
                                     dtype=torch.long)
        for i, v in enumerate(input_values):
            padded_audio[i, :v.size(0)]    = v
            attention_mask[i, :v.size(0)]  = 1

        # ── Pad labels with -100 (CTC ignore index) ─────────────────────────
        label_ids      = [f["labels"] for f in features]
        max_label_len  = max(len(l) for l in label_ids)
        labels_padded  = torch.full(
            (len(label_ids), max_label_len), fill_value=-100, dtype=torch.long
        )
        for i, ids in enumerate(label_ids):
            labels_padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

        return {
            "input_values":  padded_audio,
            "attention_mask": attention_mask,
            "labels":         labels_padded,
        }

    return collate


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics callback for HuggingFace Trainer
# ═══════════════════════════════════════════════════════════════════════════════

def make_compute_metrics(processor):
    """
    Manual CTC greedy decode → WER + CER.

    We implement decoding ourselves rather than using processor.decode() /
    batch_decode() because those APIs changed between transformers 4.x and 5.x
    (group_tokens kwarg removed, behaviour changed for non-ASCII tokens).

    CTC decode steps:
      1. argmax over logits  →  raw token id sequence  (B, T)
      2. collapse consecutive duplicate ids             (CTC merge rule)
      3. remove [PAD]=0 (CTC blank token)
      4. map each id → character via vocab lookup
      5. replace "|" word-boundary token → space
    """
    import numpy as np

    pad_id    = processor.tokenizer.pad_token_id   # 0
    vocab_map = processor.tokenizer.get_vocab()    # str → int
    vocab_sz  = processor.tokenizer.vocab_size
    id2char: List[str] = [""] * vocab_sz
    for ch, idx in vocab_map.items():
        if idx < vocab_sz:
            id2char[idx] = ch

    def _ctc_decode(ids) -> str:
        """Greedy CTC decode one sequence of token ids → string."""
        out  = []
        prev = None
        for tid in ids:
            tid = int(tid)
            if tid != prev:
                if tid != pad_id:
                    out.append(tid)
            prev = tid
        chars = []
        for tid in out:
            ch = id2char[tid] if tid < vocab_sz else ""
            if ch and ch not in ("[PAD]", "[UNK]"):
                chars.append(" " if ch == "|" else ch)
        return "".join(chars).strip()

    def _decode_labels(ids) -> str:
        """Decode ground-truth label ids (no CTC collapsing needed)."""
        chars = []
        for tid in ids:
            tid = int(tid)
            if tid in (-100, pad_id):
                continue
            ch = id2char[tid] if tid < vocab_sz else ""
            if ch and ch not in ("[PAD]", "[UNK]"):
                chars.append(" " if ch == "|" else ch)
        return "".join(chars).strip()

    def compute_metrics(pred) -> Dict:
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids

        pred_str  = [_ctc_decode(row)    for row in pred_ids]
        label_str = [_decode_labels(row) for row in label_ids]

        # Log two examples at DEBUG level (visible with --verbose)
        log = logging.getLogger("asr")
        for i in range(min(2, len(pred_str))):
            log.debug(
                f"  EVAL sample {i}\n"
                f"    REF: {label_str[i][:80]}\n"
                f"    HYP: {pred_str[i][:80]}"
            )

        wer_res = compute_wer(label_str, pred_str)
        cer_res = compute_cer(label_str, pred_str)
        return {"wer": wer_res["wer"], "cer": cer_res["cer"]}

    return compute_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Learning-curve plot
# ═══════════════════════════════════════════════════════════════════════════════

def plot_learning_curve(log_history: List[Dict], save_path: Path,
                        logger: logging.Logger) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        train_steps:  List[int]   = []
        train_losses: List[float] = []
        eval_epochs:  List[float] = []
        eval_wer:     List[float] = []
        eval_cer:     List[float] = []

        for e in log_history:
            if "loss" in e and "eval_loss" not in e:
                train_steps.append(e.get("step", 0))
                train_losses.append(e["loss"])
            if "eval_wer" in e:
                eval_epochs.append(e.get("epoch", 0))
                eval_wer.append(e["eval_wer"] * 100)
                eval_cer.append(e.get("eval_cer", 0) * 100)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(train_steps, train_losses, color="#2563eb", linewidth=1.2)
        axes[0].set_title("CTC Training Loss")
        axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)

        axes[1].plot(eval_epochs, eval_wer, marker="o",
                     color="#dc2626", linewidth=1.2)
        axes[1].set_title("Dev WER (%)")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("WER %")
        axes[1].grid(alpha=0.3)

        axes[2].plot(eval_epochs, eval_cer, marker="s",
                     color="#16a34a", linewidth=1.2)
        axes[2].set_title("Dev CER — Myanmar chars (%)")
        axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("CER %")
        axes[2].grid(alpha=0.3)

        fig.suptitle("ASR Learning Curves  —  Myanmar SLR80",
                     fontweight="bold", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Learning curve  →  {save_path}")
    except Exception as e:
        logger.warning(f"Could not plot learning curve: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training stage
# ═══════════════════════════════════════════════════════════════════════════════

def run_train(args, logger: logging.Logger) -> None:
    import torch
    from transformers import (
        Wav2Vec2ForCTC,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    import inspect

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    train_samples = load_split(args.output_dir, "train")
    dev_samples   = load_split(args.output_dir, "dev")
    logger.info(f"Train: {len(train_samples)}  Dev: {len(dev_samples)}")

    if len(train_samples) < 2:
        logger.error("Too few training samples. Re-run --stage prep.")
        sys.exit(1)

    # ── Build custom Myanmar processor ───────────────────────────────────────
    processor  = build_myanmar_processor(args.output_dir, args.model_name, logger)
    vocab_size = processor.tokenizer.vocab_size

    # ── Load Wav2Vec2ForCTC with new Myanmar vocab size ───────────────────────
    # ignore_mismatched_sizes=True re-initialises only the lm_head (classifier).
    # All encoder weights are loaded from the pretrained checkpoint.
    logger.info(f"Loading base model: {args.model_name}")
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model_name,
        ctc_loss_reduction      = "mean",
        pad_token_id            = processor.tokenizer.pad_token_id,
        vocab_size              = vocab_size,
        ignore_mismatched_sizes = True,
    )

    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()
        logger.info("  Feature encoder FROZEN (recommended for small datasets)")

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Parameters: {n_params:,}  (trainable: {n_trainable:,})")

    # ── Build HuggingFace Datasets ────────────────────────────────────────────
    logger.info("Building train dataset ...")
    train_ds = build_hf_dataset(train_samples, processor, args.max_duration, logger)
    logger.info("Building dev dataset ...")
    dev_ds   = build_hf_dataset(dev_samples,   processor, args.max_duration, logger)
    logger.info(f"  Final sizes — Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    collate_fn      = make_ctc_collator(processor)
    compute_metrics = make_compute_metrics(processor)

    # ── TrainingArguments — version-adaptive ─────────────────────────────────
    # transformers ≥ 4.41 renamed evaluation_strategy → eval_strategy
    # transformers ≥ 4.46 renamed Trainer kwarg tokenizer → processing_class
    # transformers  5.x   removed group_by_length, warmup_ratio, logging_dir
    # We introspect signatures at runtime so the code works on any version.
    import transformers as _tfm
    logger.info(f"  transformers {_tfm.__version__}")

    _ta_sig   = set(inspect.signature(TrainingArguments.__init__).parameters)
    _tr_sig   = set(inspect.signature(Trainer.__init__).parameters)

    def _ta(name: str, alt: str, value):
        if name in _ta_sig: return {name: value}
        if alt  in _ta_sig: return {alt:  value}
        return {}

    use_fp16 = args.fp16 and torch.cuda.is_available()

    _ta_kwargs: Dict = {
        "output_dir":                   str(out / "checkpoints"),
        "num_train_epochs":             args.epochs,
        "per_device_train_batch_size":  args.batch_size,
        "per_device_eval_batch_size":   args.eval_batch_size,
        "gradient_accumulation_steps":  args.grad_accum,
        "learning_rate":                args.lr,
        "weight_decay":                 args.weight_decay,
        "save_strategy":                "epoch",
        "logging_steps":                args.log_steps,
        "load_best_model_at_end":       True,
        "metric_for_best_model":        "cer",
        "greater_is_better":            False,
        "fp16":                         use_fp16,
        "dataloader_num_workers":       args.num_workers,
        "report_to":                    ["tensorboard"] if args.tensorboard
                                        else ["none"],
        "seed":                         args.seed,
        "save_total_limit":             3,
    }
    # eval_strategy (≥4.41) vs evaluation_strategy (<4.41)
    _ta_kwargs.update(_ta("eval_strategy", "evaluation_strategy", "epoch"))
    # warmup_ratio removed in 5.x → convert to warmup_steps
    _ta_kwargs.update(_ta("warmup_ratio", None, args.warmup_ratio))
    if "warmup_ratio" not in _ta_sig:
        steps = max(1, int(
            args.warmup_ratio *
            (len(train_ds) // (args.batch_size * args.grad_accum)) *
            args.epochs
        ))
        _ta_kwargs["warmup_steps"] = steps
        logger.info(f"  warmup_ratio → warmup_steps={steps}")
    # logging_dir removed in 5.x
    if "logging_dir" in _ta_sig:
        _ta_kwargs["logging_dir"] = str(out / "tb_logs")
    # group_by_length removed in 5.x
    _ta_kwargs.update(_ta("group_by_length", None, True))

    training_args = TrainingArguments(**_ta_kwargs)

    callbacks = []
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stop_patience
            )
        )
        logger.info(f"  Early stopping patience: {args.early_stop_patience}")

    _tok_key = "processing_class" if "processing_class" in _tr_sig else "tokenizer"
    trainer  = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = dev_ds,
        **{_tok_key:      processor.feature_extractor},
        data_collator   = collate_fn,
        compute_metrics = compute_metrics,
        callbacks       = callbacks,
    )

    logger.info("=" * 65)
    logger.info("  TRAINING  —  Myanmar ASR  (Wav2Vec2 CTC fine-tuning)")
    logger.info(f"  Base model   : {args.model_name}")
    logger.info(f"  Vocab size   : {vocab_size} Myanmar characters")
    logger.info(f"  Epochs       : {args.epochs}")
    logger.info(f"  LR           : {args.lr}  warmup_ratio={args.warmup_ratio}")
    logger.info(f"  Batch/device : {args.batch_size}  "
                f"grad_accum={args.grad_accum}  "
                f"eff_batch={args.batch_size * args.grad_accum}")
    logger.info(f"  FP16         : {use_fp16}")
    logger.info(f"  Output       : {out}")
    logger.info("=" * 65)

    t0           = time.time()
    train_result = trainer.train()
    elapsed_min  = (time.time() - t0) / 60

    logger.info(f"Training finished in {elapsed_min:.1f} min")
    logger.info(f"  Final train loss: {train_result.training_loss:.4f}")

    # Save best model + custom processor together
    best_path = out / "best_model"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    logger.info(f"Best model  →  {best_path}")

    log_path = out / "train_log.json"
    with open(log_path, "w") as fh:
        json.dump(trainer.state.log_history, fh, indent=2)
    logger.info(f"Training log  →  {log_path}")

    plot_learning_curve(
        trainer.state.log_history,
        out / "learning_curve_asr.png",
        logger,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation stage
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(args, logger: logging.Logger) -> None:
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    out        = Path(args.output_dir)
    model_path = out / "best_model"
    if not model_path.exists():
        logger.error(f"No model at {model_path}  —  run --stage train first.")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(str(model_path))
    model     = Wav2Vec2ForCTC.from_pretrained(str(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Device: {device}")

    test_samples = load_split(args.output_dir, "test")
    logger.info(f"Evaluating on {len(test_samples)} test utterances ...")

    # Build vocab lookup once
    pad_id   = processor.tokenizer.pad_token_id
    vocab_sz = processor.tokenizer.vocab_size
    vocab_m  = processor.tokenizer.get_vocab()
    id2char: List[str] = [""] * vocab_sz
    for ch, idx in vocab_m.items():
        if idx < vocab_sz:
            id2char[idx] = ch

    def _ctc_decode_eval(ids) -> str:
        out  = []
        prev = None
        for tid in map(int, ids):
            if tid != prev:
                if tid != pad_id:
                    out.append(tid)
            prev = tid
        chars = []
        for tid in out:
            ch = id2char[tid] if tid < vocab_sz else ""
            if ch and ch not in ("[PAD]", "[UNK]"):
                chars.append(" " if ch == "|" else ch)
        return "".join(chars).strip()

    refs: List[str] = []
    hyps: List[str] = []
    ids:  List[str] = []
    n_failed = 0

    for i, s in enumerate(test_samples):
        try:
            audio  = load_audio(s["wav"], target_sr=16_000)
            inputs = processor.feature_extractor(
                audio,
                sampling_rate  = 16_000,
                return_tensors = "pt",
                padding        = True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids  = torch.argmax(logits, dim=-1).cpu().numpy()
            pred_text = _ctc_decode_eval(pred_ids[0])

            refs.append(s["text"])
            hyps.append(pred_text)
            ids.append(s["id"])
        except Exception as e:
            logger.debug(f"  Failed {s['id']}: {e}")
            n_failed += 1
        if (i + 1) % 100 == 0:
            logger.info(f"  Decoded {i+1}/{len(test_samples)} ...")

    logger.info(f"Decoded {len(refs)}  ({n_failed} failed)")

    wer_res = compute_wer(refs, hyps)
    cer_res = compute_cer(refs, hyps)

    logger.info("")
    logger.info("=" * 65)
    logger.info("  ASR EVALUATION RESULTS  —  Myanmar SLR80 test set")
    logger.info("=" * 65)
    logger.info(f"  WER           : {wer_res['wer_pct']:6.2f}%  "
                f"({wer_res['total_words']:,} words)")
    logger.info(f"  CER (Myanmar) : {cer_res['cer_pct']:6.2f}%  "
                f"({cer_res['total_chars']:,} chars)")
    logger.info( "  CER breakdown :")
    logger.info(f"    Substitutions : {cer_res['substitutions']:,}")
    logger.info(f"    Deletions     : {cer_res['deletions']:,}")
    logger.info(f"    Insertions    : {cer_res['insertions']:,}")
    logger.info("=" * 65)

    # Per-utterance results
    utt_results = []
    for uid, ref, hyp in zip(ids, refs, hyps):
        utt_results.append({
            "id":         uid,
            "reference":  ref,
            "hypothesis": hyp,
            "cer_pct":    compute_cer([ref], [hyp])["cer_pct"],
            "wer_pct":    compute_wer([ref], [hyp])["wer_pct"],
        })
    utt_results.sort(key=lambda x: x["cer_pct"], reverse=True)

    results_path = out / "asr_test_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump({
            "summary": {
                "wer": wer_res, "cer": cer_res,
                "n_decoded": len(refs), "n_failed": n_failed,
            },
            "utterances": utt_results,
        }, fh, ensure_ascii=False, indent=2)
    logger.info(f"Results  →  {results_path}")

    # Top-5 worst utterances (teaching aid)
    logger.info("\nTop-5 worst utterances by CER:")
    for r in utt_results[:5]:
        logger.info(f"  ID : {r['id']}")
        logger.info(f"  REF: {r['reference']}")
        logger.info(f"  HYP: {r['hypothesis']}")
        logger.info(f"  CER: {r['cer_pct']:.1f}%   WER: {r['wer_pct']:.1f}%\n")

    # Markdown summary report
    report = f"""# ASR Evaluation Report — Myanmar SLR80

## Results

| Metric | Value |
|--------|-------|
| **WER** | {wer_res['wer_pct']:.2f}% |
| **CER (Myanmar)** | {cer_res['cer_pct']:.2f}% |
| Test utterances decoded | {len(refs)} |
| Total words | {wer_res['total_words']:,} |
| Total characters | {cer_res['total_chars']:,} |
| Substitutions | {cer_res['substitutions']:,} |
| Deletions | {cer_res['deletions']:,} |
| Insertions | {cer_res['insertions']:,} |
| Failed decodes | {n_failed} |

## Model
- Base checkpoint: `{args.model_name}`
- Tokeniser: Custom Myanmar character vocabulary (built from training set)
- Fine-tuned on: SLR80 Myanmar Female Speech Corpus (CC BY-SA 4.0)

## Metric notes
- **CER** is the primary metric for Myanmar. Space usage is inconsistent
  in Myanmar text, so WER is less reliable.
- CER = (Substitutions + Deletions + Insertions) / Total reference characters
- "|" in hypotheses represents the word-boundary token (maps back to space).
"""
    report_path = out / "asr_eval_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report  →  {report_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "myanmar_asr.py",
        description = (
            "Myanmar ASR tutorial — SLR80 + Wav2Vec2 CTC "
            "with custom Myanmar character tokeniser"
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python myanmar_asr.py --stage prep --data_dir ./data/slr80\n"
            "  python myanmar_asr.py --stage train --epochs 20 --fp16\n"
            "  python myanmar_asr.py --stage eval\n"
            "  python myanmar_asr.py --stage all --max_samples 500 --epochs 5\n"
        ),
    )

    p.add_argument("--stage",
                   choices=["prep", "train", "eval", "all"],
                   default="all",
                   help="Pipeline stage to run")
    p.add_argument("--check_deps", action="store_true",
                   help="Check Python dependencies and exit")

    # ── Data ──────────────────────────────────────────────────────────────
    g = p.add_argument_group("Data")
    g.add_argument("--data_dir",     default="./data/slr80",
                   help="Root directory containing wavs/ and TSV")
    g.add_argument("--tsv_file",     default="line_index_female.tsv",
                   help="TSV manifest filename inside --data_dir")
    g.add_argument("--output_dir",   default="./asr_output",
                   help="Directory for manifests, checkpoints, results")
    g.add_argument("--train_ratio",  type=float, default=0.80,
                   help="Fraction of data for training")
    g.add_argument("--dev_ratio",    type=float, default=0.10,
                   help="Fraction for dev (remainder → test)")
    g.add_argument("--max_samples",  type=int,   default=0,
                   help="Cap total samples for quick demo runs (0 = no cap)")
    g.add_argument("--max_duration", type=float, default=15.0,
                   help="Discard audio longer than this (seconds)")

    # ── Model ─────────────────────────────────────────────────────────────
    g = p.add_argument_group("Model")
    g.add_argument("--model_name",
                   default="facebook/wav2vec2-base",
                   help=(
                       "HuggingFace model to fine-tune.\n"
                       "Good alternatives:\n"
                       "  facebook/wav2vec2-large  (better, needs more GPU)\n"
                       "  facebook/wav2vec2-xls-r-300m  (best for low-resource)"
                   ))
    g.add_argument("--freeze_feature_encoder",
                   action="store_true", default=True,
                   help="Freeze CNN feature encoder (recommended for < 10h data)")
    g.add_argument("--no_freeze_feature_encoder",
                   dest="freeze_feature_encoder", action="store_false",
                   help="Unfreeze CNN encoder (more params, needs more data)")

    # ── Training hyperparameters ──────────────────────────────────────────
    g = p.add_argument_group("Training hyperparameters")
    g.add_argument("--epochs",       type=int,   default=20,
                   help="Number of training epochs")
    g.add_argument("--lr",           type=float, default=1e-4,
                   help="Peak AdamW learning rate")
    g.add_argument("--batch_size",   type=int,   default=8,
                   help="Per-device training batch size")
    g.add_argument("--eval_batch_size", type=int, default=8,
                   help="Per-device evaluation batch size")
    g.add_argument("--grad_accum",   type=int,   default=2,
                   help="Gradient accumulation steps "
                        "(effective batch = batch_size × grad_accum)")
    g.add_argument("--warmup_ratio", type=float, default=0.10,
                   help="Fraction of total steps used for LR warmup")
    g.add_argument("--weight_decay", type=float, default=0.005,
                   help="AdamW weight decay")
    g.add_argument("--fp16",         action="store_true", default=False,
                   help="FP16 mixed-precision training (requires CUDA GPU)")
    g.add_argument("--early_stop_patience", type=int, default=3,
                   help="Stop if CER does not improve for N eval rounds "
                        "(0 = disabled)")
    g.add_argument("--num_workers",  type=int,   default=2,
                   help="DataLoader worker threads")

    # ── Logging & output ──────────────────────────────────────────────────
    g = p.add_argument_group("Logging and output")
    g.add_argument("--log_file",    default=None,
                   help="Optional log file (stdout always used)")
    g.add_argument("--log_steps",   type=int, default=10,
                   help="Log training metrics every N steps")
    g.add_argument("--tensorboard", action="store_true", default=False,
                   help="Enable TensorBoard logging")
    g.add_argument("--verbose",     action="store_true", default=False,
                   help="Enable DEBUG-level logging "
                        "(shows per-file errors and sample decoding)")
    g.add_argument("--seed",        type=int, default=42,
                   help="Random seed for reproducibility")

    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    logger = setup_logging(args.log_file, args.verbose)

    logger.info("Myanmar ASR Tutorial  —  SLR80 Burmese Speech Corpus")

    if args.check_deps:
        logger.info("Checking dependencies ...")
        ok = check_dependencies(logger)
        logger.info("All dependencies OK ✓" if ok else "Missing packages — see above.")
        sys.exit(0 if ok else 1)

    logger.info(f"Stage: {args.stage}   Seed: {args.seed}")

    # Seed everything for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    if args.stage in ("prep", "all"):
        logger.info("── Stage: Data Preparation ─────────────────────────────")
        prepare_data(
            data_dir    = args.data_dir,
            tsv_file    = args.tsv_file,
            output_dir  = args.output_dir,
            train_ratio = args.train_ratio,
            dev_ratio   = args.dev_ratio,
            seed        = args.seed,
            max_samples = args.max_samples,
            logger      = logger,
        )

    if args.stage in ("train", "all"):
        logger.info("── Stage: Training ─────────────────────────────────────")
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
        except ImportError as e:
            logger.error(
                f"Missing: {e}\n"
                "  pip install torch torchaudio transformers datasets soundfile"
            )
            sys.exit(1)
        run_train(args, logger)

    if args.stage in ("eval", "all"):
        logger.info("── Stage: Evaluation ───────────────────────────────────")
        run_eval(args, logger)

    logger.info("All stages complete.")


if __name__ == "__main__":
    main()
