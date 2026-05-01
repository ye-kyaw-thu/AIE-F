"""Classical Segmental HMM for Myanmar handwritten syllable recognition.

This module is designed for:
- low-cost CPU / Raspberry Pi deployment
- weak supervision: strokes -> char_array (no explicit stroke->char alignment)
- classical training with Viterbi re-estimation
- top-k syllable decoding with scores

Main idea
---------
1. Learn a pseudo-stroke tokenizer from stroke embeddings (MiniBatchKMeans).
2. Convert each stroke into top-k pseudo-token candidates with costs.
3. Train a segmental character model:
      char_array token can consume 0..max_segment_len strokes
   using Viterbi training.
4. Decode a query sample by scoring candidate syllables' char_array and
   returning top-k syllables with scores.

Optional
--------
You can optionally use prototype.py to produce a shortlist of candidate
syllables and pass them in at decode time for faster inference.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
from collections import defaultdict, Counter
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from typing import List



# ============================================================
# Configuration
# ============================================================

@dataclass
class SegmentalHMMConfig:
    # pseudo-stroke tokenizer
    num_pseudo_tokens: int = 96
    top_k_token_candidates: int = 5
    token_temperature: float = 0.15
    kmeans_batch_size: int = 4096
    random_state: int = 42

    # segment model
    max_segment_len: int = 3
    allow_zero_len_chars: bool = True

    # training
    num_viterbi_iters: int = 5
    add_k_token: float = 0.25
    add_k_len: float = 0.25

    # scoring weights
    length_weight: float = 0.75
    token_cost_weight: float = 1.0

    # decode
    decode_top_k: int = 10

    # safety
    min_char_count_warn: int = 1


# ============================================================
# Basic helpers
# ============================================================

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(norms, eps)).astype(np.float32)


def _l2_normalize_vec(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x)
    if n < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / n).astype(np.float32)


def _logsumexp(vals: Sequence[float]) -> float:
    vals = np.asarray(list(vals), dtype=np.float64)
    if len(vals) == 0:
        return -np.inf
    m = np.max(vals)
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(vals - m))))


def _safe_log(x: float, eps: float = 1e-12) -> float:
    return float(np.log(max(float(x), eps)))


def _safe_softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-8)
    x = x - np.max(x)
    ex = np.exp(x)
    return (ex / max(np.sum(ex), 1e-12)).astype(np.float32)



def _ensure_char_array(
    row: pd.Series,
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
) -> List[str]:
    if char_array_col in row.index and isinstance(row.get(char_array_col, None), (list, tuple)):
        return [str(x) for x in row[char_array_col]]
    return list(str(row[syllable_col]))



def _group_stroke_rows(
    emb_df: pd.DataFrame,
    *,
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
) -> Dict[object, pd.DataFrame]:
    out = {}
    for sample_index, grp in emb_df.groupby(sample_index_col, sort=False):
        out[sample_index] = grp.sort_values(stroke_index_col).reset_index(drop=True)
    return out


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


# ============================================================
# Pseudo-stroke tokenizer
# ============================================================

def fit_pseudo_stroke_tokenizer(
    emb_df: pd.DataFrame,
    *,
    emb_col: str = "embedding",
    config: SegmentalHMMConfig | None = None,
) -> Dict[str, object]:
    """
    Learn a pseudo-stroke vocabulary from stroke embeddings using MiniBatchKMeans.
    """
    cfg = config or SegmentalHMMConfig()
    X = np.vstack(emb_df[emb_col].values).astype(np.float32)
    Xn = _l2_normalize_rows(X)

    km = MiniBatchKMeans(
        n_clusters=int(cfg.num_pseudo_tokens),
        random_state=int(cfg.random_state),
        batch_size=int(cfg.kmeans_batch_size),
        n_init=10,
    )
    km.fit(Xn)

    centers = _l2_normalize_rows(km.cluster_centers_.astype(np.float32))
    return {
        "config": asdict(cfg),
        "kmeans": km,
        "centers": centers,
        "num_tokens": int(centers.shape[0]),
        "embedding_dim": int(centers.shape[1]),
    }


def stroke_topk_token_candidates(
    emb: np.ndarray,
    tokenizer: Dict[str, object],
    *,
    top_k: Optional[int] = None,
    temperature: Optional[float] = None,
) -> List[Tuple[int, float]]:
    """
    Convert one stroke embedding into top-k pseudo-token candidates with costs.

    Returns:
        [(token_id, cost), ...]
    where lower cost is better.
    """
    cfg_dict = tokenizer["config"]
    centers = tokenizer["centers"]

    if top_k is None:
        top_k = int(cfg_dict["top_k_token_candidates"])
    if temperature is None:
        temperature = float(cfg_dict["token_temperature"])

    x = _l2_normalize_vec(np.asarray(emb, dtype=np.float32))
    sims = centers @ x
    order = np.argsort(-sims)[:int(top_k)]

    top_sims = sims[order]
    probs = _safe_softmax(top_sims, temperature=float(temperature))
    costs = -np.log(np.maximum(probs, 1e-12))

    out = [(int(tok), float(cost)) for tok, cost in zip(order, costs)]
    return out


def build_token_lattice_from_emb_df(
    emb_df: pd.DataFrame,
    tokenizer: Dict[str, object],
    *,
    emb_col: str = "embedding",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
) -> Dict[object, Dict[str, object]]:
    """
    Build per-sample token candidate lattices from stroke embeddings.

    Output:
        {
            sample_index: {
                "syllable": ...,
                "char_array": [...],
                "token_lattice": [
                    [(tok, cost), ...],   # stroke 0
                    [(tok, cost), ...],   # stroke 1
                    ...
                ],
            }
        }
    """
    groups = _group_stroke_rows(
        emb_df,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
    )

    out: Dict[object, Dict[str, object]] = {}
    for sample_index, grp in groups.items():
        token_lattice = []
        for _, row in grp.iterrows():
            token_lattice.append(
                stroke_topk_token_candidates(
                    np.asarray(row[emb_col], dtype=np.float32),
                    tokenizer,
                )
            )

        first = grp.iloc[0]
        syllable = str(first[syllable_col]) if syllable_col in grp.columns else ""
        char_array = _ensure_char_array(first, syllable_col=syllable_col, char_array_col=char_array_col)

        out[sample_index] = {
            "sample_index": sample_index,
            "syllable": syllable,
            "char_array": char_array,
            "token_lattice": token_lattice,
        }

    return out


# ============================================================
# Model initialization
# ============================================================

def _build_char_vocab(
    emb_df: pd.DataFrame,
    *,
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
) -> Dict[str, object]:
    chars = []
    lexicon = {}

    for _, row in emb_df.iterrows():
        syll = str(row[syllable_col])
        arr = _ensure_char_array(row, syllable_col=syllable_col, char_array_col=char_array_col)
        chars.extend(arr)
        if syll not in lexicon:
            lexicon[syll] = arr

    uniq = sorted(set(chars))
    char_to_id = {c: i for i, c in enumerate(uniq)}
    id_to_char = {i: c for c, i in char_to_id.items()}

    return {
        "chars": uniq,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "lexicon": lexicon,
    }


def _token_probs_from_candidates(cands: Sequence[Tuple[int, float]], num_tokens: int) -> np.ndarray:
    """
    Convert [(tok, cost), ...] into a sparse probability vector over tokens.
    """
    q = np.zeros(num_tokens, dtype=np.float32)
    if len(cands) == 0:
        return q
    toks = [int(t) for t, _ in cands]
    costs = np.asarray([float(c) for _, c in cands], dtype=np.float64)
    probs = np.exp(-costs)
    probs = probs / max(np.sum(probs), 1e-12)
    for t, p in zip(toks, probs):
        q[int(t)] += float(p)
    return q


def _equal_segmentation_lengths(n_strokes: int, m_chars: int) -> List:
    """
    Very simple initialization:
    distribute strokes roughly evenly across chars, allowing zeros if n<m.
    """
    if m_chars <= 0:
        return []

    base = n_strokes // m_chars
    rem = n_strokes % m_chars
    lens = [base] * m_chars
    for i in range(rem):
        lens[i] += 1
    return lens


def _init_counts_from_equal_alignment(
    train_samples: Dict[object, Dict[str, object]],
    char_to_id: Dict[str, int],
    num_tokens: int,
    cfg: SegmentalHMMConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize token->char and length counts using naive equal segmentation.
    """
    C = len(char_to_id)
    L = int(cfg.max_segment_len)

    token_counts = np.full((C, num_tokens), float(cfg.add_k_token), dtype=np.float64)
    len_counts = np.full((C, L + 1), float(cfg.add_k_len), dtype=np.float64)

    for item in train_samples.values():
        chars = item["char_array"]
        lattice = item["token_lattice"]
        N = len(lattice)
        M = len(chars)

        if M == 0:
            continue

        lens = _equal_segmentation_lengths(N, M)
        pos = 0
        for ch, seg_len in zip(chars, lens):
            cid = char_to_id[ch]
            seg_len = int(min(max(seg_len, 0), L))
            len_counts[cid, seg_len] += 1.0

            for i in range(pos, min(pos + seg_len, N)):
                q = _token_probs_from_candidates(lattice[i], num_tokens)
                token_counts[cid] += q
            pos += seg_len

    return token_counts, len_counts


def _normalize_counts_to_logprobs(counts: np.ndarray) -> np.ndarray:
    probs = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
    return np.log(np.maximum(probs, 1e-12)).astype(np.float32)


# ============================================================
# Segmental scoring / Viterbi
# ============================================================

def _precompute_stroke_char_scores(
    token_lattice: Sequence[Sequence[Tuple[int, float]]],
    token_logprob: np.ndarray,
    *,
    token_cost_weight: float = 1.0,
) -> np.ndarray:
    """
    Precompute score of assigning each individual stroke position to each char.

    stroke_char_score[i, c] =
        log sum_tok exp( log P(tok|char_c) - token_cost_weight * cost(tok|stroke_i) )
    """
    N = len(token_lattice)
    C = token_logprob.shape[0]
    out = np.full((N, C), -np.inf, dtype=np.float32)

    for i, cands in enumerate(token_lattice):
        if len(cands) == 0:
            continue
        toks = [int(t) for t, _ in cands]
        costs = [float(c) for _, c in cands]

        for c in range(C):
            vals = [float(token_logprob[c, tok] - token_cost_weight * cost) for tok, cost in zip(toks, costs)]
            out[i, c] = float(_logsumexp(vals))

    return out


def _segment_score(
    stroke_char_scores: np.ndarray,
    len_logprob: np.ndarray,
    char_id: int,
    start: int,
    end: int,
    *,
    length_weight: float = 1.0,
) -> float:
    seg_len = int(end - start)

    # bounds check
    if seg_len < 0 or seg_len >= len_logprob.shape[1]:
        return -np.inf

    score = float(length_weight * len_logprob[char_id, seg_len])
    if seg_len > 0:
        score += float(np.sum(stroke_char_scores[start:end, char_id]))

    return score



def _viterbi_align(
    token_lattice: Sequence[Sequence[Tuple[int, float]]],
    char_ids: Sequence[int],
    token_logprob: np.ndarray,
    len_logprob: np.ndarray,
    cfg: SegmentalHMMConfig,
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Viterbi alignment of a token lattice to a fixed char sequence.

    Returns:
        best_score, segments
    where segments = [(start0,end0), (start1,end1), ...] aligned to char_ids
    """
    N = len(token_lattice)
    M = len(char_ids)
    L = int(cfg.max_segment_len)
    allow_zero = bool(cfg.allow_zero_len_chars)

    stroke_char_scores = _precompute_stroke_char_scores(
        token_lattice,
        token_logprob,
        token_cost_weight=float(cfg.token_cost_weight),
    )

    dp = np.full((M + 1, N + 1), -np.inf, dtype=np.float32)
    bp = [[None for _ in range(N + 1)] for _ in range(M + 1)]
    dp[0, 0] = 0.0

    min_len = 0 if allow_zero else 1

    for j in range(M):
        cid = int(char_ids[j])
        remaining_chars = M - (j + 1)

        for i in range(N + 1):
            if not np.isfinite(dp[j, i]):
                continue

            for seg_len in range(min_len, L + 1):
                ni = i + seg_len
                if ni > N:
                    continue

                # Prune impossible future coverage
                if not allow_zero:
                    min_needed = remaining_chars * 1
                else:
                    min_needed = 0
                max_possible = remaining_chars * L
                remaining_strokes = N - ni
                if remaining_strokes < min_needed or remaining_strokes > max_possible:
                    continue

                seg_sc = _segment_score(
                    stroke_char_scores,
                    len_logprob,
                    cid,
                    i,
                    ni,
                    length_weight=float(cfg.length_weight),
                )
                cand = float(dp[j, i] + seg_sc)
                if cand > dp[j + 1, ni]:
                    dp[j + 1, ni] = cand
                    bp[j + 1][ni] = (i, ni)

    best_score = float(dp[M, N])
    if not np.isfinite(best_score):
        return -np.inf, []

    segments = []
    j, i = M, N
    while j > 0:
        prev = bp[j][i]
        if prev is None:
            return -np.inf, []
        s, e = prev
        segments.append((s, e))
        j -= 1
        i = s
    segments.reverse()
    return best_score, segments


# ============================================================
# Training
# ============================================================

def train_segmental_hmm(
    train_emb_df: pd.DataFrame,
    *,
    config: SegmentalHMMConfig | None = None,
    emb_col: str = "embedding",
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
) -> Dict[str, object]:
    """
    Train a classical Segmental HMM from stroke embeddings + char_array.

    Training:
        1. fit pseudo-stroke tokenizer
        2. build top-k token candidate lattices per stroke
        3. initialize counts with naive equal segmentation
        4. run Viterbi re-estimation
    """
    cfg = config or SegmentalHMMConfig()

    if len(train_emb_df) == 0:
        raise ValueError("train_emb_df is empty")

    # tokenizer
    tokenizer = fit_pseudo_stroke_tokenizer(
        train_emb_df,
        emb_col=emb_col,
        config=cfg,
    )

    # token lattices
    train_samples = build_token_lattice_from_emb_df(
        train_emb_df,
        tokenizer,
        emb_col=emb_col,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
        syllable_col=syllable_col,
        char_array_col=char_array_col,
    )

    # char vocab / lexicon
    vocab_info = _build_char_vocab(
        train_emb_df,
        syllable_col=syllable_col,
        char_array_col=char_array_col,
    )
    char_to_id = vocab_info["char_to_id"]
    id_to_char = vocab_info["id_to_char"]
    lexicon = vocab_info["lexicon"]

    if len(char_to_id) < cfg.min_char_count_warn:
        warnings.warn("Very small char vocabulary; check char_array preprocessing.")

    C = len(char_to_id)
    T = int(tokenizer["num_tokens"])
    L = int(cfg.max_segment_len)

    # init
    token_counts, len_counts = _init_counts_from_equal_alignment(
        train_samples,
        char_to_id=char_to_id,
        num_tokens=T,
        cfg=cfg,
    )
    token_logprob = _normalize_counts_to_logprobs(token_counts)
    len_logprob = _normalize_counts_to_logprobs(len_counts)

    # Viterbi training
    for it in range(int(cfg.num_viterbi_iters)):
        token_counts = np.full((C, T), float(cfg.add_k_token), dtype=np.float64)
        len_counts = np.full((C, L + 1), float(cfg.add_k_len), dtype=np.float64)

        num_failed = 0

        for item in train_samples.values():
            chars = item["char_array"]
            lattice = item["token_lattice"]
            char_ids = [char_to_id[ch] for ch in chars]

            best_score, segments = _viterbi_align(
                lattice,
                char_ids,
                token_logprob,
                len_logprob,
                cfg,
            )

            if not np.isfinite(best_score) or len(segments) != len(char_ids):
                num_failed += 1
                continue

            # update counts from Viterbi path
            for cid, (s, e) in zip(char_ids, segments):
                seg_len = int(e - s)
                len_counts[cid, seg_len] += 1.0

                for i in range(s, e):
                    q = _token_probs_from_candidates(lattice[i], T)
                    token_counts[cid] += q

        token_logprob = _normalize_counts_to_logprobs(token_counts)
        len_logprob = _normalize_counts_to_logprobs(len_counts)

        if num_failed > 0:
            warnings.warn(f"Viterbi iteration {it+1}: {num_failed} training samples could not be aligned.")

    model = {
        "config": asdict(cfg),
        "tokenizer": tokenizer,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "lexicon": lexicon,                    # syllable -> char_array
        "token_logprob": token_logprob,        # [num_chars, num_tokens]
        "len_logprob": len_logprob,            # [num_chars, max_seg_len+1]
    }
    return model


# ============================================================
# Shortlist helpers (optional integration with prototype.py)
# ============================================================

def build_shortlist_from_prototype_pred_df(
    pred_df: pd.DataFrame,
    *,
    sample_index_col: str = "sample_index",
    topk_col: str = "pred_topk",
    k: int = 10,
) -> Dict[object, List[str]]:
    """
    Convert prototype batch prediction dataframe into:
        {sample_index: [candidate_syllable1, ...]}
    """
    out = {}
    for _, row in pred_df.iterrows():
        sample_index = row[sample_index_col]
        vals = row[topk_col]
        if isinstance(vals, (list, tuple)):
            out[sample_index] = [str(x) for x in vals[:k]]
    return out


# ============================================================
# Decoding
# ============================================================

def score_char_array_with_segmental_hmm(
    token_lattice: Sequence[Sequence[Tuple[int, float]]],
    char_array: Sequence[str],
    model: Dict[str, object],
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Score a fixed char_array against one token lattice.
    """
    char_to_id = model["char_to_id"]
    token_logprob = model["token_logprob"]
    len_logprob = model["len_logprob"]
    cfg = SegmentalHMMConfig(**model["config"])

    # reject if unseen chars appear
    if any(ch not in char_to_id for ch in char_array):
        return -np.inf, []

    char_ids = [char_to_id[ch] for ch in char_array]
    return _viterbi_align(
        token_lattice,
        char_ids,
        token_logprob,
        len_logprob,
        cfg,
    )


def decode_one_sample_segmental_hmm(
    sample_item: Dict[str, object],
    model: Dict[str, object],
    *,
    candidate_syllables: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """
    Decode one sample and return top candidate syllables with scores.
    """
    cfg = SegmentalHMMConfig(**model["config"])
    lexicon = model["lexicon"]
    token_lattice = sample_item["token_lattice"]

    if top_k is None:
        top_k = int(cfg.decode_top_k)

    if candidate_syllables is None:
        candidate_syllables = list(lexicon.keys())

    rows = []
    for syll in candidate_syllables:
        if syll not in lexicon:
            continue
        char_array = lexicon[syll]
        score, segments = score_char_array_with_segmental_hmm(
            token_lattice,
            char_array,
            model,
        )
        rows.append({
            "syllable": str(syll),
            "char_array": list(char_array),
            "score": float(score),
            "segments": segments,
        })

    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if top_k is not None:
        out = out.head(int(top_k)).reset_index(drop=True)
    return out


def decode_segmental_hmm_dataset(
    query_emb_df: pd.DataFrame,
    model: Dict[str, object],
    *,
    emb_col: str = "embedding",
    sample_index_col: str = "sample_index",
    stroke_index_col: str = "stroke_index",
    syllable_col: str = "syllable",
    char_array_col: str = "char_array",
    shortlist_by_sample: Optional[Dict[object, Sequence[str]]] = None,
    top_k: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[object, pd.DataFrame]]:
    """
    Decode all samples in a query embedding dataframe.

    Returns:
        pred_df: one row per sample
        tables:  {sample_index: candidate_table_df}
    """
    tokenizer = model["tokenizer"]
    items = build_token_lattice_from_emb_df(
        query_emb_df,
        tokenizer,
        emb_col=emb_col,
        sample_index_col=sample_index_col,
        stroke_index_col=stroke_index_col,
        syllable_col=syllable_col,
        char_array_col=char_array_col,
    )

    pred_rows = []
    tables: Dict[object, pd.DataFrame] = {}

    for sample_index, item in items.items():
        candidates = None
        if shortlist_by_sample is not None and sample_index in shortlist_by_sample:
            candidates = shortlist_by_sample[sample_index]

        cand_df = decode_one_sample_segmental_hmm(
            item,
            model,
            candidate_syllables=candidates,
            top_k=top_k,
        )
        tables[sample_index] = cand_df

        pred_topk = cand_df["syllable"].astype(str).tolist()
        pred_scores = cand_df["score"].astype(float).tolist()
        pred_top1 = pred_topk[0] if len(pred_topk) > 0 else None
        pred_top1_chars = cand_df.iloc[0]["char_array"] if len(cand_df) > 0 else []

        pred_rows.append({
            "sample_index": sample_index,
            "true_syllable": item["syllable"],
            "true_char_array": item["char_array"],
            "pred_top1": pred_top1,
            "pred_top1_char_array": pred_top1_chars,
            "pred_topk": pred_topk,
            "topk_scores": pred_scores,
            "top1_score": pred_scores[0] if len(pred_scores) > 0 else np.nan,
            "num_query_strokes": len(item["token_lattice"]),
        })

    pred_df = pd.DataFrame(pred_rows)
    return pred_df, tables


# ============================================================
# Evaluation
# ============================================================

def evaluate_segmental_hmm_predictions(
    pred_df: pd.DataFrame,
    *,
    ks: Sequence[int] = (1, 5, 10),
) -> pd.DataFrame:
    """
    Evaluate syllable- and char-array-level accuracy.
    """
    results = []

    true_syll = pred_df["true_syllable"].astype(str).values
    pred_top1 = pred_df["pred_top1"].astype(str).values

    results.append({
        "metric": "top1_syllable_acc",
        "value": float((pred_top1 == true_syll).mean()),
    })

    for k in ks:
        hits = []
        for _, row in pred_df.iterrows():
            yt = str(row["true_syllable"])
            topk = [str(x) for x in row["pred_topk"][:k]]
            hits.append(float(yt in topk))
        results.append({
            "metric": f"top{k}_syllable_acc",
            "value": float(np.mean(hits)),
        })

    # char-array exact top1
    char_exact = []
    char_ed = []
    for _, row in pred_df.iterrows():
        ref = [str(x) for x in row["true_char_array"]]
        hyp = [str(x) for x in row["pred_top1_char_array"]]
        char_exact.append(float(ref == hyp))
        char_ed.append(float(_edit_distance(ref, hyp)))

    results.append({
        "metric": "top1_char_array_exact",
        "value": float(np.mean(char_exact)),
    })
    results.append({
        "metric": "avg_char_edit_distance_top1",
        "value": float(np.mean(char_ed)),
    })

    # mean correct rank
    ranks = []
    for _, row in pred_df.iterrows():
        yt = str(row["true_syllable"])
        topk = [str(x) for x in row["pred_topk"]]
        if yt in topk:
            ranks.append(int(topk.index(yt) + 1))
    if len(ranks) > 0:
        results.append({
            "metric": "mean_correct_rank",
            "value": float(np.mean(ranks)),
        })

    return pd.DataFrame(results)


def describe_segmental_hmm_config(config: SegmentalHMMConfig | None = None) -> Dict[str, object]:
    cfg = config or SegmentalHMMConfig()
    return asdict(cfg)


__all__ = [
    "SegmentalHMMConfig",
    "fit_pseudo_stroke_tokenizer",
    "build_token_lattice_from_emb_df",
    "train_segmental_hmm",
    "build_shortlist_from_prototype_pred_df",
    "score_char_array_with_segmental_hmm",
    "decode_one_sample_segmental_hmm",
    "decode_segmental_hmm_dataset",
    "evaluate_segmental_hmm_predictions",
    "describe_segmental_hmm_config",
]