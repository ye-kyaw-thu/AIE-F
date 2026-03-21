"""This module contains shared preprocessing helpers so train/eval/chat stay consistent.

This module depends on:
- src/preprocessing.py
- src/vocab_builder.py
"""

import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing import TextProcessor, load_stopwords
from src.vocab_builder import build_vocab, build_label_map


def drop_invalid_supervised_rows(
    df: pd.DataFrame, text_col: str, label_col: str
) -> pd.DataFrame:
    # drop rows with missing text/label or non-numeric labels (same idea as eliza_experiments dropna + astype)
    out = df.dropna(subset=[text_col, label_col]).copy()
    out[label_col] = pd.to_numeric(out[label_col], errors="coerce")
    out = out.dropna(subset=[label_col]).copy()
    if len(out) == 0:
        return out
    out[label_col] = out[label_col].astype(int)
    return out


# cache TextProcessor instances (eval/chat may call encoding repeatedly)
_processor_cache = {}


# function to fetch a TextProcessor from cache
def _get_processor(
    stopwords_path: str,
    use_char_ngrams: bool = False,
    ngram_min: int = 2,
    ngram_max: int = 3,
) -> TextProcessor:
    """
    Build a TextProcessor, then cache it to avoid re-initializing.

    This logic is shared by:
    - `encode_texts()` (online encoding for eval.py and chat.py)
    - `prepare_train_val_data()` (offline precomputation of training tensors)
    """
    key = (stopwords_path, use_char_ngrams, ngram_min, ngram_max)
    if key not in _processor_cache:
        stopwords = load_stopwords(stopwords_path)
        _processor_cache[key] = TextProcessor(
            stopwords,
            use_char_ngrams=use_char_ngrams,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
    return _processor_cache[key]


# function to convert tokens to ids, pad/truncate, and return true length for pack_padded_sequence
def _tokens_to_ids(
    tokens, word2id, max_len: int, pad_id: int = 0, unk_id: int = 1
):
    """
    Convert a token list into a fixed-length list of token ids.
    Add padding to the end of the row to `max_len`.
    Return the length of the true sequence (excluding padding).
    """
    raw = [word2id.get(t, unk_id) for t in tokens][:max_len]
    length = max(1, len(raw))
    ids = raw + [pad_id] * (max_len - len(raw))
    return ids, length


# function to convert text inputs into model-ready tensors
def encode_texts(
    texts,
    word2id,
    max_len: int,
    stopwords_path: str,
    device=None,
    use_char_ngrams: bool = False,
    ngram_min: int = 2,
    ngram_max: int = 3,
):
    """
    Used by eval.py and chat.py to encode text inputs into model-ready tensors.
    If device is specified, the tensor will be moved to the device.

    Input: texts (str or list[str]) -> normalize/tokenize/stopwords -> token ids -> truncate/pad -> tensor
    Output: (x, lengths) where x is (batch, max_len) and lengths is (batch,) for pack_padded_sequence
    """
    processor = _get_processor(
        stopwords_path,
        use_char_ngrams=use_char_ngrams,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )

    # ensure texts is a list to support both single string (used in chat.py) and list of strings (used in eval.py)
    if isinstance(texts, str):
        texts = [texts]

    # convert tokens to ids and pad/truncate
    xs = []
    lens = []
    for t in texts:
        tokens = processor.process(str(t))
        ids, length = _tokens_to_ids(tokens, word2id, max_len=max_len)
        xs.append(ids)
        lens.append(length)

    # transform into tensor
    x = torch.tensor(xs, dtype=torch.long)
    lengths = torch.tensor(lens, dtype=torch.long)

    # move to device if specified
    if device is not None:
        x = x.to(device)
        lengths = lengths.to(device)

    return x, lengths


# function to perform stratified train/val split
def _stratified_train_val_indices(labels: list[int], val_split: float, seed: int):
    """
    Per-class holdout so val contains every label when possible.
    Each label contributes to val when n>1.
    """
    grouped: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        grouped.setdefault(int(lab), []).append(idx)

    rng = random.Random(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for _label, idxs in grouped.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            split_at = 1
        else:
            val_count = max(1, int(round(n * val_split)))
            val_count = min(val_count, n - 1)
            split_at = n - val_count

        train_idx.extend(idxs[:split_at])
        val_idx.extend(idxs[split_at:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# function to prepare all offline training artifacts
def prepare_train_val_data(
    data_path: str,
    text_col: str,
    label_col: str,
    stopwords_path: str,
    seed: int,
    val_split: float,
    max_len: int,
    batch_size: int,
    max_vocab: int = 5000,  ## CHANGE HERE: vocab size cap
    tokenized_output_path: str | None = None,
    num_classes: int = 6,
    use_char_ngrams: bool = False,
    ngram_min: int = 2,
    ngram_max: int = 3,
):
    """
    Used by train.py to prepare offline training artifacts:
    - load csv or excel; require text_col and label_col
    - tokenize with TextProcessor (zawgyi handling, optional char n-grams, stopwords)
    - optionally write tokenized csv when tokenized_output_path is set
    - build word2id vocabulary (max_vocab cap)
    - encode texts to padded id sequences and per-row lengths (for pack_padded_sequence)
    - stratified train/val split (per-class holdout; shuffles use seed)
    - build TensorDatasets and DataLoaders (batched for training/validation)
    - class weights from training-label counts (num_classes; handles imbalanced classes)
    - label2id / id2label via build_label_map (fixed emotion order)
    """
    ## CHANGE HERE: add/remove supported input file extensions
    # read input file and validate required columns
    lower_path = data_path.lower()
    if lower_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    elif lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(
            f"unsupported data file format for '{data_path}'. expected .csv, .xlsx, or .xls"
        )

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"expected CSV columns '{text_col}' and '{label_col}', got: {list(df.columns)}"
        )

    n_in = len(df)
    df = drop_invalid_supervised_rows(df, text_col, label_col)
    if len(df) == 0:
        raise ValueError("no rows left after removing missing or invalid labels/text")
    n_drop = n_in - len(df)
    if n_drop:
        print(f"dropped {n_drop} row(s) with missing or invalid labels/text")

    # get the TextProcessor and tokenize
    processor = _get_processor(
        stopwords_path,
        use_char_ngrams=use_char_ngrams,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    tokenized_texts = df[text_col].apply(lambda x: processor.process(str(x)))

    ## CHANGE HERE: set tokenized_output_path=None to skip writing tokenized output
    # save tokenized dataset for debugging/inspection
    if tokenized_output_path is not None:
        df_tokenized = df.copy()
        df_tokenized["tokens"] = tokenized_texts.apply(lambda x: " ".join(x))
        df_tokenized.to_csv(tokenized_output_path, index=False)

    # build vocabulary
    word2id = build_vocab(tokenized_texts.tolist(), max_vocab=max_vocab)

    # encode labels to ids (already int after drop_invalid_supervised_rows)
    encoded_labels = df[label_col].tolist()

    # build model-ready tensors
    X = []
    lengths_list = []
    for tokens in tokenized_texts.tolist():
        ids, length = _tokens_to_ids(tokens, word2id, max_len=max_len)
        X.append(ids)
        lengths_list.append(length)
    X = torch.tensor(X, dtype=torch.long)
    lengths = torch.tensor(lengths_list, dtype=torch.long)
    y = torch.tensor(encoded_labels, dtype=torch.long)

    # stratified train/val split
    train_idx, val_idx = _stratified_train_val_indices(encoded_labels, val_split, seed)
    ti = torch.tensor(train_idx, dtype=torch.long)
    vi = torch.tensor(val_idx, dtype=torch.long)
    train_ds = TensorDataset(X[ti], y[ti], lengths[ti])
    val_ds = TensorDataset(X[vi], y[vi], lengths[vi])

    # create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_train = len(train_ds)
    n_val = len(val_ds)
    print(
        f"data: {n_in} rows in file, {len(df)} after cleaning, "
        f"train={n_train} val={n_val} (val_split={val_split}, batch_size={batch_size})"
    )

    # compute class weights
    train_labels = y[ti].tolist()
    counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=num_classes)
    eps = 1e-8
    class_weights = (counts.sum().float() / (counts.float() + eps))
    class_weights = class_weights / class_weights.mean()

    # build label map
    label2id, id2label = build_label_map()

    return train_loader, val_loader, train_ds, val_ds, word2id, id2label, label2id, class_weights