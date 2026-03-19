"""This module contains shared preprocessing helpers so train/eval/chat stay consistent.

This module depends on:
- src/preprocessing.py
- src/vocab_builder.py
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.preprocessing import TextProcessor, load_stopwords
from src.vocab_builder import build_vocab, build_label_map


# cache TextProcessor instances per stopwords file (eval/chat may call encoding repeatedly)
_processor_cache = {}

# function to fetch a TextProcessor from cache for stopwords file
def _get_processor(stopwords_path: str) -> TextProcessor:
    """
    Build a TextProcessor using stopwords_path, then cache it to avoid re-reading stopwords and re-creating the tokenizer/detector objects.

    This logic is shared by:
    - `encode_texts()` (online encoding for eval.py and chat.py)
    - `prepare_train_val_data()` (offline precomputation of training tensors)
    """
    if stopwords_path not in _processor_cache:
        stopwords = load_stopwords(stopwords_path)
        _processor_cache[stopwords_path] = TextProcessor(stopwords)
    return _processor_cache[stopwords_path]


# function to convert tokens to ids and pad/truncate
def _tokens_to_ids(tokens, word2id, max_len: int, pad_id: int = 0, unk_id: int = 1):
    """
    Convert a token list into a fixed-length list of token ids, then pad/truncate to `max_len`.
    """
    ids = [word2id.get(t, unk_id) for t in tokens][:max_len]
    ids = ids + [pad_id] * (max_len - len(ids))
    return ids


# function to convert text inputs into model-ready tensors
def encode_texts(texts, word2id, max_len: int, stopwords_path: str, device=None):
    """
    Used by eval.py and chat.py to encode text inputs into model-ready tensors.
    If device is specified, the tensor will be moved to the device.

    Input: texts (str or list[str]) -> normalize/tokenize/stopwords -> token ids -> truncate/pad -> tensor
    Output: tensor of shape (batch, max_len) of token ids
    """
    processor = _get_processor(stopwords_path)

    # ensure texts is a list to support both single line (used in chat.py) and list of lines (used in eval.py)
    if isinstance(texts, str):
        texts = [texts]

    # convert tokens to ids and pad/truncate
    xs = []
    for t in texts:
        tokens = processor.process(str(t))
        ids = _tokens_to_ids(tokens, word2id, max_len=max_len)
        xs.append(ids)

    # transform into tensor
    x = torch.tensor(xs, dtype=torch.long)
    
    # move to device if specified
    if device is not None:
        x = x.to(device)
    
    return x  # shape: (batch, max_len)


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
    max_vocab: int = 5000,
    tokenized_output_path: str | None = None,
    num_classes: int = 6,
):
    """
    Used by train.py to prepare all offline training artifacts:
    - tokenize texts
    - build a vocabulary
    - build padded X and label tensor y
    - seed and create train/val split
    - create train/val loaders
    - compute class weights from the training split
    - build dictionaries: id2label and label2id
    """
    # read the data and validate required columns
    df = pd.read_csv(data_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"expected CSV columns '{text_col}' and '{label_col}', got: {list(df.columns)}"
        )

    # get the TextProcessor and tokenize
    processor = _get_processor(stopwords_path)
    tokenized_texts = df[text_col].apply(lambda x: processor.process(str(x)))

    # (optional) save tokenized dataset for debugging/inspection
    if tokenized_output_path is not None:
        df_tokenized = df.copy()
        df_tokenized["tokens"] = tokenized_texts.apply(lambda x: " ".join(x))
        df_tokenized.to_csv(tokenized_output_path, index=False)

    # build vocabulary
    word2id = build_vocab(tokenized_texts.tolist(), max_vocab=max_vocab)

    # encode labels to ids
    encoded_labels = df[label_col].astype(int).tolist()

    # build model-ready tensor
    X = []
    for tokens in tokenized_texts.tolist():
        X.append(_tokens_to_ids(tokens, word2id, max_len=max_len))
    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(encoded_labels, dtype=torch.long)

    # create tensor dataset
    full_ds = TensorDataset(X, y)

    # train/val sizes
    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size

    # train/val split with seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)

    # create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # compute class weights
    train_labels = y[train_ds.indices].tolist()
    counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=num_classes)
    eps = 1e-8
    class_weights = (counts.sum().float() / (counts.float() + eps))
    class_weights = class_weights / class_weights.mean()

    # build label map
    label2id, id2label = build_label_map()

    return train_loader, val_loader, train_ds, val_ds, word2id, id2label, label2id, class_weights