#!/usr/bin/env python3

"""This module evaluates a trained BiLSTM model on a labeled dataset.

This module depends on:
- src/prep_data.py
- src/model.py
"""

import pandas as pd
import torch

from src.model import EmotionalBiLSTM
from src.prep_data import drop_invalid_supervised_rows, encode_texts


# function to load checkpoint
def _torch_load_checkpoint(path: str):
    """
    Pytorch 2.6+ defaults weights_only=True; full training checkpoints need False (trusted local files).
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# function to load model and preprocessing artifacts from checkpoint
def load_inference_bundle(checkpoint_path):

    # load details from checkpoint
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    word2id = checkpoint["vocab"]
    state_dict = checkpoint["state"]
    id2label = checkpoint["id2label"]
    max_len = checkpoint.get("max_len", 50)
    use_char_ngrams = checkpoint.get("use_char_ngrams", False)
    ngram_min = checkpoint.get("ngram_min", 2)
    ngram_max = checkpoint.get("ngram_max", 3)
    output_dim = checkpoint.get("output_dim", len(id2label))
    use_attention = checkpoint.get("use_attention", False)
    embed_dim = checkpoint.get("embed_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    num_layers = checkpoint.get("num_layers", 1)
    dropout = checkpoint.get("dropout", 0.2)
    pad_idx = checkpoint.get("pad_idx", 0)

    # initialize model with details from checkpoint
    model = EmotionalBiLSTM(
        vocab_size=len(word2id),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_attention=use_attention,
        pad_idx=pad_idx,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model, word2id, id2label, max_len, use_char_ngrams, ngram_min, ngram_max


# function to predict labels and confidences for one or many texts
def predict_texts(
    model,
    word2id,
    id2label,
    max_len,
    texts,
    stopwords_path,
    use_char_ngrams: bool = False,
    ngram_min: int = 2,
    ngram_max: int = 3,
):
    x, lengths = encode_texts(
        texts,
        word2id,
        max_len=max_len,
        stopwords_path=stopwords_path,
        use_char_ngrams=use_char_ngrams,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=1)
        pred_ids = probs.argmax(dim=1).tolist()
        pred_scores = probs.max(dim=1).values.tolist()

    pred_labels = [id2label[i] for i in pred_ids]
    return pred_ids, pred_labels, pred_scores


# function to run evaluation on a dataset (see group2-hybrid-eliza.py --mode eval)
def run_eval(
    checkpoint_path,                # CLI arg passed with --checkpoint_path
    data_csv,                       # CLI arg passed with --data_path
    batch_size,                     # CLI arg passed with --batch_size
    stopwords_path="../data/stopwords.txt",  # CLI arg passed with --stopwords_path
    text_col="text",                # CLI arg passed with --text_col
    label_col="label",              # CLI arg passed with --label_col
):
    # load model and preprocessing artifacts from checkpoint
    model, word2id, id2label, max_len, use_char_ngrams, ngram_min, ngram_max = (
        load_inference_bundle(checkpoint_path)
    )

    # load data
    df = pd.read_csv(data_csv)
    df = drop_invalid_supervised_rows(df, text_col, label_col)
    if len(df) == 0:
        raise ValueError("no rows left after removing missing or invalid labels/text")
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    # initialize counters
    correct = 0
    total = 0

    # evaluate model in batches
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            # encode labels and run shared prediction helper
            y = torch.tensor(batch_labels, dtype=torch.long)
            pred_ids, _, _ = predict_texts(
                model=model,
                word2id=word2id,
                id2label=id2label,
                max_len=max_len,
                texts=batch_texts,
                stopwords_path=stopwords_path,
                use_char_ngrams=use_char_ngrams,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
            )
            pred = torch.tensor(pred_ids, dtype=torch.long)

            # update counters
            total += y.size(0)
            correct += (pred == y).sum().item()

    # calculate accuracy
    acc = correct / total if total else 0.0
    print(f"accuracy: {acc:.2%} ({correct}/{total})")


# function to run evaluation with default values
def main():
    run_eval(
        ## CHANGE HERE: change default values
        checkpoint_path="../checkpoints/BiLSTM_model.pth",
        data_csv="../data/merged/Combined.csv",
        batch_size=32,
    )

# run the script
if __name__ == "__main__":
    main()