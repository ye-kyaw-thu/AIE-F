#!/usr/bin/env python3

"""This module evaluates a trained BiLSTM model on a labeled dataset.

This module depends on:
- src/prep_data.py
- src/model.py
"""

import pandas as pd
import torch

from src.model import EmotionalBiLSTM
from src.prep_data import encode_texts


# function to load model and preprocessing artifacts from checkpoint
def load_inference_bundle(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    word2id = checkpoint["vocab"]
    state_dict = checkpoint["state"]
    id2label = checkpoint["id2label"]
    max_len = checkpoint.get("max_len", 50)
    use_char_ngrams = checkpoint.get("use_char_ngrams", False)
    ngram_min = checkpoint.get("ngram_min", 2)
    ngram_max = checkpoint.get("ngram_max", 3)

    model = EmotionalBiLSTM(vocab_size=len(word2id))
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
    x = encode_texts(
        texts,
        word2id,
        max_len=max_len,
        stopwords_path=stopwords_path,
        use_char_ngrams=use_char_ngrams,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
    )
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_ids = probs.argmax(dim=1).tolist()
        pred_scores = probs.max(dim=1).values.tolist()

    pred_labels = [id2label[i] for i in pred_ids]
    return pred_ids, pred_labels, pred_scores


# function to run evaluation on a dataset
def run_eval(
    checkpoint_path,    # passable from arguments in CLI
    data_csv,           # passable from arguments in CLI
    batch_size,         # passable from arguments in CLI
    stopwords_path="../data/stopwords.txt",
    text_col="text",
    label_col="label",
):
    # load model and preprocessing artifacts from checkpoint
    model, word2id, id2label, max_len, use_char_ngrams, ngram_min, ngram_max = (
        load_inference_bundle(checkpoint_path)
    )

    # load data
    df = pd.read_csv(data_csv)
    texts = df[text_col].tolist()
    labels = df[label_col].astype(int).tolist()

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