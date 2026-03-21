#!/usr/bin/env python3

"""This module runs interactive chat-time emotion inference with a trained model.

This module depends on:
- scripts/eval.py
"""

import random

from scripts.eval import load_inference_bundle, predict_texts
from src.eliza import Eliza
from typing import Literal  


# function to run interactive chat inference (see group2-hybrid-eliza.py --mode chat)
def run_chat(
    checkpoint_path,                         # CLI arg passed with --checkpoint_path
    stopwords_path="../data/stopwords.txt",  # CLI arg passed with --stopwords_path
    language: Literal["mm", "en"] = "mm",    # CLI arg passed with --language (choices: "mm", "en")
):
    # load model and preprocessing artifacts from checkpoint
    (
        model,
        word2id,
        id2label,
        max_len,
        use_char_ngrams,
        ngram_min,
        ngram_max,
    ) = load_inference_bundle(checkpoint_path)
    eliza = Eliza(language=language)

    while True:
        try:
            text = input("You: ").strip()
        except KeyboardInterrupt:
            break

        # skip empty input
        if not text:
            continue

        # quit on quit commands (same normalization as eliza rules)
        if eliza.is_quit(text):
            print(f"Eliza: {random.choice(eliza.script['finals'])}")
            break

        # predict emotion for input text
        pred_ids, pred_labels, pred_scores = predict_texts(
            model=model,
            word2id=word2id,
            id2label=id2label,
            max_len=max_len,
            texts=[text],
            stopwords_path=stopwords_path,
            use_char_ngrams=use_char_ngrams,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )

        # get predicted emotion and confidence
        idx = pred_ids[0]
        label = pred_labels[0]
        score = float(pred_scores[0])

        print(f"Predicted emotion: {label} ({score:.2%})")
        print(f"Eliza: {eliza.respond(text)}")


# function to run chat with default values
def main():
    run_chat(checkpoint_path="../checkpoints/BiLSTM_model.pth",
             language="mm")

# run the script
if __name__ == "__main__":
    main()