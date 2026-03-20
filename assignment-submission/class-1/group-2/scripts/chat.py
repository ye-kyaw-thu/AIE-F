#!/usr/bin/env python3

"""This module runs interactive chat-time emotion inference with a trained model.

This module depends on:
- scripts/eval.py
"""

from scripts.eval import load_inference_bundle, predict_texts
from src.eliza import Eliza
from typing import Literal  
# function to run interactive chat inference
def run_chat(
    checkpoint_path,  # passable from wrapper CLI
    stopwords_path="../data/stopwords.txt",
    language: Literal["mm", "en"] = "mm",
):
    # load model and preprocessing artifacts from checkpoint
    model, word2id, id2label, max_len = load_inference_bundle(checkpoint_path)
    eliza = Eliza(language=language)
    
    quits = {"bye", "quit", "exit", "q"}
    while True:
        try:
            text = input("You: ").strip()
        except KeyboardInterrupt:
            break

        # skip empty input
        if not text:
            continue

        # quit on quit commands
        if text.lower() in quits:
            print(f"Eliza: {eliza.respond(text)}")
            break

        # predict emotion for input text
        pred_ids, pred_labels, pred_scores = predict_texts(
            model=model,
            word2id=word2id,
            id2label=id2label,
            max_len=max_len,
            texts=[text],
            stopwords_path=stopwords_path,
        )

        # get predicted emotion and confidence
        idx = pred_ids[0]
        label = pred_labels[0]
        score = float(pred_scores[0])

        print(f"Eliza: {eliza.respond(text)}")
        print(f"Emotion: {label} ({score:.2%})")


# function to run chat with default values
def main():
    run_chat(checkpoint_path="../checkpoints/BiLSTM_model.pth",
             language="mm")

# run the script
if __name__ == "__main__":
    main()