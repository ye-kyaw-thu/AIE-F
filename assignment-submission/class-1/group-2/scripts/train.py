#!/usr/bin/env python3

"""This module trains the BiLSTM emotion model using prepared training/validation data.

This module depends on:
- src/prep_data.py
- src/model.py
"""

import os
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import EmotionalBiLSTM
from src.prep_data import prepare_train_val_data


# function to run model training
def run_train(
    data_path,          # passable from arguments in CLI
    checkpoint_path,    # passable from arguments in CLI
    epochs,             # passable from arguments in CLI
    batch_size,         # passable from arguments in CLI
    val_split,          # passable from arguments in CLI
    max_len,            # passable from arguments in CLI
    tokenized_output_path=None,
    stopwords_path="../data/stopwords.txt",
    text_col="text",
    label_col="label",
    seed=42,
    lr=0.001,
    show_shape_checks=False,
    use_char_ngrams: bool = False,
    ngram_min: int = 2,
    ngram_max: int = 3,
):

    # set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # prepare loaders and artifacts so train.py stays minimal
    train_loader, val_loader, train_ds, val_ds, word2id, id2label, label2id, class_weights = (
        prepare_train_val_data(
            data_path=data_path,
            text_col=text_col,
            label_col=label_col,
            stopwords_path=stopwords_path,
            seed=seed,
            val_split=val_split,
            max_len=max_len,
            batch_size=batch_size,
            tokenized_output_path=tokenized_output_path,
            use_char_ngrams=use_char_ngrams,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
    )

    # check batch shapes of train and val; same as model sees each step
    xb, yb = next(iter(train_loader))
    xvb, yvb = next(iter(val_loader))
    print(f"[shapes] train batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}")
    print(f"[shapes] val batch: x {tuple(xvb.shape)}, y {tuple(yvb.shape)}")

    # check full shapes of stacked train and val; high ram cost
    if show_shape_checks:
        train_X, train_y = zip(*[(x, y) for x, y in train_ds])
        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)
        val_X, val_y = zip(*[(x, y) for x, y in val_ds])
        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)
        print(f"[shapes] train_X {tuple(train_X.shape)}, train_y {tuple(train_y.shape)}")
        print(f"[shapes] val_X {tuple(val_X.shape)}, val_y {tuple(val_y.shape)}")

    # choose device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU device name: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

    # initialize model (from src/model.py)
    model = EmotionalBiLSTM(vocab_size=len(word2id)).to(device)

    # initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # compute class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # train model in batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # validation
        model.eval()
        correct = 0
        total = 0

        # validate model in batches
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        # calculate accuracy
        acc = correct / total if total else 0.0

        # print epoch results
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Val Acc: {acc:.2%}")

    # save model & vocab
    os.makedirs("checkpoints", exist_ok=True)

    torch.save(
        {
        "state": model.state_dict(),
        "vocab": word2id,
        "label2id": label2id,
        "id2label": id2label,
        "text_col": text_col,
        "label_col": label_col,
        "max_len": max_len,
        "use_char_ngrams": use_char_ngrams,
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        },
        checkpoint_path,
    )

    print(f"[+] Checkpoint saved to {checkpoint_path}")


# function to run training with default values
def main():
    run_train(
        ## CHANGE HERE: change default values
        data_path="../data/merged/Combined.csv",
        checkpoint_path="../checkpoints/BiLSTM_model.pth",
        epochs=10,
        batch_size=32,
        val_split=0.1,
        max_len=50,
    )

# run the script
if __name__ == "__main__":
    main()