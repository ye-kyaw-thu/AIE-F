#!/usr/bin/env python3

"""This module trains the BiLSTM emotion model using prepared training/validation data.

This module depends on:
- src/prep_data.py
- src/model.py
"""

import copy
import os
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from src.model import EmotionalBiLSTM
from src.prep_data import prepare_train_val_data


# function to run model training (see group2-hybrid-eliza.py --mode train)
def run_train(
    data_path,                      # CLI arg passed with --data_path
    checkpoint_path,                # CLI arg passed with --checkpoint_path
    epochs,                         # CLI arg passed with --epochs
    batch_size,                     # CLI arg passed with --batch_size
    val_split,                      # CLI arg passed with --val_split
    max_len,                        # CLI arg passed with --max_len
    tokenized_output_path=None,     # not in wrapper; optional debug export path
    stopwords_path="../data/stopwords.txt",  # CLI arg passed with --stopwords_path
    text_col="text",                # CLI arg passed with --text_col
    label_col="label",              # CLI arg passed with --label_col
    seed=42,                        # CLI arg passed with --seed
    lr=0.001,                       # CLI arg passed with --lr
    show_shape_checks=False,        # NOT PASSABLE
    use_char_ngrams: bool = False,  # CLI arg passed with --use_char_ngrams / --no-use_char_ngrams
    ngram_min: int = 2,             # CLI arg passed with --ngram_min
    ngram_max: int = 3,             # CLI arg passed with --ngram_max
    weight_decay: float = 1e-5,     # CLI arg passed with --weight_decay
    patience: int = 4,              # CLI arg passed with --patience
    max_grad_norm: float = 1.0,     # CLI arg passed with --max_grad_norm
    use_attention: bool = True,     # CLI arg passed with --use_attention / --no-use_attention
    embed_dim: int = 512,           # CLI arg passed with --embed_dim
    hidden_dim: int = 256,          # CLI arg passed with --hidden_dim
    num_layers: int = 2,            # CLI arg passed with --num_layers
    dropout: float = 0.2,           # CLI arg passed with --dropout
    pad_idx: int = 0,               # CLI arg passed with --pad_idx
    max_vocab: int = 5000,          # CLI arg passed with --max_vocab
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
            max_vocab=max_vocab,
            tokenized_output_path=tokenized_output_path,
            use_char_ngrams=use_char_ngrams,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
        )
    )

    # check batch shapes of train and val; same as model sees each step
    xb, yb, lb = next(iter(train_loader))
    xvb, yvb, lvb = next(iter(val_loader))
    print(f"[shapes] train batch: x {tuple(xb.shape)}, y {tuple(yb.shape)}, lengths {tuple(lb.shape)}")
    print(f"[shapes] val batch: x {tuple(xvb.shape)}, y {tuple(yvb.shape)}, lengths {tuple(lvb.shape)}")

    # check full shapes of stacked train and val; high ram cost
    if show_shape_checks:
        train_X, train_y, train_l = zip(*[(x, y, l) for x, y, l in train_ds])
        train_X = torch.stack(train_X)
        train_y = torch.stack(train_y)
        train_l = torch.stack(train_l)
        val_X, val_y, val_l = zip(*[(x, y, l) for x, y, l in val_ds])
        val_X = torch.stack(val_X)
        val_y = torch.stack(val_y)
        val_l = torch.stack(val_l)
        print(f"[shapes] train_X {tuple(train_X.shape)}, train_y {tuple(train_y.shape)}, train_l {tuple(train_l.shape)}")
        print(f"[shapes] val_X {tuple(val_X.shape)}, val_y {tuple(val_y.shape)}, val_l {tuple(val_l.shape)}")

    # choose device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU device name: {torch.cuda.get_device_name(0)}, count: {torch.cuda.device_count()}")

    # initialize model (from src/model.py) with defaults
    ## defaults for model hparams match src/model.py when wrapper omits them from arguments
    model = EmotionalBiLSTM(
        vocab_size=len(word2id),
        output_dim=len(id2label),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        pad_idx=pad_idx,
        use_attention=use_attention,
    ).to(device)

    # initialize optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # compute class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    best_state = None
    best_val_acc = 0.0
    epochs_without_improvement = 0
    stopped_epoch = None

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # train model in batches
        for x, y, lengths in train_loader:
            x, y, lengths = x.to(device), y.to(device), lengths.to(device)

            optimizer.zero_grad()
            outputs = model(x, lengths)
            loss = criterion(outputs, y)
            loss.backward()

            # clip gradients
            if max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if train_loader else 0.0

        # validation
        model.eval()
        correct = 0
        total = 0

        # validate model in batches
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y, lengths = x.to(device), y.to(device), lengths.to(device)
                outputs = model(x, lengths)
                _, predicted = torch.max(outputs, 1)

                total += y.size(0)
                correct += (predicted == y).sum().item()

        # calculate accuracy
        acc = correct / total if total else 0.0

        # print epoch results
        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val Acc: {acc:.2%}"
        )

        # early stopping
        if acc > best_val_acc:
            best_val_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            if patience > 0:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    stopped_epoch = epoch + 1
                    print(
                        f"[*] Early stopping at epoch {stopped_epoch}. "
                        f"Best Val Acc: {best_val_acc:.2%}"
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    # save model & vocab (best weights when early stopping or best epoch improved)
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

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
        "best_val_acc": best_val_acc,
        "weight_decay": weight_decay,
        "patience": patience,
        "max_grad_norm": max_grad_norm,
        "stopped_epoch": stopped_epoch,
        "use_attention": model.use_attention,
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
        "output_dim": model.output_dim,
        "pad_idx": model.pad_idx,
        "max_vocab": max_vocab,
        },
        checkpoint_path,
    )

    print(f"[+] Checkpoint saved to {checkpoint_path} (best val acc: {best_val_acc:.2%})")


# function to run training with default values
def main():
    run_train(
        ## CHANGE HERE: change default values
        data_path="../data/merged/Combined.csv",
        checkpoint_path="../checkpoints/BiLSTM_model.pth",
        epochs=100,
        batch_size=32,
        val_split=0.1,
        max_len=50,
    )

# run the script
if __name__ == "__main__":
    main()