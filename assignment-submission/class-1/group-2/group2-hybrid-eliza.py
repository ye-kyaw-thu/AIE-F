#!/usr/bin/env python3

"""This module is the single CLI wrapper for train/eval/chat modes.

This module depends on:
- scripts/train.py
- scripts/eval.py
- scripts/chat.py
"""

import argparse

from scripts.chat import run_chat
from scripts.eval import run_eval
from scripts.train import run_train


# function to parse top-level CLI args and dispatch by mode
def main():
    # handle top-level CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "eval", "chat"],
        help="modes to run (train, eval, or chat)",
    )
    parser.add_argument("--data_path", default="../data/merged/Combined.csv")
    parser.add_argument("--checkpoint_path", default="../checkpoints/BiLSTM_model.pth")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=50)
    args = parser.parse_args()

    # run training mode
    if args.mode == "train":
        run_train(
            data_path=args.data_path,
            checkpoint_path=args.checkpoint_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            val_split=args.val_split,
            max_len=args.max_len,
        )
    # run evaluation mode
    elif args.mode == "eval":
        run_eval(
            checkpoint_path=args.checkpoint_path,
            data_csv=args.data_path,
            batch_size=args.batch_size,
        )
    # run chat mode
    else:
        run_chat(checkpoint_path=args.checkpoint_path)

# run the script
if __name__ == "__main__":
    main()