#!/usr/bin/env python3

"""This module is the single CLI wrapper for train/eval/chat modes.

This module depends on:
- scripts/train.py
- scripts/eval.py
- scripts/chat.py
"""

import argparse

from scripts.chat import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_STOPWORDS_PATH,
    launch_custom_ui,
    launch_streamlit_ui,
    run_chat,
)
from scripts.eval import run_eval
from scripts.train import run_train


# function to parse top-level CLI args and dispatch by mode
def main():
    # handle top-level CLI args and dispatch by mode
    parser = argparse.ArgumentParser(
        description="group-2 hybrid eliza: train (BiLSTM), eval, or chat",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "eval", "chat"],
        help="train, eval, or chat",
    )
    parser.add_argument(
        "--data_path",
        default="./data/merged/Combined.csv",
        help="csv for train; csv for eval (same flag)",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="checkpoint path for train save / eval load / chat load",
    )

    # training loop
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # data / preprocessing
    parser.add_argument("--stopwords_path", default=DEFAULT_STOPWORDS_PATH)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--max_vocab", type=int, default=5000)
    parser.add_argument(
        "--use_char_ngrams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="append char n-grams to token lists (default: on)",
    )
    parser.add_argument("--ngram_min", type=int, default=2)
    parser.add_argument("--ngram_max", type=int, default=2)

    # model (defaults match src/model.py when wrapper omits them from arguments)
    parser.add_argument(
        "--use_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="masked attention pooling after lstm (default: on)",
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pad_idx", type=int, default=0)

    # chat-only
    parser.add_argument(
        "--language",
        choices=["mm", "en"],
        default="mm",
        help="eliza script language (chat mode)",
    )
    parser.add_argument(
        "--chat_ui",
        choices=["terminal", "streamlit", "custom_ui"],
        default="custom_ui",
        help="chat mode: terminal | streamlit | custom_ui",
    )
    parser.add_argument(
        "--custom_ui_host",
        default="127.0.0.1",
        help="chat mode custom_ui only: bind address",
    )
    parser.add_argument(
        "--custom_ui_port",
        type=int,
        default=8765,
        help="chat mode custom_ui only: port",
    )

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
            stopwords_path=args.stopwords_path,
            text_col=args.text_col,
            label_col=args.label_col,
            seed=args.seed,
            lr=args.lr,
            use_char_ngrams=args.use_char_ngrams,
            ngram_min=args.ngram_min,
            ngram_max=args.ngram_max,
            weight_decay=args.weight_decay,
            patience=args.patience,
            max_grad_norm=args.max_grad_norm,
            use_attention=args.use_attention,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=args.pad_idx,
            max_vocab=args.max_vocab,
        )
    # run evaluation mode
    elif args.mode == "eval":
        run_eval(
            checkpoint_path=args.checkpoint_path,
            data_csv=args.data_path,
            batch_size=args.batch_size,
            stopwords_path=args.stopwords_path,
            text_col=args.text_col,
            label_col=args.label_col,
        )
    # run chat mode
    else:
        if args.chat_ui == "streamlit":
            launch_streamlit_ui(
                checkpoint_path=args.checkpoint_path,
                stopwords_path=args.stopwords_path,
                language=args.language,
            )
        elif args.chat_ui == "custom_ui":
            launch_custom_ui(
                checkpoint_path=args.checkpoint_path,
                stopwords_path=args.stopwords_path,
                language=args.language,
                host=args.custom_ui_host,
                port=args.custom_ui_port,
            )
        else:
            run_chat(
                checkpoint_path=args.checkpoint_path,
                stopwords_path=args.stopwords_path,
                language=args.language,
            )


# run the script
if __name__ == "__main__":
    main()
