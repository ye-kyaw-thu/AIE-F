#!/bin/bash

# Train (takes ~10 seconds on your RTX 3090)
time python lstm_lm.py --mode train \
                  --train_file ../data/mypos_v3.word.clean \
                  --model_path myanmar_word_lm.pt \
                  --token_level word \
                  --epochs 30 \
                  --seq_len 50
