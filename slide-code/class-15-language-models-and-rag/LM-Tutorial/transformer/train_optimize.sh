#!/bin/bash

# Fine-tuning XGLM with optimized parameters for smaller datasets
time python transformer_lm.py --mode train \
                  --train_file ../data/mypos_v3.word.clean \
                  --model_path ./myanmar_xglm_word_optimize \
                  --base_model facebook/xglm-564M \
                  --seq_len 512 \
                  --epochs 10 \
                  --batch_size 4 \
                  --lr 2e-5
