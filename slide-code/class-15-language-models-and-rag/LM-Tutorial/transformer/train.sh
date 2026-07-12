#!/bin/bash
# Fine-tuning XGLM on your data
# bf16 is enabled automatically in the python script for your RTX 3090

time python transformer_lm.py --mode train \
                  --train_file ../data/mypos_v3.word.clean \
                  --model_path ./myanmar_xglm_word \
                  --seq_len 128 \
                  --epochs 3 \
                  --batch_size 8
