#!/bin/bash
# Note the --stride 64. This gives the model context from previous sentences 
# when calculating PPL, leading to a much fairer (lower) score than stride=128.

time python transformer_lm.py --mode test \
                  --test_file ../data/otest.word.clean \
                  --model_path ./myanmar_xglm_word \
                  --seq_len 128 \
                  --stride 64
