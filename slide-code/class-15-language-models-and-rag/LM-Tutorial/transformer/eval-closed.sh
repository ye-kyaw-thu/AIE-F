#!/bin/bash

time python transformer_lm.py --mode test \
                  --test_file ../data/10k_test.txt.clean \
                  --model_path ./myanmar_xglm_word \
                  --seq_len 128 \
                  --stride 64
