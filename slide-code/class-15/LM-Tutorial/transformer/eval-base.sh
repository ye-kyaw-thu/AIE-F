#!/bin/bash
# Evaluate the PRE-TRAINED base model WITHOUT fine-tuning

time python transformer_lm.py --mode test \
                  --test_file ../data/10k_test.txt.clean \
                  --model_path facebook/xglm-564M \
                  --base_model facebook/xglm-564M \
                  --seq_len 512 \
                  --stride 256
