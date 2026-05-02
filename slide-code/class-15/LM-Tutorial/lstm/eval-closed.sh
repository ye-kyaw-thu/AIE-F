#!/bin/bash

time python lstm_lm.py --mode test \
                  --test_file ../data/10k_test.txt.clean \
                  --model_path ./myanmar_word_lm.pt \
                  --token_level word \
                  --seq_len 50
