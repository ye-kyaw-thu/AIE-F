#!/bin/bash

time python lstm_lm.py --mode test \
                  --test_file ../data/otest.word \
                  --model_path ./myanmar_word_lm.pt \
                  --token_level word \
                  --seq_len 50
