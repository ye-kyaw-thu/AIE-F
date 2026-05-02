#!/bin/bash
# Added --top_k and --top_p (Nucleus sampling) - standard practices for modern LLMs

python transformer_lm.py --mode generate \
                  --model_path ./myanmar_xglm_word_optimize \
                  --prompt "ပြည်ထောင်စု" \
                  --gen_length 200 \
                  --temperature 0.7 \
                  --top_k 50 \
                  --top_p 0.95
