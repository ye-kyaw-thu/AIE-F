python lstm_lm.py --mode generate \
                  --model_path myanmar_word_lm.pt \
                  --token_level word \
                  --prompt "ပြည်ထောင်စု" \
                  --gen_length 50 \
                  --temperature 0.4
