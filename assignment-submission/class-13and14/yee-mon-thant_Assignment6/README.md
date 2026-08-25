# G2P Conversion using Phrase-Based Statistical Machine Translation

**Yee Mon Thant**

This project converts Myanmar written syllables (graphemes) to their
pronunciation (phonemes) using the Moses Phrase-Based SMT framework —
treating G2P conversion as a machine translation problem.



## Course Context

This project was done as Assignment 6 for the AI Engineering (Fundamental)
Class by Sayar Ye Kyaw Thu, Language Understanding Lab, Myanmar.
Course repository: https://github.com/ye-kyaw-thu/AIE-F

## Project Structure

```
yee-mon-thant_Assignment6/
├── Assignment6_G2P.ipynb   # main notebook (all steps with outputs)
├── data/
│   ├── train.my / train.ph          # training pairs (20,000 lines)
│   ├── dev.my / dev.ph              # dev pairs (2,000 lines)
│   └── test.my / test.ph            # test pairs (2,802 lines)
└── README.md                        # this file
```

## Steps

### 1. Prepared the data

Data is from the **myG2P corpus** prepared by Sayar Ye Kyaw Thu.
Fields 3 and 4 were extracted to form a parallel corpus of grapheme
syllables and Romanized phoneme transcriptions.

| File | Lines | Purpose |
|---|---|---|
| `train.my` / `train.ph` | 20,000 | Train the SMT model |
| `dev.my` / `dev.ph` | 2,000 | Tune model weights (MERT) |
| `test.my` / `test.ph` | 2,802 | Final evaluation only |

Sample training pair:
```
train.my:  က က တိုး
train.ph:  ka. ga- dou:
```

### 2. Set up the environment

The Moses macOS binary (`macOS.zip`) was downloaded and tested locally,
but the `lmplz` binary was killed by the OS due to memory compatibility
issues on Apple Silicon (Mac). That's why Google Colab (Ubuntu Linux) was
used as the execution environment, with the official `ubuntu-17.04.tgz`
binary from the Moses release page. GIZA++ was compiled from source on
Colab for word alignment.

### 3. Trained language models

A trigram language model (order=3) was trained using KenLM (`lmplz`) on
the target side of each training direction:

- `train.ph` → language model for my→ph direction
- `train.my` → language model for ph→my direction

The ARPA text format output was converted to binary format (`build_binary`)
for faster loading during decoding.

### 4. Ran Moses training pipeline

`train-model.perl` was used to run the full PBSMT pipeline. This script
automatically handles all training steps internally — corpus preparation,
GIZA++ word alignment, phrase extraction, phrase scoring, and generating
the final `moses.ini` decoder config file. This was done for **both
directions**: my→ph and ph→my.

### 5. Tuned model weights with MERT

The default weights in `moses.ini` are not optimized for translation
quality. MERT (Minimum Error Rate Training) was used to automatically
find the best weight combination by running the decoder repeatedly on
the dev set and maximizing BLEU, without touching the held-out test set.

### 6. Decoded test set and evaluated with BLEU

The Moses decoder was applied to the test set for both directions.
`multi-bleu.perl` was used to compute BLEU score against the gold reference.

## Results

| Direction | Config | BLEU |
|---|---|---|
| my → ph | Baseline (default weights) | 63.35 |
| my → ph | Tuned (MERT on dev set) | **69.88** |
| ph → my | Baseline (default weights) | 70.50 |
| ph → my | Tuned (MERT on dev set) | **78.77** |

## Conclusion

This project showed that phrase-based SMT can effectively solve the G2P
conversion task for Myanmar, achieving BLEU scores above 60 even with
default baseline settings. MERT tuning gave meaningful improvements in
both directions (+6 to +8 BLEU points), confirming that weight
optimization is an important step that should not be skipped.

The my→ph direction scored lower than ph→my because a single Myanmar
grapheme syllable can map to different phonemes depending on context —
for example, the syllable က can be pronounced as ka., ga-, or
ka- depending on its position in the word, making it harder for the
model to predict the correct pronunciation.

## Credits

- SMT tutorial notebooks and myG2P corpus preparation: Sayar Ye Kyaw Thu
  (https://github.com/ye-kyaw-thu/AIE-F/tree/main/slide-code/class-13and14)
- myG2P corpus: https://github.com/ye-kyaw-thu/myG2P
- Moses SMT framework: http://www.statmt.org/moses/
- GIZA++ word aligner: https://github.com/moses-smt/giza-pp
