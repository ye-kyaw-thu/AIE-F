# Assignment 7 — Neural Machine Translation for Burmese G2P

## Overview

This project implements Neural Machine Translation (NMT) systems for Burmese
Grapheme-to-Phoneme (G2P) conversion, using the **Marian NMT** framework.
Two architectures were trained and compared: **Seq2Seq** (LSTM
encoder-decoder with attention) and **Transformer** (self-attention).

This assignment builds directly on Assignment 6 (Statistical Machine
Translation), using the same myG2P corpus and the same `ph→my` translation
direction, allowing a direct SMT vs NMT comparison.

## Folder Structure

```
AungChanNyein_Assignment7/
├── g2p-par/                          # raw data (train/dev/test, .my and .ph)
├── seq2seq/
    ├── seq2seq-baseline/                 # config.yml, train.log, valid.log, hyp.txt
    ├── seq2seq-exp1-lr/                  # LR inverse-sqrt schedule experiment
    ├── seq2seq-exp2-lr-smoothing/        # LR schedule + label smoothing + decoding fixes (BEST)
├── transformer/
    ├── transformer-baseline/             # config.yml, train.log, valid.log, hyp.txt
    ├── transformer-exp1-warmup/          # LR warmup experiment (BEST)
    ├── transformer-exp2-dropout/         # reduced dropout experiment
    ├── transformer-exp3-warmup-dropout/  # warmup + reduced dropout combined
├── Notebook/                         # Kaggle Jupyter notebooks
│   ├── compilemarian.ipynb
│   ├── assignment7-seq-seq-marian.ipynb
│   └── assignment7-transformer-v2.ipynb
├── README.md
└── requirements.txt
```

> **Note:** Trained model weight files (`model.npz`) are excluded from this
> submission due to size. Only configs, logs, and translation outputs are
> included, which is sufficient to reproduce and verify all results.

## Environment

- **Platform:** Kaggle Notebooks
- **Hardware:** 2× Tesla T4 GPU
- **OS:** Ubuntu 22.04
- **Framework:** Marian NMT v1.12.0 (compiled from source)

## Setup Process

1. **Marian compilation** (`compilemarian.ipynb`) — Marian is not available
   as a Kaggle apt package, so it was compiled from source once and saved
   as a private Kaggle dataset (`marian-nmt`) containing the `marian`,
   `marian-decoder`, and `marian-vocab` binaries. This avoids a ~15 minute
   recompile in every subsequent notebook.
2. **Data preparation** — train/dev/test `.my` and `.ph` files reused
   directly from Assignment 6's `g2p-par` corpus.
3. **Vocabulary** — built using `marian-vocab` from combined train+dev data.
   For Transformer experiments using `tied-embeddings-all`, a **shared
   combined vocabulary** (ph + my tokens together) was required, since tied
   embeddings need source and target vocabulary to be the same size.

## Experiments and Results (ph → my direction)

### Seq2Seq (LSTM)

| Experiment | Settings | Test BLEU | Best Dev BLEU |
|---|---|---|---|
| Baseline | lr=0.0001, no LR schedule, beam=12, normalize=0 | 74.74 | 76.61 |
| Exp1: LR Schedule | lr=0.0003, inv-sqrt decay @16k, beam=12, normalize=0 | 70.69 | 72.48 |
| **Exp2: LR+Smoothing+Decode** | lr=0.0003, inv-sqrt@16k, label-smoothing=0.1, beam=6, normalize=0.6, clip-norm=5 | **76.60** | **76.62** |

### Transformer (Self-Attention)

| Experiment | Settings | Test BLEU | Best Dev BLEU |
|---|---|---|---|
| Baseline | dropout=0.3, warmup=0 | 77.52 | 76.75 |
| **Exp1: LR Warmup** | dropout=0.3, warmup=4000 | 77.35 | **77.58** |
| Exp2: Dropout 0.1 | dropout=0.1, warmup=0 | 76.98 | 75.76 |
| Exp3: Warmup+Dropout | dropout=0.1, warmup=4000 | 77.09 | 76.21 |

### Best Configuration

**Transformer Exp1 (LR Warmup)** achieved the best dev BLEU (77.58),
narrowly ahead of **Seq2Seq Exp2** (76.62). Both significantly outperform
their respective unoptimized baselines/variants.

## SMT vs NMT Comparison (Assignment 6 vs 7)

| Approach | Model | ph → my BLEU |
|---|---|---|
| SMT (Assignment 6) | Moses + MGIZA baseline | 78.08 |
| SMT (Assignment 6) | Moses + MGIZA best (no pruning) | 78.58 |
| NMT (Assignment 7) | Seq2Seq best | 76.62 |
| NMT (Assignment 7) | Transformer best | 77.58 |

On this relatively small (20K sentence) G2P corpus, the phrase-based SMT
system slightly outperforms both neural approaches. This is a known pattern
for low-resource, structurally-constrained tasks like G2P, where SMT's
explicit phrase alignment can match or exceed NMT without requiring large
training data.

## Key Findings

- **LR warmup helps Transformer** — without it, the optimizer takes large
  early steps before the model has learned useful representations.
- **Flat learning rate hurts Seq2Seq** — adding an LR schedule alone
  (Exp1) without label smoothing initially looked worse; only combining it
  with label smoothing and corrected beam/normalize settings (Exp2)
  recovered and exceeded baseline performance.
- **Tied embeddings require equal vocab sizes** — a critical implementation
  detail: `tied-embeddings-all` in Transformer configs caused a hard crash
  when source (ph, 1,851 tokens) and target (my, 2,308 tokens) vocabularies
  differed in size. Fixed using one shared combined vocabulary.
- **Dropout 0.1 vs 0.3** — lower dropout did not clearly help on this
  dataset size; 0.3 baseline and warmup combination performed comparably
  or better.

## Tools Used

- **Marian NMT** v1.12.0 (compiled from source on Kaggle, x86-64 Ubuntu 22.04)
- **Kaggle Notebooks** (2× Tesla T4 GPU)
- **multi-bleu.perl** (from Moses, for BLEU scoring)
- **myG2P corpus** (same data as Assignment 6)

## References

- Marian NMT: https://github.com/marian-nmt/marian
- myG2P corpus: https://github.com/ye-kyaw-thu/myG2P
- Teacher's Seq2Seq tutorial: https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Seq2Seq-NMT-marian-ph2gp.ipynb
- Teacher's Transformer tutorial: https://github.com/ye-kyaw-thu/AIE-F/blob/main/slide-code/class-22/NMT-notebooks/Transformer-NMT-marian-ph2gp.ipynb
- Vaswani et al., "Attention Is All You Need", NeurIPS 2017

## Note

This project was done for educational purposes as an assignment for the
AI Engineering Fundamentals class taught by
[*Sayar Ye Kyaw Thu*](https://github.com/ye-kyaw-thu).
