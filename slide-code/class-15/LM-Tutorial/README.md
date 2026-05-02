# LM-Tutorial

A comprehensive, hands-on language modeling tutorial covering the full spectrum from classical statistical models to modern neural approaches and Retrieval-Augmented Generation (RAG). Prepared by **Ye Kyaw Thu**, Language Understanding Lab., Myanmar, for the AI Fundamental Class.

---

## Overview

This tutorial walks through the theory and practice of language modeling in a single self-contained Jupyter Notebook (`LM-Tutorial.ipynb`). Students build, evaluate, and compare four families of language models on real Myanmar-language data, gaining both conceptual understanding and practical experience with industry-standard tools.

| Family | Tools / Models |
|---|---|
| Statistical N-gram LM | KenLM (`lmplz`), SRILM (`ngram-count`) |
| Neural LM — RNN | PyTorch LSTM |
| Neural LM — Transformer | XGLM (fine-tuned via HuggingFace `transformers`) |
| Retrieval-Augmented Generation | Sentence-Transformers encoder + causal LLM |

A key pedagogical highlight is the **cross-model evaluation** section, which demonstrates why raw perplexity scores cannot be directly compared across different tokenization schemes (word-level vs. subword), and introduces **Bits-Per-Character (BPC)** as a fair, unified metric.

---

## Tutorial Contents

### 1. Statistical Language Models

**KenLM** — A fast, memory-efficient n-gram toolkit. The tutorial starts with a small English toy corpus to build intuition for ARPA format, smoothing, and perplexity calculation, then scales up to a 5-gram model trained on the Myanmar myPOS corpus. Key topics include the `lmplz` pipeline, ARPA file structure (log₁₀ probabilities, backoff weights), and converting to binary format for fast inference.

**SRILM** — A research-grade statistical LM toolkit. Installation from source is documented step by step (including the `tcsh` dependency and Makefile path configuration). The tutorial explains why Kneser-Ney smoothing fails on small datasets (division by zero in count-of-counts), demonstrates Witten-Bell smoothing as a robust fallback, and uses `-debug 2` to trace exactly how SRILM computes perplexity word by word. A full 5-gram KN model is then trained on the Myanmar corpus for final comparison.

### 2. Neural Language Models

**LSTM** (`lstm/lstm_lm.py`) — A word-level LSTM language model implemented in PyTorch. Training, evaluation, and text generation scripts are provided in `lstm/`. The model is trained on the Myanmar myPOS corpus and evaluated using cross-entropy loss and perplexity.

**Transformer / XGLM** (`transformer/transformer_lm.py`) — Fine-tuning of Facebook's XGLM model (a multilingual causal transformer) on the Myanmar corpus using HuggingFace `transformers`. The `transformer/` folder contains separate scripts for base training, optimized training, closed-vocabulary evaluation, and generation.

### 3. Retrieval-Augmented Generation (RAG)

Two RAG variants are demonstrated entirely within the notebook:

- **Passage RAG** — A document corpus is embedded with a Sentence-Transformer encoder; at query time, the top-k most similar passages are retrieved by cosine similarity and injected into a generative prompt.
- **Q&A RAG** — A structured question-answer database is embedded as `"Q: … A: …"` pairs. The retriever finds the closest match, and a strict generative prompt extracts the exact answer, with explicit handling for out-of-domain queries.

Both demos use Myanmar-language prompts and highlight how RAG reduces hallucinations by grounding generation in retrieved evidence.

### 4. Cross-Model Evaluation

The final section unifies all four model families in a single evaluation run against the same held-out test set (`data/otest.word.clean`). Because KenLM scores words, LSTM scores words, and XGLM scores subword tokens, perplexity is **not** directly comparable. The notebook derives BPC for each model, enabling an apples-to-apples comparison reported in a consolidated results table (PPL, Entropy in nats, BPC).

---

## Repository Structure

```
LM-Tutorial/
├── data/                        # Corpora and preprocessing
│   ├── mypos_v3.word            # myPOS v3 corpus (raw)
│   ├── mypos_v3.word.clean      # Cleaned training corpus
│   ├── otest.word / .clean      # Out-of-domain test set
│   ├── 10k_test.txt / .clean    # 10 k-sentence test set
│   └── clean_text.py            # Text cleaning script
├── lm_toy.*                     # Toy corpus files (train/val/test splits, vocab, ARPA)
├── kenlm/
│   └── eval_kenlm.py            # KenLM PPL evaluation script
├── srilm/
│   ├── myanmar_srilm.arpa       # SRILM 5-gram model (ARPA)
│   └── myanmar_srilm.binary     # KenLM-binary version for fast loading
├── lstm/
│   ├── lstm_lm.py               # LSTM model definition & dataset class
│   ├── train.sh                 # Training script
│   ├── eval.sh / eval-closed.sh # Evaluation scripts
│   └── gen.sh                   # Text generation script
├── transformer/
│   ├── transformer_lm.py        # XGLM fine-tuning & evaluation code
│   ├── train.sh / train_optimize.sh
│   ├── eval.sh / eval-base.sh / eval-closed.sh
│   ├── eval-optimize.sh / eval-optimize-10k.sh
│   └── gen.sh / gen-optimize.sh
├── pdf/                         # LaTeX source and compiled PDF
│   ├── LM-Tutorial-edit.tex
│   └── LM-Tutorial-edit.pdf
├── LM-Tutorial.ipynb            # Main tutorial notebook
├── general.txt / general.arpa   # General-domain corpus and LM
├── general.log / domain.log     # Training logs
└── README.md
```

> **Note:** Trained LSTM and Transformer model weight files (`.pt`, HuggingFace checkpoints) are **not** included due to GitHub's file-size limits. All code and data required to reproduce training from scratch are provided. The tutorial notebook is fully self-contained and can be run end-to-end.

A pre-compiled PDF version of the notebook (`pdf/LM-Tutorial-edit.pdf`) is included for convenient offline reading without requiring a Jupyter environment.

---

## Requirements

- Python ≥ 3.9, PyTorch (CUDA recommended for LSTM/Transformer sections)
- `kenlm` Python binding, `transformers`, `sentence-transformers`, `pandas`, `numpy`
- **KenLM** binaries (`lmplz`, `build_binary`) — see [https://github.com/kpu/kenlm](https://github.com/kpu/kenlm)
- **SRILM** compiled from source — see [https://github.com/BitSpeech/SRILM](https://github.com/BitSpeech/SRILM) (requires `tcsh` / `gcc` / `make`)

The notebook includes step-by-step installation notes for both SRILM and KenLM, including common pitfalls (missing `csh`, PATH configuration, Makefile edits).

---

## Getting Started

```bash
git clone https://github.com/ye-kyaw-thu/AIE-F
cd LM-Tutorial
# Install Python dependencies, then launch:
jupyter notebook LM-Tutorial.ipynb
```

Run cells in order from top to bottom. Each section is self-contained and includes explanatory comments in both English and Myanmar.

---

## License

- **Code** (scripts, notebooks): [MIT License](LICENSE)
- **myPOS v3 corpus** (`data/mypos_v3.*`): subject to the original myPOS license — see [https://github.com/ye-kyaw-thu/myPOS](https://github.com/ye-kyaw-thu/myPOS)

---

## References

- SRILM: http://www.speech.sri.com/projects/srilm/
- KenLM: https://github.com/kpu/kenlm
- myPOS Corpus v3: https://github.com/ye-kyaw-thu/myPOS
- XGLM (HuggingFace): https://huggingface.co/facebook/xglm-564M
- Hagiwara, M. — *Training an N-gram Language Model and Estimating Sentence Probability*: https://masatohagiwara.net/training-an-n-gram-language-model-and-estimating-sentence-probability.html
