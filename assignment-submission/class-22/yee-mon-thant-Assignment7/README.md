# Grapheme-to-Phoneme (G2P) Neural Machine Translation with Marian

**Yee Mon Thant**

Phoneme-to-Myanmar-grapheme neural machine translation using the Marian NMT framework, comparing a Sequence-to-Sequence (RNN) architecture against a Transformer architecture, with baseline and improved results for both.

---

## Course Context

This project was completed as Assignment No. 7 for the **AI Engineering (Fundamental)** course by Sayar Ye Kyaw Thu, Language Understanding Lab (LU Lab.), Myanmar.

Course repository: [https://github.com/ye-kyaw-thu/AIE-F](https://github.com/ye-kyaw-thu/AIE-F)

---

## Project Overview

The task is to: (1) set up the Marian NMT framework, (2) train and evaluate a Seq2Seq model for phoneme-to-grapheme (ph→my) translation, (3) train and evaluate a Transformer model on the same task, and (4) improve on both baseline results.

**Dataset:** LU Lab.'s g2p-par corpus (the same phoneme–grapheme data used in the course's earlier SMT assignment) — Myanmar syllables paired with their phonemic transcription.

**Constraint:** Marian must be compiled from source and doesn't build on macOS, and there's no local GPU available, so the whole pipeline (build → train → evaluate → improve) was run on Google Colab instead.

---

## Repository Structure

```
yee-mon-thant_Assignment7/
├── marian_build_colab-GPU.ipynb              # Builds Marian from source on Colab, saves binaries to Google Drive
├── Seq2Seq_NMT_marian_ph2gp_COLAB.ipynb       # Seq2Seq baseline + Task 4 improvement
├── Transformer_NMT_marian_ph2gp_COLAB.ipynb   # Transformer baseline + Task 4 improvement
└── README.md
```

---

## Approach

**Marian setup (Colab workaround):** Since Marian can't be built on macOS and there's no local GPU, `marian_build_colab-GPU.ipynb` compiles Marian once from source on a Colab GPU runtime and saves the `marian` / `marian-decoder` / `marian-vocab` binaries to Google Drive. Later sessions just copy the binaries back from Drive (~5 seconds) instead of rebuilding (~40 minutes).

**Seq2Seq baseline:** Same model structure and hyperparameters as the teacher's original Seq2Seq notebook — nothing changed except the Colab setup (loading Marian from Drive, uploading the data files).

**Transformer baseline:** Same model structure and hyperparameters as the teacher's original Transformer notebook, for the same reason.

**Task 4 improvements (no local GPU means limited/unpredictable Colab GPU time, so retraining from scratch wasn't a reliable option for either model):**
- *Seq2Seq:* Tried fine-tuning the converged baseline with extra label smoothing and an LR warmup first — this made BLEU worse, since the added loss term conflicted with already-confident weights. Switched instead to decoding with a mid-training checkpoint (`model.iter20000.npz`) rather than the final model, since validation BLEU had peaked at iteration 20,000 and drifted slightly afterward (mild overfitting).
- *Transformer:* Skipped fine-tuning entirely and went straight to an ensemble decode of two checkpoints (`model.iter15000.npz`, the validation-BLEU peak, + `model.iter35000.npz`) using the teacher's original beam-size/normalize decode settings — this needs no additional training at all.

---

## Results Comparison

| Model | Teacher's Baseline | My Baseline | My Improved | Improvement Method |
|---|---|---|---|---|
| Seq2Seq | 76.98 | 77.90 | **78.47** | Decode with mid-training checkpoint (iter 20,000) instead of final model |
| Transformer | 76.40 | 76.32 | **77.25** | Ensemble of 2 checkpoints (iter 15,000 + 35,000) with tuned beam-size/normalize |

BLEU scores computed with `multi-bleu.perl` on the held-out test set. Small gaps between the teacher's baseline and mine are expected given different hardware (his machine vs. Colab's T4 GPU) despite identical hyperparameters and seed.

---

## Key Findings

**1. Checkpoint selection beats retraining under limited compute.**
Marian saves a checkpoint every `--save-freq` updates by default. For both architectures, picking the checkpoint with the best validation BLEU — rather than the final one — recovered performance lost to late-training overfitting, with zero additional GPU time.

**2. Fine-tuning a converged model is risky, not free.**
Attempting to improve the Seq2Seq baseline by resuming training with new regularization (label smoothing + warmup) made BLEU worse rather than better — the model was already converged, and the new loss term fought against settled weights.

**3. Ensembling checkpoints helps more than picking a single best one.**
For the Transformer, the single best-validation checkpoint's gain didn't reliably carry over to the test set alone, but ensembling it with a second, later checkpoint did produce a real test BLEU improvement.

**4. Marian binaries can be built once and reused indefinitely.**
Since Colab resets its filesystem every session, caching the compiled Marian binaries on Google Drive turned a 40-minute rebuild into a 5-second copy for every subsequent notebook run.

---

## How to Run

### 1. Build Marian (one-time setup)

1. Open `marian_build_colab-GPU.ipynb` on a Colab **GPU runtime**.
2. Run all cells — this builds Marian from source and saves the binaries to `MyDrive/marian-build-gpu/`.
3. Skip this step on future runs if the binaries are already saved.

### 2. Seq2Seq baseline + improvement

1. Open `Seq2Seq_NMT_marian_ph2gp_COLAB.ipynb` on a Colab GPU runtime.
2. Mount Google Drive, upload `train.my`, `train.ph`, `dev.my`, `dev.ph`, `test.my`, `test.ph` when prompted.
3. Run all cells top to bottom.

### 3. Transformer baseline + improvement

Same as above, using `Transformer_NMT_marian_ph2gp_COLAB.ipynb`.

### Expected training time (Colab T4 GPU)
- Seq2Seq: ~1.5 hours (60,000 updates, early stopping)
- Transformer: ~40 minutes (55,000 updates, early stopping)

---

## Dataset

g2p-par corpus, provided by LU Lab. as part of the course:

- **Format:** parallel `.ph` (phoneme) / `.my` (Myanmar grapheme) text files, one syllable-level example per line
- **Splits:** `train`, `dev`, `test` (train+dev combined = 22,000 lines for vocab building; test set = ~2,800 sentences / ~8,047 reference tokens)
- Not included in this repository — upload the 6 data files directly into the Colab session when prompted

---

## References

- Junczys-Dowmunt, M., et al. (2018). *Marian: Fast Neural Machine Translation in C++.* ACL 2018 (System Demonstrations). [https://github.com/marian-nmt/marian](https://github.com/marian-nmt/marian)

- Ye Kyaw Thu. AI Engineering (Fundamental) Course, LU Lab., Myanmar. [https://github.com/ye-kyaw-thu/AIE-F](https://github.com/ye-kyaw-thu/AIE-F) (Assignment design, reference notebooks, g2p-par dataset)

- Koehn, P., et al. `multi-bleu.perl`, Moses SMT Toolkit. [https://github.com/moses-smt/mosesdecoder](https://github.com/moses-smt/mosesdecoder) (BLEU evaluation script)

---

## Acknowledgements

Sayar Ye Kyaw Thu (LU Lab., Myanmar) for the course design, reference notebooks, hyperparameters, and the g2p-par dataset used throughout this assignment.
