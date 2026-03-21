# Group 2 Assignment Submission

## Dataset

Our custom dataset comprises Facebook comments, synthetic data generated via generative models, and manually authored entries. It is categorized into the following six emotional classes:
- 0: Sadness
- 1: Joy
- 2: Love
- 3: Anger
- 4: Fear
- 5: Surprise

## Project Structure

```
group-2/
├── group2-hybrid-eliza.py          # single CLI entry: --mode train|eval|chat
├── data/
│   ├── raw_ungrouped/              # original team contributions (not merged)
│   ├── annotated_ungrouped/        # cleaned team contributions (not merged)
│   ├── merged/                     # merged spreadsheets before final export
│   ├── merged_preprocessed/        # final merged CSVs used for training
│   │   ├── data_before_downsampling.csv
│   │   └── data_after_downsampling.csv
│   └── stopwords.txt               # Burmese stopword list (see Sources)
├── checkpoints/                    # saved checkpoints and tokenizer/vocab artifacts
├── notebooks/                      # experimental analysis and EDA
├── scripts/
│   ├── train.py                    # model training execution
│   ├── eval.py                     # model evaluation/prediction
│   └── chat.py                     # interactive inference; reuses eval helpers
├── src/
│   ├── preprocessing.py            # Unicode normalization, lowercase, tokenization (see Sources), char n-grams, stopwords
│   ├── rabbit.py                   # Zawgyi to Unicode conversion utilities (see Sources)
│   ├── vocab_builder.py            # vocab/token-id and label-id helpers
│   ├── prep_data.py                # shared preprocessing helpers for train/eval/chat
│   ├── eliza_rules.py              # ELIZA rule data
│   ├── eliza.py                    # ELIZA engine
│   └── model.py                    # LSTM architecture and layer definitions
├── README.md                       # documentation
├── environment.yaml                # conda environment configuration
└── requirements.txt                # dependencies
```

## Project Flow

The code is organized so a single wrapper script controls the high-level mode, while `src/` contains reusable building blocks.

- `group2-hybrid-eliza.py` (primary entry point): the top-level CLI wrapper/dispatcher. It selects one of the modes and starts the corresponding script in `scripts/`.

- `scripts/train.py`: training orchestration. It uses `src/prep_data.py` and `src/model.py`, and writes a checkpoint that bundles the trained weights with the same vocabulary and label setup used for later runs.

- `scripts/eval.py`: evaluation on a labeled dataset. It loads a checkpoint from training, uses `src/prep_data.py` and `src/model.py` with that checkpoint, and reports how well the saved setup predicts labels.

- `scripts/chat.py`: interactive inference. It follows the same emotion path as `scripts/eval.py`, and pairs that with dialogue from `src/eliza.py`.

- `src/eliza_rules.py`: rule data only, paired with `src/eliza.py`.

- `src/eliza.py`: ELIZA dialogue behavior; it consumes `src/eliza_rules.py` and shares text-pattern handling with `src/preprocessing.py` where the two modules meet.

- `src/prep_data.py`: shared data preparation for training and inference. It sits between the datasets/checkpoints and the emotion pipeline, and depends on `src/preprocessing.py` plus `src/vocab_builder.py`.

- `src/preprocessing.py`: preprocessing for the **emotion model**. It depends on `src/rabbit.py` for script conversion and on external Myanmar tokenization libraries as noted under Sources. ELIZA-specific text handling is in `src/eliza.py` instead of here.

- `src/vocab_builder.py`: builds the vocabulary and fixed label order for the six emotion classes; used from `src/prep_data.py`.

- `src/model.py`: defines the neural emotion classifier; used from `scripts/train.py` and `scripts/eval.py`.

- `src/rabbit.py`: Zawgyi-to-Unicode conversion support; used from `src/preprocessing.py`.

In short: `group2-hybrid-eliza.py` starts the run; `scripts/*.py` control train/eval/chat; `src/*.py` implements the reusable preprocessing/model utilities.

## Standalone Burmese Chat UI

You can run a separate local browser chat UI for the Burmese Hybrid ELIZA bot with:

```bash
python eliza/burmese_chat_ui.py
```

Then open `http://127.0.0.1:8765` in your browser.

Optional:

```bash
python eliza/burmese_chat_ui.py --host 0.0.0.0 --port 9000 --model_path /path/to/eliza_eq_mm_lstm.pth
```

If no `.pth` checkpoint is available yet, the UI still works in rule-based mode and continues chatting in Burmese.

## Sources

- Unicode Myanmar script blocks:
    - https://www.unicode.org/charts/PDF/U1000.pdf
    - https://www.unicode.org/charts/PDF/UAA60.pdf
- Burmese grammar: https://online.fliphtml5.com/rrlzh/mbir/#p=1
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer
