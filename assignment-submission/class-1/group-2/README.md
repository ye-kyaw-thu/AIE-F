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
├── data/
│   ├── raw_ungrouped/              # Original team contributions (not merged)
│   ├── annotated_ungrouped/        # Cleaned team contributions (not merged)
│   ├── merged/                     # merged spreadsheets before final export
│   ├── merged_preprocessed/        # final merged CSVs used for training
│   │   ├── data_before_downsampling.csv
│   │   └── data_after_downsampling.csv
│   └── stopwords.txt               # Burmese stopword list
├── models/                         # Saved model weights and tokenizer artifacts
├── notebooks/                      # Experimental analysis and EDA
├── scripts/
│   ├── train.py                    # Model training execution
│   ├── eval.py                     # Evaluation metrics and performance analysis
│   └── chat.py                     # Live inference/testing script
├── src/
│   ├── preprocessing.py            # Burmese text normalization, tokenization, stopword removal
│   ├── rabbit.py                   # Zawgyi to Unicode conversion utilities (see Sources)
│   ├── vocab_builder.py            # vocab/token-id and label-id helpers
│   ├── prep_data.py                # shared preprocessing helpers for train/eval/chat
│   └── model.py                    # LSTM architecture and layer definitions
├── README.md                       # Project documentation and class labels
├── environment.yaml                # Environment configuration
└── requirements.txt                # Python dependencies
```

## Sources:
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer