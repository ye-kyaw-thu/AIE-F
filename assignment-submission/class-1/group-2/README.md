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
│   ├── raw_ungrouped/              # Original team contributions
│   ├── annotated_ungrouped/        # Cleaned individual datasets
│   ├── final_grouped/              # Consolidated master directory
│   │   ├── main_data.csv           # Merged source dataset
│   │   └── processed_main_data.csv # Preprocessed version (Tokenized/Segmented)
│   └── stopwords.txt               # Burmese stopword list
├── models/                         # Saved model weights and tokenizer artifacts
├── notebooks/                      # Experimental analysis and EDA
├── scripts/
│   ├── train.py                    # Model training execution
│   ├── eval.py                     # Evaluation metrics and performance analysis
│   └── chat.py                     # Live inference/testing script
├── src/
│   ├── preprocessing.py            # Burmese NLP cleaning and tokenization logic
│   └── model.py                    # LSTM architecture and layer definitions
├── README.md                       # Project documentation and class labels
├── environment.yaml                # Environment configuration
└── requirements.txt                # Python dependencies
```

## Sources:
- Rabbit Zawgyi to Unicode Converter: https://github.com/Rabbit-Converter/Rabbit-Python 
- MMDT Tokenizer: https://github.com/Myanmar-Data-Tech/mmdt-tokenizer