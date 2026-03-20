# Logic Flow of `hybrid-eliza-mm.py`

This document explains the step-by-step logic flow of `hybrid-eliza-mm.py`.

## 1. High-Level Purpose

`hybrid-eliza-mm.py` combines:

- Rule-based ELIZA-style conversation
- Emotion classification using a trained `LinearSVC`
- Language-specific handling for English (`en`) and Myanmar (`mm`)

The script can run in two modes:

- `train`: train and save the emotion classifier
- `chat`: load the saved model and start an interactive chatbot

## 2. Program Entry Flow

When the script is executed, Python reaches:

```python
if __name__ == "__main__":
    main()
```

From there, the logic is:

1. Parse command-line arguments in `main()`
2. Create a `HybridEliza` object
3. Check `args.mode`
4. If mode is `train`, run model training
5. If mode is `chat`, load the saved model and start the chatbot loop

## 3. Command-Line Arguments

`main()` defines these arguments:

- `--lang`: choose `en` or `mm`
- `--mode`: choose `chat` or `train`
- `--data`: CSV dataset path for training
- `--model_path`: custom path for saved model file
- `--epochs`: accepted but not used by the current `LinearSVC` training flow
- `--batch_size`: accepted but not used by the current `LinearSVC` training flow
- `--val_split`: validation split ratio
- `--seed`: random seed

## 4. Object Initialization Flow

When `HybridEliza(lang=args.lang, model_path=args.model_path)` is created, `__init__()` does the following:

1. Store the selected language in `self.lang`
2. Load the matching script config from `SCRIPTS`
3. Sort keyword rules by priority score in descending order
4. Choose the model file path
   - Myanmar default: `eliza_eq_mm.pkl`
   - English default: `eliza_eq.pkl`
5. Set device to CUDA if available, otherwise CPU
6. Initialize vocabulary with:
   - `<PAD>` -> `0`
   - `<UNK>` -> `1`
7. Prepare emotion label mapping (`id2label`)
8. Initialize placeholders for:
   - `label_to_idx`
   - `idx_to_label`
   - `num_classes`
   - `vectorizer`
   - `model`

## 5. Text Processing Helpers

These helper functions support both training and chatting.

### `normalize_text(text)`

Step-by-step:

1. Convert input to string
2. Strip leading and trailing spaces
3. Convert to lowercase
4. Replace punctuation with spaces
5. Collapse repeated spaces
6. Return cleaned text

### `build_char_ngrams(text, min_n=2, max_n=3)`

Step-by-step:

1. Remove spaces from text
2. Build character n-grams of length 2 and 3
3. Return the list of generated n-grams

### `tokenize_text(text, lang)`

Step-by-step:

1. Normalize the input text
2. If the result is empty, return an empty list
3. If language is Myanmar (`mm`):
   - Split by spaces when spaces exist
   - Otherwise tokenize with `MYANMAR_TOKEN_RE`
   - Add character n-grams for richer matching
4. If language is English (`en`):
   - Split on spaces
5. Return the token list

## 6. Training Mode Flow

This runs when:

```bash
python hybrid-eliza-mm.py --mode train
```

`main()` calls:

```python
eliza.train(args.data, args.epochs, 0.001, args.batch_size, args.val_split, args.seed)
```

### Training sequence inside `train(...)`

1. Set Python random seed
2. Set PyTorch random seed
3. Read the dataset CSV using `pandas`
4. Detect the label column:
   - use `label` if present
   - otherwise use `emotions`
5. Drop rows missing `text` or label
6. Convert labels to integers
7. Build vocabulary from the text data with `build_vocab(...)`
8. Build label-index mappings with `build_label_maps(...)`
9. Convert original labels into encoded class indices
10. Split the dataset into train/validation sets using `split_stratified(...)`
11. Create a `FeatureUnion` with:
   - word-level TF-IDF (`1-2` grams)
   - character-level TF-IDF (`2-5` grams)
12. Create the classifier:
   - `LinearSVC(class_weight="balanced", random_state=seed)`
13. Fit the vectorizer on training text
14. Transform validation text with the same vectorizer
15. Train the SVM model on training features
16. Compute validation accuracy
17. Save the trained artifacts to `self.model_path` with `pickle`

### Supporting methods used during training

#### `build_vocab(texts)`

1. Tokenize every text
2. Count token frequency with `Counter`
3. Keep up to 15,000 most common tokens
4. Add them to `self.word2id`

Note: this vocabulary is saved, but the active classifier uses TF-IDF + `LinearSVC`, not the PyTorch dataset/model classes.

#### `build_label_maps(labels)`

1. Collect unique labels
2. Sort them
3. Build:
   - `label_to_idx`
   - `idx_to_label`
4. Update `num_classes`

#### `split_stratified(texts, labels, val_split, seed)`

1. Group texts by label
2. Shuffle each label group using the seed
3. Compute how many items go to validation
4. Keep at least one training item per label when possible
5. Build train and validation lists
6. Shuffle each final split
7. Return:
   - `train_texts`
   - `train_labels`
   - `val_texts`
   - `val_labels`

## 7. Chat Mode Flow

This runs by default:

```bash
python hybrid-eliza-mm.py
```

or explicitly:

```bash
python hybrid-eliza-mm.py --mode chat
```

### Chat sequence inside `main()`

1. Call `eliza.load_model()`
2. Print a random opening message from `SCRIPTS[lang]["initials"]`
3. Enter an infinite loop
4. Read user input with `input("You: ")`
5. Normalize the input and check whether it matches any quit word
6. If user wants to quit:
   - print a random closing message
   - break the loop
7. Otherwise generate a rule-based response with `rule_respond(user_in)`
8. Run emotion prediction with `get_eq(user_in)`
9. Print:
   - the ELIZA response
   - the predicted emotion and confidence
10. If `Ctrl+C` is pressed:
   - print a closing message
   - exit the loop

## 8. Model Loading Flow

### `load_model()`

1. Check whether `self.model_path` exists
2. If it exists, open the pickle file
3. Load saved artifacts into memory:
   - `word2id`
   - `vectorizer`
   - `model`
   - `label_to_idx`
   - `idx_to_label`
   - `num_classes`
4. If no file exists, nothing is loaded

Effect in chat mode:

- rule-based responses still work
- emotion prediction falls back to `"Neutral", 0.0`

## 9. Response Generation Flow

### `rule_respond(text)`

This is the ELIZA-style response engine.

Step-by-step:

1. Normalize the input text
2. Apply preprocessing replacements from `self.script["pres"]`
3. Loop through the language-specific keyword patterns in priority order
4. For each pattern:
   - run `re.search(pattern, text)`
5. If a pattern matches:
   - choose a random response template from that pattern
   - collect captured groups from the regex
   - reflect each captured fragment using `reflect(...)`
   - insert reflected fragments into the response template
   - return the final response immediately
6. If no pattern matches, return the default fallback response

### `reflect(fragment)`

1. Tokenize the fragment
2. Replace tokens using `self.script["posts"]`
3. Join tokens back into a string
4. Return the reflected phrase

This is what allows perspective shifts like:

- `I` -> `you`
- Myanmar pronoun replacements in the `mm` script

## 10. Emotion Prediction Flow

### `get_eq(text)`

Step-by-step:

1. Check whether a trained model is loaded
2. If not loaded, return:
   - emotion: `Neutral`
   - score: `0.0`
3. Transform the input text using the saved TF-IDF vectorizer
4. Get classifier decision scores from `LinearSVC`
5. Convert scores to a PyTorch tensor
6. Apply `softmax` to approximate class probabilities
7. Find the highest-probability class index
8. Convert predicted internal index back to original label
9. Convert that label to a human-readable emotion name using `id2label`
10. Return:
   - predicted emotion name
   - probability score

## 11. Conversation Loop Summary

For each user message in chat mode, the actual flow is:

1. User types text
2. Program checks for quit command
3. Program generates an ELIZA response using regex rules
4. Program predicts the user emotion using the trained SVM model
5. Program prints both outputs

So the chatbot is hybrid because:

- the reply text comes from rule-based ELIZA logic
- the emotion analysis comes from a trained ML classifier

## 12. Important Design Notes

### Active components

The currently active ML path uses:

- `FeatureUnion`
- `TfidfVectorizer`
- `LinearSVC`
- `pickle` model saving/loading

### Defined but not used in the active flow

These are present in the file but not used by the current training/chat pipeline:

- `EmotionDataset`
- `PooledTextClassifier`
- `evaluate(...)`

These appear to be leftovers from an earlier PyTorch-based version.

### Unused training parameters

In the current implementation:

- `epochs` is parsed and passed, but not used by `LinearSVC`
- `batch_size` is parsed and passed, but not used by `LinearSVC`
- `lr` is passed into `train(...)`, but not used

## 13. Short End-to-End Flow

### If mode is `train`

1. Load CSV
2. Clean and encode labels
3. Split data
4. Build TF-IDF features
5. Train `LinearSVC`
6. Validate
7. Save pickle file

### If mode is `chat`

1. Load pickle model
2. Show greeting
3. Read user input
4. Check quit condition
5. Generate ELIZA reply
6. Predict emotion
7. Print both results
8. Repeat until exit

