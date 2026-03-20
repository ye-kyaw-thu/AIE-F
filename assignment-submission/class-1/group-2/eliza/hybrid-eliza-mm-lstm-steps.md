# `hybrid-eliza-mm-lstm.py` Step-by-Step Guide

This file explains how `hybrid-eliza-mm-lstm.py` works, from text input to ELIZA response and emotion prediction.

## 1. Purpose of the Script

`hybrid-eliza-mm-lstm.py` is a hybrid chatbot:

- Rule-based ELIZA part:
  generates the reply text using regex patterns and reflection rules.
- LSTM-based emotion classifier:
  predicts the user's emotion from the input sentence.

It is designed for:

- English (`en`)
- Myanmar (`mm`)

The Myanmar version adds custom tokenization and character n-grams to better handle Myanmar text.

## 2. Imported Libraries

The script uses:

- `os`, `re`, `copy`, `argparse`, `random`
- `pandas`
- `torch`, `torch.nn`, `torch.optim`
- `Dataset`, `DataLoader`
- `pack_padded_sequence`, `pad_packed_sequence`
- `Counter`

These support:

- file handling
- text normalization
- neural network training
- dataset batching
- LSTM sequence packing

## 3. Myanmar Text Rules

Two regex variables are defined:

- `MYANMAR_CHAR_RE`
- `MYANMAR_TOKEN_RE`

They are used to detect Myanmar script and split Myanmar text into tokens.

This is important because Myanmar text does not always behave like space-separated English text.

## 4. ELIZA Script Data

The `SCRIPTS` dictionary stores rule-based conversation content.

Each language contains:

- `initials`: opening messages
- `finals`: closing messages
- `quits`: words that stop the chat
- `pres`: preprocessing replacements
- `posts`: reflection replacements
- `keywords`: regex patterns, responses, and priority ranks

Example idea:

- user says something matching a regex
- ELIZA chooses one response template
- captured text fragments are reflected and inserted into the reply

## 5. Text Normalization

`normalize_text(text)` does basic cleanup:

- converts input to string
- trims spaces
- converts to lowercase
- removes punctuation-like symbols
- collapses repeated spaces

This makes both rule matching and model input more consistent.

## 6. Character N-gram Building

`build_char_ngrams(text, min_n=2, max_n=3)` creates character-level features.

For Myanmar text, this helps because:

- words may not be segmented cleanly
- subword patterns can still carry emotion meaning

Example:

- text is compacted by removing spaces
- 2-character and 3-character chunks are extracted

## 7. Tokenization

`tokenize_text(text, lang)` converts input into tokens.

For Myanmar:

- normalize text first
- if spaces exist, split by spaces
- otherwise use Myanmar regex token extraction
- then add character n-grams

For English:

- normalize text
- split by spaces

This tokenization is used by both:

- vocabulary building
- LSTM input preparation

## 8. Dataset Class

`EmotionDataset` prepares training samples.

Each sample returns:

- padded token id sequence
- label id
- real sequence length

Why sequence length matters:

- the LSTM uses packed sequences
- padding tokens should not be treated as real words

## 9. Attention Layer

`Attention` computes attention weights over LSTM outputs.

Flow:

- each time step gets a score
- padding positions are masked out
- softmax converts scores into weights
- weighted sum gives a single context vector

This helps the model focus on important tokens instead of averaging everything equally.

## 10. `EmotionalBiLSTM` Model

This is the neural classifier.

Architecture:

1. Embedding layer
2. Embedding dropout
3. Bidirectional LSTM
4. Attention layer
5. Dropout
6. Final linear classifier

Important settings:

- `embed_dim`: size of token embeddings
- `hidden_dim`: LSTM hidden size
- `num_layers`: number of stacked LSTM layers
- `dropout`: regularization strength

Why bidirectional:

- it reads the sequence left-to-right and right-to-left
- this gives better context for emotion classification

## 11. `HybridEliza` Controller

The `HybridEliza` class controls the whole system.

It manages:

- language selection
- ELIZA rules
- vocabulary
- label mapping
- model configuration
- training
- loading
- inference

## 12. Initialization

Inside `__init__`, the script sets:

- selected language
- sorted ELIZA keyword rules by rank
- model path
- CPU or CUDA device
- initial vocabulary:
  - `<PAD>` = 0
  - `<UNK>` = 1
- emotion label names
- training hyperparameters

The model itself starts as `None` until training or loading happens.

## 13. Vocabulary Building

`build_vocab(texts)` creates the token-to-id dictionary.

Steps:

1. tokenize every training sentence
2. count token frequency with `Counter`
3. keep the most common tokens
4. assign integer ids starting from 2

This vocabulary is later used to convert text into numeric sequences.

## 14. Label Mapping

`build_label_maps(labels)` converts dataset labels into internal class indices.

This is needed because:

- the CSV may contain labels in a specific numeric scheme
- training needs compact class ids like `0..N-1`

Two mappings are stored:

- `label_to_idx`
- `idx_to_label`

## 15. Stratified Split

`split_stratified(...)` creates train and validation sets while preserving label balance.

Why this matters:

- if classes are imbalanced, random splitting can distort validation quality
- stratification gives a more reliable validation accuracy

Steps:

1. group texts by label
2. shuffle each group with a seed
3. split each label group into train and validation parts
4. merge all groups back together
5. shuffle final train and validation lists

## 16. Training Pipeline

`train(...)` is the main model training function.

### Step 16.1 Read the Dataset

The script loads the CSV with `pandas.read_csv`.

Then it:

- chooses `label` or `emotions` column
- removes rows with missing text or label
- converts labels to integers

### Step 16.2 Prepare Vocabulary and Labels

It builds:

- vocabulary from text
- label mappings from labels

### Step 16.3 Split Data

It converts original labels to internal class ids, then creates:

- training texts and labels
- validation texts and labels

### Step 16.4 Build DataLoaders

Two datasets are created:

- `train_ds`
- `val_ds`

Then DataLoaders batch them for training and validation.

### Step 16.5 Create the Model

The script creates `EmotionalBiLSTM` using the configured hyperparameters.

### Step 16.6 Optimizer and Loss

It uses:

- `Adam` optimizer
- `CrossEntropyLoss`

It also applies `weight_decay` for regularization.

### Step 16.7 Epoch Training Loop

For each epoch:

1. switch model to training mode
2. loop through batches
3. move tensors to device
4. run forward pass
5. compute loss
6. backpropagate
7. clip gradients
8. update weights
9. accumulate loss

### Step 16.8 Validation

After each epoch:

- validation accuracy is computed with `evaluate(...)`
- average training loss is printed

### Step 16.9 Early Stopping

If validation accuracy improves:

- save the current model weights in memory as `best_state`

If validation accuracy stops improving for several epochs:

- stop training early using `patience`

This prevents continuing into stronger overfitting.

### Step 16.10 Save Best Checkpoint

After training, the best model state is restored and saved to disk.

The checkpoint contains:

- model weights
- vocabulary
- language
- label mappings
- number of classes
- model hyperparameters

## 17. Evaluation

`evaluate(loader)` runs validation without gradient updates.

Steps:

1. switch model to eval mode
2. disable gradients
3. predict class for each batch
4. count correct predictions
5. return accuracy

## 18. Loading a Saved Model

`load_model()` restores a previously trained checkpoint.

It reloads:

- vocabulary
- label mappings
- number of classes
- model dimensions and dropout
- trained weights

Then it switches the model to evaluation mode.

## 19. Emotion Prediction

`get_eq(text)` predicts emotion from one user input.

Steps:

1. tokenize input text
2. convert tokens to ids
3. truncate to max length 50
4. pad with zeros
5. pass sequence and length into the model
6. apply softmax to get probabilities
7. choose the highest-probability class
8. map it back to the emotion label

Return value:

- predicted emotion name
- prediction confidence score

## 20. Reflection

`reflect(fragment)` transforms matched user fragments using the `posts` dictionary.

This is part of classic ELIZA behavior.

Example idea:

- first-person terms can be turned into second-person terms
- the reflected fragment is inserted into a response template

## 21. Rule-Based Reply

`rule_respond(text)` creates the ELIZA response.

Steps:

1. normalize text
2. apply preprocessing substitutions from `pres`
3. test regex patterns in priority order
4. choose one response template from the matched rule
5. reflect captured groups
6. insert reflected text into the response

If no rule matches, it returns a fallback reply.

## 22. Main Function

`main()` is the CLI entry point.

It defines command-line arguments such as:

- `--lang`
- `--mode`
- `--data`
- `--model_path`
- `--epochs`
- `--batch_size`
- `--val_split`
- `--seed`
- `--lr`
- `--embed_dim`
- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--weight_decay`
- `--patience`

Then it creates `HybridEliza(...)`.

## 23. Train Mode

If `--mode train`:

- dataset is loaded
- model is trained
- best checkpoint is saved

Example:

```bash
python hybrid-eliza-mm-lstm.py --mode train --lang mm --data emotions/Combined.csv
```

## 24. Chat Mode

If `--mode chat`:

1. load trained model
2. print an initial ELIZA line
3. repeatedly read user input
4. stop if user enters a quit phrase
5. generate ELIZA rule-based response
6. generate emotion prediction
7. print both outputs

Example:

```bash
python hybrid-eliza-mm-lstm.py --mode chat --lang mm
```

## 25. Output During Chat

For each user message, the script prints:

- ELIZA response
- predicted emotion and confidence

Example format:

```text
ELIZA: ...
[EQ Analysis]: Predicted Emotion: ... (52.31%)
```

## 26. Why the Current Version Performs Better Than the First LSTM Draft

The current version improves training behavior by:

- masking padded tokens in attention
- using packed sequences
- adding dropout
- using weight decay
- clipping gradients
- stopping early when validation stops improving
- saving the best validation checkpoint instead of the last epoch

These changes keep the same model family, but usually improve generalization.

## 27. Full Flow Summary

The full pipeline is:

1. user enters text
2. text is normalized
3. ELIZA rules generate a response
4. tokenization converts text into ids
5. BiLSTM with attention predicts emotion
6. chatbot prints both:
   - rule-based response
   - emotion analysis

## 28. Suggested Next Tuning Steps

If you want to improve results further without changing model family, tune:

- `--hidden_dim`
- `--num_layers`
- `--dropout`
- `--lr`
- `--weight_decay`
- `--patience`
- `--epochs`

Good starting point:

```bash
python hybrid-eliza-mm-lstm.py --mode train --lang mm --epochs 20 --lr 0.0007 --hidden_dim 96 --num_layers 2 --dropout 0.35 --weight_decay 0.0001 --patience 4
```
