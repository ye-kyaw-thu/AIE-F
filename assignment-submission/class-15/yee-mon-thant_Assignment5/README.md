# Domain Adaptaion with Language Model 

This project trains a Myanmar language model on general text, then checks how
well it understands three different types of writing (social media, religious
text, and news). It also tries a way to make the model better at the domains
it struggled with.

## Course Context

This project was done as an assignment for the AI Engineering (Fundamental)
Class by Sayar Ye Kyaw Thu, Language Understanding Lab, Myanmar.
Course repository: https://github.com/ye-kyaw-thu/AIE-F

## Project Structure

```
yee-mon-thant_Assignment5/
├── data/
│   ├── raw/          # excluded from git (large downloads) - regenerate with download_data.py
│   ├── clean/         # excluded from git (large files) - regenerate with clean_data.py
│   ├── tokenized/      # excluded from git (large files) - regenerate with tokenize_data.py
│   └── testsets/       # fixed 200-token test sets, one per domain (included)
├── scripts/
│   ├── download_data.py         # downloads datasets from Hugging Face
│   ├── clean_data.py            # removes HTML, emoji, punctuation, etc.
│   ├── tokenize_data.py         # splits Myanmar text into syllables
│   ├── build_testsets.py        # builds the fixed-size test sets
│   ├── prepare_finetune_data.py # builds fine-tuning data (skips test tokens)
│   ├── lstm_lm.py               # the LSTM model: train / test / generate
│   ├── fine_tune.py             # continues training a model on new text
│   └── evaluate_ppl.py          # runs PPL tests on all domains, makes chart
└── results/
    ├── lstm_general.pt                    # base model (trained on Wikipedia + myPOS)
    ├── lstm_general.pt.vocab              # vocabulary for the base model
    ├── lstm_news_finetuned.pt             # fine-tuned on News only
    ├── lstm_news_finetuned.pt.vocab       # vocabulary for the News fine-tuned model
    ├── lstm_all_domains_finetuned.pt      # fine-tuned on all 3 domains together
    ├── lstm_all_domains_finetuned.pt.vocab # vocabulary for the all-domains model
    └── ppl_by_domain.png                   # results chart
```

## Steps

### 1. Collected the data
- **General text** (for training): Myanmar Wikipedia + myPOS
- **Domain text** (for testing): Facebook Flores (social media), Dhamma
  Articles (religious), and Myanmar-English News Translation (news)

### 2. Cleaned the text
Removed things that aren't useful for the model to learn, like HTML tags,
emoji, hidden/invisible characters, and Myanmar punctuation marks (`၊` `။`).
Numbers and English words were kept, since real Myanmar text often mixes
those in naturally.

### 3. Split the text into syllables
Myanmar doesn't use spaces between words, so Sayar Ye Kyaw Thu's
`sylbreak` tool was used to split text into syllables. The original tool
splits English words letter by letter (like "cat" into "c", "a", "t"),
which doesn't make sense for this project, so it was changed to keep
English words whole instead.

### 4. Built the test sets
A test file was made for each of the 3 domains, each with exactly 200
tokens, so the comparison between domains would be fair (same amount of
text for each one).

### 5. Trained the base model
Used an LSTM model (a type of neural network good at reading sequences),
based on Sayar Ye Kyaw Thu's LM-Tutorial. Two things had to be changed to
make it actually finish training in a reasonable time:

- **Smaller vocabulary**: Instead of letting the model learn every single
  word it saw (135,890 words - way too many), it was limited to the
  15,000 most common words. This made training much faster, since the
  model doesn't have to think about that many options every time it
  guesses the next word.

- **Non-overlapping training chunks**: At first, the code was cutting the
  text into chunks that overlapped a lot - like reading "A B C D", then
  "B C D E", then "C D E F", moving forward just 1 word each time. This
  makes way too many chunks that repeat almost the same words over and
  over. This was changed so each chunk starts right after the last one
  ends (like "A B C D", then "E F G H") - no repeating. This cut the
  number of training examples down by about 30 times, and training became
  much faster.

The model was trained for 5 rounds (epochs) on the general text. The
training loss went down from 4.47 to 3.18, which means the model was
learning.

### 6. Measured perplexity (PPL) on each domain
Perplexity tells us how "confused" the model is by a piece of text. Lower
number = model understands it better.

| Domain     | PPL   |
|------------|-------|
| Facebook   | 41.15 |
| Religious  | 48.31 |
| News       | 73.15 |

News was the hardest for the model, Facebook was the easiest. This was
checked against whether it was just because News has more unknown/rare
words, but actually all 3 domains had almost 0% unknown words - so the
difficulty must come from something else, like sentence style, not just
vocabulary.

### 7. Made a bar chart
See `results/ppl_by_domain.png` for a visual comparison of the 3 domains.

### 8. Tried to improve the model (fine-tuning)
Fine-tuning means taking the already-trained model and training it a
little more on specific text, so it adjusts to that style. This was tried
two ways:

**a) Fine-tuned on News text only:**

| Domain     | Before | After (News-only) |
|------------|--------|--------------------|
| Facebook   | 41.15  | 67.70 (worse)      |
| Religious  | 48.31  | 121.70 (much worse)|
| News       | 73.15  | 36.86 (much better)|

News got a lot better, but Religious got much worse. This is a known
problem called "catastrophic forgetting" - when a model focuses too much
on one new thing, it forgets some of what it knew before.

**b) Fine-tuned on all 3 domains together:**

| Domain     | Before | After (all 3 combined) |
|------------|--------|---------------------------|
| Facebook   | 41.15  | 47.72 (a little worse)    |
| Religious  | 48.31  | 38.95 (better)            |
| News       | 73.15  | 38.43 (much better)       |

This worked much better overall. News still improved a lot, Religious
actually got better too, and Facebook only got slightly worse. So mixing
all 3 domains together for fine-tuning is a much safer way to improve the
model, compared to only focusing on one domain.

(For both experiments, the fine-tuning text did not include the same 200
tokens used in the test sets, to keep the results fair.)

## Conclusion
This project showed that a Myanmar language model trained on general text (Wikipedia and myPOS) does not work equally well on every type of text. It had the most trouble with News, and the least trouble with Facebook posts.
The vocabulary was limited to 15,000 words so training would be faster, but this might have made it harder for the model to learn rare domain words (like religious/Pali words or news names). A bigger vocabulary might give better PPL. Training was also only done for 3-5 epochs because of time and hardware limits, so training longer could give better results too.
Finally, fine-tuning on just one domain helps that domain a lot, but makes the model worse on the other domains. This shows that fine-tuning on a mix of domains is a safer choice if the goal is to improve the model across multiple types of text, not just one.

## Credits

- Syllable-splitting logic (`sylbreak`): Sayar Ye Kyaw Thu
  (https://github.com/ye-kyaw-thu/sylbreak)
- LSTM training script: adapted from Sayar Ye Kyaw Thu's LM-Tutorial
  (https://github.com/ye-kyaw-thu/AIE-F)
- Datasets:
  - [Myanmar Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (config `20231101.my`)
  - [myPOS](https://huggingface.co/datasets/chuuhtetnaing/myanmar-pos-dataset)
  - [Myanmar Facebook Flores Dataset](https://huggingface.co/datasets/chuuhtetnaing/myanmar-facebook-flores-dataset)
  - [Dhamma Article Dataset](https://huggingface.co/datasets/chuuhtetnaing/dhamma-article-dataset)
  - [Myanmar-English News Translation Dataset](https://huggingface.co/datasets/chuuhtetnaing/myanmar-english-news-translation-dataset)
