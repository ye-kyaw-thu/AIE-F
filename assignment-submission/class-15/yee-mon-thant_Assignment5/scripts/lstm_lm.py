#!/usr/bin/env python3


import os
import math
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. DATA HANDLING & VOCABULARY
# ==========================================

class Vocabulary:
    
    def __init__(self):
        # 4 special reserved tokens every vocabulary needs:
        # <pad>  = filler, used to pad shorter sequences
        # <unk>  = "unknown" - used for any word not in our vocabulary
        # <s>    = marks the start of a sentence
        # </s>   = marks the end of a sentence
        self.token2idx = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.idx2token = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}

    def build(self, tokens, max_vocab_size=15000):

       # Build the vocabulary from training text with only the most frequent tokens.

        from collections import Counter
        counts = Counter(tokens)  # count how many times each token appears
        most_common = counts.most_common(max_vocab_size)  # keep only the top N
        for token, _ in most_common:
            if token not in self.token2idx:
                # assign each new token the next available number
                self.token2idx[token] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = token
        return self

    def __len__(self):
        return len(self.token2idx)


class TextDataset(Dataset):
    
    def __init__(self, tokens, vocab, seq_len=20):
        self.vocab = vocab
        self.seq_len = seq_len  # how many tokens per training example
        # convert every token to its number; unknown tokens become <unk> (index 1)
        self.indices = [vocab.token2idx.get(t, 1) for t in tokens]

    def __len__(self):
        # how many non-overlapping chunks of size seq_len fit in our data
        return max(0, (len(self.indices) - 1) // self.seq_len)

    def __getitem__(self, idx):
        # find where this chunk starts (chunk 0 starts at 0, chunk 1 starts at seq_len, etc.)
        start = idx * self.seq_len
        # input = this chunk of tokens
        x = torch.tensor(self.indices[start : start + self.seq_len], dtype=torch.long)
        # target = the SAME chunk, shifted one position later
        y = torch.tensor(self.indices[start + 1 : start + 1 + self.seq_len], dtype=torch.long)
        return x, y


def read_and_tokenize(file_path, token_level="word"):
    #Reads a text file and splits it into tokens (words or characters).
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if token_level == "char":
        # character-level: every single character is its own token
        tokens = list(text.replace("\n", " "))
    else:
        # word-level: split on whitespace (this is what we use for Myanmar,
        # since our text is already syllable-tokenized with spaces)
        tokens = text.split()
    return tokens


# ==========================================
# 2. MODEL DEFINITION
# ==========================================

class LSTM_LM(nn.Module):
    """
    Three stages:
    1. embedding - turns each token number into a list of numbers (its "meaning")
    2. lstm            - reads through the sequence, keeping a running "memory"
    3. fc                 - turns that memory into a prediction for the next token
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        # embedding: vocab_size possible tokens -> each becomes a vector of embed_dim numbers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # lstm: the memory mechanism. hidden_dim = size of its "notebook".
        # num_layers=2 means two LSTM layers stacked for extra learning power.
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # fc (fully connected): converts the LSTM's memory into a score
        # for every possible next token in the vocabulary
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x = a batch of token-number sequences
        embeds = self.embedding(x)           # step 1: numbers -> meaning vectors
        out, hidden = self.lstm(embeds, hidden)  # step 2: read through the sequence
        logits = self.fc(out)                # step 3: predict the next token
        return logits, hidden


# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================

def train_model(args):
    # Trains the LSTM on a text file and saves the resulting model.
    print(f"Reading training data from {args.train_file} (Level: {args.token_level})...")
    train_tokens = read_and_tokenize(args.train_file, args.token_level)

    print("Building vocabulary...")
    vocab = Vocabulary().build(train_tokens)
    print(f"Vocabulary size: {len(vocab)} tokens")

    # wrap our tokens into (input, target) training examples
    dataset = TextDataset(train_tokens, vocab, seq_len=args.seq_len)
    print(f"Number of training examples per epoch: {len(dataset)}")
    # DataLoader automatically groups examples into batches and shuffles them
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # build the actual model
    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    # criterion = how we measure "how wrong" a prediction was
    # ignore_index=0 means we don't penalize the model for <pad> positions
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = the algorithm that adjusts the model's internal numbers to reduce error
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()  # tell PyTorch we're in "training mode"
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)  # move data to GPU if available
            optimizer.zero_grad()               # clear old gradients
            logits, _ = model(x)                 # step 1: model makes a guess
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))  # step 2: measure error
            loss.backward()                      # step 3: figure out how to improve
            optimizer.step()                     # step 4: actually improve
            total_loss += loss.item()

        # print average loss for this epoch - this number should go DOWN over time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")

    # save the trained model's weights, plus the vocabulary it was trained with
    print(f"Saving model to {args.model_path}...")
    torch.save(model.state_dict(), args.model_path)
    with open(args.model_path + '.vocab', 'w', encoding='utf-8') as f:
        json.dump(vocab.token2idx, f, ensure_ascii=False)
    print("Training complete!")


def test_model(args):
    # Loads a trained model and computes perplexity on new (unseen) text.
    print(f"Loading model from {args.model_path}...")
    # load the saved vocabulary so token numbers match what the model was trained with
    with open(args.model_path + '.vocab', 'r', encoding='utf-8') as f:
        token2idx = json.load(f)
    vocab = Vocabulary()
    vocab.token2idx = token2idx
    vocab.idx2token = {int(v): k for k, v in token2idx.items()}

    # rebuild the model architecture, then load the trained weights into it
    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()  # tell PyTorch we're in "evaluation mode" (turns off dropout etc.)

    print(f"Reading test data from {args.test_file}...")
    test_tokens = read_and_tokenize(args.test_file, args.token_level)
    dataset = TextDataset(test_tokens, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=1)

    total_nll, total_tokens = 0.0, 0
    # reduction='sum' means we get the TOTAL error, not the average - we'll
    # divide by the total token count ourselves below
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

    print("Evaluating...")
    # torch.no_grad() = don't calculate gradients, we're not training here,
    # just measuring how good the model's predictions are
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            total_nll += loss.item()                       # accumulate total error
            total_tokens += (y != 0).sum().item()           # count real (non-pad) tokens

    avg_nll = total_nll / total_tokens   # average "surprise" per token
    ppl = math.exp(avg_nll)              # perplexity = e^(average surprise)
    print(f"Test Results -> Tokens: {total_tokens}, Avg NLL: {avg_nll:.4f}, PPL: {ppl:.2f}")


def generate_text(args):
    """Uses a trained model to generate new text, one token at a time (bonus feature)."""
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path + '.vocab', 'r', encoding='utf-8') as f:
        token2idx = json.load(f)
    vocab = Vocabulary()
    vocab.token2idx = token2idx
    vocab.idx2token = {int(v): k for k, v in token2idx.items()}

    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    prompt = args.prompt if args.prompt else "<s>"
    print(f"\nPrompt: {prompt}")
    print("Generating: ")

    # turn the starting prompt into token numbers
    if args.token_level == "char":
        tokens = list(prompt)
    else:
        tokens = prompt.split()
    indices = [vocab.token2idx.get(t, 1) for t in tokens]
    input_seq = torch.tensor([indices], dtype=torch.long).to(device)
    hidden = None

    generated_tokens = list(tokens)

    with torch.no_grad():
        for _ in range(args.gen_length):
            logits, hidden = model(input_seq, hidden)
            # temperature controls randomness: lower = safer/more predictable,
            # higher = more random/creative
            last_token_logits = logits[0, -1, :] / args.temperature
            probs = torch.softmax(last_token_logits, dim=-1)
            # randomly sample the next token according to its predicted probability
            next_token_idx = torch.multinomial(probs, num_samples=1).item()

            if next_token_idx == 3:  # </s> - model decided to end the sentence
                break

            generated_tokens.append(vocab.idx2token.get(next_token_idx, "<unk>"))
            # feed this new token back in as input for predicting the NEXT one
            input_seq = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)

    if args.token_level == "char":
        print("".join(generated_tokens))
    else:
        print(" ".join(generated_tokens))


# ==========================================
# 4. MAIN ARGUMENTS PARSER
# ==========================================

if __name__ == "__main__":
    # this section defines all the --flags you can use on the command line
    parser = argparse.ArgumentParser(description="LSTM Language Model Trainer/Evaluator/Generator")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "generate"])
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--model_path", type=str, default="lstm_model.pt")
    parser.add_argument("--token_level", type=str, default="char", choices=["word", "char"])
    parser.add_argument("--embed_dim", type=int, default=128)      # size of each token's "meaning vector"
    parser.add_argument("--hidden_dim", type=int, default=256)     # size of the LSTM's memory
    parser.add_argument("--num_layers", type=int, default=2)       # how many LSTM layers stacked
    parser.add_argument("--seq_len", type=int, default=30)         # tokens per training example
    parser.add_argument("--epochs", type=int, default=20)          # how many full passes over the data
    parser.add_argument("--batch_size", type=int, default=32)      # how many examples processed at once
    parser.add_argument("--lr", type=float, default=0.001)         # learning rate (how big each adjustment step is)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)

    args = parser.parse_args()
    # use GPU if available, otherwise fall back to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # call the right function based on which --mode was chosen
    if args.mode == "train":
        if not args.train_file: parser.error("--train_file is required for training")
        train_model(args)
    elif args.mode == "test":
        if not args.test_file: parser.error("--test_file is required for testing")
        test_model(args)
    elif args.mode == "generate":
        generate_text(args)