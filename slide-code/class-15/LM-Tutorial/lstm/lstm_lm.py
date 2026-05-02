#!/usr/bin/env python3
"""
A complete, standalone LSTM Language Model with Train, Test, and Generate modes.
Specifically designed to handle OOVs safely and support Character-level modeling (ideal for Myanmar).
"""

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
        self.token2idx = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.idx2token = {0: "<pad>", 1: "<unk>", 2: "<s>", 3: "</s>"}
    
    def build(self, tokens):
        """Build vocab from a list of tokens (words or characters)"""
        for token in set(tokens):
            if token not in self.token2idx:
                self.token2idx[token] = len(self.token2idx)
                self.idx2token[len(self.idx2token)] = token
        return self

    def __len__(self):
        return len(self.token2idx)

class TextDataset(Dataset):
    def __init__(self, tokens, vocab, seq_len=20):
        self.vocab = vocab
        self.seq_len = seq_len
        # Convert tokens to indices, mapping unseen tokens to <unk> (1)
        self.indices = [vocab.token2idx.get(t, 1) for t in tokens]
        
    def __len__(self):
        # We need seq_len for input, and 1 for target. So total length is seq_len + 1.
        return max(0, len(self.indices) - self.seq_len)
        
    def __getitem__(self, idx):
        # Input sequence: [i to i+seq_len]
        x = torch.tensor(self.indices[idx : idx + self.seq_len], dtype=torch.long)
        # Target sequence: [i+1 to i+seq_len+1] (shifted by 1)
        y = torch.tensor(self.indices[idx + 1 : idx + 1 + self.seq_len], dtype=torch.long)
        return x, y

def read_and_tokenize(file_path, token_level="word"):
    """Reads a file and returns a flat list of tokens."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if token_level == "char":
        # Character level: list of characters (Great for Myanmar without spaces!)
        tokens = list(text.replace("\n", " ")) # replace newlines with space char
    else:
        # Word level: split by whitespace
        tokens = text.split()
    return tokens

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)
        # out shape: (batch_size, seq_len, hidden_dim)
        out, hidden = self.lstm(embeds, hidden)
        # logits shape: (batch_size, seq_len, vocab_size)
        logits = self.fc(out)
        return logits, hidden

# ==========================================
# 3. TRAINING & EVALUATION FUNCTIONS
# ==========================================
def train_model(args):
    print(f"Reading training data from {args.train_file} (Level: {args.token_level})...")
    train_tokens = read_and_tokenize(args.train_file, args.token_level)
    
    print("Building vocabulary...")
    vocab = Vocabulary().build(train_tokens)
    print(f"Vocabulary size: {len(vocab)} tokens")
    
    dataset = TextDataset(train_tokens, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore <pad> in loss calculation
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            # Reshape for CrossEntropy: (Batch * SeqLen, VocabSize)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")

    # Save model and vocabulary
    print(f"Saving model to {args.model_path}...")
    torch.save(model.state_dict(), args.model_path)
    with open(args.model_path + '.vocab', 'w', encoding='utf-8') as f:
        json.dump(vocab.token2idx, f, ensure_ascii=False)
    print("Training complete!")

def test_model(args):
    print(f"Loading model from {args.model_path}...")
    with open(args.model_path + '.vocab', 'r', encoding='utf-8') as f:
        token2idx = json.load(f)
    vocab = Vocabulary()
    vocab.token2idx = token2idx
    vocab.idx2token = {int(v): k for k, v in token2idx.items()}
    
    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    print(f"Reading test data from {args.test_file}...")
    test_tokens = read_and_tokenize(args.test_file, args.token_level)
    dataset = TextDataset(test_tokens, vocab, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=1) # Batch size 1 for accurate PPL tracking
    
    total_nll, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    
    print("Evaluating...")
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            
            # Calculate loss only on non-padding tokens
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            total_nll += loss.item()
            total_tokens += (y != 0).sum().item() # Count non-pad tokens

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)
    print(f"Test Results -> Tokens: {total_tokens}, Avg NLL: {avg_nll:.4f}, PPL: {ppl:.2f}")

def generate_text(args):
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
    
    # Convert prompt to indices
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
            
            # Get logits for the last token in the sequence
            last_token_logits = logits[0, -1, :] / args.temperature
            probs = torch.softmax(last_token_logits, dim=-1)
            
            # Sample from the distribution (prevents repetitive loops)
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_idx == 3: # </s> token
                break
                
            generated_tokens.append(vocab.idx2token.get(next_token_idx, "<unk>"))
            
            # Feed the predicted token back in for the next step
            input_seq = torch.tensor([[next_token_idx]], dtype=torch.long).to(device)
            
    # Join characters or words based on mode
    if args.token_level == "char":
        print("".join(generated_tokens))
    else:
        print(" ".join(generated_tokens))

# ==========================================
# 4. MAIN ARGUMENTS PARSER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Language Model Trainer/Evaluator/Generator")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test", "generate"], 
                        help="What to do?")
    parser.add_argument("--train_file", type=str, help="Path to training text file")
    parser.add_argument("--test_file", type=str, help="Path to test text file")
    parser.add_argument("--model_path", type=str, default="lstm_model.pt", help="Path to save/load model")
    
    # Data arguments
    parser.add_argument("--token_level", type=str, default="char", choices=["word", "char"], 
                        help="'word' for English, 'char' for Myanmar or unsegmented text")
    
    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=30, help="Length of sequence to feed LSTM")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, default="", help="Text to start generation")
    parser.add_argument("--gen_length", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Higher = more random, Lower = stricter")
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.mode == "train":
        if not args.train_file: parser.error("--train_file is required for training")
        train_model(args)
    elif args.mode == "test":
        if not args.test_file: parser.error("--test_file is required for testing")
        test_model(args)
    elif args.mode == "generate":
        generate_text(args)

