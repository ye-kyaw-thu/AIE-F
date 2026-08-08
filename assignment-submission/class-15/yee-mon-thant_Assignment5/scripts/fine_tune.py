
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Reuse the same building blocks from lstm_lm.py
from lstm_lm import Vocabulary, TextDataset, LSTM_LM, read_and_tokenize

def fine_tune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the EXISTING vocabulary 
    print(f"Loading vocabulary from {args.base_model}.vocab...")
    with open(args.base_model + '.vocab', 'r', encoding='utf-8') as f:
        token2idx = json.load(f)
    vocab = Vocabulary()
    vocab.token2idx = token2idx
    vocab.idx2token = {int(v): k for k, v in token2idx.items()}
    print(f"Vocabulary size: {len(vocab)} tokens")

    # Load the existing trained model weights
    print(f"Loading base model from {args.base_model}...")
    model = LSTM_LM(len(vocab), args.embed_dim, args.hidden_dim, args.num_layers).to(device)
    model.load_state_dict(torch.load(args.base_model, map_location=device))

    # Read the domain-specific fine-tuning data
    print(f"Reading fine-tuning data from {args.finetune_file}...")
    finetune_tokens = read_and_tokenize(args.finetune_file, args.token_level)
    dataset = TextDataset(finetune_tokens, vocab, seq_len=args.seq_len)
    print(f"Number of fine-tuning examples per epoch: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    # Use a SMALLER learning rate than original training 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Fine-tuning for {args.epochs} epochs (lr={args.lr})...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.view(-1, len(vocab)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {total_loss/len(dataloader):.4f}")

    print(f"Saving fine-tuned model to {args.output_model}...")
    torch.save(model.state_dict(), args.output_model)
    # Save the SAME vocabulary alongside it
    with open(args.output_model + '.vocab', 'w', encoding='utf-8') as f:
        json.dump(vocab.token2idx, f, ensure_ascii=False)
    print("Fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an existing LSTM model on domain-specific text")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the already-trained model (e.g. results/lstm_general.pt)")
    parser.add_argument("--finetune_file", type=str, required=True, help="Domain-specific text file to fine-tune on")
    parser.add_argument("--output_model", type=str, required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--token_level", type=str, default="word", choices=["word", "char"])
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)  # smaller than the original 0.001

    args = parser.parse_args()
    fine_tune(args)