## Hybrid-ELIZA
## Demo code of applying rules+LSTM for AI Engineering Class (Fundamental)
## Written by Ye, Language Understanding Lab., Myanmar
## Last updated: 14 Mar 2026
## Reference code: https://www.kaggle.com/code/wjburns/eliza 
## Dataset link: https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter
import argparse
import random
from sklearn.metrics import classification_report

# --- 1. GLOBAL SCRIPT DATA ---
# Added 'Rank' (3rd element in list). Higher = Higher Priority.
SCRIPTS = {
    "en": {
        "initials": ["How do you do. Please tell me your problem.", "Is something troubling you?"],
        "finals": ["Goodbye. It was nice talking to you.", "Your terminal will self-destruct in 5s."],
        "quits": ["bye", "quit", "exit"],
        "pres": {"don't": "dont", "i'm": "i am", "recollect": "remember", "machine": "computer"},
        "posts": {"am": "are", "i": "you", "my": "your", "me": "you", "your": "my"},
        "synons": {
            "be": ["am", "is", "are", "was", "were"],
            "joy": ["happy", "glad", "better", "fine"],
            "sadness": ["sad", "depressed", "sick", "gloomy"]
        },
        "keywords": [
            # [Regex, [Responses], Rank]
            [r'(.*) die (.*)', ["Please don't talk like that. Tell me more about your feelings."], 10],
            [r'i need (.*)', ["Why do you need {0}?", "Would it help you to get {0}?"], 5],
            [r'i am (.*)', ["Is it because you are {0} that you came to me?", "How long have you been {0}?"], 5],
            [r'(.*) problem (.*)', ["Tell me more about this problem.", "How does it make you feel?"], 8],
            [r'(.*)', ["Please tell me more.", "I see.", "Can you elaborate?"], 0]
        ]
    }
}

# --- 2. NEURAL ENGINE COMPONENTS ---
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word2id, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word2id = word2id
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx]).lower().split()
        seq = [self.word2id.get(w, 1) for w in text][:self.max_len]
        padding = [0] * (self.max_len - len(seq))
        return torch.tensor(seq + padding), torch.tensor(self.labels[idx])

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(x * weights, dim=1), weights

class EmotionalBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        return self.fc(context)

# --- 3. THE HYBRID CONTROLLER ---
class HybridEliza:
    def __init__(self, lang="en", model_path="eliza_eq.pth"):
        self.lang = lang
        # Sort keywords by Rank (index 2) descending immediately
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)
        
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        # Updated Kaggle-compliant mapping
        self.id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        self.model = None

    def build_vocab(self, texts):
        words = Counter([w for t in texts for w in str(t).lower().split()])
        for i, (w, _) in enumerate(words.most_common(5000), 2):
            self.word2id[w] = i

    def train(self, data_path, epochs, lr, batch_size, val_split=0.1):
        df = pd.read_csv(data_path)
        self.build_vocab(df['text'])
        label_col = 'label' if 'label' in df.columns else 'emotions'
        
        full_dataset = EmotionDataset(df['text'].tolist(), df[label_col].tolist(), self.word2id)
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        
        self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"[*] Training on {self.device} (6 Classes)...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Epoch Evaluation
            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%}")
        
        torch.save({'state': self.model.state_dict(), 'vocab': self.word2id}, self.model_path)

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.word2id = checkpoint['vocab']
            self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6).to(self.device)
            self.model.load_state_dict(checkpoint['state'])
            self.model.eval()

    def get_eq(self, text):
        if not self.model: return "Neutral", 0.0
        # Strip punctuation to ensure "Happy!" becomes "happy"
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [self.word2id.get(w, 1) for w in text.lower().split()][:50]
        tokens += [0] * (50 - len(tokens))
        with torch.no_grad():
            output = self.model(torch.tensor([tokens]).to(self.device))
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs).item()
            return self.id2label[idx], probs[0][idx].item()

    def rule_respond(self, text):
        text = text.lower()
        for k, v in self.script["pres"].items(): text = text.replace(k, v)
        # Because keywords were sorted by rank in __init__, we take the first match
        for pattern, resps, rank in self.script["keywords"]:
            match = re.search(pattern, text)
            if match:
                resp = random.choice(resps)
                frags = [self.reflect(g) for g in match.groups() if g]
                return resp.format(*frags) if frags else resp
        return "Please continue."

    def reflect(self, fragment):
        return " ".join([self.script["posts"].get(w, w) for w in fragment.split()])

# --- 4. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--mode", default="chat", choices=["chat", "train"])
    parser.add_argument("--data", default="emotions.csv")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    eliza = HybridEliza(lang=args.lang)

    if args.mode == "train":
        eliza.train(args.data, args.epochs, 0.001, args.batch_size, args.val_split)
    else:
        eliza.load_model()
        print(f"ELIZA: {random.choice(SCRIPTS[args.lang]['initials'])}")
        while True:
            try:
                user_in = input("You: ")
                if user_in.lower() in SCRIPTS[args.lang]["quits"]: break
                resp = eliza.rule_respond(user_in)
                emotion, score = eliza.get_eq(user_in)
                print(f"ELIZA: {resp}")
                print(f"[EQ Analysis]: Predicted Emotion: {emotion} ({score:.2%})")
            except KeyboardInterrupt: break

if __name__ == "__main__":
    main()

