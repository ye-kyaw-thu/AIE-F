## Hybrid-ELIZA (Myanmar-aware LSTM version)
## Derived from hybrid-eliza-mm.py without modifying the original file

import os
import re
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import argparse
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


MYANMAR_CHAR_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]")
MYANMAR_TOKEN_RE = re.compile(r"[\u1000-\u109F\uAA60-\uAA7F]+|[a-zA-Z0-9]+")


SCRIPTS = {
    "en": {
        "initials": ["How do you do. Please tell me your problem.", "Is something troubling you?"],
        "finals": ["Goodbye. It was nice talking to you.", "Take care."],
        "quits": ["bye", "quit", "exit"],
        "pres": {"don't": "dont", "i'm": "i am", "recollect": "remember", "machine": "computer"},
        "posts": {"am": "are", "i": "you", "my": "your", "me": "you", "your": "my"},
        "keywords": [
            [r"(.*) die (.*)", ["Please don't talk like that. Tell me more about your feelings."], 10],
            [r"i need (.*)", ["Why do you need {0}?", "Would it help you to get {0}?"], 5],
            [r"i am (.*)", ["Is it because you are {0} that you came to me?", "How long have you been {0}?"], 5],
            [r"(.*) problem (.*)", ["Tell me more about this problem.", "How does it make you feel?"], 8],
            [r"(.*)", ["Please tell me more.", "I see.", "Can you elaborate?"], 0],
        ],
    },
     "mm": {
        "initials": [
            "မင်္ဂလာပါ။ သင့်စိတ်ထဲမှာရှိတာကို ပြောပြပါ။",
            "ဘာက သင့်ကို စိတ်အနှောင့်အယှက်ဖြစ်စေတာလဲ။",
        ],
        "finals": [
            "နှုတ်ဆက်ပါတယ်။ ပြောပြပေးတာ ကျေးဇူးတင်ပါတယ်။",
            "ဒီနေ့အတွက် ဒီလောက်နဲ့ရပ်မယ်နော်။",
        ],
        "quits": ["တာ့တာ", "ထွက်မယ်", "ပြီးပြီ", "bye", "quit", "exit"],
        "pres": {
            "ကျွန်တော်": "ကျွန်တော်",
            "ကျွန်မ": "ကျွန်မ",
            "ငါ": "ငါ",
            "မသိဘူး": "မသိ",
            "မပျော်ဘူး": "မပျော်",
        },
        "posts": {
            "ကျွန်တော်": "သင်",
            "ကျွန်မ": "သင်",
            "ကျွန်ုပ်": "သင်",
            "ငါ": "သင်",
            "ကျွန်တော့်": "သင့်",
            "ကျွန်မရဲ့": "သင့်",
            "ငါ့": "သင့်",
            "ငါ့ရဲ့": "သင့်",
            "သင်": "ကျွန်တော်",
            "သင့်": "ကျွန်တော့်",
        },
        "keywords": [
            [r"(.*)(သေချင်|မနေချင်တော့|ကိုယ့်ကိုယ်ကို သတ်ချင်)(.*)", ["အဲဒီလို ခံစားနေရတာကို ပိုပြောပြပါ။ အခု ဘာဖြစ်နေတယ်လို့ ထင်သလဲ။"], 10],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)လိုအပ်(?:တယ်|ပါတယ်)", ["ဘာကြောင့် {0} လိုအပ်တာလဲ။", "{0} ရရင် သက်သာမယ်လို့ ထင်သလား။"], 6],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ခံစားရ(?:တယ်|ပါတယ်)", ["{0} လို့ ခံစားရတာ ဘယ်အချိန်က စတာလဲ။", "{0} လို့ ခံစားရတာကို ပိုရှင်းပြပါ။"], 6],
            [r"(?:ကျွန်တော်|ကျွန်မ|ငါ)\s?(.*)ဖြစ်နေ(?:တယ်|ပါတယ်)", ["{0} ဖြစ်နေတာ ဘာကြောင့်လို့ ထင်သလဲ။", "{0} ဖြစ်နေတာကို ပိုပြောပြပါ။"], 5],
            [r"(.*)(ပြဿနာ|အခက်အခဲ)(.*)", ["ဒီပြဿနာအကြောင်း ပိုပြောပြပါ။", "ဒီအရာက သင့်ကို ဘယ်လို ခံစားရစေလဲ။"], 8],
            [r"(.*)", ["ဆက်ပြောပြပါ။", "နားလည်ပါတယ်။", "အဲဒါကို နည်းနည်းပိုရှင်းပြပါ။"], 0],
        ],
    },
}


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[၊။!?,;:\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_char_ngrams(text, min_n=2, max_n=3):
    compact = text.replace(" ", "")
    ngrams = []
    for n in range(min_n, max_n + 1):
        if len(compact) < n:
            continue
        for i in range(len(compact) - n + 1):
            ngrams.append(compact[i : i + n])
    return ngrams


def tokenize_text(text, lang):
    text = normalize_text(text)
    if not text:
        return []

    if lang == "mm":
        if " " in text:
            base_tokens = [tok for tok in text.split() if tok]
        else:
            base_tokens = MYANMAR_TOKEN_RE.findall(text)
        return base_tokens + build_char_ngrams(text)

    return text.split()


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, word2id, tokenizer, max_len=50):
        self.texts = texts
        self.labels = labels
        self.word2id = word2id
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])[: self.max_len]
        seq = [self.word2id.get(w, 1) for w in tokens]
        padding = [0] * (self.max_len - len(seq))
        length = max(1, len(seq))
        return torch.tensor(seq + padding), torch.tensor(int(self.labels[idx])), torch.tensor(length)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, mask=None):
        scores = self.attn(x)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1), weights


class EmotionalBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, lengths):
        mask = x != 0
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))

        context, _weights = self.attention(lstm_out, mask=mask)
        context = self.dropout(context)
        return self.fc(context)


class HybridEliza:
    def __init__(
        self,
        lang="mm",
        model_path=None,
        embed_dim=128,
        hidden_dim=96,
        num_layers=2,
        dropout=0.35,
        weight_decay=1e-4,
        patience=4,
    ):
        self.lang = lang
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)
        self.model_path = model_path or ("eliza_eq_mm_lstm.pth" if lang == "mm" else "eliza_eq_lstm.pth")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        self.id2label = {
            0: "ဝမ်းနည်းမှု",
            1: "ပျော်ရွှင်မှု",
            2: "ချစ်ခင်မှု",
            3: "ဒေါသ",
            4: "ကြောက်ရွံ့မှု",
            5: "အံ့အားသင့်မှု",
        }
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.num_classes = len(self.id2label)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.model = None

    def tokenize(self, text):
        return tokenize_text(text, self.lang)

    def build_vocab(self, texts):
        words = Counter(token for text in texts for token in self.tokenize(text))
        for i, (word, _) in enumerate(words.most_common(15000), 2):
            self.word2id[word] = i

    def build_label_maps(self, labels):
        unique_labels = sorted({int(label) for label in labels})
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

    def split_stratified(self, texts, labels, val_split, seed):
        grouped = {}
        for text, label in zip(texts, labels):
            grouped.setdefault(label, []).append(text)

        rng = random.Random(seed)
        train_texts, train_labels, val_texts, val_labels = [], [], [], []

        for label, label_texts in grouped.items():
            rng.shuffle(label_texts)
            if len(label_texts) == 1:
                split_at = 1
            else:
                val_count = max(1, int(round(len(label_texts) * val_split)))
                val_count = min(val_count, len(label_texts) - 1)
                split_at = len(label_texts) - val_count

            train_part = label_texts[:split_at]
            val_part = label_texts[split_at:]

            train_texts.extend(train_part)
            train_labels.extend([label] * len(train_part))
            val_texts.extend(val_part)
            val_labels.extend([label] * len(val_part))

        train_pairs = list(zip(train_texts, train_labels))
        val_pairs = list(zip(val_texts, val_labels))
        rng.shuffle(train_pairs)
        rng.shuffle(val_pairs)

        train_texts, train_labels = zip(*train_pairs)
        val_texts, val_labels = zip(*val_pairs)
        return list(train_texts), list(train_labels), list(val_texts), list(val_labels)

    def train(self, data_path, epochs, lr, batch_size, val_split=0.1, seed=42):
        import pandas as pd

        random.seed(seed)
        torch.manual_seed(seed)

        df = pd.read_csv(data_path)
        label_col = "label" if "label" in df.columns else "emotions"
        df = df.dropna(subset=["text", label_col]).copy()
        df[label_col] = df[label_col].astype(int)

        self.build_vocab(df["text"])
        self.build_label_maps(df[label_col].tolist())

        encoded_labels = [self.label_to_idx[label] for label in df[label_col].tolist()]
        train_texts, train_labels, val_texts, val_labels = self.split_stratified(
            df["text"].tolist(), encoded_labels, val_split, seed
        )

        train_ds = EmotionDataset(train_texts, train_labels, self.word2id, self.tokenize)
        val_ds = EmotionDataset(val_texts, val_labels, self.word2id, self.tokenize)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        self.model = EmotionalBiLSTM(
            len(self.word2id),
            self.embed_dim,
            self.hidden_dim,
            self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()
        best_state = None
        best_val_acc = 0.0
        epochs_without_improvement = 0

        print(f"[*] Training EmotionalBiLSTM on {self.device} ({self.num_classes} classes)...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_x, batch_y, lengths in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(batch_x, lengths), batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()

            val_acc = self.evaluate(val_loader)
            avg_loss = total_loss / len(train_loader) if train_loader else 0.0
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"[*] Early stopping at epoch {epoch + 1}. Best Val Acc: {best_val_acc:.2%}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        torch.save(
            {
                "state": self.model.state_dict(),
                "vocab": self.word2id,
                "lang": self.lang,
                "label_to_idx": self.label_to_idx,
                "idx_to_label": self.idx_to_label,
                "num_classes": self.num_classes,
                "embed_dim": self.embed_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            self.model_path,
        )

    def evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, lengths in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x, lengths)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        return correct / total if total else 0.0

    def load_model(self):
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.word2id = checkpoint["vocab"]
            self.label_to_idx = checkpoint.get("label_to_idx", {i: i for i in range(6)})
            self.idx_to_label = checkpoint.get("idx_to_label", {i: i for i in range(6)})
            self.num_classes = checkpoint.get("num_classes", len(self.idx_to_label))
            self.embed_dim = checkpoint.get("embed_dim", self.embed_dim)
            self.hidden_dim = checkpoint.get("hidden_dim", self.hidden_dim)
            self.num_layers = checkpoint.get("num_layers", self.num_layers)
            self.dropout = checkpoint.get("dropout", self.dropout)
            self.model = EmotionalBiLSTM(
                len(self.word2id),
                self.embed_dim,
                self.hidden_dim,
                self.num_classes,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
            self.model.load_state_dict(checkpoint["state"])
            self.model.eval()

    def get_eq(self, text):
        if not self.model:
            return "Neutral", 0.0

        tokens = [self.word2id.get(w, 1) for w in self.tokenize(text)][:50]
        length = max(1, len(tokens))
        tokens += [0] * (50 - len(tokens))
        with torch.no_grad():
            output = self.model(torch.tensor([tokens]).to(self.device), torch.tensor([length]))
            probs = torch.softmax(output, dim=1)
            idx = int(torch.argmax(probs).item())
            original_label = self.idx_to_label.get(idx, idx)
            return self.id2label.get(original_label, str(original_label)), probs[0][idx].item()

    def reflect(self, fragment):
        tokens = self.tokenize(fragment)
        reflected = [self.script["posts"].get(tok, tok) for tok in tokens]
        return " ".join(reflected).strip()

    def rule_respond(self, text):
        text = normalize_text(text)
        for src, dst in self.script["pres"].items():
            text = text.replace(src, dst)

        for pattern, responses, _rank in self.script["keywords"]:
            match = re.search(pattern, text)
            if match:
                response = random.choice(responses)
                fragments = [self.reflect(group) for group in match.groups() if group and group.strip()]
                return response.format(*fragments) if fragments else response
        return "ဆက်ပြောပြပါ။" if self.lang == "mm" else "Please continue."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="mm", choices=["en", "mm"])
    parser.add_argument("--mode", default="chat", choices=["chat", "train"])
    parser.add_argument("--data", default="./../data/merged_preporcessed/data_before_downsampling.csv")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    args = parser.parse_args()

    eliza = HybridEliza(
        lang=args.lang,
        model_path=args.model_path,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    if args.mode == "train":
        eliza.train(args.data, args.epochs, args.lr, args.batch_size, args.val_split, args.seed)
    else:
        eliza.load_model()
        print(f"ELIZA: {random.choice(SCRIPTS[args.lang]['initials'])}")
        while True:
            try:
                user_in = input("You: ")
                if normalize_text(user_in) in SCRIPTS[args.lang]["quits"]:
                    print(f"ELIZA: {random.choice(SCRIPTS[args.lang]['finals'])}")
                    break
                resp = eliza.rule_respond(user_in)
                emotion, score = eliza.get_eq(user_in)
                print(f"ELIZA: {resp}")
                print(f"[EQ Analysis]: Predicted Emotion: {emotion} ({score:.2%})")
            except KeyboardInterrupt:
                print()
                print(f"ELIZA: {random.choice(SCRIPTS[args.lang]['finals'])}")
                break


if __name__ == "__main__":
    main()
