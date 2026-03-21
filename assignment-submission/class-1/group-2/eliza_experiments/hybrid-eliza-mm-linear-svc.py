## Hybrid-ELIZA (Myanmar-aware version)
## Derived from hybrid-eliza.py without modifying the original file

import os
import pickle
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import argparse
import random
import copy
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


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
        return torch.tensor(seq + padding), torch.tensor(int(self.labels[idx]))


class PooledTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout=0.35):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).unsqueeze(-1)
        masked = embedded * mask

        lengths = mask.sum(dim=1).clamp_min(1)
        mean_pool = masked.sum(dim=1) / lengths

        max_pool = embedded.masked_fill(~mask, float("-inf")).max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        features = torch.cat([mean_pool, max_pool], dim=1)
        features = self.dropout(features)
        hidden = torch.relu(self.hidden(features))
        hidden = self.dropout(hidden)
        return self.fc(hidden)


class HybridEliza:
    def __init__(self, lang="mm", model_path=None):
        self.lang = lang
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)
        self.model_path = model_path or ("eliza_eq_mm.pkl" if lang == "mm" else "eliza_eq.pkl")
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
        self.vectorizer = None
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

        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        token_pattern=r"(?u)\b\w+\b",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(2, 5),
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        self.model = LinearSVC(class_weight="balanced", random_state=seed)

        x_train = self.vectorizer.fit_transform(train_texts)
        x_val = self.vectorizer.transform(val_texts)

        print(f"[*] Training LinearSVC on {len(train_texts)} train / {len(val_texts)} val samples ({self.num_classes} classes)...")
        self.model.fit(x_train, train_labels)
        val_acc = self.model.score(x_val, val_labels)
        print(f"Validation Accuracy: {val_acc:.2%}")

        with open(self.model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "vectorizer": self.vectorizer,
                    "vocab": self.word2id,
                    "lang": self.lang,
                    "label_to_idx": self.label_to_idx,
                    "idx_to_label": self.idx_to_label,
                    "num_classes": self.num_classes,
                },
                f,
            )

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
        return correct / total if total else 0.0

    def load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                checkpoint = pickle.load(f)
            self.word2id = checkpoint["vocab"]
            self.vectorizer = checkpoint["vectorizer"]
            self.model = checkpoint["model"]
            self.label_to_idx = checkpoint.get("label_to_idx", {i: i for i in range(6)})
            self.idx_to_label = checkpoint.get("idx_to_label", {i: i for i in range(6)})
            self.num_classes = checkpoint.get("num_classes", len(self.idx_to_label))

    def get_eq(self, text):
        if not self.model:
            return "Neutral", 0.0

        features = self.vectorizer.transform([text])
        scores = self.model.decision_function(features)
        scores = torch.tensor(scores[0], dtype=torch.float32)
        probs = torch.softmax(scores, dim=0)
        idx = int(torch.argmax(probs).item())
        original_label = self.idx_to_label.get(idx, idx)
        return self.id2label.get(original_label, str(original_label)), probs[idx].item()

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
    parser.add_argument("--data", default="./../data/merged/Combined.csv")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eliza = HybridEliza(lang=args.lang, model_path=args.model_path)

    if args.mode == "train":
        eliza.train(args.data, args.epochs, 0.001, args.batch_size, args.val_split, args.seed)
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
