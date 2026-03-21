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
#Chatbot Personality Manual
SCRIPTS = {
    "my": {
        "initials": ["နေကောင်းလား။ သင့်ရဲ့ အခက်အခဲကို ပြောပြပေးပါ။", "တစ်ခုခု စိတ်အနှောင့်အယှက် ဖြစ်နေတာ ရှိလား။"],
        "finals": ["သွားတော့မယ်နော်။ စကားပြောရတာ ဝမ်းသာပါတယ်။", "သင်၏ terminal သည် ၅ စက္ကန့်အတွင်း အလိုအလျောက် ပျက်စီးသွားပါလိမ့်မည်။"],
        "quits": ["သွားပြီ", "ထွက်မယ်", "ပိတ်မယ်"],
        "pres": {"မလုပ်နဲ့": "မလုပ်ပါနဲ့", "ကျွန်တော်က": "ကျွန်တော်ဖြစ်သည်", "မှတ်မိတယ်": "သတိရတယ်", "စက်": "ကွန်ပျူတာ"},
        "posts": {"ဖြစ်သည်": "ဖြစ်ကြသည်", "ကျွန်တော်": "သင်", "ကျွန်တော့်ရဲ့": "သင့်ရဲ့", "ကျွန်တော့်ကို": "သင့်ကို", "သင့်ရဲ့": "ကျွန်တော့်ရဲ့"},
        "synons": {
            "ရှိနေခြင်း": ["ဖြစ်သည်", "ရှိသည်", "ဖြစ်နေသည်"],
            "ပျော်ရွှင်မှု": ["ပျော်တယ်", "ဝမ်းသာတယ်", "အဆင်ပြေတယ်", "ကောင်းတယ်"],
            "ဝမ်းနည်းမှု": ["ဝမ်းနည်းတယ်", "စိတ်ညစ်တယ်", "နေမကောင်းဘူး", "မှိုင်းနေတယ်"]
        },
        "keywords": [
            # [Regex, [Responses], Rank]
            [r'(.*) သေ (.*)', ["အဲဒီလို မပြောပါနဲ့။ သင့်ရဲ့ ခံစားချက်တွေကို ပိုပြီး ပြောပြပေးပါဦး။"], 10],
            [r'(.*) ပြဿနာ (.*)', ["ဒီပြဿနာအကြောင်း ပိုပြောပြပါဦး။", "အဲဒါက သင့်ကို ဘယ်လို ခံစားရစေသလဲ။"], 8],
            [r'(.*) ချစ် (.*)', ["သူ့ကို ဘယ်လောက်ထိ ချစ်လဲ ပြောပြနိုင်မလား။", "ချစ်ခြင်းမေတ္တာက အေးချမ်းပါတယ်၊ ဒါပေမယ့် တစ်ခါတလေ နာကျင်ရတတ်ပါတယ်။"], 7],
            [r'(.*) လွမ်း (.*)', ["လွမ်းရတဲ့ ဝေဒနာက ခံစားရခက်ပါတယ်။ သူ့ကို ပြန်တွေ့ဖို့ မျှော်လင့်နေသလား။", "အတိတ်က အမှတ်တရတွေကို ပြန်တွေးနေမိတာလား။"], 7],
            [r'(.*) ကြောက် (.*)', ["မကြောက်ပါနဲ့။ ကျွန်တော် ဒီမှာ ရှိပါတယ်။ ဘာကို အကြောက်ဆုံးလဲ။", "ကြောက်ရွံ့စိတ်တွေကို ရင်ဆိုင်နိုင်ဖို့ ဘယ်လို ကြိုးစားမလဲ။"], 7],
            [r'(.*) ဒေါသ (.*)|(.*) စိတ်တို (.*)', ["ဒေါသထွက်နေတာကို နားလည်ပါတယ်။ ဘာက သင့်ကို ဒီလောက် ဒေါသထွက်စေတာလဲ။", "စိတ်ကို ခဏလောက် လျှော့ချလိုက်ပါ။ ဒေါသက ကိုယ့်ကိုပဲ ပိုပင်ပန်းစေပါတယ်။"], 7],
            [r'(.*) ပျော် (.*)', ["သင် ပျော်ရွှင်နေလို့ ကျွန်တော်လည်း ဝမ်းသာပါတယ်။", "ဒီထက်ပိုပြီး ပျော်ရွှင်ရတဲ့ အခိုက်အတန့်တွေကို မျှဝေပေးပါဦး။"], 7],
            [r'(.*) သီချင်း (.*)', ["သီချင်းနားထောင်တာက စိတ်ကို သက်သာစေပါတယ်။ ဘယ်လိုသီချင်းမျိုးကို အကြိုက်ဆုံးလဲ။", "ဒီသီချင်းလေးက သင့်အတွက် အဓိပ္ပါယ်တစ်ခုခု ရှိနေလို့လား။"], 6],
            [r'(.*) စိတ်ညစ် (.*)|(.*) ဝမ်းနည်း (.*)', ["စိတ်ညစ်နေတယ်ဆိုရင် ရင်ဖွင့်လိုက်ပါ။ ကျွန်တော် နားထောင်ပေးပါ့မယ်။", "ဘာတွေက သင့်ကို ဝမ်းနည်းအောင် လုပ်နေတာလဲ။"], 7],
            [r'(.*) သရဲ (.*)', ["မကြောက်ပါနဲ့။ ကျွန်တော် ဒီမှာ ရှိပါတယ်။ ဘာကို အကြောက်ဆုံးလဲ။", "ကြောက်ရွံ့စိတ်တွေကို ရင်ဆိုင်နိုင်ဖို့ ဘယ်လို ကြိုးစားမလဲ။"], 7],
            [r'ကျွန်တော် (.*) လိုအပ်တယ်', ["{0} ကို ဘာကြောင့် လိုအပ်တာလဲ။", "{0} ကို ရလိုက်ရင် သင့်အတွက် အကူအညီ ဖြစ်မလား။"], 5],
            [r'ကျွန်တော် (.*) ဖြစ်နေတယ်', ["သင် {0} ဖြစ်နေလို့ ကျွန်တော့်ဆီ ရောက်လာတာလား။", "{0} ဖြစ်နေတာ ဘယ်လောက် ကြာပြီလဲ။"], 5],
            [r'(.*)', ["ပိုပြီး ပြောပြပေးပါဦး။", "ဟုတ်ကဲ့ နားလည်ပါတယ်။", "အသေးစိတ်လေး ရှင်းပြပေးလို့ ရမလား။"], 0]
        ]
    }
}

# --- 2. NEURAL ENGINE COMPONENTS ---
# Neural Brain
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
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        # integer => dense vectors (array of floating point numbers)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        lstm_out, _ = self.lstm(x)
        context, weights = self.attention(lstm_out)
        context = self.dropout(context)
        return self.fc(context)

# --- 3. THE HYBRID CONTROLLER ---
# The Manager
class HybridEliza:
    def __init__(self, lang="en", model_path="./models/eliza_mm.pth"):
        self.lang = lang
        # Sort keywords by Rank (index 2) descending immediately
        self.script = SCRIPTS[lang]
        self.script["keywords"].sort(key=lambda x: x[2], reverse=True)
        
        self.model_path = model_path
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS is available")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.word2id = {"<PAD>": 0, "<UNK>": 1}
        # Updated Kaggle-compliant mapping
        self.id2label = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
        self.model = None

    def build_vocab(self, texts):
        words = Counter([w for t in texts for w in str(t).lower().split()])
        for i, (w, _) in enumerate(words.most_common(2000), 2):
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
        
        self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6, dropout_rate=0.3).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
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
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2%} | Learnig Rate: {lr}")
        
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
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.word2id = checkpoint['vocab']
            self.model = EmotionalBiLSTM(len(self.word2id), 128, 64, 6, dropout_rate=0.3).to(self.device)
            self.model.load_state_dict(checkpoint['state'])
            self.model.eval()

    def get_eq(self, text):
        if not self.model: return "Neutral", 0.0
        # Strip punctuation to ensure "Happy!" becomes "happy"
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = [self.word2id.get(w, 1) for w in text.lower().split()][:50]
        tokens += [0] * (50 - len(tokens))
        # we are just using the model to get the answer, not teaching it anything new
        with torch.no_grad():
            
            output = self.model(torch.tensor([tokens]).to(self.device))
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs).item()
            return self.id2label[idx], probs[0][idx].item()

    def rule_respond(self, text):
        #Hello, I'm happy at the moment
        text = text.lower()
        for k, v in self.script["pres"].items(): text = text.replace(k, v)
        #text = Hello, Im happy at the moment
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
    parser.add_argument("--model_path", type=str, default="./models/eliza_mm.pth")
    args = parser.parse_args()

    eliza = HybridEliza(lang=args.lang, model_path=args.model_path)

    if args.mode == "train":
        eliza.train(args.data, args.epochs, 0.003, args.batch_size, args.val_split)
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

