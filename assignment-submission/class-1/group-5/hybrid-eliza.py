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
# from oppa_word import HybridDAGSegmenter
import os

# class CustomSegmenter(HybridDAGSegmenter):
#     def _get_dict_score(self, word):
#         # Override dict_score weighting to reward longer words without Language Model
#         return self.dict_weight * (len(word) ** 2) if word in self.word_dict else 0.0

# dict_path = os.path.join(os.path.dirname(__file__), "oppaWord", "data", "myg2p_mypos.dict")
# segmenter = CustomSegmenter(dict_path=dict_path)

import word_segment as wseg

# Initialize myword segmenter (using unigram-word.bin & bigram-word.bin)
uni_dict_bin = os.path.join(os.path.dirname(__file__), 'dict_ver1', 'unigram-word.bin')
bi_dict_bin = os.path.join(os.path.dirname(__file__), 'dict_ver1', 'bigram-word.bin')
wseg.P_unigram = wseg.ProbDist(uni_dict_bin, True)
wseg.P_bigram = wseg.ProbDist(bi_dict_bin, False)

def tokenize_myanmar(text):
    """Segment Myanmar text using myword and return list of words."""
    text = str(text)
    # myword viterbi requires spaces to be removed first
    text_nospace = text.replace(" ", "").strip()
    if not text_nospace:
        return []
    # wseg.viterbi returns (probability, [list_of_words])
    listString = wseg.viterbi(text_nospace)
    return listString[1]

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
    },
    "my": {
        "initials": [
            "မင်္ဂလာပါ။ သင့်ရဲ့ စိတ်အခြေအနေကို ပြောပြပါ။",
            "ဘာများ စိတ်ပူစရာ ရှိလဲ ပြောပြပါ။",
            "တစ်ခုခုများ အဆင်မပြေဖြစ်နေတာလား"
        ],
        "finals": [
            "နောက်မှ ပြန်ဆက်ပြောကြမယ်။",
            "ဒီနေ့အတွက် ဒီလောက်နဲ့ ရပ်လိုက်မယ်။"
        ],
        "quits": ["တာ့တာ", "ထွက်မယ်", "ပြီးပြီ", "bye", "quit", "exit"],
        "pres": {"ကျနော်": "ကျွန်တော်", "ကျမ": "ကျွန်မ", "ကွန်ပျူတာ": "စက်"},
        "posts": {
            "ငါ": "သင်", "ငါ့": "သင့်", "ကျွန်တော်": "သင်", "ကျွန်မ": "သင်",
            "ငါ့ကို": "သင့်ကို", "ကျွန်တော့်ကို": "သင့်ကို", "ကျွန်မကို": "သင့်ကို",
            "ငါ့ရဲ့": "သင့်ရဲ့", "ကျွန်တော့်ရဲ့": "သင့်ရဲ့", "ကျွန်မရဲ့": "သင့်ရဲ့"
        },
        "synons": {
            "ပျော်": ["ပျော်", "ဝမ်းသာ", "စိတ်ချမ်းသာ"],
            "ဝမ်းနည်း": ["ဝမ်းနည်း", "စိတ်မကောင်း", "စိတ်ညစ်", "မပျော်"],
            "ဒေါသ": ["ဒေါသ", "စိတ်ဆိုး", "စိတ်တို", "မုန်း", "လီး", "စောက်", "ဖာ", "ချီး", "ခွေးကောင်", "ခွေးမသား"],
            "ကြောက်": ["ကြောက်", "စိုးရိမ်", "စိတ်ပူ", "လန့်"],
            "အံ့ဩ": ["အံ့ဩ", "မထင်ထား", "အံ့အားသင့်"]
        },
        "keywords": [
            # [Regex, [Responses], Rank]
            [r'(.*)(လီး|စောက်|ဖာ|ချီး|ခွေးကောင်|ခွေးမသား)(.*)', ["ဆဲရေးတိုင်းထွာတဲ့ စကားတွေကို ကျွန်တော် နားမထောင်ချင်ပါဘူး။ ဘာတွေစိတ်တိုနေလဲဆိုတာ သေချာပြောပြပါ။", "မိုက်ရိုင်းတဲ့ စကားတွေ မသုံးပါနဲ့။ ဒေါသထွက်နေရင်လည်း အေးအေးဆေးဆေး စကားပြောလို့ ရပါတယ်။"], 15],
            [r"(.*)သေ(.*)", ["အဲဒီလို မပြောပါနဲ့။ ဘာကြောင့် အဲဒီလို ခံစားရတာလဲ။"], 10],
            [r"(.*)လိုအပ်(.*)", ["ဘာကြောင့် {0} ကို လိုအပ်တာလဲ။", "{0} ကို ရရင် အဆင်ပြေမယ်လို့ ထင်လား။"], 6],
            [r"(.*)စိတ်မကောင်း(.*)", ["ဘာက သင့်ကို စိတ်မကောင်းဖြစ်စေတာလဲ။", "ဒီခံစားချက်ကို ပိုပြောပြပါ။"], 8],
            [r"(.*)ဝမ်းနည်း(.*)", ["ဘာကြောင့် ဝမ်းနည်းတာလဲ။", "ဘယ်အရာက ဒီလို ခံစားရစေတာလဲ။"], 8],
            [r"(.*)မပျော်(.*)", ["ဘာကြောင့် မပျော်ရတာလဲ။", "ဘယ်အရာက ဒီလို ခံစားရစေတာလဲ။"], 8],
            [r"(.*)စိတ်ညစ်(.*)", ["စိတ်ညစ်စရာက ဘာလဲ။", "ဒီအကြောင်း နည်းနည်း ပိုရှင်းပြပါ။"], 8],
            [r"(.*)ပျော်(.*)", ["ဘာကြောင့် ပျော်တာလဲ။", "ပျော်စရာ ဖြစ်ခဲ့တာကို ပြောပြပါ။"], 7],
            [r"(.*)ဝမ်းသာ(.*)", ["ဘာက သင့်ကို ဝမ်းသာစေတာလဲ။", "ဒီကောင်းတဲ့ အကြောင်းကို ထပ်ပြောပါ။"], 7],
            [r"(.*)ချစ်(.*)", ["{0} အပေါ် ဘာကြောင့် ဒီလို ခံစားရတာလဲ။", "ဒီချစ်ခြင်းအကြောင်း ပိုပြောပြပါ။"], 7],
            [r"(.*)စိတ်ဆိုး(.*)", ["ဘာက သင့်ကို စိတ်ဆိုးစေတာလဲ။", "စိတ်ဆိုးရတဲ့ အကြောင်းကို ပြောပြပါ။"], 8],
            [r"(.*)စိတ်တို(.*)", ["ဘာကြောင့် စိတ်တိုတာလဲ။", "ဒီအခြေအနေကို ပိုရှင်းပြပါ။"], 8],
            [r"(.*)ဒေါသ(.*)", ["ဒေါသဖြစ်ရတဲ့ အကြောင်းက ဘာလဲ။", "ဒီဒေါသကို ဖြစ်စေတဲ့ အရာက ဘာလဲ။"], 8],
            [r"(.*)ကြောက်(.*)", ["ဘာကို ကြောက်နေတာလဲ။", "ဒီကြောက်ရွံ့မှုအကြောင်း ပိုပြောပြပါ။"], 8],
            [r"(.*)စိုးရိမ်(.*)", ["ဘာအတွက် စိုးရိမ်နေတာလဲ။", "ဒီစိုးရိမ်မှုရဲ့ အကြောင်းရင်းကို ပြောပြပါ။"], 8],
            [r"(.*)စိတ်ပူ(.*)", ["ဘာက သင့်ကို စိတ်ပူစေတာလဲ။", "ဒီစိတ်ပူမှုကို ပိုရှင်းပြပါ။"], 8],
            [r"(.*)အံ့ဩ(.*)", ["ဘာကြောင့် အံ့ဩတာလဲ။", "ဒီအံ့ဩစရာ အကြောင်းကို ပြောပြပါ။"], 7],
            [r"(.*)မထင်ထား(.*)", ["ဘာကို မထင်ထားတာလဲ။", "ဒီအကြောင်းက ဘာကြောင့် ထူးဆန်းတာလဲ။"], 7],
            [r"(.*)နေကောင်း(.*)", ["နေကောင်းပါတယ်။ သတိရပြီး မေးပေးတဲ့အတွက် ကျေးဇူးပါ။ သင့်ရော ဘယ်လိုနေလဲ?", "ကျွန်တော်ကတော့ အမြဲတမ်း နေကောင်းနေတဲ့ AI လေးပါ။ သင့် ကျန်းမာရေးကော အဆင်ပြေရဲ့လား?"], 9],
            [r"(.*)လိုတယ်(.*)", ["ဘာလို့ {0} ကို လိုချင်ရတာလဲ?", "{0} ရလိုက်ရင်ကော တကယ် အဆင်ပြေသွားမှာမို့လို့လား?"], 5],
            [r"(.*)ဖြစ်နေ(.*)", ["{0}ဖြစ်နေလို့ ကျွန်တော့်ဆီ လာခဲ့တာလား?", "ဒီလို {0}ဖြစ်နေတာ ဘယ်လောက်ကြာပြီလဲ?"], 5],
            [r"(.*)ပြဿနာ(.*)", ["ဒီပြဿနာအကြောင်းလေး ပိုပြီး ပြောပြပေးပါလား?", "အဲ့ဒါက သင့်ကို ဘယ်လို ခံစားရစေလဲ?"], 8],
            [r"(.*)", ["ဆက်ပြောပါ။", "နည်းနည်း ပိုရှင်းပြပါ။", "အဲဒီအကြောင်း ထပ်ပြောပါ။"], 0]
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
        text_tokens = tokenize_myanmar(str(self.texts[idx]).lower())
        seq = [self.word2id.get(w, 1) for w in text_tokens][:self.max_len]
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
        self.dropout = nn.Dropout(0.4)  # Regularization: prevents overfitting on frequent patterns
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.dropout(self.embedding(x))  # Apply dropout after embedding
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
        if self.lang == "my":
            self.id2label = {
                0: "ဝမ်းနည်း", 1: "ပျော်ရွှင်", 2: "ချစ်ခြင်း", 3: "ဒေါသ", 4: "ကြောက်ရွံ့", 5: "အံ့ဩ"
            }
        else:
            self.id2label = {
                0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"
            }
        self.model = None

    def build_vocab(self, texts):
        words = Counter([w for t in texts for w in tokenize_myanmar(str(t).lower())])
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

    def evaluate_test(self, test_data_path):
        self.load_model()
        if not self.model:
            print("Error: Model not found. Train first.")
            return

        print(f"[*] Evaluating model on {test_data_path}...")
        try:
            df = pd.read_csv(test_data_path)
        except Exception as e:
            print(f"Error reading {test_data_path}: {e}")
            return
            
        label_col = 'label' if 'label' in df.columns else 'emotions'
        y_true, y_pred = [], []
        self.model.eval()
        
        for idx, row in df.iterrows():
            text = str(row['text']).lower()
            true_label = int(row[label_col])
            
            clean_text = re.sub(r'[^\w\s\u1000-\u109F]', '', text)
            text_tokens = tokenize_myanmar(clean_text)
            tokens = [self.word2id.get(w, 1) for w in text_tokens][:50]
            tokens += [0] * (50 - len(tokens))
            
            with torch.no_grad():
                output = self.model(torch.tensor([tokens]).to(self.device))
                pred_idx = torch.argmax(torch.softmax(output, dim=1)).item()
                
            y_true.append(true_label)
            y_pred.append(pred_idx)
            
        print("\n--- BiLSTM Model Evaluation Results ---")
        target_names = [self.id2label.get(i, f"Class_{i}") for i in range(6)]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    def get_eq(self, text):
        if not self.model: return "Neutral", 0.0
        
        text_lower = text.lower()
        
        # Hybrid AI: Direct keyword detection bypass
        # Skip keyword matching if text is a question (ends with 'လား', 'ဘူးလား', '?')
        is_question = text_lower.rstrip().endswith(("လား", "?", "ဘူးလား", "လား။"))
        
        # Detect pure neutral greetings (questions with no emotion word)
        neutral_greetings = ["နေကောင်းလား", "ဘာလား", "မလား", "ဘသာလား", "ကောင်းလား", "စကားလား"]
        if is_question and any(g in text_lower for g in neutral_greetings):
            return "Neutral", 1.0
        
        if not is_question:
            emotion_label_map = {
                "joy": "Joy",
                "sadness": "Sadness",
                "fear": "Fear",
                "anger": "Anger",
                "surprise": "Surprise"
            }
            for emotion_key, syn_list in self.script.get("synons", {}).items():
                for keyword in syn_list:
                    idx = text_lower.find(keyword)
                    if idx != -1:
                        preceded_by_negation = (idx > 0 and text_lower[idx - 1] == "မ")
                        if not preceded_by_negation:
                            label = emotion_label_map.get(emotion_key, emotion_key.capitalize())
                            return label, 1.0  # Rule-based → confident 100%
                    
        # No keyword found → fall back to LSTM Model
        clean_text = re.sub(r'[^\w\s\u1000-\u109F]', '', text_lower)
        text_tokens = tokenize_myanmar(clean_text)
        tokens = [self.word2id.get(w, 1) for w in text_tokens][:50]
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
    parser.add_argument("--lang", default="my")
    parser.add_argument("--mode", default="chat", choices=["chat", "train", "evaluate"])
    parser.add_argument("--data", default="myanmar_emotion_dataset.csv")
    parser.add_argument("--test_data", default="test_emotions.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    eliza = HybridEliza(lang=args.lang)

    if args.mode == "train":
        eliza.train(args.data, args.epochs, 0.001, args.batch_size, args.val_split)
    elif args.mode == "evaluate":
        eliza.evaluate_test(args.test_data)
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


