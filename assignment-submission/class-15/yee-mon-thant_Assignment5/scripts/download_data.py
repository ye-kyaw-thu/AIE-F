from datasets import load_dataset
import os

os.makedirs("data/raw", exist_ok=True)

def save_texts(texts, filename):
    path = f"data/raw/{filename}"
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            t = t.strip().replace("\n", " ")
            if t:
                f.write(t + "\n")
    print(f"Saved {len(texts)} lines to {path}")

# ---- General-domain training corpus ----
print("Loading Myanmar Wikipedia...")
wiki = load_dataset("wikimedia/wikipedia", "20231101.my", split="train")
save_texts(wiki["text"], "general_wikipedia.txt")

print("Loading myPOS...")
mypos = load_dataset("chuuhtetnaing/myanmar-pos-dataset", split="train")
mypos_sentences = [" ".join(row) for row in mypos["tokens"]]
save_texts(mypos_sentences, "general_mypos.txt")

# ---- Domain test sets ----
print("Loading Facebook Flores (social media)...")
fb = load_dataset("chuuhtetnaing/myanmar-facebook-flores-dataset", split="train")
save_texts(fb["sentence"], "domain_facebook.txt")

print("Loading Dhamma dataset (religious)...")
dhamma = load_dataset("chuuhtetnaing/dhamma-article-dataset", split="train")
save_texts(dhamma["body"], "domain_religious.txt")

print("Loading News translation dataset (news)...")
news = load_dataset("chuuhtetnaing/myanmar-english-news-translation-dataset", split="train")
save_texts(news["burmese"], "domain_news.txt")

print("All done! Check data/raw/ for the saved files.")
