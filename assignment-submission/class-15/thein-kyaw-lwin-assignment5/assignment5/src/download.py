import os
import sys
import string
from datasets import load_dataset

def clean_sentence(text):
    english_punct = string.punctuation
    burmese_punct = "၊။"
    all_to_remove = english_punct + burmese_punct
    table = str.maketrans('', '', all_to_remove)
    cleaned = text.translate(table)
    return " ".join(cleaned.split())

def main():
    print("=== Step 1: Downloading & Gathering All Datasets ===")
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)

    # 1. Load general text corpus (BBC News from kalixlouiis/burmese-text-corpus)
    print("Loading kalixlouiis/burmese-text-corpus from HF...")
    gen_sentences = []
    try:
        ds = load_dataset("kalixlouiis/burmese-text-corpus", split="train")
        for item in ds:
            text = item.get("text", "")
            if text:
                cleaned = clean_sentence(text)
                if cleaned:
                    gen_sentences.append(cleaned)
        print(f"Loaded {len(gen_sentences):,} sentences from burmese-text-corpus.")
    except Exception as e:
        print(f"Error loading burmese-text-corpus: {e}")
        sys.exit(1)

    # Split general text corpus: first 100 for general test, rest for train
    test_general_candidates = gen_sentences[:100]
    train_general_part1 = gen_sentences[100:]

    # 2. Load ALT Treebank (Wikipedia translations from mutiyama/alt)
    print("Loading mutiyama/alt from HF...")
    alt_sentences = []
    try:
        ds = load_dataset("mutiyama/alt", split="train+validation+test")
        for item in ds:
            my_text = item['translation'].get('my', '')
            if my_text:
                cleaned = clean_sentence(my_text)
                if cleaned:
                    alt_sentences.append(cleaned)
        print(f"Loaded {len(alt_sentences):,} sentences from ALT Treebank.")
    except Exception as e:
        print(f"Error loading ALT: {e}")
        sys.exit(1)

    # Split ALT Treebank: first 100 for wikipedia test, rest for train
    test_wiki_candidates = alt_sentences[:100]
    train_general_part2 = alt_sentences[100:]

    # 3. Stream DatarrX/myX-Mega-Corpus (500k sentences)
    print("Streaming DatarrX/myX-Mega-Corpus (First 500,000 sentences)...")
    mega_sentences = []
    try:
        ds = load_dataset("DatarrX/myX-Mega-Corpus", split="train", streaming=True)
        count = 0
        for item in ds:
            text = item.get("text", "")
            if text:
                cleaned = clean_sentence(text)
                if cleaned:
                    mega_sentences.append(cleaned)
                    count += 1
                    if count >= 500000:
                        break
            if count % 100000 == 0 and count > 0 and len(mega_sentences) == count:
                print(f"Streamed {count:,} sentences...")
        print(f"Loaded {len(mega_sentences):,} sentences from myX-Mega-Corpus.")
    except Exception as e:
        print(f"Error streaming myX-Mega-Corpus: {e}")
        sys.exit(1)

    # 4. Load local myPOS corpus
    print("Loading local myPOS corpus...")
    mypos_sentences = []
    mypos_path = "../LM-Tutorial/data/mypos_v3.word.clean"
    if os.path.exists(mypos_path):
        with open(mypos_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = clean_sentence(line.strip())
                if cleaned:
                    mypos_sentences.append(cleaned)
        print(f"Loaded {len(mypos_sentences):,} sentences from myPOS.")
    else:
        print(f"Warning: myPOS clean file not found at {mypos_path}.")

    # 5. Load target domain candidates: news and conversational
    print("Loading mteb/MyanmarNews (News target domain)...")
    news_sentences = []
    try:
        ds = load_dataset("mteb/MyanmarNews", split="train")
        for item in ds:
            text = item.get("text", "")
            if text:
                cleaned = clean_sentence(text)
                if cleaned:
                    news_sentences.append(cleaned)
        print(f"Loaded {len(news_sentences):,} sentences from MyanmarNews.")
    except Exception as e:
        print(f"Error loading MyanmarNews: {e}")
        sys.exit(1)

    print("Loading local conversational corpus (otest)...")
    conversational_sentences = []
    otest_path = "../LM-Tutorial/data/otest.word.clean"
    if os.path.exists(otest_path):
        with open(otest_path, 'r', encoding='utf-8') as f:
            for line in f:
                cleaned = clean_sentence(line.strip())
                if cleaned:
                    conversational_sentences.append(cleaned)
        print(f"Loaded {len(conversational_sentences):,} sentences from otest.")
    else:
        print(f"Error: otest.word.clean not found at {otest_path}")
        sys.exit(1)

    # === Save Raw Split Files ===
    print("\n=== Saving Raw Split Files ===")
    
    # 1. Merge general train files: gen_train + alt_train + mega_train + mypos
    all_train = mypos_sentences + train_general_part1 + train_general_part2 + mega_sentences
    print(f"Saving merged raw training set: {len(all_train):,} sentences...")
    with open("data/raw/train_general_raw.txt", 'w', encoding='utf-8') as f:
        for s in all_train:
            f.write(s + "\n")

    # 2. General validation split
    print(f"Saving raw general validation set: {len(test_general_candidates)} sentences...")
    with open("data/raw/test_general_raw.txt", 'w', encoding='utf-8') as f:
        for s in test_general_candidates:
            f.write(s + "\n")

    # 3. News domain raw test candidates
    print(f"Saving raw news test candidates: {len(news_sentences)} sentences...")
    with open("data/raw/test_news_raw.txt", 'w', encoding='utf-8') as f:
        for s in news_sentences:
            f.write(s + "\n")

    # 4. Wikipedia domain raw test candidates
    print(f"Saving raw Wikipedia test candidates: {len(test_wiki_candidates)} sentences...")
    with open("data/raw/test_wikipedia_raw.txt", 'w', encoding='utf-8') as f:
        for s in test_wiki_candidates:
            f.write(s + "\n")

    # 5. Conversational domain raw test candidates
    print(f"Saving raw conversational test candidates: {len(conversational_sentences)} sentences...")
    with open("data/raw/test_conversational_raw.txt", 'w', encoding='utf-8') as f:
        for s in conversational_sentences:
            f.write(s + "\n")

    # 6. Save full raw files for adaptation access
    with open("data/raw/news_raw.txt", 'w', encoding='utf-8') as f:
        for s in news_sentences:
            f.write(s + "\n")
    with open("data/raw/alt_raw.txt", 'w', encoding='utf-8') as f:
        for s in alt_sentences:
            f.write(s + "\n")
    with open("data/raw/conversational_raw.txt", 'w', encoding='utf-8') as f:
        for s in conversational_sentences:
            f.write(s + "\n")

    print("\nData gathering and split candidates complete!")

if __name__ == "__main__":
    main()
