import re
import os

def clean_text(text):

    # keep only Myanmar, English, digits, spaces
    text = re.sub(r'[^\u1000-\u109F\uAA60-\uAA7Fa-zA-Z0-9 ]', ' ', text)

    # Remove Myanmar punctuation marks
    text = text.replace('၊', '').replace('။', '')

    # collapse extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def clean_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            cleaned = clean_text(line)
            if cleaned:
                fout.write(cleaned + '\n')

os.makedirs("data/clean", exist_ok=True)
for f in ["general_wikipedia.txt", "general_mypos.txt", "domain_facebook.txt",
          "domain_religious.txt", "domain_news.txt"]:
    clean_file(f"data/raw/{f}", f"data/clean/{f}")
    print(f"Cleaned {f}")

print("Done.")