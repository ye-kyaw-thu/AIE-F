import os

def prepare_finetune_data(tokenized_path, skip_tokens=200):
    with open(tokenized_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = text.split()
    finetune_tokens = tokens[skip_tokens:]
    print(f"{tokenized_path}: {len(tokens)} total, {len(finetune_tokens)} available for fine-tuning")
    return finetune_tokens

os.makedirs("data/clean", exist_ok=True)

domains = ["facebook", "religious", "news"]
all_tokens = []

for domain in domains:
    tokens = prepare_finetune_data(f"data/tokenized/domain_{domain}.txt", skip_tokens=200)
    all_tokens.extend(tokens)

print(f"\nTotal combined fine-tuning tokens: {len(all_tokens)}")

with open("data/clean/all_domains_finetune.txt", "w", encoding="utf-8") as f:
    for i in range(0, len(all_tokens), 20):
        f.write(' '.join(all_tokens[i:i+20]) + '\n')

print("Saved to data/clean/all_domains_finetune.txt")
