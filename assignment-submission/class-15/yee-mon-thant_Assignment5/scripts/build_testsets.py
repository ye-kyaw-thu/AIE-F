
import os

def extract_fixed_tokens(input_path, output_path, n_tokens=200, tokens_per_line=20):
    # Read the whole tokenized file as one string
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into a flat list of individual tokens (syllables/words)
    tokens = text.split()

    # Check the file if it has fewer tokens than requested
    if len(tokens) < n_tokens:
        print(f"WARNING: {input_path} only has {len(tokens)} tokens, need {n_tokens}")
        n_tokens = len(tokens)

    # Keep only the first n_tokens
    selected = tokens[:n_tokens]

    # Write tokens back out
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(selected), tokens_per_line):
            line = ' '.join(selected[i:i+tokens_per_line])
            f.write(line + '\n')

    print(f"Wrote {len(selected)} tokens ({len(selected)//tokens_per_line} lines) to {output_path}")

os.makedirs("data/testsets", exist_ok=True)

# Map each tokenized domain file to its output filename in data/testsets/
domains = {
    "domain_facebook.txt": "facebook.txt",
    "domain_religious.txt": "religious.txt",
    "domain_news.txt": "news.txt",
}

# Build a fixed 200-token test set for each of the 3 domains
for src, dst in domains.items():
    extract_fixed_tokens(f"data/tokenized/{src}", f"data/testsets/{dst}", n_tokens=200, tokens_per_line=20)

print("Done building domain test sets.")
