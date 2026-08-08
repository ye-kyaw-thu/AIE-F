import os
import sys
import subprocess

def run_sylbreak(input_path, output_path):
    print(f"Syllable segmenting {input_path} -> {output_path} ...")
    cmd = [
        sys.executable,
        "sylbreak/python/sylbreak.py",
        "-i", input_path,
        "-o", output_path,
        "-s", " "
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error running sylbreak on {input_path}: {result.stderr}")
        sys.exit(1)

def slice_balanced_set(tokenized_path, output_path, target_sentences=10, target_words=20):
    print(f"Slicing balanced set from {tokenized_path} -> {output_path} ...")
    sliced = []
    with open(tokenized_path, 'r', encoding='utf-8') as f:
        for line in f:
            syllables = line.strip().split()
            syllables = [s for s in syllables if s.strip()]
            if len(syllables) >= target_words:
                sliced.append(" ".join(syllables[:target_words]))
                if len(sliced) == target_sentences:
                    break
                    
    if len(sliced) < target_sentences:
        print(f"Warning: Only found {len(sliced)} valid sentences with >= {target_words} syllables in {tokenized_path}.")
        while len(sliced) < target_sentences and len(sliced) > 0:
            sliced.append(sliced[-1])
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sliced:
            f.write(s + "\n")
    print(f"Saved balanced set ({len(sliced)} sents, {target_words} syllables each) to {output_path}")

def main():
    print("=== Step 2: Running Myanmar Syllable Segmentation ===")
    
    # Create output directories
    os.makedirs("data/tokenized", exist_ok=True)
    os.makedirs("data/balanced_tests", exist_ok=True)
    
    # 1. Run sylbreak on raw files (segmenting into syllables)
    run_sylbreak("data/raw/train_general_raw.txt", "data/tokenized/train_general.txt")
    run_sylbreak("data/raw/test_general_raw.txt", "data/tokenized/test_general_intermediate.txt")
    run_sylbreak("data/raw/test_news_raw.txt", "data/tokenized/test_news_intermediate.txt")
    run_sylbreak("data/raw/test_wikipedia_raw.txt", "data/tokenized/test_wikipedia_intermediate.txt")
    run_sylbreak("data/raw/test_conversational_raw.txt", "data/tokenized/test_conversational_intermediate.txt")

    # 2. Slice and build balanced final test sets
    slice_balanced_set("data/tokenized/test_general_intermediate.txt", "data/balanced_tests/test_general.txt")
    slice_balanced_set("data/tokenized/test_news_intermediate.txt", "data/balanced_tests/test_news.txt")
    slice_balanced_set("data/tokenized/test_wikipedia_intermediate.txt", "data/balanced_tests/test_wikipedia.txt")
    slice_balanced_set("data/tokenized/test_conversational_intermediate.txt", "data/balanced_tests/test_conversational.txt")
    
    print("Syllable segmentation and test set slicing complete!")

if __name__ == "__main__":
    main()
