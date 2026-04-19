import sys
from collections import Counter
from jiwer import wer, mer

def calculate_wil_wip(source, hypothesis):
    source_words = set(source.split())
    hypothesis_words = set(hypothesis.split())
    wil = len(source_words - hypothesis_words) / len(source_words)
    wip = len(source_words & hypothesis_words) / len(source_words)
    return wil, wip

def calculate_cer(source, hypothesis):
    cer = wer(source, hypothesis)
    return cer

def get_error_pairs(source, hypothesis):
    source_words = source.split()
    hypothesis_words = hypothesis.split()

    min_len = min(len(source_words), len(hypothesis_words))

    word_errors = [
        (source_words[i], hypothesis_words[i])
        for i in range(min_len)
        if source_words[i] != hypothesis_words[i]
    ]

    char_errors = [
        (s, h)
        for s, h in zip(source, hypothesis)
        if s != h
    ]
    return word_errors, char_errors

def main():
    if len(sys.argv) != 3:
        print("Usage: python cal_err_scores.py <source_file> <hypothesis_file>")
        sys.exit(1)

    source_file = sys.argv[1]
    hypothesis_file = sys.argv[2]

    # Read the contents of the source file
    with open(source_file, 'r', encoding='utf-8') as f:
        source = f.read().strip()

    # Read the contents of the hypothesis file
    with open(hypothesis_file, 'r', encoding='utf-8') as f:
        hypothesis = f.read().strip()

    # Calculate WER
    wer_score = wer(source, hypothesis)

    # Calculate MER
    mer_score = mer(source, hypothesis)

    # Calculate WIL and WIP
    wil, wip = calculate_wil_wip(source, hypothesis)

    # Calculate CER
    cer = calculate_cer(source, hypothesis)

    print("Word Error Rate (WER): {:.2f}%".format(wer_score * 100))
    print("Match Error Rate (MER): {:.2f}%".format(mer_score * 100))
    print("Word Information Lost (WIL): {:.2f}%".format(wil * 100))
    print("Word Information Preserved (WIP): {:.2f}%".format(wip * 100))
    print("Character Error Rate (CER): {:.2f}%".format(cer * 100))

    # Get error pairs
    word_errors, char_errors = get_error_pairs(source, hypothesis)

    # Count frequencies of word-level and character-level error pairs
    word_errors_freq = Counter(word_errors)
    char_errors_freq = Counter(char_errors)

#    # Print most frequent word-level error pairs
#    print("\nMost frequent word-level error pairs:")
#    for pair, freq in word_errors_freq.most_common():
#        print(f"{pair}: {freq} times")
#
#    # Print most frequent character-level error pairs
#    print("\nMost frequent character-level error pairs:")
#    for pair, freq in char_errors_freq.most_common():
#        print(f"{pair}: {freq} times")

if __name__ == "__main__":
    main()