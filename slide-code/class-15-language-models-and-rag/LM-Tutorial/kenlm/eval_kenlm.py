import kenlm, sys, math

model_path = sys.argv[1]
test_path = sys.argv[2]

model = kenlm.LanguageModel(model_path)
sum_log10 = 0.0
n_words = 0

with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        for prob, length, oov in model.full_scores(line):
            sum_log10 += prob
        n_words += len(line.split())

# Convert log10 to natural log, then calculate PPL
#sum_nats = sum_log10 * math.log(10)
#ppl = math.exp(-sum_nats / n_words) if n_words else float("inf")
#print(f"PPL: {ppl:.2f}")

# Cleaner, unambiguous version
sum_nats = -sum_log10 * math.log(10)
ppl = math.exp(sum_nats / n_words)
entropy = sum_nats / n_words
print(f"PPL: {ppl:.2f}")
print(f"Entropy (nats): {entropy:.4f}")
