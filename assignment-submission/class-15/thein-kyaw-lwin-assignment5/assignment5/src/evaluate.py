import os
import sys
import math
import string
import subprocess

def evaluate_kenlm(model_path, test_path):
    import sys
    notebook_dir = os.path.abspath(".")
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(notebook_dir)]
    import kenlm
    sys.path.insert(0, notebook_dir)
    
    model = kenlm.LanguageModel(model_path)
    sum_log10 = 0.0
    n_words = 0
    oov_count = 0
    
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for prob, length, oov in model.full_scores(line):
                sum_log10 += prob
                if oov:
                    oov_count += 1
            n_words += len(line.split())
            
    sum_nats = -sum_log10 * math.log(10)
    ppl = math.exp(sum_nats / n_words) if n_words else float("inf")
    entropy = sum_nats / n_words if n_words else float("inf")
    oov_rate = (oov_count / n_words * 100) if n_words else 0.0
    return ppl, entropy, oov_rate

def clean_word(word):
    english_punct = string.punctuation
    burmese_punct = "၊။"
    all_to_remove = english_punct + burmese_punct
    table = str.maketrans('', '', all_to_remove)
    return word.translate(table).strip()

def find_tools():
    local_bin = os.path.abspath("kenlm_src/build/bin")
    lmplz = os.path.join(local_bin, "lmplz")
    build_binary = os.path.join(local_bin, "build_binary")
    
    if os.path.exists(lmplz) and os.path.exists(build_binary):
        return lmplz, build_binary
    return "/workspace/assignment5/kenlm_src/build/bin/lmplz", "/workspace/assignment5/kenlm_src/build/bin/build_binary"

def main():
    print("=== Step 4: Evaluating Model on Target Domains ===")
    
    test_sets = {
        "General Validation": "data/balanced_tests/test_general.txt",
        "News Articles": "data/balanced_tests/test_news.txt",
        "Wikipedia / Formal": "data/balanced_tests/test_wikipedia.txt",
        "Conversational": "data/balanced_tests/test_conversational.txt"
    }
    
    base_model_path = "data/models/general.binary"
    if not os.path.exists(base_model_path):
        print(f"Error: Base model binary not found at {base_model_path}")
        sys.exit(1)
        
    results = {}
    print("\n--- Base Model Perplexity & Entropy Evaluation ---")
    for name, path in test_sets.items():
        if not os.path.exists(path):
            print(f"Warning: Test set {path} not found.")
            continue
        ppl, entropy, oov_rate = evaluate_kenlm(base_model_path, path)
        results[name] = {"ppl": ppl, "entropy": entropy, "oov_rate": oov_rate}
        print(f"Domain: {name:<20} | PPL: {ppl:<10.2f} | Entropy (nats): {entropy:<6.4f} | OOV: {oov_rate:.2f}%")

    # Determine hardest domain (excluding General Validation)
    target_domains = {k: v for k, v in results.items() if k != "General Validation"}
    if not target_domains:
        print("Error: No target domains found for evaluation.")
        sys.exit(1)
        
    hardest_domain = max(target_domains, key=lambda k: target_domains[k]["ppl"])
    print(f"\nHardest Domain Identified: {hardest_domain} (PPL: {results[hardest_domain]['ppl']:.2f})")

    # Load adaptation data for hardest domain from local verified raw files
    print(f"\n--- Preparing Adaptation Data for {hardest_domain} ---")
    adaptation_sentences = []
    
    if hardest_domain == "News Articles":
        print("Extracting training sentences from local news_raw.txt...")
        with open("data/raw/news_raw.txt", 'r', encoding='utf-8') as f:
            # Skip the first 100 sentences to prevent any possible test set leakage
            adaptation_sentences = [line.strip() for line in f if line.strip()][100:]
    elif hardest_domain == "Wikipedia / Formal":
        print("Extracting training sentences from local alt_raw.txt...")
        with open("data/raw/alt_raw.txt", 'r', encoding='utf-8') as f:
            adaptation_sentences = [line.strip() for line in f if line.strip()][100:]
    elif hardest_domain == "Conversational":
        print("Extracting training sentences from local conversational_raw.txt...")
        with open("data/raw/conversational_raw.txt", 'r', encoding='utf-8') as f:
            adaptation_sentences = [line.strip() for line in f if line.strip()][10:]

    # 1. Clean the adaptation sentences
    cleaned_adaptation = []
    for s in adaptation_sentences:
        parts = s.strip().split()
        cleaned_words = [clean_word(w) for w in parts if clean_word(w)]
        if cleaned_words:
            cleaned_adaptation.append(" ".join(cleaned_words))
            
    print(f"Found {len(cleaned_adaptation):,} training sentences for {hardest_domain}.")
    
    # Save raw adaptation to file
    raw_adapt_path = "data/tokenized/train_adapted_raw.txt"
    with open(raw_adapt_path, 'w', encoding='utf-8') as f:
        for s in cleaned_adaptation:
            f.write(s + "\n")
            
    # 2. Tokenize adaptation sentences using sylbreak
    tokenized_adapt_path = "data/tokenized/train_adapted_tokenized.txt"
    print("Tokenizing adaptation sentences with sylbreak...")
    cmd_tok = [
        sys.executable,
        "sylbreak/python/sylbreak.py",
        "-i", raw_adapt_path,
        "-o", tokenized_adapt_path,
        "-s", " "
    ]
    subprocess.run(cmd_tok)
    
    # Read tokenized adaptation sentences
    with open(tokenized_adapt_path, 'r', encoding='utf-8') as f:
        adapted_train_sentences = f.readlines()
        
    # Read base general training data
    with open("data/tokenized/train_general.txt", 'r', encoding='utf-8') as f:
        base_train_sentences = f.readlines()

    # 3. Create merged adapted corpus: Base + 1x upweighted adaptation data (optimal for Kneser-Ney)
    merged_train_path = "data/tokenized/train_adapted.txt"
    print(f"Creating merged corpus (Base General + 1x {hardest_domain}) ...")
    final_sentences = base_train_sentences + (adapted_train_sentences * 1)
    with open(merged_train_path, 'w', encoding='utf-8') as f:
        for s in final_sentences:
            f.write(s.strip() + "\n")

    # 4. Train Adapted Model
    print("\n--- Training Adapted Language Model ---")
    lmplz, build_binary = find_tools()
    
    arpa_adapted = "data/models/adapted.arpa"
    binary_adapted = "data/models/adapted.binary"
    
    cmd_train = f"{lmplz} -o 5 -S 2G -T /tmp --discount_fallback < {merged_train_path} > {arpa_adapted}"
    subprocess.run(cmd_train, shell=True)
    
    cmd_binary = f"{build_binary} {arpa_adapted} {binary_adapted}"
    subprocess.run(cmd_binary, shell=True)

    # 5. Evaluate Adapted Model on the Hardest Domain
    adapted_ppl, adapted_entropy, adapted_oov = evaluate_kenlm(binary_adapted, test_sets[hardest_domain])
    base_ppl = results[hardest_domain]["ppl"]
    base_oov = results[hardest_domain]["oov_rate"]
    improvement_ppl = base_ppl - adapted_ppl
    rel_improvement = (improvement_ppl / base_ppl * 100) if base_ppl else 0.0
    
    print("\n--- Evaluation of Adapted Model ---")
    print(f"Domain                  : {hardest_domain}")
    print(f"Base LM Perplexity      : {base_ppl:.2f} (OOV: {base_oov:.2f}%)")
    print(f"Adapted LM Perplexity   : {adapted_ppl:.2f} (OOV: {adapted_oov:.2f}%)")
    print(f"Absolute PPL Reduction  : {improvement_ppl:.2f}")
    print(f"Relative PPL Reduction  : {rel_improvement:.2f}%")

    # Save visualization bar chart
    try:
        import matplotlib.pyplot as plt
        domains = list(results.keys())
        base_ppls = [results[d]["ppl"] for d in domains]
        
        plt.figure(figsize=(9, 5.5))
        bars = plt.bar(domains, base_ppls, color=['#4285F4', '#EA4335', '#FBBC05', '#34A853'])
        plt.ylabel('Perplexity (PPL)')
        plt.title('KenLM Base Model Perplexity Across Domains (Syllable-Level)')
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        plt.tight_layout()
        chart_path = "data/models/ppl_comparison.png"
        plt.savefig(chart_path)
        print(f"Saved bar chart visualization to: {chart_path}")
    except Exception as e:
        print(f"Warning: Could not save Matplotlib plot: {e}")

    # Generate Markdown Table for README
    readme_table = f"""| Domain | Base LM PPL | Base LM Entropy | Base LM OOV Rate | Adapted LM PPL | Adapted LM OOV Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Validation** | {results['General Validation']['ppl']:.2f} | {results['General Validation']['entropy']:.4f} | {results['General Validation']['oov_rate']:.2f}% | - | - | Reference |
| **News Articles** | {results['News Articles']['ppl']:.2f} | {results['News Articles']['entropy']:.4f} | {results['News Articles']['oov_rate']:.2f}% | {f"{adapted_ppl:.2f} (Adapted)" if hardest_domain == "News Articles" else "-"} | {f"{adapted_oov:.2f}%" if hardest_domain == "News Articles" else "-"} | {"🔴 Hardest" if hardest_domain == "News Articles" else "Normal"} |
| **Wikipedia / Formal** | {results['Wikipedia / Formal']['ppl']:.2f} | {results['Wikipedia / Formal']['entropy']:.4f} | {results['Wikipedia / Formal']['oov_rate']:.2f}% | {f"{adapted_ppl:.2f} (Adapted)" if hardest_domain == "Wikipedia / Formal" else "-"} | {f"{adapted_oov:.2f}%" if hardest_domain == "Wikipedia / Formal" else "-"} | {"🔴 Hardest" if hardest_domain == "Wikipedia / Formal" else "Normal"} |
| **Conversational** | {results['Conversational']['ppl']:.2f} | {results['Conversational']['entropy']:.4f} | {results['Conversational']['oov_rate']:.2f}% | {f"{adapted_ppl:.2f} (Adapted)" if hardest_domain == "Conversational" else "-"} | {f"{adapted_oov:.2f}%" if hardest_domain == "Conversational" else "-"} | {"🔴 Hardest" if hardest_domain == "Conversational" else "Normal"} |
"""
    # Write report file to save final metrics
    metrics_path = "data/models/metrics.txt"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(readme_table)
    print(f"\nMetrics table saved to {metrics_path}")

if __name__ == "__main__":
    main()
