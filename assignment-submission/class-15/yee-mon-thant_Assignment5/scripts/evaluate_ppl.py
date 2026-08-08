import os
import subprocess
import re
import matplotlib.pyplot as plt

def run_ppl_test(test_file, model_path="results/lstm_general.pt", token_level="word"):
    """Runs lstm_lm.py in test mode on one domain file, and extracts the PPL number."""
    result = subprocess.run(
        ["python", "scripts/lstm_lm.py", "--mode", "test",
         "--test_file", test_file, "--model_path", model_path,
         "--token_level", token_level],
        capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    print(output)

    match = re.search(r"PPL:\s*([\d.]+)", output)
    if match:
        return float(match.group(1))
    else:
        print(f"WARNING: could not find PPL in output for {test_file}")
        return None


def plot_ppl(domain_ppl, output_path="results/ppl_by_domain.png"):
    domains = list(domain_ppl.keys())
    values = list(domain_ppl.values())

    plt.figure(figsize=(7, 5))
    bars = plt.bar(domains, values, color="#2a78d6")

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{value:.2f}", ha="center", fontsize=11)

    plt.title("Perplexity by Domain (General LSTM Language Model)")
    plt.xlabel("Domain")
    plt.ylabel("Perplexity (PPL)")
    plt.ylim(0, max(values) + 15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved chart to {output_path}")


os.makedirs("results", exist_ok=True)

test_files = {
    "Facebook": "data/testsets/facebook.txt",
    "Religious": "data/testsets/religious.txt",
    "News": "data/testsets/news.txt",
}

print("Running PPL evaluation on each domain...")
print("=" * 50)

domain_ppl = {}
for domain_name, test_path in test_files.items():
    print(f"\n--- Testing on {domain_name} ---")
    ppl = run_ppl_test(test_path)
    if ppl is not None:
        domain_ppl[domain_name] = ppl

if domain_ppl:
    print("\nFinal PPL results:")
    for domain, ppl in domain_ppl.items():
        print(f"  {domain}: {ppl}")
    plot_ppl(domain_ppl)
else:
    print("No PPL results collected - check that results/lstm_general.pt exists.")
