# stroke_confusion_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_BLOCKS = {
    "geom":            slice(0, 26),
    "turn":            slice(26, 30),
    "progression":     slice(30, 42),
    "zone_pct_cov":    slice(42, 48),
    "zone_active":     slice(48, 51),
    "line_rel":        slice(51, 57),
    "dot":             slice(57, 69),
    "direction_hist":  slice(69, 77),
    "curvature_hist":  slice(77, 85),
    "projection_hist": slice(85, 101),
    "local_grid":      slice(101, None),
}


def get_embeddings_for_label(stroke_df, label):
    """Return stacked embeddings for one syllable."""
    rows = stroke_df[stroke_df["syllable_label"] == label]
    if len(rows) == 0:
        raise ValueError(f"No strokes found for label: {label}")
    return np.vstack(rows["embedding"].values)


def block_effect_size(A, B):
    """Mean Cohen-style effect over block dimensions."""
    mA, mB = A.mean(axis=0), B.mean(axis=0)
    vA, vB = A.var(axis=0), B.var(axis=0)
    pooled = np.sqrt(0.5 * (vA + vB)) + 1e-6
    return float(np.mean(np.abs(mA - mB) / pooled))


def compare_confusion_pair_blocks(
    stroke_df,
    label_a: str,
    label_b: str,
):
    """
    Compare two syllables by feature blocks.
    Returns a ranked dataframe.
    """
    A = get_embeddings_for_label(stroke_df, label_a)
    B = get_embeddings_for_label(stroke_df, label_b)

    rows = []
    for name, sl in FEATURE_BLOCKS.items():
        effect = block_effect_size(A[:, sl], B[:, sl])
        rows.append({
            "feature_block": name,
            "effect_size": effect,
            "dim": A[:, sl].shape[1],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("effect_size", ascending=False)
        .reset_index(drop=True)
    )

def plot_block_histogram(
    stroke_df,
    label_a: str,
    label_b: str,
    block_name: str,
    bins: int = 40,
):
    """
    Plot histogram comparison for ONE feature block.
    """
    if block_name not in FEATURE_BLOCKS:
        raise KeyError(f"Unknown block: {block_name}")

    sl = FEATURE_BLOCKS[block_name]

    A = get_embeddings_for_label(stroke_df, label_a)
    B = get_embeddings_for_label(stroke_df, label_b)

    # aggregate per-stroke (mean over block dims)
    A_block = A[:, sl].mean(axis=1)
    B_block = B[:, sl].mean(axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(A_block, bins=bins, alpha=0.6, density=True, label=label_a)
    plt.hist(B_block, bins=bins, alpha=0.6, density=True, label=label_b)

    plt.title(f"{block_name} distribution: {label_a} vs {label_b}")
    plt.xlabel("Feature value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_confusion_pair(
    stroke_df,
    label_a: str,
    label_b: str,
    top_k: int = 4,
):
    """
    Full confusion analysis:
    - ranks feature blocks
    - plots histograms for top-K blocks
    """
    table = compare_confusion_pair_blocks(stroke_df, label_a, label_b)

    print(f"\nConfusion analysis: {label_a} vs {label_b}")
    print(table.head(top_k))

    for block in table["feature_block"].head(top_k):
        plot_block_histogram(stroke_df, label_a, label_b, block)

