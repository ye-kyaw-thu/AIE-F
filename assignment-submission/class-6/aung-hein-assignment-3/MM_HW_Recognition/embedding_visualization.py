# embedding_visualization.py

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False


# ============================================================
# Helpers
# ============================================================

def stack_embeddings(emb_df: pd.DataFrame, emb_col="embedding"):
    """Stack embedding column into a NumPy array"""
    return np.vstack(emb_df[emb_col].values)


def attach_labels(emb_df: pd.DataFrame, meta_df: pd.DataFrame):
    """Merge syllable/base labels into stroke-level df"""
    return emb_df.merge(
        meta_df.drop_duplicates("sample_index"),
        on="sample_index",
        how="left",
    )


# ============================================================
# 1) PCA visualization
# ============================================================

def plot_pca(
    emb_df,
    meta_df=None,
    label_col=None,
    emb_col="embedding",
    max_points=5000,
    save_path=None,
):
    """
    PCA plot of stroke embeddings.
    Optionally color by label (syllable or base).
    """

    df = emb_df.sample(n=min(max_points, len(emb_df)), random_state=42)

    if meta_df is not None:
        df = attach_labels(df, meta_df)

    X = stack_embeddings(df, emb_col)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))

    if label_col and label_col in df.columns:
        for label, g in df.groupby(label_col):
            idx = g.index
            plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.6, label=label)
        plt.legend(markerscale=2, fontsize=8)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.6)

    plt.title(f"PCA of Stroke Embeddings ({pca.explained_variance_ratio_.sum():.2%} var)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)

    plt.show()


# ============================================================
# 2) UMAP visualization
# ============================================================

def plot_umap(
    emb_df,
    meta_df=None,
    label_col=None,
    emb_col="embedding",
    max_points=5000,
    n_neighbors=15,
    min_dist=0.1,
    save_path=None,
):
    if not _HAS_UMAP:
        raise RuntimeError("Install umap-learn first")

    df = emb_df.sample(n=min(max_points, len(emb_df)), random_state=42)

    if meta_df is not None:
        df = attach_labels(df, meta_df)

    X = stack_embeddings(df, emb_col)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
    )
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(6, 5))

    if label_col and label_col in df.columns:
        for label, g in df.groupby(label_col):
            idx = g.index
            plt.scatter(Z[idx, 0], Z[idx, 1], s=6, alpha=0.6, label=label)
        plt.legend(markerscale=2, fontsize=8)
    else:
        plt.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.6)

    plt.title("UMAP of Stroke Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)

    plt.show()


# ============================================================
# 3) Feature distribution / histogram
# ============================================================

def plot_feature_histograms(
    emb_df,
    emb_col="embedding",
    dims=None,
    bins=40,
    save_path=None,
):
    """
    Plot histogram of selected embedding dimensions.
    If dims=None, auto-select first 8 dims.
    """

    X = stack_embeddings(emb_df, emb_col)

    D = X.shape[1]
    if dims is None:
        dims = list(range(min(8, D)))

    n = len(dims)
    fig, axes = plt.subplots(
        nrows=(n + 1) // 2,
        ncols=2,
        figsize=(10, 4 * ((n + 1) // 2)),
    )
    axes = axes.flatten()

    for ax, d in zip(axes, dims):
        ax.hist(X[:, d], bins=bins, alpha=0.7)
        ax.set_title(f"Embedding dim {d}")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Embedding Feature Distributions", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)

    plt.show()


# ============================================================
# 4) Compare two syllables / bases
# ============================================================

def compare_two_labels(
    emb_df,
    meta_df,
    label_col,
    label_a,
    label_b,
    emb_col="embedding",
):
    """
    Compare embedding distributions of two labels.
    Prints cosine similarity & plots feature means.
    """

    df = attach_labels(emb_df, meta_df)
    A = stack_embeddings(df[df[label_col] == label_a], emb_col)
    B = stack_embeddings(df[df[label_col] == label_b], emb_col)

    mean_A = A.mean(axis=0)
    mean_B = B.mean(axis=0)

    cosine_sim = np.dot(mean_A, mean_B) / (
        np.linalg.norm(mean_A) * np.linalg.norm(mean_B)
    )

    print(f"Cosine similarity ({label_a} vs {label_b}): {cosine_sim:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(mean_A, label=label_a)
    plt.plot(mean_B, label=label_b)
    plt.title(f"Mean Embedding Comparison: {label_a} vs {label_b}")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()