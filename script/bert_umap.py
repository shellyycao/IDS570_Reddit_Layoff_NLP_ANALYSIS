"""
Part 3 — UMAP visualization of BERT embeddings
Two scatter plots: colored by cluster and by label category.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import umap

EMB_FILE    = os.path.join("processed", "bert", "bert_embeddings.npy")
CLUSTER_FILE = os.path.join("processed", "bert", "bert_clusters.csv")
OUT_IMAGE   = os.path.join("image", "bert", "bert_umap.png")

LABEL_ORDER  = ["personal_experience", "media_public", "worker_perspective"]
RANDOM_STATE = 42


def main():
    print("Loading embeddings ...")
    embeddings = np.load(EMB_FILE)
    df = pd.read_csv(CLUSTER_FILE)

    print("Running UMAP (n_components=2) ...")
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
    coords  = reducer.fit_transform(embeddings)
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    # Save updated coords back
    df.to_csv(CLUSTER_FILE, index=False, encoding="utf-8")

    cluster_palette = sns.color_palette("Set1", df["cluster"].nunique())
    label_palette   = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left — colored by cluster
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]
        axes[0].scatter(sub["umap_x"], sub["umap_y"],
                        c=[cluster_palette[cid]], alpha=0.5, s=8, label=f"Cluster {cid}")
    axes[0].set_title("UMAP — colored by K-means cluster", fontsize=13)
    axes[0].legend(title="Cluster", markerscale=3)
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")

    # Right — colored by label
    for label in LABEL_ORDER:
        sub = df[df["label"] == label]
        axes[1].scatter(sub["umap_x"], sub["umap_y"],
                        c=[label_palette[label]], alpha=0.5, s=8,
                        label=label.replace("_", " ").title())
    axes[1].set_title("UMAP — colored by label category", fontsize=13)
    axes[1].legend(title="Label", markerscale=3)
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")

    plt.suptitle("BERT Embeddings — UMAP Projection", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_IMAGE, dpi=150, bbox_inches="tight")
    print(f"  Saved → {OUT_IMAGE}")


if __name__ == "__main__":
    main()
