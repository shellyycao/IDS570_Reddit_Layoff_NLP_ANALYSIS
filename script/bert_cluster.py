"""
Part 2 — K-means Clustering on BERT embeddings (k=3)
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

EMB_FILE  = os.path.join("processed", "bert", "bert_embeddings.npy")
META_FILE = os.path.join("processed", "bert", "bert_metadata.csv")
OUT_FILE  = os.path.join("processed", "bert", "bert_clusters.csv")

K = 3
RANDOM_STATE = 42


def main():
    print(f"Loading embeddings from {EMB_FILE} ...")
    embeddings = np.load(EMB_FILE)
    print(f"  Shape: {embeddings.shape}")

    print(f"Loading metadata from {META_FILE} ...")
    df = pd.read_csv(META_FILE)

    print(f"Running K-means with k={K} ...")
    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    df["cluster"] = kmeans.fit_predict(embeddings)

    df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved → {OUT_FILE}")

    print("\n── Cluster × Label crosstab ─────────────────────────────────")
    crosstab = pd.crosstab(df["cluster"], df["label"])
    print(crosstab.to_string())


if __name__ == "__main__":
    main()
