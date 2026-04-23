"""
Part 4 — Representative examples per cluster
Finds the 3 rows per cluster whose BERT embedding is closest to the centroid.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_FILE     = os.path.join(ROOT_DIR, "processed", "bert", "bert_embeddings.npy")
CLUSTER_FILE = os.path.join(ROOT_DIR, "processed", "bert", "bert_clusters.csv")
OUT_FILE     = os.path.join(ROOT_DIR, "processed", "bert", "bert_cluster_examples.csv")

K            = 3
TOP_N        = 3
RANDOM_STATE = 42


def main():
    print("Loading embeddings and cluster assignments ...")
    embeddings = np.load(EMB_FILE)
    df         = pd.read_csv(CLUSTER_FILE).reset_index(drop=True)

    # Refit K-means to recover centroids
    kmeans     = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(embeddings)
    centroids  = kmeans.cluster_centers_

    examples = []
    print("\n── Representative examples per cluster ──────────────────────")
    for cid in range(K):
        idx      = df[df["cluster"] == cid].index.tolist()
        sub_emb  = embeddings[idx]
        dists    = euclidean_distances(sub_emb, [centroids[cid]]).flatten()
        top_idx  = [idx[i] for i in np.argsort(dists)[:TOP_N]]

        print(f"\nCluster {cid}:")
        for rank, row_i in enumerate(top_idx, 1):
            row = df.loc[row_i]
            print(f"  [{rank}] subreddit={row['subreddit']}  label={row['label']}")
            print(f"       text: {str(row['text_clean'])[:200]}")
            examples.append({
                "cluster":   cid,
                "rank":      rank,
                "subreddit": row["subreddit"],
                "label":     row["label"],
                "text_clean": row["text_clean"],
                "window":    row.get("window", ""),
            })

    out_df = pd.DataFrame(examples)
    out_df.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
