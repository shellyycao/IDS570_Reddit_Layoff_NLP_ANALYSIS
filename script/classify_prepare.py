"""
Part 1 — Prepare training data
Merges bert_clusters.csv with text_tfidf.csv on id, encodes labels,
and produces a stratified 80/20 train/test split.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUSTER_FILE = os.path.join(ROOT_DIR, "processed", "bert", "bert_clusters.csv")
TFIDF_FILE   = os.path.join(ROOT_DIR, "processed", "text_tfidf.csv")
EMB_FILE     = os.path.join(ROOT_DIR, "processed", "bert", "bert_embeddings.npy")
OUT_DIR      = os.path.join(ROOT_DIR, "processed", "classification")

LABEL_MAP = {
    "personal_experience": 0,
    "media_public":        1,
    "worker_perspective":  2,
}
RANDOM_STATE = 42


def main():
    print("Loading data ...")
    clusters = pd.read_csv(CLUSTER_FILE)
    tfidf    = pd.read_csv(TFIDF_FILE)[["id", "text_tfidf"]]
    emb      = np.load(EMB_FILE)

    # Merge on id — keeps only rows that have BERT embeddings
    df = clusters.merge(tfidf, on="id", how="left")
    df["label_int"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_int", "text_tfidf"]).reset_index(drop=True)
    print(f"  Merged rows: {len(df)}")

    # Align embeddings to merged df (bert_clusters rows are in the same order as emb)
    # Re-index embeddings to match df after dropna
    emb_aligned = emb[df.index.tolist()]

    y = df["label_int"].astype(int).values

    # Stratified 80/20 split
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Save split index + metadata
    df["split"] = "train"
    df.loc[te_idx, "split"] = "test"
    df.to_csv(os.path.join(OUT_DIR, "classification_data.csv"), index=False, encoding="utf-8")

    # Save aligned embeddings for train/test
    np.save(os.path.join(OUT_DIR, "X_bert_train.npy"), emb_aligned[tr_idx])
    np.save(os.path.join(OUT_DIR, "X_bert_test.npy"),  emb_aligned[te_idx])
    np.save(os.path.join(OUT_DIR, "y_train.npy"),       y[tr_idx])
    np.save(os.path.join(OUT_DIR, "y_test.npy"),         y[te_idx])

    # Save tfidf text splits
    df.loc[tr_idx, ["id", "text_tfidf", "label_int"]].to_csv(
        os.path.join(OUT_DIR, "train_tfidf.csv"), index=False)
    df.loc[te_idx, ["id", "text_tfidf", "label_int"]].to_csv(
        os.path.join(OUT_DIR, "test_tfidf.csv"), index=False)

    print("\n── Class distribution ───────────────────────────────────────")
    inv_map = {v: k for k, v in LABEL_MAP.items()}
    for split_name, split_idx in [("TRAIN", tr_idx), ("TEST", te_idx)]:
        counts = pd.Series(y[split_idx]).value_counts().sort_index()
        print(f"\n  {split_name} (n={len(split_idx)}):")
        for k, v in counts.items():
            print(f"    {inv_map[k]:25s}: {v}  ({v/len(split_idx)*100:.1f}%)")


if __name__ == "__main__":
    main()
