"""
Part 5 — Apply the best model to the full dataset.
Adds a predicted_label column and saves to layoffs_predictions.csv.
"""

import os
import pickle
import numpy as np
import pandas as pd

MDL_DIR  = "models"
IN_DIR   = os.path.join("processed", "classification")
OUT_FILE = os.path.join("processed", "classification", "layoffs_predictions.csv")

LABEL_MAP_INV = {0: "personal_experience", 1: "media_public", 2: "worker_perspective"}
LABEL_MAP     = {v: k for k, v in LABEL_MAP_INV.items()}


def main():
    # Read which model won
    with open(os.path.join(IN_DIR, "best_model.txt")) as f:
        winner = f.read().strip()
    print(f"Best model: {winner}")

    full_df = pd.read_csv(os.path.join("processed", "text_tfidf.csv"))

    if winner == "TF-IDF":
        with open(os.path.join(MDL_DIR, "model_tfidf_lr.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MDL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vec = pickle.load(f)
        X = vec.transform(full_df["text_tfidf"].fillna(""))
        preds = model.predict(X)

    else:  # BERT — embeddings only exist for the 2532 matched rows
        with open(os.path.join(MDL_DIR, "model_bert_lr.pkl"), "rb") as f:
            model = pickle.load(f)
        bert_meta = pd.read_csv(os.path.join("processed", "bert", "bert_metadata.csv"))
        bert_emb  = np.load(os.path.join("processed", "bert", "bert_embeddings.npy"))

        # Predict for matched rows; leave others as NaN
        bert_preds = model.predict(bert_emb)
        bert_meta["predicted_int"] = bert_preds
        merged = full_df.merge(
            bert_meta[["id", "predicted_int"]], on="id", how="left"
        )
        preds = merged["predicted_int"].map(LABEL_MAP_INV).values
        full_df = merged.drop(columns=["predicted_int"])

    if winner == "TF-IDF":
        full_df["predicted_label"] = [LABEL_MAP_INV[p] for p in preds]
    else:
        full_df["predicted_label"] = preds

    out = full_df[["id", "subreddit", "label", "predicted_label",
                   "type", "text_clean", "time_period"]].copy()
    out.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved {len(out)} rows → {OUT_FILE}")

    print("\n── Prediction distribution ──────────────────────────────────")
    print(out["predicted_label"].value_counts().to_string())

    print("\n── Agreement (predicted == original label) ──────────────────")
    agree = (out["label"] == out["predicted_label"]).sum()
    total = out["predicted_label"].notna().sum()
    print(f"  {agree}/{total}  ({agree/total*100:.1f}%)")


if __name__ == "__main__":
    main()
