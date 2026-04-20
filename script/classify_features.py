"""
Part 4 — Feature importance (TF-IDF model only)
Top 15 most positive and negative feature weights per class.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MDL_DIR = "models"
IMG_DIR = os.path.join("image", "classification")

LABEL_NAMES  = ["personal_experience", "media_public", "worker_perspective"]
LABEL_COLORS = dict(zip(LABEL_NAMES, sns.color_palette("Set2", 3)))
TOP_N = 15


def main():
    print("Loading TF-IDF model and vectorizer ...")
    with open(os.path.join(MDL_DIR, "model_tfidf_lr.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MDL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)

    feature_names = vec.get_feature_names_out()
    coef          = model.coef_           # shape: (3, n_features)

    fig, axes = plt.subplots(3, 1, figsize=(12, 15))

    for i, (label, ax) in enumerate(zip(LABEL_NAMES, axes)):
        weights = coef[i]

        top_pos_idx = np.argsort(weights)[-TOP_N:][::-1]
        top_neg_idx = np.argsort(weights)[:TOP_N]

        top_idx   = np.concatenate([top_neg_idx, top_pos_idx[::-1]])
        top_feats = feature_names[top_idx]
        top_vals  = weights[top_idx]

        colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_vals]

        ax.barh(range(len(top_feats)), top_vals, color=colors, edgecolor="none")
        ax.set_yticks(range(len(top_feats)))
        ax.set_yticklabels(top_feats, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(
            f"{label.replace('_', ' ').title()} — top {TOP_N} positive / negative weights",
            fontsize=12
        )
        ax.set_xlabel("Logistic regression coefficient")

        # Legend proxy
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#d62728", label="Positive (pushes toward this class)"),
            Patch(color="#1f77b4", label="Negative (pushes away from this class)"),
        ], fontsize=8, loc="lower right")

    plt.suptitle("TF-IDF Logistic Regression — Feature Weights per Class",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "tfidf_feature_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
