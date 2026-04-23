"""
Parts 2 & 3 — Train logistic regression classifiers and evaluate.
Model A: BERT embeddings
Model B: TF-IDF vectors (max 5000 features)
Saves models, reports, and confusion matrix heatmaps.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR   = os.path.join(ROOT_DIR, "processed", "classification")
IMG_DIR  = os.path.join(ROOT_DIR, "image", "classification")
MDL_DIR  = os.path.join(ROOT_DIR, "models")
OUT_DIR  = os.path.join(ROOT_DIR, "processed", "classification")

LABEL_NAMES = ["personal_experience", "media_public", "worker_perspective"]
RANDOM_STATE = 42
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MDL_DIR, exist_ok=True)


def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[lbl.replace("_", "\n") for lbl in LABEL_NAMES],
        yticklabels=[lbl.replace("_", "\n") for lbl in LABEL_NAMES],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=13)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=LABEL_NAMES)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\n{report}")

    # Save report as text
    report_path = os.path.join(OUT_DIR, f"report_{name.lower().replace(' ', '_')}.txt")
    with open(report_path, "w") as f:
        f.write(f"{name}\n{'='*55}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
    print(f"  Report → {report_path}")

    return y_pred, cm, acc


def main():
    # ── Load splits ──────────────────────────────────────────────────────────
    X_bert_train = np.load(os.path.join(IN_DIR, "X_bert_train.npy"))
    X_bert_test  = np.load(os.path.join(IN_DIR, "X_bert_test.npy"))
    y_train      = np.load(os.path.join(IN_DIR, "y_train.npy"))
    y_test       = np.load(os.path.join(IN_DIR, "y_test.npy"))

    train_tfidf = pd.read_csv(os.path.join(IN_DIR, "train_tfidf.csv"))
    test_tfidf  = pd.read_csv(os.path.join(IN_DIR, "test_tfidf.csv"))

    # ── Model A: BERT logistic regression ────────────────────────────────────
    print("Training Model A (BERT embeddings) ...")
    lr_bert = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr_bert.fit(X_bert_train, y_train)
    with open(os.path.join(MDL_DIR, "model_bert_lr.pkl"), "wb") as f:
        pickle.dump(lr_bert, f)
    print("  Model A saved → models/model_bert_lr.pkl")

    _, cm_bert, acc_bert = evaluate("Model A — BERT + LR", lr_bert, X_bert_test, y_test)
    plot_confusion_matrix(cm_bert, "Confusion Matrix — BERT + Logistic Regression",
                          "confusion_matrix_bert.png")

    # ── Model B: TF-IDF logistic regression ──────────────────────────────────
    print("\nTraining Model B (TF-IDF vectors) ...")
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_tfidf_train = vec.fit_transform(train_tfidf["text_tfidf"].fillna(""))
    X_tfidf_test  = vec.transform(test_tfidf["text_tfidf"].fillna(""))

    lr_tfidf = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE
    )
    lr_tfidf.fit(X_tfidf_train, y_train)

    with open(os.path.join(MDL_DIR, "model_tfidf_lr.pkl"), "wb") as f:
        pickle.dump(lr_tfidf, f)
    with open(os.path.join(MDL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vec, f)
    print("  Model B saved → models/model_tfidf_lr.pkl")
    print("  Vectorizer  → models/tfidf_vectorizer.pkl")

    _, cm_tfidf, acc_tfidf = evaluate("Model B — TF-IDF + LR", lr_tfidf, X_tfidf_test, y_test)
    plot_confusion_matrix(cm_tfidf, "Confusion Matrix — TF-IDF + Logistic Regression",
                          "confusion_matrix_tfidf.png")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Accuracy comparison ──────────────────────────────────────")
    print(f"  Model A (BERT)  : {acc_bert:.4f}")
    print(f"  Model B (TF-IDF): {acc_tfidf:.4f}")
    winner = "BERT" if acc_bert >= acc_tfidf else "TF-IDF"
    print(f"  Better model    : {winner}")

    # Save winner flag for Part 5
    with open(os.path.join(OUT_DIR, "best_model.txt"), "w") as f:
        f.write(winner)


if __name__ == "__main__":
    main()
