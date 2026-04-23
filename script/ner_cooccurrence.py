"""
Co-occurrence matrix builder — ±2-sentence windows around layoff keywords.
Generates cooccurrence_personal.csv, cooccurrence_media.csv,
cooccurrence_worker.csv, and image/cooccurrence/cooccurrence_combined.png.
"""

import re
import itertools
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("stopwords",  quiet=True)

CUSTOM_STOPWORDS = [
    "im", "ive", "id", "youre", "theyre", "hes", "shes", "weve",
    "dont", "cant", "wont", "didnt", "wasnt", "isnt", "wouldnt",
    "couldnt", "shouldnt", "doesnt", "hadnt", "havent", "hasnt",
    "get", "got", "getting",
    "one", "last", "new", "like", "time",
    "000",
]
STOP_WORDS = set(stopwords.words("english")) | set(CUSTOM_STOPWORDS)

KEYWORD_PAT = re.compile(r"\blayoffs?\b|\blaid off\b", re.IGNORECASE)
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COOC_DIR = os.path.join(BASE, "processed")
IMG_COOC = os.path.join(BASE, "image", "cooccurrence")

LABEL_ORDER = ["media_public", "personal_experience", "worker_perspective"]
LABEL_MAP   = {
    "personal_experience": "personal",
    "media_public":        "media",
    "worker_perspective":  "worker",
}
HEATMAP_CMAPS = {
    "personal_experience": "YlOrRd",
    "media_public":        "Blues",
    "worker_perspective":  "Greens",
}
LABEL_TITLES = {
    "personal_experience": "Personal Experience",
    "media_public":        "Media / Public",
    "worker_perspective":  "Worker Perspective",
}


def extract_window_pm2(text):
    if not isinstance(text, str) or not text.strip():
        return None
    sentences = SENT_SPLIT.split(text.strip())
    for i, sent in enumerate(sentences):
        if KEYWORD_PAT.search(sent):
            lo = max(0, i - 2)
            hi = min(len(sentences), i + 3)
            return " ".join(sentences[lo:hi])
    return None


def tokenize_window(window):
    tokens = re.sub(r"[^a-z0-9\s]", " ", window.lower()).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) >= 2 and not t.isdigit()]


def build_cooccurrence(token_lists, top_n=30):
    freq  = Counter(tok for toks in token_lists for tok in toks)
    vocab = [w for w, _ in freq.most_common(top_n)]
    cooc  = defaultdict(int)
    for toks in token_lists:
        window_set = set(toks) & set(vocab)
        for w1, w2 in itertools.combinations(sorted(window_set), 2):
            cooc[(w1, w2)] += 1
            cooc[(w2, w1)] += 1
    mat = pd.DataFrame(0, index=vocab, columns=vocab)
    for (w1, w2), cnt in cooc.items():
        if w1 in mat.index and w2 in mat.columns:
            mat.loc[w1, w2] = cnt
    return mat, freq


def main():
    raw = pd.read_csv(os.path.join(BASE, "processed", "text_clean.csv"))
    raw["window"] = raw["text_clean"].apply(extract_window_pm2)
    matched = raw.dropna(subset=["window"]).copy()
    matched["tokens"] = matched["window"].apply(tokenize_window)
    print(f"Rows with keyword window: {len(matched)} / {len(raw)}")

    cooc_matrices = {}
    for label, short in LABEL_MAP.items():
        subset = matched[matched["label"] == label]["tokens"].tolist()
        mat, _ = build_cooccurrence(subset, top_n=30)
        cooc_matrices[label] = mat
        out_path = os.path.join(COOC_DIR, f"cooccurrence_{short}.csv")
        mat.to_csv(out_path)
        print(f"  Saved {label} matrix {mat.shape} -> {out_path}")

    print("\nAll co-occurrence matrices saved.")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    for ax, label in zip(axes, LABEL_ORDER):
        full_mat  = cooc_matrices[label]
        top_words = full_mat.sum(axis=1).nlargest(15).index
        mat       = full_mat.loc[top_words, top_words]
        sns.heatmap(
            mat,
            cmap=HEATMAP_CMAPS[label],
            linewidths=0.3,
            linecolor="white",
            annot=False,
            ax=ax,
            cbar_kws={"label": "Co-occurrence", "shrink": 0.75},
        )
        ax.set_title(f"{LABEL_TITLES[label]}\n(top 15 words)", fontsize=12, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    fig.suptitle(
        "Word Co-occurrence Matrices: ±2-sentence windows around layoff keywords",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    out_png = os.path.join(IMG_COOC, "cooccurrence_combined.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
