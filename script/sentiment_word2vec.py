"""
Word2Vec Sentiment Scoring (window=5)
--------------------------------------
1. Trains Word2Vec on the text_clean corpus (window=5, CBOW).
2. Computes a document vector per row by averaging token vectors.
3. Derives a sentiment score via cosine similarity to positive vs.
   negative seed word centroids — producing a [-1, 1] sentiment_w2v score.
4. Saves results to processed/layoffs_sentiment_w2v.csv.
   No visualisations produced here.
"""

import os
import re
import numpy as np
import pandas as pd
import word2vec

INPUT_FILE  = os.path.join("processed", "text_clean.csv")
OUTPUT_FILE = os.path.join("processed", "layoffs_sentiment_w2v.csv")
MODEL_DIR   = "models"
CORPUS_TMP  = os.path.join(MODEL_DIR, "w2v_corpus.txt")
MODEL_PATH  = os.path.join(MODEL_DIR, "word2vec_w5.bin")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Seed lexicon (domain-tuned for layoff discourse) ──────────────────────────
POSITIVE_SEEDS = [
    "hopeful", "grateful", "opportunity", "support", "hired",
    "recover", "optimistic", "growth", "better", "strong",
    "resilient", "improved", "confidence", "success", "offer",
]
NEGATIVE_SEEDS = [
    "devastated", "struggling", "desperate", "worried", "anxious",
    "unemployed", "rejected", "depressed", "fearful", "miserable",
    "stressed", "hopeless", "overwhelmed", "terrified", "broke",
]

# ── Tokenisation (same logic as TF-IDF prep but minimal) ─────────────────────
def tokenise(text):
    if not isinstance(text, str):
        return []
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def seed_centroid(model, seeds):
    """Average vector of seed words that exist in the vocabulary."""
    vecs = [model[w] for w in seeds if w in model]
    if not vecs:
        raise ValueError("No seed words found in vocabulary.")
    return np.mean(vecs, axis=0)


def doc_vector(model, tokens):
    """Average vector of tokens present in the vocabulary."""
    vecs = [model[t] for t in tokens if t in model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows: {len(df)}")

    tokens_col = df["text_clean"].apply(tokenise)

    # ── Write corpus file (one doc per line, space-separated) ─────────────────
    print(f"Writing corpus → {CORPUS_TMP}")
    with open(CORPUS_TMP, "w", encoding="utf-8") as f:
        for tokens in tokens_col:
            if tokens:
                f.write(" ".join(tokens) + "\n")

    # ── Train Word2Vec ─────────────────────────────────────────────────────────
    print("Training Word2Vec (CBOW, window=5, size=200, min_count=3) ...")
    word2vec.word2vec(
        train=CORPUS_TMP,
        output=MODEL_PATH,
        size=200,
        window=5,
        cbow=1,            # CBOW (0 = skip-gram)
        min_count=3,
        negative=5,
        iter_=10,
        threads=4,
        binary=False,
        verbose=False,
    )
    print(f"  Model saved → {MODEL_PATH}")

    # ── Load model (parse text format manually — avoids np.float bug) ────────
    print("Loading model vectors ...")
    model = {}
    with open(MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()          # first line: "vocab_size vector_size"
        vocab_size, vector_size = map(int, header.strip().split())
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec  = np.array(parts[1:], dtype=np.float32)
            if len(vec) == vector_size:
                model[word] = vec
    vocab = set(model.keys())
    print(f"  Vocabulary size: {len(vocab):,}")

    # ── Seed centroids ────────────────────────────────────────────────────────
    pos_vec = seed_centroid(model, POSITIVE_SEEDS)
    neg_vec = seed_centroid(model, NEGATIVE_SEEDS)

    pos_found = [w for w in POSITIVE_SEEDS if w in vocab]
    neg_found = [w for w in NEGATIVE_SEEDS if w in vocab]
    print(f"  Positive seeds found ({len(pos_found)}): {pos_found}")
    print(f"  Negative seeds found ({len(neg_found)}): {neg_found}")

    # ── Score each document ───────────────────────────────────────────────────
    print("Scoring documents ...")
    sentiments = []
    for tokens in tokens_col:
        dv = doc_vector(model, tokens)
        if dv is None:
            sentiments.append(None)
        else:
            sim_pos = cosine_sim(dv, pos_vec)
            sim_neg = cosine_sim(dv, neg_vec)
            # Normalised difference: ranges from -1 (fully negative) to +1 (fully positive)
            score = (sim_pos - sim_neg) / (sim_pos + sim_neg + 1e-9)
            sentiments.append(round(score, 6))

    df["sentiment_w2v"] = sentiments

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved → {OUTPUT_FILE}")

    # ── Summary ───────────────────────────────────────────────────────────────
    scored = df.dropna(subset=["sentiment_w2v"])
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"  Rows scored      : {len(scored)} / {len(df)}")
    print(f"  Mean score       : {scored['sentiment_w2v'].mean():.4f}")
    print(f"  Std              : {scored['sentiment_w2v'].std():.4f}")
    print(f"  Min / Max        : {scored['sentiment_w2v'].min():.4f} / {scored['sentiment_w2v'].max():.4f}")
    print("\nMean sentiment_w2v by label:")
    print(scored.groupby("label")["sentiment_w2v"].mean().round(4).to_string())

    print("\n── First 3 rows ─────────────────────────────────────────────")
    print(df[["label", "subreddit", "text_clean", "sentiment_w2v"]].head(3).to_string())


if __name__ == "__main__":
    main()
