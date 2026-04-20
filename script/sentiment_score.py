"""
Part 1 — VADER sentiment scoring
Loads text_clean.csv, computes neg/neu/pos/compound per row, saves result.
"""

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT_FILE  = os.path.join("processed", "text_clean.csv")
OUTPUT_FILE = os.path.join("processed", "layoffs_sentiment.csv")


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    analyzer = SentimentIntensityAnalyzer()

    scores = df["text_clean"].fillna("").apply(analyzer.polarity_scores)
    df["neg"]      = scores.apply(lambda s: s["neg"])
    df["neu"]      = scores.apply(lambda s: s["neu"])
    df["pos"]      = scores.apply(lambda s: s["pos"])
    df["compound"] = scores.apply(lambda s: s["compound"])

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved → {OUTPUT_FILE}")

    print("\n── First 3 rows ──────────────────────────────────────────────")
    print(df[["text_clean", "label", "neg", "neu", "pos", "compound"]].head(3).to_string())


if __name__ == "__main__":
    main()
