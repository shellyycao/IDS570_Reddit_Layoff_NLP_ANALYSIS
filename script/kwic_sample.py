"""
KWIC — Keyword in Context
Extracts a ±1 sentence window around the first occurrence of
"layoff," "layoffs," or "laid off" per row. Samples 15 examples per label.
"""

import os
import re
import pandas as pd

INPUT_FILE  = os.path.join("processed", "text_clean.csv")
OUTPUT_FILE = os.path.join("processed", "kwic_examples.csv")

KEYWORD_PAT = re.compile(r"\blayoffs?\b|\blaid off\b", re.IGNORECASE)
SENT_SPLIT  = re.compile(r"(?<=[.!?])\s+")
LABEL_ORDER = ["personal_experience", "media_public", "worker_perspective"]
N_SAMPLE    = 15
RANDOM_STATE = 42


def extract_window(text):
    """Return ±1 sentence window around the first keyword hit, or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    sentences = SENT_SPLIT.split(text.strip())
    for i, sent in enumerate(sentences):
        if KEYWORD_PAT.search(sent):
            lo = max(0, i - 1)
            hi = min(len(sentences), i + 2)
            return " ".join(sentences[lo:hi])
    return None


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    df["keyword_context"] = df["text_clean"].apply(extract_window)
    matched = df.dropna(subset=["keyword_context"]).reset_index(drop=True)
    print(f"  Rows with keyword window: {len(matched)}")

    samples = []
    for label in LABEL_ORDER:
        group = matched[matched["label"] == label]
        n = min(N_SAMPLE, len(group))
        sampled = group.sample(n, random_state=RANDOM_STATE)
        samples.append(sampled[["label", "subreddit", "keyword_context"]])

    result = pd.concat(samples, ignore_index=True)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved {len(result)} rows → {OUTPUT_FILE}")

    # Print 5 examples per label
    print("\n" + "=" * 70)
    for label in LABEL_ORDER:
        print(f"\n── {label} ({'─' * (50 - len(label))})")
        sub = result[result["label"] == label].head(5)
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            print(f"\n  [{i}] {row['subreddit']}")
            print(f"       {row['keyword_context'][:250]}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
