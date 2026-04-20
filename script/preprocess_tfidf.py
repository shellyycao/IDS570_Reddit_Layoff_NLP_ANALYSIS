import os
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data if not already present
nltk.download("punkt",        quiet=True)
nltk.download("punkt_tab",    quiet=True)
nltk.download("stopwords",    quiet=True)

INPUT_FILE  = os.path.join("processed", "text_clean.csv")
OUTPUT_FILE = os.path.join("processed", "text_tfidf.csv")

CUSTOM_STOPWORDS = [
    # Generic Reddit filler words
    "going", "make", "said", "want", "also", "one", "even",
    "still", "back", "every", "thing", "way", "lot",
    "good", "great", "bad", "big", "real", "well", "better", "less",
    "most", "those", "these", "some", "any",

    # Time words too generic to be useful
    "time", "day", "month", "year",

    # Generic action words
    "get", "got", "like", "just",

    # Generic people/subject words
    "people", "guy", "someone", "everyone", "anyone",

    # Generic work words that appear across ALL categories equally
    "work", "job", "company",

    # Added round 2
    "000", "im", "dont", "cant", "wont", "didnt", "wasnt", "isnt",
    "ive", "id", "go", "got", "get", "getting", "us", "new", "know",
    "think", "take", "feel", "need", "would", "much", "really", "something",

    # Added round 3
    "youre", "could", "never", "keep", "says", "sounds",
]


# ── Cleaning function ─────────────────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english")) | set(CUSTOM_STOPWORDS)

def preprocess_for_tfidf(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation (keep letters, digits, whitespace)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3. Tokenize with NLTK word tokenizer
    tokens = word_tokenize(text)

    # 4. Remove stopwords (NLTK English + custom list)
    tokens = [t for t in tokens if t not in STOP_WORDS]

    # 5. Remove tokens shorter than 2 characters
    tokens = [t for t in tokens if len(t) >= 2]

    # 6. Join back into a single string
    return " ".join(tokens)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    df["text_tfidf"] = df["text_clean"].apply(preprocess_for_tfidf)

    empty = (df["text_tfidf"] == "").sum()
    if empty:
        print(f"  Warning: {empty} rows are empty after preprocessing")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} rows → {OUTPUT_FILE}")

    # Sanity check
    print("\n── Sample before / after ────────────────────────────────────")
    sample = df[df["text_tfidf"] != ""][["text_clean", "text_tfidf"]].sample(3, random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  CLEAN : {str(row['text_clean'])[:200]}")
        print(f"  TFIDF : {str(row['text_tfidf'])[:200]}")


if __name__ == "__main__":
    main()
