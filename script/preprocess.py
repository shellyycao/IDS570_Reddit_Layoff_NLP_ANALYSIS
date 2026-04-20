import pandas as pd
import re
import html
import unicodedata
import os

INPUT_FILE  = os.path.join("data_ready", "layoffs_reddit_data.csv")
OUTPUT_DIR  = "processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "text_clean.csv")


# ── Cleaning function ─────────────────────────────────────────────────────────

def clean_text(text):
    # Handle non-string / NaN values
    if not isinstance(text, str):
        return ""

    # 1. Decode encoding errors — encode to UTF-8 ignoring bad bytes, decode back
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # 2. Remove Reddit-specific artifacts before unescaping so they don't
    #    survive as bare punctuation (e.g. &gt; → > would stay in text)
    text = re.sub(r"&gt;|&lt;|&amp;|&nbsp;", " ", text)

    # 3. Unescape any remaining HTML entities, then strip all &...; patterns
    text = html.unescape(text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)

    # 4. Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Remove emojis and non-ASCII symbols
    #    Normalize to NFKD, then keep only ASCII-compatible characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii", errors="ignore")

    # 6. Collapse multiple whitespace (spaces, tabs, newlines) into a single space
    text = re.sub(r"\s+", " ", text)

    # 7. Strip leading and trailing whitespace
    text = text.strip()

    return text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    # Apply cleaning to the text column
    df["text_clean"] = df["text"].apply(clean_text)

    # Report rows where cleaning produced an empty string
    empty = (df["text_clean"] == "").sum()
    if empty:
        print(f"  Warning: {empty} rows have empty text after cleaning")

    # Create output directory if needed
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save — keep all original columns plus the new text_clean column
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} rows → {OUTPUT_FILE}")

    # Quick sanity check: print a before/after sample
    print("\n── Sample before / after ────────────────────────────────────")
    sample = df[["text", "text_clean"]].dropna().sample(3, random_state=42)
    for _, row in sample.iterrows():
        print(f"\n  BEFORE: {str(row['text'])[:200]}")
        print(f"  AFTER : {str(row['text_clean'])[:200]}")


if __name__ == "__main__":
    main()
