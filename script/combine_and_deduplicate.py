import os
import pandas as pd

DATA_DIR  = "data_ready"
MAIN_FILE = os.path.join(DATA_DIR, "layoffs_reddit_data.csv")

STOPWORDS = {
    "the","a","an","is","it","in","on","at","to","of","and","or","but",
    "with","for","this","that","was","are","be","have","has","had",
    "i","you","he","she","they","we","my","your","his","her","their",
    "just","like","really","so","very","also","do","did","not","no",
    "if","can","will","would","could","should","what","how","when",
    "get","got","think","know","people","one","don","t","s","re","ve",
    "deleted","removed","www","http","reddit","com","amp",
    "layoff","layoffs","laid","lay","job","jobs","cut","cuts",
    "work","company","companies","employee","employees","fired","firing",
}


def generate_wordclouds(df):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        print("Run: pip install wordcloud matplotlib")
        return

    os.makedirs(os.path.join("image", "wordcloud"), exist_ok=True)
    for label, group in df.groupby("label"):
        corpus = " ".join(group["text"].dropna().astype(str))
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            stopwords=STOPWORDS,
            max_words=150,
            colormap="viridis",
        ).generate(corpus)
        fname = os.path.join("image", "wordcloud", f"wordcloud_{label}.png")
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud — {label.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved → {fname}")


def print_summary(df):
    print("\nBy label:")
    for k, v in sorted(df["label"].value_counts().items()):
        print(f"  {k:30s}: {v}")
    print("\nBy subreddit:")
    for k, v in sorted(df["subreddit"].value_counts().items()):
        print(f"  {k:30s}: {v}")
    print("\nBy time period:")
    for k, v in sorted(df["time_period"].value_counts().items(), key=lambda x: str(x[0])):
        print(f"  {str(k):10s}: {v}")


def main():
    source_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".csv") and f != "layoffs_reddit_data.csv"
    ]
    print(f"Found {len(source_files)} per-subreddit files")

    frames = []
    for f in sorted(source_files):
        df = pd.read_csv(f)
        print(f"  {os.path.basename(f)}: {len(df)} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined before dedup: {len(combined)} rows")
    combined.drop_duplicates(subset="id", inplace=True)
    print(f"After dedup: {len(combined)} rows")

    combined.to_csv(MAIN_FILE, index=False)
    print(f"Saved → {MAIN_FILE}")

    print_summary(combined)
    generate_wordclouds(combined)


if __name__ == "__main__":
    main()
