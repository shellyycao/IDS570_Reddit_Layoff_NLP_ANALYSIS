"""
Part 2 — Sentiment visualizations
- Box plot of compound score per label
- Box plot of compound score per subreddit
- Bar chart of mean compound score per label with error bars
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = os.path.join("processed", "layoffs_sentiment.csv")
IMG_DIR    = os.path.join("image", "sentiment")

LABEL_ORDER    = ["personal_experience", "media_public", "worker_perspective"]
LABEL_PALETTE  = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))
sns.set_theme(style="whitegrid")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # ── Box plot: compound per label ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df, x="label", y="compound", order=LABEL_ORDER,
        hue="label", palette=LABEL_PALETTE, legend=False,
        ax=ax, width=0.5, linewidth=1.2,
        flierprops=dict(marker="o", markersize=2, alpha=0.3)
    )
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels([lbl.replace("_", "\n") for lbl in LABEL_ORDER])
    ax.set_xlabel("")
    ax.set_ylabel("Compound sentiment score")
    ax.set_title("Compound Sentiment Score by Label Category")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "boxplot_compound_by_label.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # ── Box plot: compound per subreddit ──────────────────────────────────────
    sub_order = df.groupby("subreddit")["compound"].median().sort_values().index.tolist()
    palette   = sns.color_palette("tab10", len(sub_order))

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=df, x="subreddit", y="compound", order=sub_order,
        hue="subreddit", palette=dict(zip(sub_order, palette)), legend=False,
        ax=ax, width=0.55, linewidth=1.2,
        flierprops=dict(marker="o", markersize=2, alpha=0.3)
    )
    ax.set_xticks(range(len(sub_order)))
    ax.set_xticklabels(sub_order, rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Compound sentiment score")
    ax.set_title("Compound Sentiment Score by Subreddit (sorted by median)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "boxplot_compound_by_subreddit.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # ── Bar chart: mean compound per label with error bars ───────────────────
    stats = (
        df.groupby("label")["compound"]
          .agg(mean="mean", sem=lambda x: x.std() / len(x) ** 0.5)
          .reindex(LABEL_ORDER)
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        stats["label"], stats["mean"],
        yerr=stats["sem"], capsize=5,
        color=[LABEL_PALETTE[lbl] for lbl in LABEL_ORDER],
        edgecolor="white", linewidth=0.8,
        error_kw=dict(elinewidth=1.2, ecolor="black")
    )
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels([lbl.replace("_", "\n") for lbl in LABEL_ORDER])
    ax.set_ylabel("Mean compound sentiment score")
    ax.set_title("Average Compound Sentiment Score per Label (±1 SE)")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + row["sem"] + 0.005,
                f"{row['mean']:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "bar_mean_compound_by_label.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
