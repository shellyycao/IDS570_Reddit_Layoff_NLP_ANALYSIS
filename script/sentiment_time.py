"""
Parts 3–6 — Temporal analysis
- Convert created_utc to datetime, extract year_month
- Line chart: post volume over time per label
- Line chart: mean compound sentiment over time per label
- Line chart: subreddit activity over time
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = os.path.join("processed", "layoffs_sentiment.csv")
IMG_DIR    = os.path.join("image", "sentiment")

LABEL_ORDER   = ["personal_experience", "media_public", "worker_perspective"]
LABEL_PALETTE = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))
sns.set_theme(style="whitegrid")


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # ── Part 3: time preparation ─────────────────────────────────────────────
    df["datetime"]   = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df["year_month"] = df["datetime"].dt.to_period("M").astype(str)

    valid = df.dropna(subset=["datetime"])
    print("\n── Date range ───────────────────────────────────────────────")
    print(f"  Earliest : {valid['datetime'].min().strftime('%Y-%m-%d')}")
    print(f"  Latest   : {valid['datetime'].max().strftime('%Y-%m-%d')}")
    print(f"  Span     : {valid['year_month'].nunique()} unique year-month periods")

    # Save updated file back with year_month column
    df.to_csv(INPUT_FILE, index=False, encoding="utf-8")
    print(f"  year_month column added and saved back to {INPUT_FILE}")

    # ── Part 4: post volume over time per label ───────────────────────────────
    vol = (
        df.groupby(["year_month", "label"])
          .size()
          .reset_index(name="count")
    )
    vol = vol[vol["label"].isin(LABEL_ORDER)]
    vol["year_month"] = pd.PeriodIndex(vol["year_month"], freq="M")
    vol = vol.sort_values("year_month")
    vol["year_month"] = vol["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(13, 5))
    for label in LABEL_ORDER:
        sub = vol[vol["label"] == label]
        ax.plot(sub["year_month"], sub["count"],
                label=label.replace("_", " ").title(),
                color=LABEL_PALETTE[label], marker="o", markersize=3, linewidth=1.8)
    _format_time_axis(ax, vol["year_month"].unique())
    ax.set_ylabel("Number of posts / comments")
    ax.set_title("Post Volume Over Time by Label Category")
    ax.legend(title="Label")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "volume_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved → {path}")

    # ── Part 5: sentiment over time per label ────────────────────────────────
    sent = (
        df.groupby(["year_month", "label"])["compound"]
          .mean()
          .reset_index()
    )
    sent = sent[sent["label"].isin(LABEL_ORDER)]
    sent["year_month"] = pd.PeriodIndex(sent["year_month"], freq="M")
    sent = sent.sort_values("year_month")
    sent["year_month"] = sent["year_month"].astype(str)

    fig, ax = plt.subplots(figsize=(13, 5))
    for label in LABEL_ORDER:
        sub = sent[sent["label"] == label]
        ax.plot(sub["year_month"], sub["compound"],
                label=label.replace("_", " ").title(),
                color=LABEL_PALETTE[label], marker="o", markersize=3, linewidth=1.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    _format_time_axis(ax, sent["year_month"].unique())
    ax.set_ylabel("Mean compound sentiment score")
    ax.set_title("Sentiment Trajectory Over Time by Label Category")
    ax.legend(title="Label")
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "sentiment_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # ── Part 6: subreddit activity over time ──────────────────────────────────
    sub_vol = (
        df.groupby(["year_month", "subreddit"])
          .size()
          .reset_index(name="count")
    )
    sub_vol["year_month"] = pd.PeriodIndex(sub_vol["year_month"], freq="M")
    sub_vol = sub_vol.sort_values("year_month")
    sub_vol["year_month"] = sub_vol["year_month"].astype(str)

    subreddits   = sorted(df["subreddit"].unique())
    sub_palette  = dict(zip(subreddits, sns.color_palette("tab10", len(subreddits))))

    fig, ax = plt.subplots(figsize=(14, 6))
    for sub in subreddits:
        s = sub_vol[sub_vol["subreddit"] == sub]
        ax.plot(s["year_month"], s["count"],
                label=sub, color=sub_palette[sub],
                marker="o", markersize=3, linewidth=1.6)
    _format_time_axis(ax, sub_vol["year_month"].unique())
    ax.set_ylabel("Number of posts / comments")
    ax.set_title("Subreddit Activity Over Time")
    ax.legend(title="Subreddit", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "subreddit_volume_over_time.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def _format_time_axis(ax, periods):
    """Show every 3rd tick to avoid crowding."""
    ticks = list(periods)
    step  = max(1, len(ticks) // 10)
    ax.set_xticks(ticks[::step])
    ax.set_xticklabels(ticks[::step], rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Year-Month")


if __name__ == "__main__":
    main()
