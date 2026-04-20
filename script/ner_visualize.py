"""
Part 4 — NER visualizations
- Top 15 ORG entities per label category (3 horizontal bar charts)
- Top 10 entity types overall (1 horizontal bar chart)
"""

import pandas as pd
import ast
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

INPUT_FILE = os.path.join("processed", "ner", "layoffs_ner_results.csv")
OUT_DIR    = os.path.join("image", "ner")

LABEL_ORDER  = ["personal_experience", "media_public", "worker_perspective"]
LABEL_COLORS = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))

sns.set_theme(style="whitegrid")


def parse_entities(series):
    all_ents = []
    for val in series:
        try:
            all_ents.extend(ast.literal_eval(val))
        except Exception:
            pass
    return all_ents


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)

    # ── Top 15 ORG per label — three separate charts ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, label in zip(axes, LABEL_ORDER):
        subset = df[df["label"] == label]
        ents   = parse_entities(subset["entities"])
        orgs   = [text for text, lbl in ents if lbl == "ORG"]
        top    = pd.DataFrame(Counter(orgs).most_common(15), columns=["org", "count"])

        top.sort_values("count").plot(
            kind="barh", x="org", y="count", ax=ax,
            color=LABEL_COLORS[label], legend=False
        )
        ax.set_title(label.replace("_", " ").title(), fontsize=12)
        ax.set_xlabel("Count")
        ax.set_ylabel("")

    fig.suptitle("Top 15 ORG Entities per Label Category", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "top15_org_per_label.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")

    # ── Top 10 entity types overall ──────────────────────────────────────────
    all_ents  = parse_entities(df["entities"])
    type_counts = Counter(lbl for _, lbl in all_ents)
    top_types = pd.DataFrame(
        type_counts.most_common(10), columns=["entity_type", "count"]
    ).sort_values("count")

    fig, ax = plt.subplots(figsize=(8, 5))
    top_types.plot(
        kind="barh", x="entity_type", y="count", ax=ax,
        color=sns.color_palette("Blues_d", len(top_types)), legend=False
    )
    ax.set_title("Top 10 Entity Types Across Full Corpus", fontsize=13)
    ax.set_xlabel("Total occurrences")
    ax.set_ylabel("")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "top10_entity_types.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved → {path}")


if __name__ == "__main__":
    main()
