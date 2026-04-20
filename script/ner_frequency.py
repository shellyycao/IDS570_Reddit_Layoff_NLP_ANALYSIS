"""
Part 2 — NER frequency tables
Builds top-20 entity tables overall, per entity type, and top-20 ORG per label.
"""

import pandas as pd
import ast
import os
from collections import Counter

INPUT_FILE = os.path.join("processed", "ner", "layoffs_ner_results.csv")
OUT_DIR    = os.path.join("processed", "ner")

TARGET_TYPES  = ["PERSON", "ORG", "GPE", "DATE", "MONEY"]
LABEL_ORDER   = ["personal_experience", "media_public", "worker_perspective"]


def parse_entities(series):
    """Return a flat list of (text, label) tuples from the entities column."""
    all_ents = []
    for val in series:
        try:
            ents = ast.literal_eval(val)
            all_ents.extend(ents)
        except Exception:
            pass
    return all_ents


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows: {len(df)}")

    # ── Top 20 overall ───────────────────────────────────────────────────────
    all_ents = parse_entities(df["entities"])
    top_overall = pd.DataFrame(
        Counter(all_ents).most_common(20),
        columns=["entity", "count"]
    )
    path = os.path.join(OUT_DIR, "top20_overall.csv")
    top_overall.to_csv(path, index=False)
    print("\n── Top 20 entities overall ──────────────────────────────────")
    print(top_overall.to_string(index=False))

    # ── Top 20 per entity type ───────────────────────────────────────────────
    for etype in TARGET_TYPES:
        filtered = [(text, lbl) for text, lbl in all_ents if lbl == etype]
        top = pd.DataFrame(
            Counter(t for t, _ in filtered).most_common(20),
            columns=["entity", "count"]
        )
        path = os.path.join(OUT_DIR, f"top20_{etype.lower()}.csv")
        top.to_csv(path, index=False)
        print(f"\n── Top 20 {etype} ──────────────────────────────────────────")
        print(top.to_string(index=False))

    # ── Top 20 ORG per label category ───────────────────────────────────────
    for label in LABEL_ORDER:
        subset = df[df["label"] == label]
        orgs   = parse_entities(subset["entities"])
        orgs   = [text for text, lbl in orgs if lbl == "ORG"]
        top = pd.DataFrame(
            Counter(orgs).most_common(20),
            columns=["entity", "count"]
        )
        path = os.path.join(OUT_DIR, f"top20_org_{label}.csv")
        top.to_csv(path, index=False)
        print(f"\n── Top 20 ORG — {label} ─────────────────────────────────")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
