"""
Part 3 — ORG co-occurrence with layoff keywords
Finds rows where text_clean contains layoff/layoffs/laid off AND has ≥1 ORG entity.
"""

import pandas as pd
import ast
import re
import os

INPUT_FILE  = os.path.join("processed", "ner", "layoffs_ner_results.csv")
OUTPUT_FILE = os.path.join("processed", "ner", "layoff_org_cooccurrence.csv")

LAYOFF_PATTERN = re.compile(r"\blayoffs?\b|\blaid off\b", re.IGNORECASE)


def get_orgs(entity_str):
    try:
        ents = ast.literal_eval(entity_str)
        return [text for text, lbl in ents if lbl == "ORG"]
    except Exception:
        return []


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows: {len(df)}")

    df["orgs"] = df["entities"].apply(get_orgs)

    mask = (
        df["text_clean"].fillna("").str.contains(LAYOFF_PATTERN) &
        df["orgs"].apply(lambda x: len(x) > 0)
    )
    result = df[mask].copy()
    print(f"  Rows matching (layoff keyword + ≥1 ORG): {len(result)}")

    # Save full result
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved → {OUTPUT_FILE}")

    # Print 10 examples
    print("\n── 10 example rows ──────────────────────────────────────────")
    for _, row in result.head(10).iterrows():
        snippet = str(row["text_clean"])[:160]
        print(f"\n  ORGs : {row['orgs']}")
        print(f"  text : {snippet}")


if __name__ == "__main__":
    main()
