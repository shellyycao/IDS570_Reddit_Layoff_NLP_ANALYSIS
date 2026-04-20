"""
Part 1 — spaCy NER extraction
Loads text_clean.csv, runs en_core_web_lg NER on the text_clean column,
adds an `entities` column (list of (text, label) pairs), and saves the result.
"""

import pandas as pd
import spacy
import ast
import os

INPUT_FILE  = os.path.join("processed", "text_clean.csv")
OUTPUT_FILE = os.path.join("processed", "ner", "layoffs_ner_results.csv")


def extract_entities(nlp, texts, batch_size=256):
    entities = []
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["tagger", "parser", "lemmatizer"]):
        ents = [(ent.text.strip(), ent.label_) for ent in doc.ents if ent.text.strip()]
        entities.append(str(ents))   # store as string for CSV compatibility
    return entities


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    print("Loading spaCy model en_core_web_lg ...")
    nlp = spacy.load("en_core_web_lg")

    # Use text_clean column; fill NaN with empty string
    texts = df["text_clean"].fillna("").tolist()

    print(f"Running NER on {len(texts)} texts (this may take a few minutes) ...")
    df["entities"] = extract_entities(nlp, texts)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\n  Saved → {OUTPUT_FILE}")

    print("\n── First 3 rows ──────────────────────────────────────────────")
    for _, row in df.head(3).iterrows():
        ents = ast.literal_eval(row["entities"])
        print(f"\n  text_clean : {str(row['text_clean'])[:120]}")
        print(f"  entities   : {ents[:6]}")


if __name__ == "__main__":
    main()
