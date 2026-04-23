"""
Part 1 — BERT Embeddings
Extracts a ±2 sentence window around the first layoff keyword per row,
runs it through bert-base-uncased, and saves the CLS token embeddings.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE   = os.path.join(ROOT_DIR, "processed", "text_clean.csv")
EMB_FILE     = os.path.join(ROOT_DIR, "processed", "bert", "bert_embeddings.npy")
META_FILE    = os.path.join(ROOT_DIR, "processed", "bert", "bert_metadata.csv")

KEYWORD_PAT  = re.compile(r"\blayoffs?\b|\blaid off\b", re.IGNORECASE)
SENT_SPLIT   = re.compile(r"(?<=[.!?])\s+")
WINDOW       = 2          # sentences on each side of the keyword sentence
BATCH_SIZE   = 32
MAX_LEN      = 512


def extract_window(text):
    """Return ±WINDOW sentences around the first keyword match, or None."""
    sentences = SENT_SPLIT.split(text.strip())
    for i, sent in enumerate(sentences):
        if KEYWORD_PAT.search(sent):
            lo = max(0, i - WINDOW)
            hi = min(len(sentences), i + WINDOW + 1)
            return " ".join(sentences[lo:hi])
    return None


def get_cls_embeddings(texts, tokenizer, model, device):
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        cls = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls)
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} rows", flush=True)
    return np.vstack(embeddings)


def main():
    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Rows loaded: {len(df)}")

    # Extract keyword windows
    df["window"] = df["text_clean"].fillna("").apply(extract_window)
    matched = df.dropna(subset=["window"]).reset_index(drop=True)
    print(f"  Rows with keyword window: {len(matched)}")

    # Load BERT
    print("Loading bert-base-uncased ...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertModel.from_pretrained("bert-base-uncased")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"  Running on: {device}")

    # Generate embeddings
    print("Extracting CLS embeddings ...")
    embeddings = get_cls_embeddings(matched["window"].tolist(), tokenizer, model, device)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Save
    np.save(EMB_FILE, embeddings)
    print(f"  Saved embeddings → {EMB_FILE}")

    meta = matched[["id", "subreddit", "label", "type", "text_clean", "window"]].copy()
    meta.to_csv(META_FILE, index=False, encoding="utf-8")
    print(f"  Saved metadata  → {META_FILE}")
    print(f"\nEmbeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    main()
