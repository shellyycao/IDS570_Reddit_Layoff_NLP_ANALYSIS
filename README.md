# IDS 570: Layoffs on Reddit — Computational Discourse Analysis

**Course:** IDS 570 — Natural Language Processing
**Dataset:** Reddit posts & comments scraped from 9 subreddits (2020–2026, n = 7,643)

---

## Research Question

Do lexical, semantic, and sentiment signals systematically differ across three discourse communities discussing the 2020–2026 tech layoff wave?

| Label | Subreddits | Framing Style |
|---|---|---|
| `media_public` | r/news, r/business, r/technology | Third-person, institutional, scale-focused, company-anchored |
| `personal_experience` | r/layoffs, r/cscareerquestions, r/careeradvice | First-person, financially anxious, job-search oriented |
| `worker_perspective` | r/antiwork, r/WorkReform, r/jobs | Collective, politically charged, structural critique |

---

## Pipeline Overview

```
Data Collection → Preprocessing → EDA → NER → BERT → Classification → Sentiment → Synthesis
```

1. **EDA** — TF-IDF profiling, cosine similarity heatmap, keyword occurrence, temporal trends, word clouds, KWIC
2. **NER** — spaCy entity extraction (ORG, GPE, DATE, MONEY, PERSON), co-occurrence networks
3. **BERT** — `bert-base-uncased` [CLS] embeddings (768-dim), K-means (k=3), UMAP projection
4. **Classification** — Logistic Regression on TF-IDF vs. BERT features, confusion matrices, feature weights
5. **Sentiment** — Word2Vec semantic scoring, time-series trends, AI + layoff subset analysis
6. **Synthesis** — Cross-method findings, central claim, limitations, future directions

---

## Key Results

| Method | Key Finding |
|---|---|
| TF-IDF / EDA | Subreddit membership is a reliable weak-label proxy; cosine similarity confirms distinct lexical clusters (career vs. media) |
| NER | *AI* and *Amazon* appear across all labels; DATE entities dominate; institutional vs. insider vs. labor framing is distinct by community |
| BERT + UMAP | Cluster 0 cleanly separates media_public; Clusters 1 & 2 overlap — personal experience and worker advocacy share semantic space |
| Logistic Regression | TF-IDF LR (62.1%) outperforms BERT LR (53%); bag-of-words stronger than contextual embeddings for this weak-label task |
| Word2Vec Sentiment | Worker discourse peaks most negative in 2023–2024; media discourse stays institutionally neutral across the period |
| AI + Layoff Extension | Posts mentioning both AI and layoff terms are more negative than the corpus baseline; strongest effect in worker_perspective |

---

## Central Claim

The word *layoff* carries structurally distinct meanings across Reddit communities. The media/non-media split is cleanly recoverable by both BERT and TF-IDF. The personal-experience vs. worker-perspective boundary is fuzzier — shared semantic vocabulary requires lexical supervision to distinguish. Post-2023, AI becomes explicit in discourse; worker communities sharpen negatively while media remains neutral.

---

## Repository Structure

```
IDS570_NLP/
├── notebook/
│   └── reddit_layoffs_discourse_analysis.ipynb   # Main analysis notebook
│
├── script/                     # Python scripts (run in order within each group)
│   ├── reddit_scraper_layoffs.py       # 1. Scrape Reddit posts
│   ├── combine_and_deduplicate.py      # 2. Merge per-subreddit CSVs, drop duplicates
│   ├── preprocess.py                   # 3. Clean text (HTML, unicode, stopwords)
│   ├── preprocess_tfidf.py             # 4. Build TF-IDF features
│   ├── ner_extract.py                  # NER: extract entities with spaCy
│   ├── ner_frequency.py                # NER: entity frequency analysis
│   ├── ner_visualize.py                # NER: charts
│   ├── ner_cooccurrence.py             # NER: co-occurrence matrices & network graphs
│   ├── kwic_sample.py                  # KWIC: keyword-in-context samples
│   ├── bert_embeddings.py              # BERT: extract [CLS] embeddings
│   ├── bert_cluster.py                 # BERT: K-means clustering
│   ├── bert_umap.py                    # BERT: UMAP projection & plots
│   ├── bert_examples.py                # BERT: representative examples per cluster
│   ├── classify_prepare.py             # CLF: train/test split & feature prep
│   ├── classify_train_eval.py          # CLF: train LR models & evaluate
│   ├── classify_features.py            # CLF: TF-IDF feature weight charts
│   ├── classify_predict.py             # CLF: predict on full dataset
│   ├── sentiment_word2vec.py           # Sentiment: Word2Vec semantic scoring
│   ├── sentiment_viz.py                # Sentiment: plots by label & subreddit
│   └── sentiment_time.py               # Sentiment: time-series trends
│
├── data_ready/                 # Raw scraped CSVs (one per subreddit + combined)
│   ├── layoffs_reddit_data.csv         # Full combined dataset
│   └── layoffs_r_<subreddit>.csv       # Per-subreddit files
│
├── processed/                  # Intermediate outputs from scripts
│   ├── text_clean.csv                  # Cleaned text
│   ├── text_tfidf.csv                  # TF-IDF processed text
│   ├── ner/                            # NER entity data
│   ├── bert/                           # BERT embeddings & cluster assignments
│   ├── classification/                 # Train/test splits & predictions
│   ├── cooccurrence_*.csv              # Co-occurrence matrices per label
│   ├── kwic_examples.csv               # KWIC samples
│   └── layoffs_sentiment_w2v.csv       # Word2Vec sentiment scores
│
├── models/                     # Saved model artifacts
│   ├── model_bert_lr.pkl               # BERT + Logistic Regression
│   ├── model_tfidf_lr.pkl              # TF-IDF + Logistic Regression
│   ├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
│   ├── word2vec_w5.bin                 # Word2Vec model (window=5)
│   └── w2v_corpus.txt                  # Corpus used to train Word2Vec
│
├── image/                      # All saved figures (organized by step)
│   ├── bert/
│   ├── classification/
│   ├── cooccurrence/
│   ├── ner/
│   ├── network/
│   ├── sentiment/
│   └── wordcloud/
│
└── Instruction.pdf             # Assignment instructions
```

---

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn
spacy (en_core_web_sm)
transformers torch
umap-learn
nltk networkx wordcloud
gensim
```

Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```
