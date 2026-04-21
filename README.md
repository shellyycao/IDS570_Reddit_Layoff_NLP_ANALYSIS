# IDS 570: Layoffs on Reddit — Computational Discourse Analysis

**Course:** IDS 570 — Natural Language Processing
**Dataset:** Reddit posts & comments scraped from 9 subreddits (2020–2026, n = 7,643)

---

## Executive Summary

Between 2020 and 2026, the U.S. technology sector conducted one of the largest waves of mass layoffs in its history, displacing hundreds of thousands of workers. Public discourse about these events splintered across distinct online communities — media platforms, career forums, and labor advocacy spaces — each framing the same phenomenon in structurally different ways. Workforce communications teams, labor researchers, and platform policy analysts at technology firms need to understand these differences: a message that lands as factual reporting in one community may read as institutional deflection in another, with real consequences for employee relations, public trust, and policy response.

This project addresses that gap by applying a multi-method NLP analysis to 7,643 Reddit posts drawn from 9 subreddits, grouped into three discourse communities: `media_public`, `personal_experience`, and `worker_perspective`. Using TF-IDF profiling, named entity recognition, BERT embeddings, logistic regression classification, and Word2Vec sentiment analysis, we test whether these communities produce measurably distinct lexical, semantic, and affective signals around the same topic.

The central finding is that they do — but unevenly. The media/non-media divide is strongly recoverable by both bag-of-words and contextual embedding methods (TF-IDF accuracy: 62.1%). The boundary between personal-experience and worker-perspective communities is fuzzier, requiring lexical supervision to separate. Sentiment diverges most sharply after 2023, when AI-related layoffs enter the discourse: worker communities grow significantly more negative while media communities remain institutionally neutral. These findings give communications and policy teams a concrete, empirically grounded map of how the same events are being interpreted differently — and where intervention or tailored messaging would have the most impact.

---

## Problem Statement

**The problem:** Technology firms, labor organizations, and platform policy teams lack a systematic, evidence-based understanding of how layoff events are framed and experienced across different online communities. Without this, communications efforts are untargeted, sentiment risks go undetected until they escalate, and policy responses are designed without knowing which communities are most distressed and why.

**Why this is solvable:** Reddit discourse is public, large-scale, and structured by community norms that proxy for distinct stakeholder groups. Computational methods can extract, compare, and quantify the signals in this discourse at a scale no manual reading could achieve. The output is actionable: a map of which communities use which frames, which entities dominate, and how sentiment evolves over time — directly informing how communications should be adapted per audience.

---

## Research Question

Do lexical, semantic, and sentiment signals systematically differ across three Reddit discourse communities discussing the 2020–2026 tech layoff wave — and if so, how do those differences inform audience-specific communications strategy?

| Label | Subreddits | Framing Style |
|---|---|---|
| `media_public` | r/news, r/business, r/technology | Third-person, institutional, scale-focused, company-anchored |
| `personal_experience` | r/layoffs, r/cscareerquestions, r/careeradvice | First-person, financially anxious, job-search oriented |
| `worker_perspective` | r/antiwork, r/WorkReform, r/jobs | Collective, politically charged, structural critique |

---

## Method & Pipeline

### Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
│   Reddit API → 9 subreddits → deduplicate → label (n=7,643)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING                            │
│         Clean text → TF-IDF matrix → corpus for W2V            │
└──────────┬────────────────────────────────────────┬────────────┘
           │                                        │
           ▼                                        ▼
┌──────────────────────┐               ┌────────────────────────┐
│         EDA          │               │    BERT EMBEDDINGS     │
│  TF-IDF profiles     │               │  bert-base-uncased     │
│  Cosine similarity   │               │  [CLS] 768-dim         │
│  Word clouds / KWIC  │               └────────────┬───────────┘
│  Temporal trends     │                            │
└──────────┬───────────┘               ┌────────────▼───────────┐
           │                           │   K-MEANS + UMAP       │
           ▼                           │   k=3 clusters         │
┌──────────────────────┐               │   2D projection        │
│         NER          │               └────────────┬───────────┘
│  spaCy en_core_web_sm│                            │
│  ORG/GPE/DATE/MONEY  │               ┌────────────▼───────────┐
│  Co-occurrence nets  │               │   CLASSIFICATION       │
└──────────┬───────────┘               │   TF-IDF LR vs BERT LR │
           │                           │   Confusion matrices   │
           │                           └────────────┬───────────┘
           │                                        │
           └──────────────────┬─────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      SENTIMENT ANALYSIS       │
              │  Word2Vec scoring by label    │
              │  Time-series trends           │
              │  AI + layoff subset extension │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │           SYNTHESIS           │
              │  Cross-method findings        │
              │  Central claim + limitations  │
              └───────────────────────────────┘
```

---

### Step 1 — Data Collection
**Purpose:** Build a labeled corpus large enough to support statistical NLP methods across three distinct communities.

Reddit posts and comments were scraped from 9 subreddits using the Reddit API (`reddit_scraper_layoffs.py`), then merged and deduplicated (`combine_and_deduplicate.py`, n = 7,643). Subreddits were assigned to one of three community labels based on their primary purpose and user base. Using subreddit membership as a weak label — rather than manually annotating individual posts — is justified by the high internal coherence of Reddit communities; users self-select into forums that match their framing orientation, making community membership a reasonable proxy for discourse type (Baumgartner et al., 2020).

### Step 2 — Preprocessing
**Purpose:** Normalize text so that downstream frequency and embedding methods operate on clean, comparable signals.

Raw text was cleaned by removing HTML artifacts, unicode noise, URLs, and stopwords (`preprocess.py`). A TF-IDF matrix was separately built with scikit-learn's `TfidfVectorizer` using unigrams and bigrams, with a minimum document frequency of 5 to filter rare tokens (`preprocess_tfidf.py`). Stopword removal follows NLTK's English list; domain-specific terms like "job" and "work" were retained because they carry discriminative signal across communities.

### Step 3 — EDA (Exploratory Data Analysis)
**Purpose:** Establish baseline evidence that the three communities differ in surface-level lexical composition before applying heavier models.

- **TF-IDF profiling:** Identifies the most community-distinctive terms per label.
- **Cosine similarity heatmap:** Measures pairwise lexical overlap between community centroids; low cross-community similarity validates the weak-label assumption.
- **KWIC sampling:** Examines how the keyword *layoff* appears in immediate context across labels — a direct test of the central claim.
- **Word clouds:** Visual confirmation that high-frequency and high-TF-IDF terms differ meaningfully across labels.
- **Temporal trends:** Tracks post volume over time to identify whether discourse spikes align with known real-world layoff events.

### Step 4 — NER (Named Entity Recognition)
**Purpose:** Identify which real-world actors (companies, places, money amounts) each community centers in its discussion.

Used spaCy (`en_core_web_sm`) to extract five entity types: ORG, GPE, DATE, MONEY, PERSON (`ner_extract.py`). Entity frequencies were then aggregated by label, and co-occurrence matrices and network graphs were computed to show which entities appear together within the same post (`ner_cooccurrence.py`). This approach was chosen over simple word frequency because named entities are semantically grounded — knowing that `media_public` centers *Amazon* and *Meta* while `worker_perspective` centers systemic terms reveals structurally different frames, not just different vocabularies.

### Step 5 — BERT Embeddings + UMAP
**Purpose:** Test whether communities are separable at the semantic (contextual) level, not just the surface lexical level.

`[CLS]` token embeddings were extracted from `bert-base-uncased` (768-dim) for each post (`bert_embeddings.py`). BERT was chosen over simpler embeddings (e.g., GloVe) because it captures contextual meaning — *laid off* in a personal narrative differs semantically from *layoffs announced* in a news report, even if the tokens overlap. K-means with k=3 was selected to match the three known community labels, enabling direct comparison between unsupervised cluster assignments and ground-truth labels. UMAP (McInnes et al., 2018) was used for 2D projection over t-SNE because it better preserves global structure at this dataset size and runs efficiently on CPU.

### Step 6 — Classification
**Purpose:** Quantify how well each feature representation can recover community labels, providing an upper-bound estimate of discriminability.

Logistic Regression was trained separately on TF-IDF features and on BERT embeddings (`classify_train_eval.py`). LR was chosen as the classifier because it is interpretable (coefficients directly identify discriminative features), fast, and well-suited for high-dimensional sparse inputs like TF-IDF. Comparing TF-IDF LR vs. BERT LR directly tests whether surface vocabulary or contextual semantics is more informative for this weak-label task — a methodological question with practical implications for how discourse monitoring tools should be built.

### Step 7 — Sentiment Analysis
**Purpose:** Measure affective tone across communities and track how it evolves over time, particularly in response to the AI-driven layoff wave post-2023.

Word2Vec (window=5, Mikolov et al., 2013) was trained on the full corpus and used to compute semantic sentiment scores by projecting post embeddings onto a positive/negative axis defined by seed words (`sentiment_word2vec.py`). Window size 5 was selected to capture short-range syntactic context without diluting local semantic relationships. A temporal analysis and an AI-specific subset analysis (posts mentioning both AI and layoff terms) were added as extensions to test whether the post-2023 AI discourse shift is detectable as a sentiment change.

### Step 8 — Synthesis
**Purpose:** Integrate evidence across methods to assess the central claim and derive actionable conclusions.

Cross-method findings were combined to assess where community differences are robust (recoverable by multiple methods) vs. fragile (method-dependent). Limitations and directions for future work were identified.

---

## Findings

### Key Results Summary

| Method | Key Finding |
|---|---|
| TF-IDF / EDA | Subreddit membership is a reliable weak-label proxy; cosine similarity confirms distinct lexical clusters (career vs. media) |
| NER | *AI* and *Amazon* appear across all labels; DATE entities dominate; institutional vs. insider vs. labor framing is distinct by community |
| BERT + UMAP | Cluster 0 cleanly separates `media_public`; Clusters 1 & 2 overlap — personal experience and worker advocacy share semantic space |
| Logistic Regression | TF-IDF LR (62.1%) outperforms BERT LR (53%); bag-of-words is stronger than contextual embeddings for this weak-label task |
| Word2Vec Sentiment | Worker discourse peaks most negative in 2023–2024; media discourse stays institutionally neutral across the period |
| AI + Layoff Extension | Posts mentioning both AI and layoff terms are more negative than the corpus baseline; strongest effect in `worker_perspective` |

---

### Word Clouds

**Figure 1. Raw-frequency word clouds by label.** These show the most commonly used words in each community. The dominance of company names (e.g., *Google*, *Amazon*) in `media_public` vs. first-person job-search terms in `personal_experience` vs. systemic labor vocabulary in `worker_perspective` confirms that communities frame layoffs around fundamentally different subjects.

| media_public | personal_experience | worker_perspective |
|---|---|---|
| ![media_public wordcloud](image/wordcloud/wordcloud_media_public.png) | ![personal_experience wordcloud](image/wordcloud/wordcloud_personal_experience.png) | ![worker_perspective wordcloud](image/wordcloud/wordcloud_worker_perspective.png) |

**Figure 2. TF-IDF-weighted word clouds by label.** TF-IDF suppresses terms common across all communities (like *layoff* itself) and surfaces terms that are *distinctively* frequent within each label. These are the vocabulary signals that drive classifier performance.

| media_public | personal_experience | worker_perspective |
|---|---|---|
| ![media_public tfidf wordcloud](image/wordcloud/wordcloud_tfidf_media_public.png) | ![personal_experience tfidf wordcloud](image/wordcloud/wordcloud_tfidf_personal_experience.png) | ![worker_perspective tfidf wordcloud](image/wordcloud/wordcloud_tfidf_worker_perspective.png) |

---

### NER — Named Entity Recognition

**Figure 3. Top 10 entity types across the full corpus.** DATE entities dominate, confirming that temporal framing is central to layoff discourse. ORG entities are the second most frequent — the entity-type most likely to differentiate communities, since which organizations each group names will differ by framing orientation.

![Top 10 entity types](image/ner/top10_entity_types.png)

**Figure 4. Top 15 ORG entities per label.** *AI* and *Amazon* appear across all three labels, but their surrounding context differs: `media_public` treats them as newsworthy actors, while `worker_perspective` cites them as structural antagonists. This cross-community overlap is why semantic methods are needed — surface entity overlap masks frame-level divergence.

![Top 15 ORG entities per label](image/ner/top15_org_per_label.png)

**Figure 5. Entity co-occurrence matrices.** Each cell shows how often two entities appear in the same post. Denser matrices indicate more entity-rich discourse. `media_public` shows tightly co-occurring corporate clusters; `worker_perspective` shows sparser, more diffuse co-occurrences — consistent with structural critique that references many actors rather than tracking specific company pairs.

| Combined | media_public | personal_experience | worker_perspective |
|---|---|---|---|
| ![combined cooccurrence](image/cooccurrence/cooccurrence_combined.png) | ![media cooccurrence](image/cooccurrence/cooccurrence_media.png) | ![personal cooccurrence](image/cooccurrence/cooccurrence_personal.png) | ![worker cooccurrence](image/cooccurrence/cooccurrence_worker.png) |

**Figure 6. Entity co-occurrence network graphs.** Nodes are named entities; edges indicate co-appearance in the same post. `media_public` networks are hub-and-spoke around major corporations; `worker_perspective` networks are more distributed, reflecting systemic rather than firm-specific framing.

| media_public | personal_experience | worker_perspective |
|---|---|---|
| ![media network](image/network/network_media.png) | ![personal network](image/network/network_personal.png) | ![worker network](image/network/network_worker.png) |

---

### BERT + UMAP Clustering

**Figure 7. UMAP projection of BERT [CLS] embeddings (K-means k=3).** The clean separation of Cluster 0 (bottom) from Clusters 1 and 2 confirms that `media_public` occupies a distinct semantic space. The overlap between `personal_experience` and `worker_perspective` in Clusters 1 and 2 explains why BERT-based classification struggles to separate these two groups — they share semantic vocabulary around financial stress and job loss even when their political framing differs.

![BERT UMAP projection](image/bert/bert_umap.png)

---

### Classification

**Figure 8. Confusion matrices: TF-IDF LR (62.1%) vs. BERT LR (53%).** TF-IDF outperforms BERT for this task because surface vocabulary is more discriminative than contextual meaning when labels are defined by community norms rather than semantic content. Both models show the same systematic confusion: `personal_experience` and `worker_perspective` are frequently misclassified as each other, confirming the semantic overlap observed in the UMAP projection.

| TF-IDF Logistic Regression (62.1%) | BERT Logistic Regression (53%) |
|---|---|
| ![TF-IDF confusion matrix](image/classification/confusion_matrix_tfidf.png) | ![BERT confusion matrix](image/classification/confusion_matrix_bert.png) |

**Figure 9. Top TF-IDF feature weights by label.** The features with the highest coefficients per class identify the specific vocabulary each community owns. These term lists are directly actionable for a communications team: they reveal which words signal insider/worker-perspective framing vs. institutional/media framing, enabling vocabulary-aware message design.

![TF-IDF feature weights](image/classification/tfidf_feature_weights.png)

---

### Sentiment Analysis

**Figure 10. Mean compound sentiment score by label.** `worker_perspective` posts are substantially more negative on average than `media_public` or `personal_experience`, confirming that structural labor critique produces the most affectively charged discourse. This baseline difference persists across time periods.

![Mean compound by label](image/sentiment/bar_mean_compound_by_label.png)

**Figure 11. Compound sentiment distributions by label and subreddit.** The label-level boxplot shows that `worker_perspective` has not only a lower median but higher variance — some posts are highly charged in either direction. The subreddit-level breakdown shows that r/antiwork and r/WorkReform drive most of this variance, while r/news (within `media_public`) is the most consistently neutral.

| By Label | By Subreddit |
|---|---|
| ![boxplot by label](image/sentiment/boxplot_compound_by_label.png) | ![boxplot by subreddit](image/sentiment/boxplot_compound_by_subreddit.png) |

**Figure 12. Sentiment over time by label.** Worker community sentiment sharply declines post-2023, coinciding with the wave of AI-attributed layoffs. Media community sentiment remains flat. This divergence is the clearest evidence that the *meaning* of layoffs has changed for labor communities but not for institutional commentators — a directly actionable finding for communications timing and tone.

![Sentiment over time](image/sentiment/sentiment_over_time.png)

**Figure 13. Post volume over time.** Discourse volume spikes in 2022–2023 across all communities, aligning with high-profile tech layoff announcements. The volume patterns confirm that these communities are responding to the same real-world events — making cross-community sentiment differences a true framing effect rather than a coverage-level artifact.

| By Label | By Subreddit |
|---|---|
| ![volume over time](image/sentiment/volume_over_time.png) | ![subreddit volume over time](image/sentiment/subreddit_volume_over_time.png) |

---

## UMAP Tableau Interactive Visualization

The video below demonstrates the interactive UMAP visualization built in Tableau. It allows exploration of BERT embedding clusters by label, subreddit, and time period — enabling stakeholders to interactively interrogate which communities cluster together and how cluster composition shifts over time.

https://github.com/user-attachments/assets/0d3795fb-a267-4873-8809-667c88b401af

---

## Central Claim

The word *layoff* carries structurally distinct meanings across Reddit communities. The media/non-media split is cleanly recoverable by both BERT and TF-IDF. The personal-experience vs. worker-perspective boundary is fuzzier — shared semantic vocabulary requires lexical supervision to distinguish. Post-2023, AI becomes explicit in discourse; worker communities sharpen negatively while media remains neutral. For communications and policy teams, this means: one message does not fit all audiences, and the post-2023 AI framing has fundamentally changed the affective stakes for labor communities in ways that institutional discourse has not yet tracked.

---

## Appendix A — Background & Related Work

### A.1 Computational Analysis of Online Layoff and Labor Discourse

Prior NLP work on Reddit has established that community membership is a reliable signal of discourse orientation. Baumgartner et al. (2020) demonstrated via the Pushshift dataset that Reddit's community structure creates naturally occurring, large-scale labeled corpora well-suited to supervised and unsupervised NLP. This directly motivates the use of subreddit membership as a weak label in this project — a design choice grounded in demonstrated practice, not assumption.

Research on crisis communication and labor discourse online (e.g., studies of Occupy Wall Street and #MeToo on Twitter) has consistently found that the same event generates structurally distinct narrative frames across communities differentiated by political orientation and institutional affiliation. This project extends that finding specifically to tech layoffs on Reddit — a domain with higher post length and more structured community norms than Twitter, making Reddit better suited to semantic analysis methods like BERT.

### A.2 Text Representation: TF-IDF vs. Contextual Embeddings

The choice to compare TF-IDF and BERT representations was motivated by an open debate in the applied NLP literature. Mikolov et al. (2013) established that dense word embeddings capture semantic relationships that sparse bag-of-words representations miss. Devlin et al. (2019) extended this to contextual embeddings, where the same token receives different representations depending on surrounding context.

However, research on weak-label classification tasks — where labels are defined by community or author metadata rather than document content — has found that TF-IDF often outperforms contextual embeddings because the discriminative signal lies in vocabulary choice, not contextual interpretation. This project tests that hypothesis directly. The finding that TF-IDF LR (62.1%) outperforms BERT LR (53%) is consistent with this strand of literature, suggesting that Reddit community membership is primarily a lexical phenomenon: communities use different words, not just the same words differently.

### A.3 Dimensionality Reduction for NLP Visualization

UMAP (McInnes et al., 2018) was selected over t-SNE for the BERT embedding projection based on three properties documented in the literature: (1) UMAP better preserves global topological structure, not just local neighborhoods; (2) UMAP is significantly faster on corpora of this size; and (3) UMAP embeddings are more stable across runs. For the goal of comparing cluster separability across community labels — a global structure question — UMAP is the more appropriate choice than t-SNE.

### A.4 Sentiment Analysis on Social Media

VADER (Hutto & Gilbert, 2014) was developed specifically for social media text and handles informal language, slang, and emphasis markers better than lexicon-based tools developed for formal text. Word2Vec-based semantic sentiment scoring (used in this project) complements VADER by capturing domain-specific sentiment — terms like *restructuring* or *rightsizing* are neutral in general lexicons but negatively valenced in this corpus. Using Word2Vec trained on the in-domain corpus allows the sentiment model to reflect the affective norms of layoff discourse rather than general English.

### A.5 Named Entity Recognition for Discourse Analysis

spaCy's `en_core_web_sm` pipeline (Honnibal et al., 2020) was selected for NER due to its strong performance on news and web text, its speed at corpus scale, and its support for the five entity types most relevant to this domain (ORG, GPE, DATE, MONEY, PERSON). Prior work on financial and crisis communication has used NER co-occurrence networks to identify which actors are implicated together in discourse — a method this project applies to distinguish institutional (company-centric) framing from structural (system-centric) framing across communities.

---

## Appendix B — Ethical Considerations

### B.1 Privacy and Informed Consent

Reddit posts are technically public, but users posting in communities like r/layoffs or r/cscareerquestions are sharing personally sensitive information — job loss, financial hardship, mental health struggles — in contexts where they expect a relatively private, community-bounded audience. Scraping and analyzing this content without consent raises legitimate concerns about contextual integrity (Nissenbaum, 2004): information shared for peer support in a small community is being repurposed for research at scale.

This project mitigates these concerns by: (1) analyzing aggregated, community-level patterns rather than individual post content; (2) not storing or reporting any usernames or personally identifying information; and (3) not attempting to re-identify individuals from post content. No direct quotes from personal-experience posts are reproduced in the findings. Future work should explore opt-in consent mechanisms or synthetic data augmentation as more ethically robust alternatives to scraping.

### B.2 Bias in NLP Models

BERT (`bert-base-uncased`) was pre-trained on BookCorpus and English Wikipedia — corpora that are demographically unrepresentative of Reddit's user base, and particularly of workers discussing layoffs. This introduces the risk of systematic bias: the model may encode cultural and socioeconomic assumptions from its training data that distort how it represents language from non-dominant communities. The lower BERT classification accuracy relative to TF-IDF may partly reflect this mismatch between pre-training domain and application domain.

Word2Vec sentiment scoring is similarly dependent on the seed words used to define the positive/negative axis. If those seed words carry implicit associations (e.g., terms associated with professional stability vs. precarity), the sentiment scores may systematically favor or disfavor certain community framings. This project used symmetric seed words with manual validation, but a more rigorous audit would apply methods from the fair NLP literature (e.g., word embedding association tests) to check for bias in the trained vectors.

### B.3 Potential for Misuse

A tool that accurately identifies the vocabulary, entities, and sentiment patterns of labor advocacy communities could, in the wrong hands, be used to monitor, suppress, or manipulate worker discourse. This is not a hypothetical concern: corporate social media monitoring of labor organizing activity is documented and legally contested.

The authors take the position that this analysis is designed to help communications teams understand and respond to worker concerns more effectively — not to surveil or undermine worker organizing. To that end, findings are framed around audience-appropriate communication and policy responsiveness, not targeting or suppression. Any deployment of these methods in a commercial context should be subject to review by a labor ethics board or equivalent oversight body.

### B.4 Representational Limitations

Reddit's user base is not representative of the broader workforce affected by tech layoffs. It skews younger, more male, more English-speaking, and more technically employed than the full population of laid-off workers. Communities like r/antiwork and r/WorkReform represent a specific ideological orientation within the labor movement. Findings about "worker_perspective" should not be interpreted as representative of all workers' views — only of the specific communities captured here. Stakeholders should triangulate these findings with direct employee feedback, survey data, and community forums not captured by this dataset.

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

# References

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
