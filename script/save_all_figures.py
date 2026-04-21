"""
Regenerates and saves all notebook visualizations to the image/ directory.
Run from the project root: python script/save_all_figures.py
"""

import ast
import os
import pickle
import re
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ── Directories ───────────────────────────────────────────────────────────────
os.makedirs("image/ner", exist_ok=True)
os.makedirs("image/bert", exist_ok=True)
os.makedirs("image/classification", exist_ok=True)
os.makedirs("image/sentiment", exist_ok=True)
os.makedirs("image/eda", exist_ok=True)

sns.set_theme(style="whitegrid", palette="Set2")
LABEL_ORDER = ["media_public", "personal_experience", "worker_perspective"]
LABEL_COLORS = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))
LABEL_NAMES = ["personal_experience", "media_public", "worker_perspective"]

# ── Load main dataframe ───────────────────────────────────────────────────────
df = pd.read_csv("processed/text_tfidf.csv")
df["time_period"] = df["time_period"].astype(str)
EXCLUDE_PERIODS = {"2009","2011","2012","2013","2014","2015","2016","2017","2018","2019","other"}
df = df[~df["time_period"].isin(EXCLUDE_PERIODS)].reset_index(drop=True)
print(f"Main df: {df.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# EDA
# ──────────────────────────────────────────────────────────────────────────────

# Row count by subreddit x label
pivot = (
    df.groupby(["subreddit", "label"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=LABEL_ORDER)
)
fig, ax = plt.subplots(figsize=(11, 5))
pivot.plot(kind="bar", stacked=True, ax=ax, color=[LABEL_COLORS[l] for l in LABEL_ORDER])
ax.set_xlabel("Subreddit")
ax.set_ylabel("Row count")
ax.set_title("Row count by subreddit, colored by label")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Label", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("image/eda/subreddit_label_counts.png", dpi=150)
plt.close()
print("Saved: image/eda/subreddit_label_counts.png")

# Keyword hits
pattern = r"\blayoffs?\b|\blaid off\b"
df["kw_hits"] = df["text_clean"].fillna("").str.count(pattern, flags=re.IGNORECASE)
kw_sub = df.groupby("subreddit")["kw_hits"].sum().sort_values(ascending=False)
kw_label = df.groupby("label")["kw_hits"].sum().reindex(LABEL_ORDER)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
kw_sub.plot(kind="bar", ax=axes[0], color=sns.color_palette("Blues_d", len(kw_sub)))
axes[0].set_title("Keyword hits by subreddit")
axes[0].set_ylabel("Total occurrences")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
kw_label.plot(kind="bar", ax=axes[1], color=[LABEL_COLORS[l] for l in LABEL_ORDER])
axes[1].set_title("Keyword hits by label")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=20, ha="right")
fig.suptitle('Occurrences of "layoff", "layoffs", "laid off" in text_clean')
plt.tight_layout()
plt.savefig("image/eda/keyword_hits.png", dpi=150)
plt.close()
print("Saved: image/eda/keyword_hits.png")


def top_tfidf_terms(corpus, n=15):
    vec_tmp = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    mat = vec_tmp.fit_transform(corpus)
    scores = np.asarray(mat.mean(axis=0)).flatten()
    terms = vec_tmp.get_feature_names_out()
    idx = scores.argsort()[::-1][:n]
    return pd.Series(scores[idx], index=terms[idx])


label_groups = {l: df[df["label"] == l]["text_tfidf"].dropna() for l in LABEL_ORDER}

# Top TF-IDF per label
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, LABEL_ORDER):
    top = top_tfidf_terms(label_groups[label])
    top[::-1].plot(kind="barh", ax=ax, color=LABEL_COLORS[label])
    ax.set_title(label.replace("_", " ").title())
    ax.set_xlabel("Mean TF-IDF score")
fig.suptitle("Top 15 TF-IDF terms per label category", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("image/eda/tfidf_top_terms_per_label.png", dpi=150)
plt.close()
print("Saved: image/eda/tfidf_top_terms_per_label.png")

# Top TF-IDF per subreddit
subreddits = sorted(df["subreddit"].unique())
ncols = 3
nrows = -(-len(subreddits) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
axes_flat = axes.flatten()
palette = sns.color_palette("tab10", len(subreddits))
for i, sub in enumerate(subreddits):
    corpus = df[df["subreddit"] == sub]["text_tfidf"].dropna()
    top = top_tfidf_terms(corpus)
    top[::-1].plot(kind="barh", ax=axes_flat[i], color=palette[i])
    axes_flat[i].set_title(sub)
    axes_flat[i].set_xlabel("Mean TF-IDF score")
for j in range(i + 1, len(axes_flat)):
    axes_flat[j].set_visible(False)
fig.suptitle("Top 15 TF-IDF terms per subreddit", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("image/eda/tfidf_top_terms_per_subreddit.png", dpi=150)
plt.close()
print("Saved: image/eda/tfidf_top_terms_per_subreddit.png")

# Volume by time period
period_order = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
vol = df.groupby("time_period").size().reindex(period_order, fill_value=0)
fig, ax = plt.subplots(figsize=(9, 4))
vol.plot(kind="bar", ax=ax, color=sns.color_palette("Blues_d", len(vol)), edgecolor="none")
ax.set_xlabel("Time period")
ax.set_ylabel("Row count")
ax.set_title("Post/comment volume by time period")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("image/eda/volume_by_period.png", dpi=150)
plt.close()
print("Saved: image/eda/volume_by_period.png")

# Cosine similarity heatmap
agg = df.groupby("subreddit")["text_tfidf"].apply(lambda x: " ".join(x.dropna()))
vec_cs = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
mat_cs = vec_cs.fit_transform(agg)
sim = cosine_similarity(mat_cs)
sim_df = pd.DataFrame(sim, index=agg.index, columns=agg.index)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Cosine similarity"})
ax.set_title("Cosine similarity between subreddits (aggregated TF-IDF)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("image/eda/cosine_similarity_heatmap.png", dpi=150)
plt.close()
print("Saved: image/eda/cosine_similarity_heatmap.png")


# ──────────────────────────────────────────────────────────────────────────────
# NER
# ──────────────────────────────────────────────────────────────────────────────
NER_DF = pd.read_csv("processed/ner/layoffs_ner_results.csv")
NER_DF["entities_parsed"] = NER_DF["entities"].apply(
    lambda v: ast.literal_eval(v) if isinstance(v, str) else []
)

def flat_ents(series):
    out = []
    for ents in series:
        out.extend(ents)
    return out

all_ents = flat_ents(NER_DF["entities_parsed"])

# Top 10 entity types
type_counts = Counter(lbl for _, lbl in all_ents)
top_types = pd.DataFrame(
    type_counts.most_common(10), columns=["entity_type", "count"]
).sort_values("count")
fig, ax = plt.subplots(figsize=(8, 5))
top_types.plot(kind="barh", x="entity_type", y="count", ax=ax,
               color=sns.color_palette("Blues_d", len(top_types)), legend=False)
ax.set_title("Top 10 Entity Types Across Full Corpus", fontsize=13)
ax.set_xlabel("Total occurrences")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("image/ner/top10_entity_types.png", dpi=150)
plt.close()
print("Saved: image/ner/top10_entity_types.png")

# Top 15 ORG per label
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, label in zip(axes, LABEL_ORDER):
    subset = NER_DF[NER_DF["label"] == label]
    orgs = [t for t, l in flat_ents(subset["entities_parsed"]) if l == "ORG"]
    top = pd.DataFrame(Counter(orgs).most_common(15), columns=["org", "count"])
    top.sort_values("count").plot(kind="barh", x="org", y="count", ax=ax,
                                  color=LABEL_COLORS[label], legend=False)
    ax.set_title(label.replace("_", " ").title(), fontsize=12)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
fig.suptitle("Top 15 ORG Entities per Label Category", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("image/ner/top15_org_per_label.png", dpi=150)
plt.close()
print("Saved: image/ner/top15_org_per_label.png")


# ──────────────────────────────────────────────────────────────────────────────
# BERT UMAP
# ──────────────────────────────────────────────────────────────────────────────
BERT_CLUST = pd.read_csv("processed/bert/bert_clusters.csv")

CLUSTER_PALETTE = sns.color_palette("Set1", 3)
LABEL_PALETTE = dict(zip(LABEL_ORDER, sns.color_palette("Set2", 3)))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for cid in sorted(BERT_CLUST["cluster"].unique()):
    sub = BERT_CLUST[BERT_CLUST["cluster"] == cid]
    axes[0].scatter(sub["umap_x"], sub["umap_y"], c=[CLUSTER_PALETTE[cid]],
                    alpha=0.5, s=8, label=f"Cluster {cid}")
axes[0].set_title("UMAP: colored by K-means cluster", fontsize=13)
axes[0].legend(title="Cluster", markerscale=3)
axes[0].set_xlabel("UMAP 1")
axes[0].set_ylabel("UMAP 2")

for label in LABEL_ORDER:
    sub = BERT_CLUST[BERT_CLUST["label"] == label]
    axes[1].scatter(sub["umap_x"], sub["umap_y"], c=[LABEL_PALETTE[label]],
                    alpha=0.5, s=8, label=label.replace("_", " ").title())
axes[1].set_title("UMAP: colored by label category", fontsize=13)
axes[1].legend(title="Label", markerscale=3)
axes[1].set_xlabel("UMAP 1")
axes[1].set_ylabel("UMAP 2")

plt.suptitle("BERT Embeddings: UMAP Projection", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("image/bert/bert_umap.png", dpi=150)
plt.close()
print("Saved: image/bert/bert_umap.png")


# ──────────────────────────────────────────────────────────────────────────────
# Classification
# ──────────────────────────────────────────────────────────────────────────────
CLS_DIR = "processed/classification"
MDL_DIR = "models"

X_bert_train = np.load(f"{CLS_DIR}/X_bert_train.npy")
X_bert_test  = np.load(f"{CLS_DIR}/X_bert_test.npy")
y_train = np.load(f"{CLS_DIR}/y_train.npy")
y_test  = np.load(f"{CLS_DIR}/y_test.npy")
train_tfidf = pd.read_csv(f"{CLS_DIR}/train_tfidf.csv")
test_tfidf  = pd.read_csv(f"{CLS_DIR}/test_tfidf.csv")

with open(f"{MDL_DIR}/model_bert_lr.pkl", "rb") as f:
    lr_bert = pickle.load(f)
with open(f"{MDL_DIR}/model_tfidf_lr.pkl", "rb") as f:
    lr_tfidf = pickle.load(f)
with open(f"{MDL_DIR}/tfidf_vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

X_tfidf_train = vec.transform(train_tfidf["text_tfidf"].fillna(""))
X_tfidf_test  = vec.transform(test_tfidf["text_tfidf"].fillna(""))

results = {}
for name, model, X_tr, X_te in [
    ("BERT + LR", lr_bert, X_bert_train, X_bert_test),
    ("TF-IDF + LR", lr_tfidf, X_tfidf_train, X_tfidf_test),
]:
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"acc": acc, "pred": y_pred, "cm": confusion_matrix(y_test, y_pred)}

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, res) in zip(axes, results.items()):
    short_labels = [l.replace("_", "\n") for l in LABEL_NAMES]
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=short_labels, yticklabels=short_labels, ax=ax)
    ax.set_title(f"Confusion Matrix: {name}\n(acc={res['acc']:.3f})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("image/classification/confusion_matrices.png", dpi=150)
plt.close()
print("Saved: image/classification/confusion_matrices.png")

# Individual confusion matrices
for name, res in results.items():
    fig, ax = plt.subplots(figsize=(7, 5))
    short_labels = [l.replace("_", "\n") for l in LABEL_NAMES]
    sns.heatmap(res["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=short_labels, yticklabels=short_labels, ax=ax)
    fname = "bert" if "BERT" in name else "tfidf"
    ax.set_title(f"Confusion Matrix: {name} (acc={res['acc']:.3f})", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"image/classification/confusion_matrix_{fname}.png", dpi=150)
    plt.close()
    print(f"Saved: image/classification/confusion_matrix_{fname}.png")

# TF-IDF feature weights
feature_names = vec.get_feature_names_out()
coef = lr_tfidf.coef_
TOP_N = 15

fig, axes = plt.subplots(3, 1, figsize=(12, 15))
for i, (label, ax) in enumerate(zip(LABEL_NAMES, axes)):
    weights = coef[i]
    top_pos = np.argsort(weights)[-TOP_N:][::-1]
    top_neg = np.argsort(weights)[:TOP_N]
    top_idx = np.concatenate([top_neg, top_pos[::-1]])
    top_feats = feature_names[top_idx]
    top_vals = weights[top_idx]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_vals]
    ax.barh(range(len(top_feats)), top_vals, color=colors, edgecolor="none")
    ax.set_yticks(range(len(top_feats)))
    ax.set_yticklabels(top_feats, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"{label.replace('_', ' ').title()}: top {TOP_N} positive / negative weights", fontsize=12)
    ax.set_xlabel("Coefficient")
axes[0].legend(handles=[
    Patch(color="#d62728", label="Positive (toward class)"),
    Patch(color="#1f77b4", label="Negative (away from class)"),
], fontsize=9)
plt.suptitle("TF-IDF LR Feature Weights per Class", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("image/classification/tfidf_feature_weights.png", dpi=150)
plt.close()
print("Saved: image/classification/tfidf_feature_weights.png")


# ──────────────────────────────────────────────────────────────────────────────
# Sentiment
# ──────────────────────────────────────────────────────────────────────────────
w2v = pd.read_csv("processed/layoffs_sentiment_w2v.csv")
w2v["datetime"] = pd.to_datetime(w2v["created_utc"], unit="s", errors="coerce")
w2v["year"] = w2v["datetime"].dt.year.astype("Int64")
valid_years = [2021, 2022, 2023, 2024, 2025, 2026]
w2v = w2v[w2v["year"].isin(valid_years)].copy()

# Box plot by label
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=w2v, x="label", y="sentiment_w2v", order=LABEL_ORDER,
            hue="label", palette=LABEL_COLORS, legend=False, ax=ax,
            width=0.5, linewidth=1.2,
            flierprops=dict(marker="o", markersize=2, alpha=0.3))
ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_xticklabels([l.replace("_", "\n") for l in LABEL_ORDER])
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("")
ax.set_ylabel("Word2Vec sentiment score")
ax.set_title("Word2Vec Sentiment Score by Label Category")
plt.tight_layout()
plt.savefig("image/sentiment/boxplot_compound_by_label.png", dpi=150)
plt.close()
print("Saved: image/sentiment/boxplot_compound_by_label.png")

# Box plot by subreddit
fig, ax = plt.subplots(figsize=(11, 5))
sub_order = w2v.groupby("subreddit")["sentiment_w2v"].mean().sort_values().index
sns.boxplot(data=w2v, x="subreddit", y="sentiment_w2v", order=sub_order,
            hue="subreddit", palette="tab10", legend=False, ax=ax,
            width=0.5, linewidth=1.2,
            flierprops=dict(marker="o", markersize=2, alpha=0.3))
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_xlabel("")
ax.set_ylabel("Word2Vec sentiment score")
ax.set_title("Word2Vec Sentiment Score by Subreddit")
plt.tight_layout()
plt.savefig("image/sentiment/boxplot_compound_by_subreddit.png", dpi=150)
plt.close()
print("Saved: image/sentiment/boxplot_compound_by_subreddit.png")

# Bar chart: mean sentiment per label
stats = (
    w2v.groupby("label")["sentiment_w2v"]
    .agg(mean="mean", sem=lambda x: x.std() / len(x) ** 0.5)
    .reindex(LABEL_ORDER).reset_index()
)
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(stats["label"], stats["mean"], yerr=stats["sem"], capsize=5,
              color=[LABEL_COLORS[l] for l in LABEL_ORDER], edgecolor="white",
              error_kw=dict(elinewidth=1.2, ecolor="black"))
ax.set_xticks(range(len(LABEL_ORDER)))
ax.set_xticklabels([l.replace("_", "\n") for l in LABEL_ORDER])
ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
ax.set_ylabel("Mean Word2Vec sentiment score")
ax.set_title("Average Word2Vec Sentiment per Label (±1 SE)")
for bar, (_, row) in zip(bars, stats.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row["sem"] + 0.001,
            f"{row['mean']:.4f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig("image/sentiment/bar_mean_compound_by_label.png", dpi=150)
plt.close()
print("Saved: image/sentiment/bar_mean_compound_by_label.png")

# Sentiment over time by label
w2v_annual = (
    w2v.groupby(["year", "label"])["sentiment_w2v"]
    .mean().reset_index().rename(columns={"sentiment_w2v": "w2v"})
)
w2v_annual["year"] = w2v_annual["year"].astype(str)

fig, ax = plt.subplots(figsize=(11, 5))
for label in LABEL_ORDER:
    sub = w2v_annual[w2v_annual["label"] == label]
    ax.plot(sub["year"], sub["w2v"], marker="o", linewidth=2,
            color=LABEL_COLORS[label], label=label.replace("_", " ").title())
ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
ax.set_title("Word2Vec Sentiment over Time: by Label", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Mean W2V Sentiment")
ax.legend(title="Label")
plt.tight_layout()
plt.savefig("image/sentiment/sentiment_over_time.png", dpi=150)
plt.close()
print("Saved: image/sentiment/sentiment_over_time.png")

# Volume over time (from main df)
period_order = ["2020", "2021", "2022", "2023", "2024", "2025", "2026"]
vol_label = df.groupby(["time_period", "label"]).size().unstack(fill_value=0).reindex(period_order, fill_value=0)
fig, ax = plt.subplots(figsize=(11, 5))
for label in LABEL_ORDER:
    if label in vol_label.columns:
        ax.plot(vol_label.index, vol_label[label], marker="o", linewidth=2,
                color=LABEL_COLORS[label], label=label.replace("_", " ").title())
ax.set_title("Post Volume over Time by Label", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Row count")
ax.legend(title="Label")
plt.tight_layout()
plt.savefig("image/sentiment/volume_over_time.png", dpi=150)
plt.close()
print("Saved: image/sentiment/volume_over_time.png")

# Volume over time by subreddit
vol_sub = df.groupby(["time_period", "subreddit"]).size().unstack(fill_value=0).reindex(period_order, fill_value=0)
fig, ax = plt.subplots(figsize=(13, 5))
palette_sub = sns.color_palette("tab10", len(vol_sub.columns))
for col, color in zip(vol_sub.columns, palette_sub):
    ax.plot(vol_sub.index, vol_sub[col], marker="o", linewidth=1.5,
            color=color, label=col)
ax.set_title("Post Volume over Time by Subreddit", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Row count")
ax.legend(title="Subreddit", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("image/sentiment/subreddit_volume_over_time.png", dpi=150)
plt.close()
print("Saved: image/sentiment/subreddit_volume_over_time.png")

# Subreddit mean sentiment bar
sub_means = (
    w2v.groupby(["subreddit", "label"])["sentiment_w2v"]
    .mean().sort_values().reset_index()
)
fig, ax = plt.subplots(figsize=(11, 5))
colors = [LABEL_COLORS[l] for l in sub_means["label"]]
ax.barh(sub_means["subreddit"], sub_means["sentiment_w2v"], color=colors)
ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
ax.set_title("Mean W2V Sentiment by Subreddit", fontsize=13)
ax.set_xlabel("Mean W2V Sentiment")
plt.tight_layout()
plt.savefig("image/sentiment/mean_sentiment_by_subreddit.png", dpi=150)
plt.close()
print("Saved: image/sentiment/mean_sentiment_by_subreddit.png")

# Posts vs comments sentiment
type_means = (
    w2v.groupby(["label", "type"])["sentiment_w2v"]
    .mean().unstack().reindex(LABEL_ORDER)
)
fig, ax = plt.subplots(figsize=(9, 5))
type_means.plot(kind="bar", ax=ax, color=["#aaaaaa", "#e84545"], width=0.6)
ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
ax.set_title("Mean W2V Sentiment: Posts vs Comments by Label", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("Mean W2V Sentiment")
ax.set_xticklabels([l.replace("_", " ").title() for l in LABEL_ORDER], rotation=15)
ax.legend(title="Type")
plt.tight_layout()
plt.savefig("image/sentiment/posts_vs_comments_sentiment.png", dpi=150)
plt.close()
print("Saved: image/sentiment/posts_vs_comments_sentiment.png")

# AI mention trend
df["has_ai"] = df["text_clean"].str.contains(
    r"\b(ai|artificial intelligence|automation|chatgpt|llm)\b", case=False, regex=True
)
ai_trend = (
    df[df["time_period"].isin([str(y) for y in range(2020, 2027)])]
    .groupby(["time_period", "label"])["has_ai"]
    .mean().unstack().reindex(columns=LABEL_ORDER)
)
fig, ax = plt.subplots(figsize=(11, 5))
for label in LABEL_ORDER:
    ax.plot(ai_trend.index, ai_trend[label], marker="o", linewidth=2,
            color=LABEL_COLORS[label], label=label.replace("_", " ").title())
ax.set_title("Proportion of Posts Mentioning AI: by Label and Year", fontsize=13)
ax.set_xlabel("Year")
ax.set_ylabel("Proportion of posts with AI mention")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.legend(title="Label")
ax.grid(axis="y", linewidth=0.5, alpha=0.5)
plt.tight_layout()
plt.savefig("image/sentiment/ai_mention_trend.png", dpi=150)
plt.close()
print("Saved: image/sentiment/ai_mention_trend.png")

# AI + Layoff sentiment comparison
AI_PAT = r"\b(ai|artificial intelligence|automation|chatgpt|llm)\b"
LAYOFF_PAT = r"\blayoffs?\b|\blaid off\b"
w2v["has_ai_layoff"] = (
    w2v["text_clean"].str.contains(AI_PAT, case=False, regex=True) &
    w2v["text_clean"].str.contains(LAYOFF_PAT, case=False, regex=True)
)
ai_layoff = w2v[w2v["has_ai_layoff"]].copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=ai_layoff, x="label", y="sentiment_w2v", order=LABEL_ORDER,
            hue="label", palette=LABEL_COLORS, ax=axes[0])
axes[0].axhline(0, color="grey", linewidth=0.6, linestyle=":")
axes[0].set_title("W2V Sentiment: AI + Layoff posts only", fontsize=12)
axes[0].set_xlabel("")
axes[0].set_ylabel("Mean W2V Sentiment")
axes[0].set_xticklabels([l.replace("_", " ").title() for l in LABEL_ORDER], rotation=15)

compare = pd.DataFrame({
    "All posts": w2v.groupby("label")["sentiment_w2v"].mean(),
    "AI + Layoff posts": ai_layoff.groupby("label")["sentiment_w2v"].mean(),
}).reindex(LABEL_ORDER)
compare.plot(kind="bar", ax=axes[1], color=["#aaaaaa", "#e84545"], width=0.6)
axes[1].axhline(0, color="grey", linewidth=0.6, linestyle=":")
axes[1].set_title("Sentiment Comparison: All vs AI+Layoff posts", fontsize=12)
axes[1].set_xlabel("")
axes[1].set_ylabel("Mean W2V Sentiment")
axes[1].set_xticklabels([l.replace("_", " ").title() for l in LABEL_ORDER], rotation=15)
axes[1].legend(fontsize=9)
plt.suptitle("Word2Vec Sentiment: Posts Mentioning Both AI and Layoff Terms", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("image/sentiment/ai_layoff_sentiment.png", dpi=150)
plt.close()
print("Saved: image/sentiment/ai_layoff_sentiment.png")

print("\nDone — all figures saved.")
