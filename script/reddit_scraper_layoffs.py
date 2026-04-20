"""
Reddit .json endpoint scraper — no API credentials needed.
For IDS 570 Term Project: Layoff discourse analysis, 2020–2026.

Fetches historically distributed data using Reddit's search endpoint
with date (before/after) filtering to reach back to 2020.

Discourse groups (weak labels):
    personal_experience  — r/layoffs, r/cscareerquestions, r/careeradvice
    media_public         — r/technology, r/business, r/news
    worker_perspective   — r/antiwork, r/WorkReform, r/jobs

Usage:
    python reddit_scraper_layoffs.py

Output:
    layoffs_reddit_data.csv            — combined dataset (saved after every subreddit)
    layoffs_r_layoffs.csv              — per-subreddit CSV
    layoffs_r_cscareerquestions.csv    — per-subreddit CSV
    ... etc for all 9 subreddits
    wordcloud_personal_experience.png
    wordcloud_media_public.png
    wordcloud_worker_perspective.png
"""

import requests
import time
import csv
import re
from collections import defaultdict

# ── Configuration ────────────────────────────────────────────────────────────

SUBREDDITS = {
    
    # media_public: layoffs as reported news events, general public reaction
    "r/technology":        "media_public",
    "r/business":          "media_public",
    "r/news":              "media_public",
    # worker_perspective: layoffs viewed through labor and worker rights lens
    "r/antiwork":          "worker_perspective",
    "r/WorkReform":        "worker_perspective",
    "r/jobs":              "worker_perspective",
}

# Strict keywords — post must contain at least one as a whole word
# e.g. "layoffs" matches but "outlaying" does not
KEYWORDS = ["layoff", "layoffs", "laid off", "lay off", "job cut", "job cuts"]

POSTS_PER_SUBREDDIT   = 100
INCLUDE_COMMENTS      = True   # comments add richer NER and BERT signal
MAX_COMMENTS_PER_POST = 5    # fetch up to 20 comments per post

HEADERS = {"User-Agent": "NLP-academic-research/1.0 (IDS570 layoffs study)"}

OUTPUT_CSV = "layoffs_reddit_data.csv"

# ── Time windows for historical search ───────────────────────────────────────

TIME_WINDOWS = [
    ("2020", 1577836800, 1609459199),   # 2020-01-01 to 2020-12-31
    ("2021", 1609459200, 1640995199),   # 2021-01-01 to 2021-12-31
    ("2022", 1640995200, 1672531199),   # 2022-01-01 to 2022-12-31
    ("2023", 1672531200, 1704067199),   # 2023-01-01 to 2023-12-31
    ("2024", 1704067200, 1735689599),   # 2024-01-01 to 2024-12-31
    ("2025", 1735689600, 1767225599),   # 2025-01-01 to 2025-12-31
    ("2026", 1767225600, 1798761599),   # 2026-01-01 to 2026-12-31
]

def assign_time_period(created_utc):
    """Return year label for a UTC timestamp."""
    for label, start, end in TIME_WINDOWS:
        if start <= int(created_utc) <= end:
            return label
    return "other"

# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            elif r.status_code == 403:
                print("  403 Forbidden — skipping.")
                return None
            else:
                print(f"  HTTP {r.status_code}")
                return None
        except Exception as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return None


def is_relevant(text):
    """Return True if text contains at least one keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def make_record(id_, subreddit, label, type_, text, title, score, created_utc, permalink):
    return {
        "id":          id_,
        "subreddit":   subreddit,
        "label":       label,
        "type":        type_,
        "text":        text[:2000],
        "title":       title,
        "score":       score,
        "created_utc": int(created_utc) if created_utc else 0,
        "time_period": assign_time_period(created_utc) if created_utc else "other",
        "url":         f"https://reddit.com{permalink}",
    }

# ── Scraping strategies ───────────────────────────────────────────────────────

def fetch_recent_posts(sub_clean, label, subreddit_name, limit=100):
    """Fetch recent posts via hot/top/new."""
    records = []
    for sort in ["hot", "top", "new"]:
        url    = f"https://www.reddit.com/r/{sub_clean}/{sort}.json"
        params = {"limit": limit, "t": "all"}
        data   = fetch_json(url, params=params)
        if not data:
            continue

        posts = data.get("data", {}).get("children", [])
        print(f"    [{sort}] {len(posts)} posts fetched")

        for post in posts:
            pd_data   = post.get("data", {})
            title     = pd_data.get("title", "")
            selftext  = pd_data.get("selftext", "")
            post_id   = pd_data.get("id", "")
            permalink = pd_data.get("permalink", "")
            created   = pd_data.get("created_utc", 0)
            combined  = f"{title} {selftext}"

            if not is_relevant(combined):
                continue

            records.append(make_record(
                post_id, subreddit_name, label, "post",
                clean_text(combined), title,
                pd_data.get("score", 0), created, permalink
            ))

            if INCLUDE_COMMENTS and post_id:
                records += fetch_comments(sub_clean, post_id, label, subreddit_name, permalink)

        time.sleep(2)
    return records


def fetch_historical_posts(sub_clean, label, subreddit_name, year_label, after_ts, before_ts, limit=100):
    """Fetch posts from a specific year window using Reddit's search endpoint."""
    records = []
    search_terms = ["layoff", "laid off", "job cut", "job cuts"]

    for term in search_terms:
        url    = f"https://www.reddit.com/r/{sub_clean}/search.json"
        params = {
            "q":           term,
            "sort":        "new",
            "restrict_sr": "true",
            "limit":       limit,
            "after":       f"t_{after_ts}",
            "before":      f"t_{before_ts}",
            "t":           "all",
        }

        data = fetch_json(url, params=params)
        if not data:
            time.sleep(2)
            continue

        posts = data.get("data", {}).get("children", [])
        print(f"    [{year_label}] '{term}': {len(posts)} posts")

        for post in posts:
            pd_data   = post.get("data", {})
            title     = pd_data.get("title", "")
            selftext  = pd_data.get("selftext", "")
            post_id   = pd_data.get("id", "")
            permalink = pd_data.get("permalink", "")
            created   = pd_data.get("created_utc", 0)
            combined  = f"{title} {selftext}"

            if not is_relevant(combined):
                continue

            records.append(make_record(
                post_id, subreddit_name, label, "post",
                clean_text(combined), title,
                pd_data.get("score", 0), created, permalink
            ))

            if INCLUDE_COMMENTS and post_id:
                records += fetch_comments(sub_clean, post_id, label, subreddit_name, permalink)

        time.sleep(2)

    return records


def fetch_comments(sub_clean, post_id, label, subreddit_name, permalink):
    url  = f"https://www.reddit.com/r/{sub_clean}/comments/{post_id}.json"
    data = fetch_json(url)
    if not data or len(data) < 2:
        return []

    comments = []
    for i, c in enumerate(data[1].get("data", {}).get("children", [])):
        if i >= MAX_COMMENTS_PER_POST:
            break
        cd   = c.get("data", {})
        body = cd.get("body", "")
        if not body or body in ("[deleted]", "[removed]"):
            continue
        # Keep all comments — parent post already passed relevance check
        comments.append(make_record(
            cd.get("id", ""), subreddit_name, label, "comment",
            clean_text(body), "",
            cd.get("score", 0), cd.get("created_utc", 0), permalink
        ))

    time.sleep(2)
    return comments

# ── Main scraping loop ────────────────────────────────────────────────────────

def scrape_subreddit(subreddit_name, label):
    sub_clean = subreddit_name.lstrip("r/")
    records   = []

    print(f"\n══ {subreddit_name} ({label}) ══")

    print("  → Fetching recent posts (hot/top/new):")
    records += fetch_recent_posts(sub_clean, label, subreddit_name)

    print("  → Fetching historical posts by year:")
    for year_label, after_ts, before_ts in TIME_WINDOWS:
        print(f"    Searching {year_label}...")
        records += fetch_historical_posts(
            sub_clean, label, subreddit_name,
            year_label, after_ts, before_ts
        )

    print(f"  → Subtotal (before dedup): {len(records)}")
    return records

# ── Output helpers ────────────────────────────────────────────────────────────

def save_csv(records, path):
    if not records:
        print(f"  No records to save for {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✓ Saved {len(records)} records → {path}")


def dedup(records):
    seen, out = set(), []
    for r in records:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    return out


def generate_wordclouds(records):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        print("Run: pip install wordcloud matplotlib")
        return

    stopwords = {
        "the","a","an","is","it","in","on","at","to","of","and","or","but",
        "with","for","this","that","was","are","be","have","has","had",
        "i","you","he","she","they","we","my","your","his","her","their",
        "just","like","really","so","very","also","do","did","not","no",
        "if","can","will","would","could","should","what","how","when",
        "get","got","think","know","people","one","don","t","s","re","ve",
        "deleted","removed","www","http","reddit","com","amp",
        "layoff","layoffs","laid","lay","job","jobs","cut","cuts",
        "work","company","companies","employee","employees","fired","firing",
    }

    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r["text"])

    for label, texts in by_label.items():
        corpus = " ".join(texts)
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            stopwords=stopwords,
            max_words=150,
            colormap="viridis",
        ).generate(corpus)

        fname = f"wordcloud_{label}.png"
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud — {label.replace('_', ' ').title()}", fontsize=16)
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  Saved → {fname}")


def print_summary(records):
    by_label  = defaultdict(int)
    by_sub    = defaultdict(int)
    by_period = defaultdict(int)

    for r in records:
        by_label[r["label"]]        += 1
        by_sub[r["subreddit"]]      += 1
        by_period[r["time_period"]] += 1

    print("\n── Summary ──────────────────────────────────")
    print("By discourse group (weak label):")
    for k, v in sorted(by_label.items()):
        print(f"  {k:30s}: {v}")
    print("\nBy subreddit:")
    for k, v in sorted(by_sub.items()):
        print(f"  {k:30s}: {v}")
    print("\nBy time period:")
    for k, v in sorted(by_period.items()):
        print(f"  {k:10s}: {v}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    all_records = []

    for subreddit, label in SUBREDDITS.items():
        records = scrape_subreddit(subreddit, label)

        # Save individual subreddit CSV
        sub_filename = f"layoffs_{subreddit.replace('/', '_')}.csv"
        clean_records = dedup(records)
        save_csv(clean_records, sub_filename)

        all_records.extend(records)

        # Save combined CSV after every subreddit
        combined = dedup(all_records)
        save_csv(combined, OUTPUT_CSV)
        print(f"  ✓ Combined total so far: {len(combined)} records")

    # Final summary
    final = dedup(all_records)
    removed = len(all_records) - len(final)
    print(f"\nFinal deduplication: {len(all_records)} → {len(final)} ({removed} duplicates removed)")
    print_summary(final)

    print("\nGenerating word clouds...")
    generate_wordclouds(final)
    print("\nAll done! ✓")


if __name__ == "__main__":
    main()
