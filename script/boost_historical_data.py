"""
Boost 2025-2026 data using Reddit's native API.

Strategy:
  - Fetch posts via search (sort=new + sort=top) and hot/top/new feeds
  - Filter locally: only keep posts from 2025 or 2026
  - Paginate using Reddit's after=t3_<post_id> cursor until posts go older than 2025
  - New posts are ADDED to per-subreddit CSVs; nothing is deleted

Estimated runtime: ~15-20 minutes for all 9 subreddits.

Run from project root OR the VS Code interactive window:
    conda run -n ML python script/boost_historical_data.py
"""

import os
import re
import time
import warnings
import requests
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
try:
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    ROOT_DIR = "/Users/shellyy/Desktop/570/IDS570_Reddit_Layoff_NLP_ANALYSIS"
DATA_DIR = os.path.join(ROOT_DIR, "data_ready")
PROC_DIR = os.path.join(ROOT_DIR, "processed")
HEADERS  = {"User-Agent": "NLP-academic-research/1.0 (IDS570 layoffs study)"}

SUBREDDITS = {
    "r/technology":        "media_public",
    "r/business":          "media_public",
    "r/news":              "media_public",
    "r/antiwork":          "worker_perspective",
    "r/WorkReform":        "worker_perspective",
    "r/jobs":              "worker_perspective",
    "r/layoffs":           "personal_experience",
    "r/cscareerquestions": "personal_experience",
    "r/careeradvice":      "personal_experience",
}

TARGET_YEARS  = {2024}
CUTOFF_TS     = int(pd.Timestamp("2025-01-01").timestamp())  # stop paginating before this
SEARCH_TERMS  = ["layoff", "layoffs", "laid off", "job cut", "job cuts", "downsizing"]
KEYWORDS      = ["layoff", "layoffs", "laid off", "lay off",
                 "job cut", "job cuts", "downsizing", "workforce reduction"]
MAX_COMMENTS  = 5
MAX_PAGES     = 10   # 100 posts × 10 pages = up to 1000 posts per term


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_json(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    Rate limited — waiting {wait}s ...")
                time.sleep(wait)
            elif r.status_code in (403, 404):
                return None
            else:
                time.sleep(4 * (attempt + 1))
        except Exception as e:
            print(f"    Request error (attempt {attempt+1}): {e}")
            time.sleep(5 * (attempt + 1))
    return None


def is_relevant(text):
    return any(kw in text.lower() for kw in KEYWORDS)


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def make_record(id_, subreddit, label, type_, text, title, score, created_utc, permalink):
    dt     = pd.to_datetime(created_utc, unit="s", errors="coerce")
    period = str(dt.year) if pd.notna(dt) else "other"
    return {
        "id":          id_,
        "subreddit":   subreddit,
        "label":       label,
        "type":        type_,
        "text":        text[:2000],
        "title":       title,
        "score":       score,
        "created_utc": int(created_utc) if created_utc else 0,
        "time_period": period,
        "url":         f"https://reddit.com{permalink}",
    }


def fetch_comments(sub_clean, post_id, label, subreddit_name, permalink, existing_ids):
    url  = f"https://www.reddit.com/r/{sub_clean}/comments/{post_id}.json"
    data = fetch_json(url)
    if not data or len(data) < 2:
        return []
    out = []
    for i, c in enumerate(data[1].get("data", {}).get("children", [])):
        if i >= MAX_COMMENTS:
            break
        cd   = c.get("data", {})
        body = cd.get("body", "")
        cid  = cd.get("id", "")
        if not body or body in ("[deleted]", "[removed]") or cid in existing_ids:
            continue
        existing_ids.add(cid)
        out.append(make_record(
            cid, subreddit_name, label, "comment",
            clean_text(body), "",
            cd.get("score", 0), cd.get("created_utc", 0), permalink
        ))
    time.sleep(0.8)
    return out


def paginate_search(sub_clean, label, subreddit_name, term, sort, existing_ids):
    """Paginate Reddit search, keeping only TARGET_YEARS posts."""
    records = []
    after   = None

    for _ in range(MAX_PAGES):
        params = {
            "q":           term,
            "sort":        sort,
            "restrict_sr": "true",
            "limit":       100,
            "t":           "all",
        }
        if after:
            params["after"] = after

        data = fetch_json(f"https://www.reddit.com/r/{sub_clean}/search.json", params)
        time.sleep(1.2)

        if not data:
            break

        posts = data.get("data", {}).get("children", [])
        if not posts:
            break

        oldest_ts = None
        for post in posts:
            pd_data   = post.get("data", {})
            created   = pd_data.get("created_utc", 0)
            post_id   = pd_data.get("id", "")
            title     = pd_data.get("title", "")
            selftext  = pd_data.get("selftext", "") or ""
            permalink = pd_data.get("permalink", "")
            combined  = f"{title} {selftext}"
            yr        = pd.to_datetime(created, unit="s", errors="coerce").year
            oldest_ts = created

            if yr not in TARGET_YEARS or not is_relevant(combined) or post_id in existing_ids:
                continue

            existing_ids.add(post_id)
            records.append(make_record(
                post_id, subreddit_name, label, "post",
                clean_text(combined), title,
                pd_data.get("score", 0), created, permalink
            ))
            for c in fetch_comments(sub_clean, post_id, label,
                                    subreddit_name, permalink, existing_ids):
                records.append(c)

        cursor = data.get("data", {}).get("after")
        if cursor:
            after = cursor
        else:
            break

        # Stop once we've scrolled past the target window
        if oldest_ts and oldest_ts < CUTOFF_TS:
            break

    return records


def fetch_feed(sub_clean, label, subreddit_name, feed, existing_ids):
    """Fetch hot/top/new feed, keeping only TARGET_YEARS posts."""
    records = []
    data    = fetch_json(
        f"https://www.reddit.com/r/{sub_clean}/{feed}.json",
        {"limit": 100, "t": "all"}
    )
    time.sleep(1.2)
    if not data:
        return records

    for post in data.get("data", {}).get("children", []):
        pd_data   = post.get("data", {})
        created   = pd_data.get("created_utc", 0)
        post_id   = pd_data.get("id", "")
        title     = pd_data.get("title", "")
        selftext  = pd_data.get("selftext", "") or ""
        permalink = pd_data.get("permalink", "")
        combined  = f"{title} {selftext}"
        yr        = pd.to_datetime(created, unit="s", errors="coerce").year

        if yr not in TARGET_YEARS or not is_relevant(combined) or post_id in existing_ids:
            continue

        existing_ids.add(post_id)
        records.append(make_record(
            post_id, subreddit_name, label, "post",
            clean_text(combined), title,
            pd_data.get("score", 0), created, permalink
        ))
        for c in fetch_comments(sub_clean, post_id, label,
                                subreddit_name, permalink, existing_ids):
            records.append(c)

    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_updated = []
    total_new   = 0

    for subreddit_name, label in SUBREDDITS.items():
        sub_clean = subreddit_name.lstrip("r/")
        fname     = f"layoffs_{subreddit_name.replace('/', '_')}.csv"
        fpath     = os.path.join(DATA_DIR, fname)

        print(f"\n{'═'*60}")
        print(f"  {subreddit_name}  ({label})")
        print(f"{'═'*60}")

        existing     = pd.read_csv(fpath)
        existing_ids = set(existing["id"].astype(str))
        print(f"  Existing: {len(existing)} rows")

        all_new = []

        # Paginated search: multiple terms × 2 sort orders
        for term in SEARCH_TERMS:
            for sort in ["new", "top"]:
                recs = paginate_search(sub_clean, label, subreddit_name,
                                       term, sort, existing_ids)
                if recs:
                    print(f"    search [{sort}] '{term}': +{len(recs)}")
                all_new.extend(recs)

        # Feed scrape: hot/top/new (catches posts search misses)
        for feed in ["hot", "top", "new"]:
            recs = fetch_feed(sub_clean, label, subreddit_name, feed, existing_ids)
            if recs:
                print(f"    feed  [{feed}]: +{len(recs)}")
            all_new.extend(recs)

        added = len(all_new)
        total_new += added

        if all_new:
            updated = pd.concat([existing, pd.DataFrame(all_new)], ignore_index=True)
            updated = updated.drop_duplicates(subset=["id"])
        else:
            updated = existing

        print(f"  → {len(updated)} rows total  (+{added} new)")
        updated.to_csv(fpath, index=False)
        all_updated.append(updated)

    # Rebuild combined CSV
    combined      = pd.concat(all_updated, ignore_index=True).drop_duplicates(subset=["id"])
    combined_path = os.path.join(DATA_DIR, "layoffs_reddit_data.csv")
    combined.to_csv(combined_path, index=False)

    combined["_yr"] = pd.to_datetime(combined["created_utc"], unit="s",
                                      errors="coerce").dt.year
    print(f"\nCombined: {len(combined)} rows total  (+{total_new} new)")
    print("\nYear distribution (2022+):")
    print(combined[combined["_yr"] >= 2022]["_yr"].value_counts().sort_index().to_string())

    print("\nRebuilding processed/text_tfidf.csv ...")
    rebuild_processed(combined)
    print("\nDone.")


def rebuild_processed(combined):
    df = combined.copy()
    df["datetime"]    = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    df["time_period"] = df["datetime"].dt.year.astype("Int64").astype(str)

    def basic_clean(t):
        if not isinstance(t, str):
            return ""
        t = re.sub(r"http\S+", "", t)
        t = re.sub(r"[^a-zA-Z\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip().lower()

    STOPWORDS = {
        "the","a","an","is","it","in","on","at","to","of","and","or","but",
        "with","for","this","that","was","are","be","have","has","had",
        "i","you","he","she","they","we","my","your","his","her","their",
        "just","like","really","so","very","also","do","did","not","no",
        "if","can","will","would","could","should","what","how","when",
        "get","got","think","know","people","one","dont","im","ive",
    }

    def tfidf_clean(t):
        return " ".join(w for w in t.split() if len(w) > 2 and w not in STOPWORDS)

    df["text_clean"] = df["text"].apply(basic_clean)
    df["text_tfidf"] = df["text_clean"].apply(tfidf_clean)

    existing_cols = pd.read_csv(os.path.join(PROC_DIR, "text_tfidf.csv"), nrows=0).columns.tolist()
    keep = [c for c in existing_cols if c in df.columns]
    out  = df[keep] if keep else df
    out_path = os.path.join(PROC_DIR, "text_tfidf.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved {len(out)} rows → {out_path}")

    EXCLUDE  = {"2009","2011","2012","2013","2014","2015","2016","2017",
                "2018","2019","2020","2021","other"}
    filtered = out[~out["time_period"].isin(EXCLUDE)]
    print("\n  Distribution in text_tfidf.csv (2022+):")
    print(filtered["time_period"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
