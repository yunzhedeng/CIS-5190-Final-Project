import os
import gzip
import json
import csv
import re
from urllib.parse import urlparse

# ======================
# Fixed configuration
# ======================
BASE_DIR = '3DLNews2'
PLATFORMS = ['1-Google']
MEDIA_TYPES = ['1-Newspaper']

# Write output directly to current working directory
OUTPUT_DIR = ""

QUOTE_CHARS = "\"'“”‘’`´"

def clean_headline(text: str) -> str:
    """
    Remove leading/trailing quotation marks and extra whitespace
    while preserving meaningful punctuation inside the headline.
    """
    if not text:
        return ""

    t = str(text).strip()

    # Iteratively strip quotation marks from both ends
    while t and t[0] in QUOTE_CHARS:
        t = t[1:].lstrip()
    while t and t[-1] in QUOTE_CHARS:
        t = t[:-1].rstrip()

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t



def get_url(record) -> str:
    """Extract URL from a record."""
    return (record.get("expanded_url") or record.get("link") or "").strip()


def get_source(record) -> str:
    """Extract source as the domain name from URL."""
    url = get_url(record)
    if not url:
        return ""

    domain = urlparse(url).netloc.lower().replace("www.", "")
    return domain



def extract_google_newspaper():
    for platform in PLATFORMS:
        for media_type in MEDIA_TYPES:
            full_media_path = os.path.join(BASE_DIR, platform, media_type)
            input_dir = os.path.join(full_media_path, 'preprocessed_state')

            if not os.path.exists(input_dir):
                print(f"[WARN] Missing directory: {input_dir}")
                continue

            # Output file written directly to current directory
            out_csv = os.path.join(
                OUTPUT_DIR, "cleaned_headlines.csv"
            )

            rows = []

            for state in sorted(os.listdir(input_dir)):
                state_path = os.path.join(input_dir, state)
                if not os.path.isdir(state_path):
                    continue

                for file_name in os.listdir(state_path):
                    file_path = os.path.join(state_path, file_name)

                    try:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    record = json.loads(line)
                                except json.JSONDecodeError:
                                    continue

                                # Keep only valid news articles
                                if not record.get("is_news_article"):
                                    continue

                                url = get_url(record)
                                headline = clean_headline(record.get("title") or "")
                                source = get_source(record)

                                # Drop incomplete records
                                if not url or not headline or not source:
                                    continue

                                rows.append({
                                    "url": url,
                                    "headline": headline,
                                    "source": source
                                })

                    except Exception as e:
                        print(f"[ERROR] {file_path}: {e}")

            if rows:
                with open(out_csv, 'w', newline='', encoding='utf-8') as w:
                    writer = csv.DictWriter(
                        w,
                        fieldnames=["url", "headline", "source"],
                        quoting=csv.QUOTE_MINIMAL
                    )
                    writer.writeheader()
                    writer.writerows(rows)

                print(f"[OK] Wrote {len(rows)} rows → {out_csv}")
            else:
                print(f"[INFO] No data found for {platform}/{media_type}")


if __name__ == "__main__":
    print("[INFO] Extracting Google Newspaper URLs, headlines, and sources...")
    extract_google_newspaper()
    print("[INFO] Done.")
