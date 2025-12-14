# -*- coding: utf-8 -*-
import re
import time
import requests
import pandas as pd
from typing import Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# =========================
# Config and Constants
# =========================
FOX_HOME = "https://www.foxnews.com/"
FOX_HOST = "foxnews.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

SLEEP_S = 0.2
MAX_PAGES = 10         # 翻页安全上限（?page=N）
MAX_SHOWMORE_STEPS = 60  # show more 安全上限（from += size）

OUT_CSV = "fox_articles.csv"

# Show More only for these two categories
SHOW_MORE_CATEGORIES = {
    "https://www.foxnews.com/category/us/campus-radicals",
    "https://www.foxnews.com/category/us/immigration",
}

# Fox "Show More" API
ARTICLE_SEARCH_API = "https://www.foxnews.com/api/article-search"

# 1) Strong article signal: URLs containing /YYYY/MM/DD/
DATE_PATH_RE = re.compile(r"/\d{4}/\d{2}/\d{2}/")

# 2) Explicitly excluded non-article paths
BAD_PATH_PREFIXES = (
    "/shows/", "/games/", "/person/", "/story/", "/deals/",
    "/category/", "/topic/", "/elections/", "/live-news/", "/live/",
    "/podcasts/", "/watch/"
)

# Substrings indicating external or unwanted Fox properties
BAD_SUBSTRINGS = (
    "foxnation", "outkick", "radio.foxnews.com", "foxweather.com", "foxbusiness.com",
)

# For extracting tag value used by /api/article-search (from HTML state)
TAG_FROM_OFFSETDATA_RE = re.compile(
    r'offsetData"\s*:\s*\{\s*"(?P<tag>fox-news/[^"]+)"\s*:\s*\d+\s*\}'
)

# =========================
# Utility functions
# =========================
def normalize_url(url: str) -> str:
    """Normalize article URLs by removing print suffixes and fragments."""
    if url.endswith(".print"):
        url = url[:-6]
    return url.split("#")[0].rstrip("/")

def normalize_category_url(url: str) -> str:
    """Normalize category URL for matching show-more list."""
    return url.split("#")[0].rstrip("/")

def is_good_article(url: str) -> bool:
    """Determine whether a URL corresponds to a Fox News article page."""
    u = urlparse(url)
    host = (u.netloc or "").lower().replace("www.", "")
    path = (u.path or "").lower()

    if host != FOX_HOST:
        return False

    full = url.lower()
    if any(s in full for s in BAD_SUBSTRINGS):
        return False

    # Strong signal: Date path exists
    if DATE_PATH_RE.search(path):
        return True

    # Exclude known non-article paths
    if any(path.startswith(p) for p in BAD_PATH_PREFIXES):
        return False

    # Exclude too-shallow paths
    if path.count("/") <= 1:
        return False

    # Slug length heuristic
    last = path.strip("/").split("/")[-1]
    if len(last) < 12:
        return False

    return True

# =========================
# Phase 1: discover nav links
# =========================
def fetch_nav_links(home_url: str = FOX_HOME, headers: dict = HEADERS) -> dict:
    """
    Fetch all URLs found in the main navigation bar (including dropdowns).
    Returns: {nav_text: full_url}
    """
    print("--- Phase 1: Discovering Navigation Links ---")
    try:
        r = requests.get(home_url, headers=headers, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching homepage during discovery: {e}")
        return {}

    nav_container = soup.select_one(".nav-row-upper .primary-nav ul")
    if not nav_container:
        print("Error: Failed to locate the primary navigation list (ul inside .primary-nav).")
        return {}

    links = {}
    for a in nav_container.select("a[href]"):
        href = (a.get("href") or "").strip()
        text = a.get_text(" ", strip=True)
        if not href or not text or len(text) < 3:
            continue

        full_url = urljoin(home_url, href)
        u = urlparse(full_url)
        if FOX_HOST not in (u.netloc or "").lower().replace("www.", ""):
            continue

        links[text] = normalize_category_url(full_url)

    print(f"-> Successfully found {len(links)} unique links from the navigation bar.")
    return links

# =========================
# Phase 2A: normal paging crawl (?page=N)
# =========================
def deep_crawl_category_with_pagination(base_url: str, rows: list, seen_urls: set, max_pages: int = MAX_PAGES):
    """
    Visits a category/topic page and continues to crawl through pages using ?page=N.
    Saves only {url,label}.
    """
    page = 1
    print(f"-> Starting Paging Crawl: {base_url}")

    while page <= max_pages:
        current_url = base_url if page == 1 else f"{base_url}?page={page}"

        try:
            r = requests.get(current_url, headers=HEADERS, timeout=20)
            if r.status_code == 404:
                break
            if r.status_code != 200:
                break

            soup = BeautifulSoup(r.text, "html.parser")
            new_on_page = 0

            for a in soup.select("a[href]"):
                href = (a.get("href") or "").strip()
                if not href:
                    continue

                article_url = normalize_url(urljoin(FOX_HOME, href))

                if is_good_article(article_url) and article_url not in seen_urls:
                    seen_urls.add(article_url)
                    rows.append({"url": article_url, "label": "fox"})
                    new_on_page += 1

            if page > 1 and new_on_page == 0:
                break

            print(f"   -> Page {page} found {new_on_page} new article URLs.")
            page += 1
            time.sleep(SLEEP_S)

        except requests.exceptions.RequestException as e:
            print(f"   -> Stopping: Error fetching {current_url} - {e}")
            break
        except Exception as e:
            print(f"   -> Stopping: Unexpected error - {e}")
            break

    print(f"-> Finished paging crawl for {base_url} after {page-1} pages.")

# =========================
# Phase 2B: show-more crawl (/api/article-search)
# =========================
def extract_tag_value_from_category_html(category_url: str, headers: dict) -> Optional[str]:

    """
    Extract 'fox-news/us/xxx' tag value from the category page HTML.
    We match OffsetManager.offsetData {"fox-news/...": 30} pattern you pasted.
    """
    r = requests.get(category_url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text

    m = TAG_FROM_OFFSETDATA_RE.search(html)
    if m:
        return m.group("tag")

    # fallback (looser)
    m2 = re.search(r'"(fox-news/us/[^"]+)"', html)
    if m2:
        return m2.group(1)

    return None

def extract_urls_from_article_search_json(data) -> list[str]:
    """
    Robustly extract foxnews URLs from the /api/article-search JSON.
    Different builds may store URLs under different keys.
    """
    urls = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() in {"url", "link", "weburl", "canonical", "canonicalurl"} and isinstance(v, str):
                    urls.add(v)
                walk(v)
        elif isinstance(x, list):
            for it in x:
                walk(it)

    walk(data)

    out = []
    for u in urls:
        full = normalize_url(urljoin(FOX_HOME, u))
        if full.startswith("https://www.foxnews.com/"):
            out.append(full)
    return sorted(set(out))

def deep_crawl_category_with_show_more_api(
    category_url: str,
    rows: list,
    seen_urls: set,
    size: int = 11,
    max_steps: int = MAX_SHOWMORE_STEPS,
):
    """
    Use /api/article-search?searchBy=tags&values=...&size=...&from=...
    offset increases by size each request.
    """
    category_url = normalize_category_url(category_url)

    tag_value = extract_tag_value_from_category_html(category_url, headers=HEADERS)
    if not tag_value:
        print(f"   -> [ShowMore] Could not find tag_value on page, fallback to paging.")
        deep_crawl_category_with_pagination(category_url, rows, seen_urls)
        return

    print(f"-> Starting ShowMore API Crawl: {category_url}")
    print(f"   -> tag_value = {tag_value}")

    headers = dict(HEADERS)
    headers["Accept"] = "application/json, text/plain, */*"
    headers["Referer"] = category_url

    sess = requests.Session()
    offset = 0

    for step in range(max_steps):
        params = {
            "searchBy": "tags",
            "values": tag_value,
            "excludeBy": "tags",
            "excludeValues": "",
            "size": size,
            "from": offset,
        }

        r = sess.get(ARTICLE_SEARCH_API, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()

        urls = extract_urls_from_article_search_json(data)

        new = 0
        for u in urls:
            if is_good_article(u) and u not in seen_urls:
                seen_urls.add(u)
                rows.append({"url": u, "label": "fox"})
                new += 1

        print(f"   -> [ShowMore] from={offset} got={len(urls)} new={new} total={len(seen_urls)}")

        if new == 0:
            break

        offset += size
        time.sleep(SLEEP_S)

    print(f"-> Finished ShowMore API crawl: {category_url}")

# =========================
# Router: choose showmore vs paging
# =========================
def deep_crawl_category_smart(category_url: str, rows: list, seen_urls: set):
    u = normalize_category_url(category_url)
    if u in SHOW_MORE_CATEGORIES:
        deep_crawl_category_with_show_more_api(u, rows, seen_urls, size=11)
    else:
        deep_crawl_category_with_pagination(u, rows, seen_urls)

# =========================
# Main
# =========================
def crawl_all_dynamically():
    # Step 1: discover nav links
    category_links = fetch_nav_links()

    # Step 2: filter URLs suitable for crawling
    target_urls = {}
    for nav_name, url in category_links.items():
        path = urlparse(url).path.lower()

        # skip some sections you excluded
        if path.startswith(("/video", "/ai", "/opinion", "/games")):
            continue

        # keep these major sections
        if path.startswith((
            "/us", "/politics", "/world", "/media", "/entertainment",
            "/sports", "/lifestyle", "/health", "/category"
        )):
            target_urls[nav_name] = normalize_category_url(url)

    print(f"--- Phase 2: Starting Crawl on {len(target_urls)} Filtered Categories ---")

    rows = []
    seen_urls = set()

    for category_name, category_url in target_urls.items():
        print(f"\n[SECTION] Crawling {category_name}...")
        deep_crawl_category_smart(category_url, rows, seen_urls)

    df = pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates(subset=["url"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\n--- DONE ---\nSaved a total of {len(df)} unique article rows -> {OUT_CSV}")
    return df

if __name__ == "__main__":
    crawl_all_dynamically()
