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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

SLEEP_S = 0.3          # Slightly increased sleep to be polite
MAX_PAGES = 10         # Safety limit for pagination crawl (?page=N)
MAX_SHOWMORE_STEPS = 60  # Safety limit for API-based crawl (from += size)

OUT_CSV = "fox_articles.csv"

# Categories that require API-based "Show More" crawling instead of standard pagination.
# These pages load content dynamically via JavaScript and an internal API.
SHOW_MORE_CATEGORIES = {
    "https://www.foxnews.com/category/us/campus-radicals",
    "https://www.foxnews.com/category/us/immigration",
}

# Fox internal API endpoint for fetching additional articles ("Show More" button functionality).
ARTICLE_SEARCH_API = "https://www.foxnews.com/api/article-search"

# 1) Strong article signal: URLs containing date patterns like /YYYY/MM/DD/.
DATE_PATH_RE = re.compile(r"/\d{4}/\d{2}/\d{2}/")

# 2) Prefixes of URL paths to explicitly exclude, as they are not standard text articles.
BAD_PATH_PREFIXES = (
    "/shows/", "/games/", "/person/", "/story/", "/deals/",
    "/category/", "/topic/", "/elections/", "/live-news/", "/live/",
    "/podcasts/", "/watch/", "/video/"
)

# Substrings indicating external, affiliated, or non-news Fox properties to avoid.
BAD_SUBSTRINGS = (
    "foxnation", "outkick", "radio.foxnews.com", "foxweather.com", "foxbusiness.com",
)

# Regex to extract the specific 'tag' value required by the /api/article-search endpoint.
# This value is typically embedded within a JSON object in the category page's HTML source.
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
    """Normalize category URLs for consistent matching against SHOW_MORE_CATEGORIES."""
    return url.split("#")[0].rstrip("/")

def is_good_article(url: str) -> bool:
    """
    Determine whether a given URL likely corresponds to a valid Fox News text article.
    Checks hostname, excluded substrings, date patterns, path prefixes, depth, and slug length.
    """
    try:
        u = urlparse(url)
    except ValueError:
        return False

    host = (u.netloc or "").lower().replace("www.", "")
    path = (u.path or "").lower()

    # Ensure the article belongs to the main Fox News domain.
    if host != FOX_HOST:
        return False

    full_url_lower = url.lower()
    if any(s in full_url_lower for s in BAD_SUBSTRINGS):
        return False

    # Strong signal: Presence of a date structure in the URL path.
    if DATE_PATH_RE.search(path):
        return True

    # Exclude known non-article sections based on path prefixes.
    if any(path.startswith(p) for p in BAD_PATH_PREFIXES):
        return False

    # Exclude paths that are too shallow (likely section hubs, not articles).
    # e.g., /us/ or /politics/ instead of /us/article-slug.
    if path.count("/") <= 2 and not path.endswith(".html"): # Slightly relaxed check
         # Basic check: must have at least /section/slug
         parts = [p for p in path.split("/") if p]
         if len(parts) < 2:
             return False

    # Heuristic check on the slug length to filter out short, non-article URLs.
    last_segment = path.strip("/").split("/")[-1]
    if len(last_segment) < 10: # Slightly lowered threshold
        return False

    return True

# =========================
# Phase 1: Discover Navigation Links
# =========================
def fetch_nav_links(home_url: str = FOX_HOME, headers: dict = HEADERS) -> dict:
    """
    Fetch and parse the homepage to discover URLs found in the main navigation bar.
    Returns a dictionary mapping navigation text to its normalized full URL.
    """
    print("--- Phase 1: Discovering Navigation Links ---")
    try:
        r = requests.get(home_url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching homepage during discovery: {e}")
        return {}

    # Locate the primary navigation container. Selector might need adjustments if site updates.
    nav_container = soup.select_one(".nav-row-upper .primary-nav ul")
    if not nav_container:
        print("Warning: Could not locate primary navigation list (ul inside .primary-nav). Trying fallback...")
        # Fallback: Try a broader selector if specific one fails
        nav_container = soup.select_one("nav.primary-nav ul")
        if not nav_container:
             print("Error: Failed to locate navigation container even with fallback.")
             return {}

    links = {}
    for a_tag in nav_container.select("a[href]"):
        href = (a_tag.get("href") or "").strip()
        text = a_tag.get_text(" ", strip=True)

        # Filter out empty links or extremely short text labels.
        if not href or not text or len(text) < 2:
            continue

        full_url = urljoin(home_url, href)
        u = urlparse(full_url)

        # Ensure links point to internal Fox News pages.
        if FOX_HOST not in (u.netloc or "").lower().replace("www.", ""):
            continue

        links[text] = normalize_category_url(full_url)

    print(f"-> Successfully found {len(links)} unique links from the navigation bar.")
    return links

# =========================
# Phase 2A: Standard Pagination Crawl (?page=N)
# =========================
def deep_crawl_category_with_pagination(base_url: str, rows: list, seen_urls: set, max_pages: int = MAX_PAGES):
    """
    Crawls a category page using standard pagination parameters (?page=1, ?page=2, ...).
    Extracts valid article URLs and appends them to the 'rows' list.
    """
    page = 1
    print(f"-> Starting Pagination Crawl: {base_url}")

    # Use a session for connection pooling
    with requests.Session() as session:
        session.headers.update(HEADERS)

        while page <= max_pages:
            current_url = base_url if page == 1 else f"{base_url}?page={page}"

            try:
                r = session.get(current_url, timeout=15)
                if r.status_code == 404:
                    print(f"   -> Page {page} not found (404). Stopping.")
                    break
                r.raise_for_status() # Raise exception for other bad status codes

                soup = BeautifulSoup(r.text, "html.parser")
                new_articles_on_page = 0

                # Find all links on the page
                for a_tag in soup.select("a[href]"):
                    href = (a_tag.get("href") or "").strip()
                    if not href:
                        continue

                    article_url = normalize_url(urljoin(FOX_HOME, href))

                    if is_good_article(article_url) and article_url not in seen_urls:
                        seen_urls.add(article_url)
                        rows.append({"url": article_url, "label": "fox"})
                        new_articles_on_page += 1

                # Stop if a non-first page yields no new unique articles.
                if page > 1 and new_articles_on_page == 0:
                    print(f"   -> No new articles found on page {page}. Stopping pagination.")
                    break

                print(f"   -> Page {page}: Found {new_articles_on_page} new article URLs.")
                page += 1
                time.sleep(SLEEP_S)

            except requests.exceptions.RequestException as e:
                print(f"   -> Error fetching {current_url}: {e}. Stopping pagination for this category.")
                break
            except Exception as e:
                print(f"   -> Unexpected error on page {page}: {e}. Stopping.")
                break

    print(f"-> Finished pagination crawl for {base_url} after checking {page-1} pages.")

# =========================
# Phase 2B: API-Based "Show More" Crawl
# =========================
def extract_tag_value_from_category_html(category_url: str, headers: dict) -> Optional[str]:
    """
    Fetches the category page HTML and attempts to extract the required 'tag' value
    used by the internal /api/article-search endpoint.
    Looks for patterns like offsetData: {"fox-news/..." : ...}
    """
    try:
        r = requests.get(category_url, headers=headers, timeout=15)
        r.raise_for_status()
        html_content = r.text
    except requests.exceptions.RequestException as e:
        print(f"   -> Error fetching category page for tag extraction: {e}")
        return None

    # Attempt 1: Specific regex matching the known 'offsetData' structure.
    match = TAG_FROM_OFFSETDATA_RE.search(html_content)
    if match:
        return match.group("tag")

    # Attempt 2: Fallback, looser regex looking for the tag pattern anywhere.
    match_fallback = re.search(r'"(fox-news/[a-z-]+/[^"]+)"', html_content)
    if match_fallback:
        return match_fallback.group(1)

    return None

def extract_urls_from_article_search_json(data_json) -> list[str]:
    """
    Recursively traverses the JSON response from the API to find article URLs.
    This is robust against variations in JSON structure across different site sections.
    """
    found_urls = set()

    # Potential keys holding URLs in the JSON response
    url_keys = {"url", "link", "weburl", "canonical", "canonicalurl"}

    def recursive_walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in url_keys and isinstance(value, str):
                    found_urls.add(value)
                # Continue recursion into nested dictionaries or lists
                elif isinstance(value, (dict, list)):
                    recursive_walk(value)
        elif isinstance(node, list):
            for item in node:
                recursive_walk(item)

    recursive_walk(data_json)

    # Normalize and filter found URLs
    valid_urls = []
    for u in found_urls:
        # Ensure absolute URL
        full_url = normalize_url(urljoin(FOX_HOME, u))
        # Double-check host
        if full_url.startswith(FOX_HOME) or full_url.startswith(FOX_HOME.replace("www.", "")):
             valid_urls.append(full_url)

    return sorted(set(valid_urls))

def deep_crawl_category_with_show_more_api(
    category_url: str,
    rows: list,
    seen_urls: set,
    batch_size: int = 11,
    max_steps: int = MAX_SHOWMORE_STEPS,
):
    """
    Crawls a category by mimicking the "Show More" button behavior via internal API calls.
    Iterates by increasing the 'from' offset parameter.
    """
    category_url = normalize_category_url(category_url)

    # Step 1: Get the necessary tag value from the page HTML.
    tag_value = extract_tag_value_from_category_html(category_url, headers=HEADERS)
    if not tag_value:
        print(f"   -> [ShowMore] Failed to extract 'tag_value' from HTML. Falling back to pagination crawl.")
        deep_crawl_category_with_pagination(category_url, rows, seen_urls)
        return

    print(f"-> Starting API-Based 'Show More' Crawl: {category_url}")
    print(f"   -> Using tag_value: {tag_value}")

    # Prepare headers for API requests
    api_headers = dict(HEADERS)
    api_headers["Accept"] = "application/json, text/plain, */*"
    api_headers["Referer"] = category_url
    api_headers["X-Requested-With"] = "XMLHttpRequest" # Often required for internal APIs

    session = requests.Session()
    offset = 0

    for step in range(max_steps):
        # Parameters for the internal search API endpoint
        params = {
            "searchBy": "tags",
            "values": tag_value,
            "excludeBy": "tags",
            "excludeValues": "", # Sometimes needed, sometimes empty is fine
            "size": batch_size,
            "from": offset,
        }

        try:
            r = session.get(ARTICLE_SEARCH_API, headers=api_headers, params=params, timeout=15)
            if r.status_code == 404:
                 print("   -> [ShowMore] API returned 404 (End of results). Stopping.")
                 break
            r.raise_for_status()
            
            try:
                data_json = r.json()
            except ValueError:
                 print("   -> [ShowMore] Error decoding JSON response. Stopping.")
                 break

            # Extract URLs from the JSON response
            urls_in_batch = extract_urls_from_article_search_json(data_json)

            new_in_batch = 0
            for u in urls_in_batch:
                if is_good_article(u) and u not in seen_urls:
                    seen_urls.add(u)
                    rows.append({"url": u, "label": "fox"})
                    new_in_batch += 1

            print(f"   -> [ShowMore] Step {step+1}: offset={offset}, fetched={len(urls_in_batch)}, new={new_in_batch}, total seen={len(seen_urls)}")

            # Stop if no new valid articles are returned in a batch.
            if new_in_batch == 0 and len(urls_in_batch) < batch_size:
                 print("   -> [ShowMore] No new articles found and batch is small. Stopping.")
                 break

            # Prepare next offset and pause gently.
            offset += batch_size
            time.sleep(SLEEP_S)

        except requests.exceptions.RequestException as e:
            print(f"   -> [ShowMore] API Request Error at offset {offset}: {e}. Stopping.")
            break
        except Exception as e:
            print(f"   -> [ShowMore] Unexpected Error at offset {offset}: {e}. Stopping.")
            break

    print(f"-> Finished 'Show More' API crawl for {category_url} after {step+1} steps.")

# =========================
# Crawling Strategy Router
# =========================
def deep_crawl_category_smart(category_url: str, rows: list, seen_urls: set):
    """
    Determines the appropriate crawling strategy (API-based vs. Pagination) for a given URL.
    """
    normalized_url = normalize_category_url(category_url)
    if normalized_url in SHOW_MORE_CATEGORIES:
        # Use specific API crawler for known dynamic pages.
        deep_crawl_category_with_show_more_api(normalized_url, rows, seen_urls, batch_size=11)
    else:
        # Default to standard pagination crawler for other pages.
        deep_crawl_category_with_pagination(normalized_url, rows, seen_urls)

# =========================
# Main Execution
# =========================
def crawl_all_dynamically():
    """
    Main function to orchestrate the crawling process: discover links, filter targets, and crawl.
    """
    # Step 1: Discover navigation links from the homepage.
    category_links = fetch_nav_links()

    # Step 2: Filter discovered URLs to select suitable crawling targets.
    target_urls = {}
    # Define path prefixes to exclude from crawling.
    excluded_sections = ("/video", "/ai", "/opinion", "/games", "/person", "/shows")
    # Define major sections to prioritize for crawling.
    target_sections = (
        "/us", "/politics", "/world", "/media", "/entertainment",
        "/sports", "/lifestyle", "/health", "/category", "/science"
    )

    for nav_name, url in category_links.items():
        path = urlparse(url).path.lower()

        # Skip explicitly excluded sections.
        if path.startswith(excluded_sections):
            continue

        # Keep URLs belonging to targeted major sections.
        if path.startswith(target_sections):
            # Use nav name as key to handle potential duplicate URLs under different labels
            target_urls[nav_name] = normalize_category_url(url)

    print(f"--- Phase 2: Starting Deep Crawl on {len(target_urls)} Filtered Categories ---")

    rows = []
    seen_urls = set()

    # Iterate through filtered categories and apply smart crawling strategy.
    for category_name, category_url in target_urls.items():
        print(f"\n[SECTION] Crawling '{category_name}' ({category_url})...")
        deep_crawl_category_smart(category_url, rows, seen_urls)
        time.sleep(SLEEP_S * 2) # Extra pause between major sections

    # Create DataFrame and save results to CSV.
    # drop_duplicates by URL just in case, though seen_urls set handles most.
    df = pd.DataFrame(rows, columns=["url", "label"]).drop_duplicates(subset=["url"])
    
    try:
        df.to_csv(OUT_CSV, index=False, encoding="utf-8")
        print(f"\n--- DONE ---\nSuccessfully saved {len(df)} unique article rows to '{OUT_CSV}'.")
    except IOError as e:
         print(f"\n--- ERROR ---\nFailed to save CSV file: {e}")

    return df

if __name__ == "__main__":
    # Entry point of the script.
    crawl_all_dynamically()