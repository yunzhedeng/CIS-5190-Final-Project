# -*- coding: utf-8 -*-
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ===== Config and Constants =====
FOX_HOME = "https://www.foxnews.com/"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SLEEP_S = 0.2 
MAX_PAGES = 10 # Safety cap to prevent infinite crawling

# 1) Strong article signal: URLs containing /YYYY/MM/DD/
DATE_PATH_RE = re.compile(r"/\d{4}/\d{2}/\d{2}/")

# 2) Explicitly excluded non-article paths 
BAD_PATH_PREFIXES = (
    "/shows/", "/games/", "/person/", "/story/", "/deals/", "/video/",
    "/category/", "/topic/", "/elections/", "/live-news/", "/live/"
)
# Substrings indicating external or unwanted Fox properties
BAD_SUBSTRINGS = (
    "foxnation", "outkick", "radio.foxnews.com", "foxweather.com", "foxbusiness.com",
)


# --- Utility Functions ---

def is_good_article(url: str) -> bool:
    """Determine whether a URL corresponds to a Fox News article page."""
    u = urlparse(url)
    host = (u.netloc or "").lower().replace("www.", "")
    path = (u.path or "").lower()

    if host != "foxnews.com":
        return False
    
    # Strong signal: Date path is present
    if DATE_PATH_RE.search(path):
        return True

    # Exclude known non-article paths
    if any(path.startswith(p) for p in BAD_PATH_PREFIXES):
        return False

    # Exclude non-article type URLs (e.g., /politics/ /world/ without date)
    if path.count("/") <= 1:
        return False

    # Heuristic filter: check slug length
    last = path.strip("/").split("/")[-1]
    if len(last) < 12:
        return False

    return True

def normalize_url(url: str) -> str:
    """Normalize article URLs by removing print suffixes and fragments."""
    if url.endswith(".print"):
        url = url[:-6]
    return url


# ----------------------------------------------------
# Function A: Automatically discover all navigation bar links
# ----------------------------------------------------
def fetch_nav_links(home_url: str = FOX_HOME, headers: dict = HEADERS) -> dict:
    """
    Fetches all URLs found in the main navigation bar (including dropdowns).
    """
    print(f"--- Phase 1: Discovering Navigation Links ---")
    
    try:
        r = requests.get(home_url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching homepage during discovery: {e}")
        return {}
    
    # === CRITICAL FIX: Use the accurate CSS selector based on provided HTML ===
    # Target: ul list inside .primary-nav
    nav_container = soup.select_one('.nav-row-upper .primary-nav ul') 

    if not nav_container:
         # Log the error if the selector fails
         print("Error: Failed to locate the primary navigation list (ul inside .primary-nav).")
         return {}
    
    links = {}
    
    # Iterate through all links within the navigation container
    for a in nav_container.select("a[href]"):
        href = a.get("href", "").strip()
        headline = a.get_text(" ", strip=True) 

        if not href or not headline or len(headline) < 3:
            continue

        full_url = urljoin(home_url, href)

        # Filter external links and anchor links
        u = urlparse(full_url)
        if 'foxnews.com' not in u.netloc.lower().replace("www.", ""):
            continue
        if not u.path and u.fragment:
            continue
            
        links[headline] = full_url

    print(f"-> Successfully found {len(links)} unique links from the navigation bar.")
    return links

# ----------------------------------------------------
# Function B: Deep crawl a category with pagination
# ----------------------------------------------------
def deep_crawl_category_with_pagination(
    base_url: str, 
    rows: list, 
    seen: set,
    max_pages: int = MAX_PAGES  
):
    """
    Visits a category/topic page and continues to crawl through all pages 
    using the correct ?page=N query parameter format.
    """
    page = 1
    
    print(f"-> Starting Deep Crawl: {base_url}")

    while page <= max_pages:
        current_url = base_url
        if page > 1:
            # Using the correct ?page=N format for pagination
            current_url = f"{base_url}?page={page}" 
        
        try:
            r = requests.get(current_url, headers=HEADERS, timeout=15)
            
            if r.status_code == 404:
                # 404 typically means the end of the category pages
                break
            if r.status_code != 200:
                break
            
            soup = BeautifulSoup(r.text, "html.parser")
            article_found_on_page = 0 

            for a in soup.select("a[href]"):
                href = a.get("href", "").strip()
                if not href:
                    continue

                article_url = normalize_url(urljoin(FOX_HOME, href))
                
                # Only save if the link is a "good article"
                if is_good_article(article_url):
                    headline = a.get_text(" ", strip=True)
                    if not headline or len(headline) < 8:
                        continue
                        
                    key = (article_url, headline)
                    if key not in seen:
                        seen.add(key)
                        rows.append({
                            "url": article_url,
                            "headline": headline,
                            "label": "fox"
                        })
                        article_found_on_page += 1
            
            # Stop condition: if we found no new articles after the first page
            if page > 1 and article_found_on_page == 0:
                 break
            
            print(f"   -> Page {page} found {article_found_on_page} new articles.")
            page += 1
            time.sleep(SLEEP_S) 
            
        except requests.exceptions.RequestException as e:
            print(f"   -> Stopping: Error fetching {current_url} - {e}")
            break
        except Exception as e:
            print(f"   -> Stopping: An unexpected error occurred - {e}")
            break
            
    print(f"-> Finished deep crawl for {base_url} after {page-1} pages.")

# ----------------------------------------------------
# Main execution function (Dynamic Crawling)
# ----------------------------------------------------
def crawl_all_dynamically():
    
    # Step 1: Automatically discover all category links
    category_links = fetch_nav_links()
    
    # Step 2: Filter URLs suitable for deep pagination crawling
    target_urls = {}
    
    for headline, url in category_links.items():
        # Exclude pages unlikely to have paginated articles: Video, AI, Opinion, Games
        u = urlparse(url).path.lower()
        if u.startswith(('/video', '/ai', '/opinion', '/games')):
            continue
            
        # Only keep main sections or subcategory paths that list articles
        if u.startswith(('/us', '/politics', '/world', '/media', '/entertainment', '/sports', '/lifestyle', '/health', '/category')):
            target_urls[headline] = url
    
    print(f"--- Phase 2: Starting Deep Crawl on {len(target_urls)} Filtered Categories ---")
    
    # Step 3: Iterate through targets and perform deep, paginated crawl
    rows = []
    seen = set()

    for category_name, category_url in target_urls.items():
        print(f"\n[SECTION] Crawling {category_name}...")
        deep_crawl_category_with_pagination(category_url, rows, seen)

    out_csv = "fox_articles.csv"
    df = pd.DataFrame(rows, columns=["url", "headline", "label"])
    df = df.drop_duplicates(subset=['url']) 
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\n--- DONE ---\nSaved a total of {len(df)} unique article rows from all dynamic categories -> {out_csv}")
    return df

if __name__ == "__main__":
    crawl_all_dynamically()