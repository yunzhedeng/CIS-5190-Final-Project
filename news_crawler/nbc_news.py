import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
import time
import random

def scrape_nbc_archive():
    """
    Scrapes the NBC News 2025 archive for article URLs across all months,
    saves only {url, label} to a CSV.
    """
    BASE_URL = "https://www.nbcnews.com"
    ARCHIVE_BASE = f"{BASE_URL}/archive/articles/2025/"
    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    source_name = "nbc"
    output_filename = "nbc_news_2025_archive.csv"

    rows = []
    seen_urls = set()

    print("Starting to scrape NBC News 2025 article archive...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    for month in MONTHS:
        month_url = ARCHIVE_BASE + month
        print(f"\n>>> Now scraping: {month_url}")

        try:
            response = requests.get(month_url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            new_count = 0
            for link in soup.find_all("a", href=True):
                href = (link.get("href") or "").strip()
                if not href:
                    continue

                # Exclude month navigation links
                if "/archive/articles/2025/" in href.lower():
                    continue

                # Heuristic article-path filter (keep yours)
                is_article_link = any(p in href for p in [
                    "/news/", "/video/", "/business/", "/politics/", "/health/"
                ])
                if not is_article_link:
                    continue

                full_url = urljoin(BASE_URL, href)

                # de-dup
                if full_url in seen_urls:
                    continue

                seen_urls.add(full_url)
                rows.append({
                    "url": full_url,
                    "label": source_name
                })
                new_count += 1

            print(f"  - Added {new_count} new article URLs")

            time.sleep(random.uniform(1, 3))

        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", "unknown")
            print(f"  - Failed to scrape {month} (HTTP Error: {status})")
            continue
        except requests.exceptions.RequestException as e:
            print(f"  - Failed to scrape {month} (Connection/Timeout Error): {e}")
            continue

    if rows:
        print(f"\n--- Scraping Complete ---")
        print(f"Total {len(rows)} unique article URLs found. Writing: {output_filename}")

        fieldnames = ["url", "label"]
        try:
            with open(output_filename, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"CSV file '{output_filename}' written successfully.")
        except IOError as e:
            print(f"Failed to write CSV file: {e}")
    else:
        print("\nCould not scrape any article data.")

if __name__ == "__main__":
    scrape_nbc_archive()
