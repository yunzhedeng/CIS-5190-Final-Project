import requests
from bs4 import BeautifulSoup
import csv
from urllib.parse import urljoin
import time
import random

def scrape_nbc_archive():
    """
    Scrapes the NBC News 2025 archive for articles across all months, 
    extracts the article URL, headline, and source, and saves them to a CSV file.
    """
    
    # Base settings
    BASE_URL = "https://www.nbcnews.com"
    ARCHIVE_BASE = f"{BASE_URL}/archive/articles/2025/"
    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]
    
    all_articles = []
    source_name = "nbc"
    output_filename = "nbc_news_2025_archive.csv"

    print("Starting to scrape NBC News 2025 article archive...")
    
    # Iterate through all months
    for month in MONTHS:
        month_url = ARCHIVE_BASE + month
        print(f"\n>>> Now scraping: {month_url}")
        
        try:
            # Set request headers to mimic a browser visit
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            # Send HTTP request, get page content, and set timeout
            response = requests.get(month_url, headers=headers, timeout=15)
            # Check response status code and raise an error for bad responses (4xx or 5xx)
            response.raise_for_status() 
            
            # Parse HTML using BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_count_for_month = 0
            
            # Find all <a> tags (links) on the page
            # Based on the provided screenshot, article titles are the link texts
            all_links = soup.find_all('a', href=True)
            
            # Filter for article links
            for link in all_links:
                href = link['href']
                headline = link.text.strip()
                
                # --- Link Filtering Logic ---
                
                # 1. Headline must not be empty and should have a reasonable length (to exclude very short text links)
                if not headline or len(headline) < 10:
                    continue
                    
                # 2. Exclude month navigation links (they contain "/archive/articles/2025/")
                # Month links on the archive page have this structure, which is different from news article links.
                if "/archive/articles/2025/" in href.lower():
                    continue

                # 3. Exclude other non-article links (e.g., SITE MAP, navigation links, etc.)
                # Check if the URL matches common path structures for news articles
                is_article_link = any(p in href for p in ["/news/", "/video/", "/business/", "/politics/", "/health/"])
                
                # Only extract information if the link appears to be a news article
                if is_article_link:
                    
                    # Combine to form the complete URL (handles relative paths)
                    full_url = urljoin(BASE_URL, href)
                    
                    all_articles.append({
                        "url": full_url,
                        "headline": headline,
                        "source": source_name
                    })
                    article_count_for_month += 1
                        
            print(f"  - Successfully scraped {article_count_for_month} articles")
            
            # Add a random delay to avoid excessive load on the server
            time.sleep(random.uniform(1, 3))
            
        except requests.exceptions.HTTPError as e:
            print(f"  - Failed to scrape {month} (HTTP Error: {e.response.status_code})")
            continue
        except requests.exceptions.RequestException as e:
            print(f"  - Failed to scrape {month} (Connection or Timeout Error): {e}")
            continue

    # Write to CSV file
    if all_articles:
        print(f"\n--- Scraping Complete ---")
        print(f"Total {len(all_articles)} articles found. Writing to CSV file: {output_filename}")
        
        # Define column names for the CSV file
        fieldnames = ["url", "headline", "label"]
        
        try:
            # Write to file with utf-8 encoding; newline='' prevents writing extra blank rows
            with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
                # Use DictWriter to handle the list of dictionaries
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader() # Write the header row
                writer.writerows(all_articles) # Write the data rows
            
            print(f"CSV file '{output_filename}' written successfully. Check the directory where the script was run.")
            
        except IOError as e:
            print(f"Failed to write CSV file: {e}")
            
    else:
        print("\nCould not scrape any article data.")

# Run the scraping function
if __name__ == "__main__":
    scrape_nbc_archive()