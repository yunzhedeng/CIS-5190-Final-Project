import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def extract_label(url):
    try:
        parts = url.split("/")
        return parts[3]  # foxnews.com/<label>/...
    except:
        return "unknown"

def extract_headline(url):
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={"user-agent": "Mozilla/5.0"}
        )
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.content, 'html.parser')

        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title.get("content")

        if soup.title:
            return soup.title.text

        h1 = soup.find("h1")
        if h1:
            return h1.text

        return None

    except:
        return None


def scrape_urls(input_path, output_path):
    df = pd.read_csv(input_path)

    results = []
    total = len(df)

    for index, row in df.iterrows():
        url = row["url"]
        print(f"[{index+1}/{total}] Fetching:", url)

        headline = extract_headline(url)
        label = extract_label(url)

        print("  headline:", headline)
        print("  label:", label)

        results.append({
            "url": url,
            "headline": headline,
            "label": label
        })

        time.sleep(0.05)

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"\nFinished! Cleaned data saved to {output_path}")


if __name__ == "__main__":
    scrape_urls("url_only_data.csv", "cleaned_headlines.csv")
