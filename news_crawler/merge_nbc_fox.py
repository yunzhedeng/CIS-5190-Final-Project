import csv

def merge_csv_files(input_files: list, output_file: str):
    """
    Merge multiple CSV files that contain at least columns:
    - url
    - label

    Output CSV will contain only: url, label
    Duplicate URLs will be removed.
    """

    merged_rows = []
    seen_urls = set()

    fieldnames = ["url", "label"]

    print("--- Starting Data Merging Process ---")

    for file_path in input_files:
        try:
            with open(file_path, "r", newline="", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)

                # check required columns
                if not all(col in reader.fieldnames for col in fieldnames):
                    print(f"Skipping {file_path}: missing required columns {fieldnames}")
                    continue

                print(f"Processing: {file_path}")

                for row in reader:
                    url = row["url"].strip()
                    label = row["label"].strip()

                    if not url or url in seen_urls:
                        continue

                    seen_urls.add(url)
                    merged_rows.append({
                        "url": url,
                        "label": label
                    })

        except FileNotFoundError:
            print(f"Error: file not found -> {file_path}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if not merged_rows:
        print("No data merged. Output file not created.")
        return

    print(f"\nWriting {len(merged_rows)} unique rows to {output_file}...")

    try:
        with open(output_file, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)

        print(f"Success: merged file saved as '{output_file}'")

    except IOError as e:
        print(f"Error writing output file: {e}")


# --- Main Execution ---
if __name__ == "__main__":

    input_files_list = [
        "fox_articles.csv",
        "nbc_news_2025_archive.csv"
    ]

    output_file_name = "merged_news_urls.csv"

    merge_csv_files(input_files_list, output_file_name)
