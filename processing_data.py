import csv

def merge_csv_files(input_files: list, output_file: str):
    """
    Reads data from multiple CSV files that share the same structure 
    and merges the content into a single output CSV file.
    
    The expected columns in the input files are: 'url', 'headline', 'source'.
    """
    
    all_merged_data = []
    
    # Define the common column names for the output file
    fieldnames = ['url', 'headline', 'source']
    
    print("--- Starting Data Merging Process ---")

    for file_path in input_files:
        try:
            # Open the input file for reading
            with open(file_path, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                
                # Verify that the required headers are present
                if not all(col in reader.fieldnames for col in fieldnames):
                    print(f"Skipping {file_path}: Missing one or more required columns (url, headline, source).")
                    continue
                    
                print(f"Processing and merging data from: {file_path}...")
                
                # Iterate through each row and append relevant data
                for row in reader:
                    # Select only the required columns and append to the list
                    all_merged_data.append({
                        'url': row['url'],
                        'headline': row['headline'],
                        'label': row['label']
                    })
                    
        except FileNotFoundError:
            print(f"Error: Input file not found: {file_path}")
        except Exception as e:
            print(f"An unexpected error occurred while reading {file_path}: {e}")

    # Write the merged data to the output CSV file
    if all_merged_data:
        print(f"\nWriting {len(all_merged_data)} merged entries to {output_file}...")
        
        try:
            # Open the output file for writing
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                
                writer.writeheader() # Write the header row
                writer.writerows(all_merged_data) # Write all the data rows
                
            print(f"Success: Merged data saved to '{output_file}'.")
        except IOError as e:
            print(f"Error writing output file: {e}")
    else:
        print("No data was successfully merged. Output file not created.")


# --- Main Execution ---
if __name__ == "__main__":
    
    # List of input files to process
    input_files_list = [
        'fox_articles.csv', 
        'nbc_news_2025_archive.csv'
    ]
    
    # Output file name
    output_file_name = 'cleaned_headlines.csv'
    
    merge_csv_files(input_files_list, output_file_name)