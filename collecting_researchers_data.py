import requests
import pandas as pd
import os

# Create a folder to store CSV files
output_folder = "orcid_data"
os.makedirs(output_folder, exist_ok=True)


# Function to fetch and save CSV data
def fetch_and_save_orcid_data(base_url, start, rows, output_folder, file_prefix):
    url = f"{base_url}&start={start}&rows={rows}"
    headers = {
        "Accept": "text/csv",  # Specify the Accept header for CSV data
        "Content-Type": "application/json",  # Adding content type for JSON, which is often used in APIs
    }
    print(f"Requesting URL: {url}")  # Debugging: Print the full request URL

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_path = os.path.join(output_folder, f"{file_prefix}_{start}.csv")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        print(f"Data saved to {file_path}")
        return True
    else:
        print(f"Failed to fetch data for start={start}. HTTP status code: {response.status_code}")
        return False


# Base URL and parameters
base_url = "https://pub.orcid.org/v3.0/csv-search/?q=affiliation-org-name:(\"Syracuse University\")"
rows = 1000  # Number of rows to fetch per request
max_records = 10000  # Total number of records to fetch

# Fetch data in chunks
for start in range(0, max_records, rows):
    success = fetch_and_save_orcid_data(base_url, start, rows, output_folder, "syracuse_data")
    if not success:
        break

# Check if any CSV files were downloaded
csv_files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".csv")]

if not csv_files:
    print("No data fetched. Exiting.")
else:
    # Read and concatenate all CSV files
    all_data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

    # Save the merged data to a single CSV
    merged_file = os.path.join(output_folder, "merged_syracuse_data.csv")
    all_data.to_csv(merged_file, index=False)
    print(f"Merged data saved to {merged_file}")


import pandas as pd
data = pd.read_csv('orcid_data/merged_syracuse_data.csv')
print(data.columns)
# Check for duplicates
print("Checking for duplicates...")
duplicates = data[data.duplicated()]
if not duplicates.empty:
    print(f"Found {len(duplicates)} duplicate rows.")
    print(duplicates)
else:
    print("No duplicates found.")