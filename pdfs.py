"""
pdfs.py

This module handles downloading PDF files from a CSV file containing PDF URLs.
It reads the CSV, downloads each PDF, and saves them in a designated folder.
"""

import os
import pandas as pd
import requests

# Define the path to the CSV file containing PDF URLs
pdf_download_csv_path = r"C:\Users\arapte\Downloads\cleaned_author_works.csv"
# Define the folder where PDFs will be saved
pdf_folder = r"C:\codes\t5-db\download_pdfs"

# Ensure the download directory exists
os.makedirs(pdf_folder, exist_ok=True)

def download_pdfs():
    """
    Downloads PDFs listed in the CSV file specified by 'pdf_download_csv_path'.
    Each PDF is saved in the 'pdf_folder' with a unique name based on its index.
    
    The function:
      - Reads the CSV file.
      - Iterates through each row to get the URL.
      - Downloads the PDF using the requests library with a timeout.
      - Handles timeouts, HTTP errors, and other request exceptions.
    """
    df = pd.read_csv(pdf_download_csv_path)
    
    for index, row in df.iterrows():
        url = row['pdf_url']
        # Check if the URL is valid (non-empty and non-null)
        if pd.notna(url) and url.strip():
            # Construct a filename for saving the PDF
            file_name = os.path.join(pdf_folder, f"file_{index}.pdf")
            try:
                # Attempt to download the PDF file with a 5-second timeout
                response = requests.get(url, stream=True, timeout=5)
                response.raise_for_status()  # Raise exception for HTTP errors
                # Write the PDF in chunks to avoid memory issues
                with open(file_name, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                print(f"Downloaded: {file_name}")
            except requests.exceptions.Timeout:
                print(f"⚠️ Skipping (timeout): {url}")
            except requests.exceptions.HTTPError as e:
                if response.status_code in [403, 401]:
                    print(f"⚠️ Skipping (restricted access, manual download needed): {url}")
                else:
                    print(f"⚠️ Failed to download {url}: HTTP Error {e}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Failed to download {url}: {e}")
    
    print("PDF download process completed.")
