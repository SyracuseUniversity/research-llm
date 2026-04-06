"""
pdfs.py

Download PDFs listed in cleaned_author_works.csv,
with robust error handling and skipping already downloaded files.
"""

import os
import pandas as pd
import requests
from tqdm import tqdm

CSV_PATH = r"C:\Users\arapte\Downloads\Application\cleaned_author_works.csv"
DOWNLOAD_DIR = r"C:\codes\t5-db\download_pdfs"


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    try:
        df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
    except Exception as e:
        print(f"Failed to read CSV at {CSV_PATH}: {e}")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    total = len(df)
    for idx, row in tqdm(df.iterrows(), total=total, desc="Downloading PDFs"):
        url = row.get("pdf_url", "")
        if not isinstance(url, str) or not url.strip():
            continue

        url = url.strip()
        file_name = os.path.join(DOWNLOAD_DIR, f"file_{idx}.pdf")

        if os.path.isfile(file_name):
            continue

        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()

            with open(file_name, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        except requests.exceptions.Timeout:
            print(f"Timeout while downloading: {url}")
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status in (401, 403):
                print(f"Access denied for: {url} (status {status})")
            else:
                print(f"HTTP error {status} for: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Request error for {url}: {e}")
        except Exception as e:
            print(f"Error saving {url}: {e}")

    print("PDF download complete.")


if __name__ == "__main__":
    main()
