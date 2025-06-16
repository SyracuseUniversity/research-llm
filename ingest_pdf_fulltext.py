"""
ingest_pdf_fulltext.py  –  Extract full text from each downloaded PDF and insert into `works`.
Skips duplicates and corrupted files gracefully.
"""

import os
import glob
import sqlite3
from pdf_pre import extract_raw_text_from_pdf, clean_text

DB_PATH = r"C:\codes\t5-db\researchers_all.db"
PDF_DOWNLOAD_DIR = r"C:\codes\t5-db\download_pdfs"


def work_exists(file_name: str, conn: sqlite3.Connection) -> bool:
    """
    Return True if a row with this file_name already exists in works.
    """
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM works WHERE file_name = ? LIMIT 1;", (file_name,))
    return cur.fetchone() is not None


def main():
    if not os.path.isdir(PDF_DOWNLOAD_DIR):
        print(f"No download directory found: {PDF_DOWNLOAD_DIR}")
        return

    pdf_files = glob.glob(os.path.join(PDF_DOWNLOAD_DIR, "*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} PDFs. Ingesting full text into 'works'…")

    conn = sqlite3.connect(DB_PATH)
    for idx, pdf_path in enumerate(pdf_files, start=1):
        file_name = os.path.basename(pdf_path)

        # Skip if already exists in works
        try:
            if work_exists(file_name, conn):
                continue
        except Exception as e:
            print(f"ERROR checking existence for {file_name}: {e}")
            continue

        # Extract text
        try:
            raw_text = extract_raw_text_from_pdf(pdf_path)
            if not raw_text or not raw_text.strip():
                print(f"Skipping empty PDF: {file_name}")
                continue
            cleaned = clean_text(raw_text)
        except Exception as e:
            print(f"Failed to extract from {pdf_path}: {e}")
            continue

        # Insert into works
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO works (file_name, full_text, summary, summary_status, progress)
                VALUES (?, ?, '', 'unsummarized', 0)
                """,
                (file_name, cleaned)
            )
        except sqlite3.IntegrityError:
            print(f"Skipped duplicate insertion for: {file_name}")
        except Exception as e:
            print(f"ERROR inserting {file_name}: {e}")

        # Progress log
        if idx % 50 == 0 or idx == total:
            print(f"  → {idx}/{total} PDFs processed")

    conn.commit()
    conn.close()
    print("All PDF full‐text ingested.")
    

if __name__ == "__main__":
    main()
