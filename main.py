"""
main.py

This is the main entry point for the entire pipeline. It performs the following steps:
  1. Populates the 'works' table from PDFs by extracting full text.
  1b. Populates the 'research_info' table using the CSV file 'merged_cleaned.csv'.
  2. Generates summaries for unsummarized works using the T5 model.
  3. Fine-tunes the T5 model on the summarized data (resuming from the latest checkpoint if available).
  4. Fine-tunes the LLaMA model on the same summarized data (resuming from the latest checkpoint if available).

To run the pipeline, execute: python main.py
"""

import os
import sqlite3
import gc
import torch
import pandas as pd
from pdf_pre import extract_text_from_pdf, extract_research_info_from_pdf
from database_handler import (
    setup_database, insert_work, fetch_unsummarized_works, update_summary,
    count_entries_in_table, check_missing_files_in_db, close_connection, remove_duplicates,
    setup_research_info_table, insert_research_info
)
from model import summarize_text, fine_tune_t5_on_papers

# Paths for PDFs, CSV, and model output
pdf_folder = r"C:\codes\t5-db\download_pdfs"
csv_path = r"C:\codes\llama-bot-pipeline\merged_cleaned.csv"
output_model_dir = r"C:\codes\t5-db\fine_tuned_t5"
FINE_TUNED_MODEL_PATH = r"C:\codes\llama32\fine_tuned_llama"  # Added definition

# Set device for torch (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def clear_memory():
    """Clear GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared memory and cache.")

def populate_database_from_pdfs():
    """
    STEP 1: Process all PDFs in the folder and store their full text in the 'works' table.
    """
    setup_database()        # Ensure the 'works' table exists.
    remove_duplicates()     # Remove duplicate entries if any.

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    if total_files == 0:
        print("No PDF files found in the directory.")
        return

    print(f"Found {total_files} PDF files in the folder.")
    missing_files = check_missing_files_in_db(pdf_files)
    print(f"{len(missing_files)} files missing from the database. Processing these files...")

    for idx, file_name in enumerate(missing_files, start=1):
        file_path = os.path.join(pdf_folder, file_name)
        print(f"[{idx}/{len(missing_files)}] Processing: {file_name}")
        extracted_text = extract_text_from_pdf(file_path)
        if extracted_text:
            insert_work(
                file_name=file_name,
                full_text=extracted_text,
                summary=None,
                summary_status="unsummarized",
                progress=0
            )

    print(f"Database populated with all PDFs. Total entries in 'works': {count_entries_in_table('works')}")

def populate_research_info_from_csv():
    """
    STEP 1b: Populate the 'research_info' table using the CSV file 'merged_cleaned.csv'.
    
    Reads the CSV, extracts researcher information, and inserts records into the table.
    Uses:
      - 'author_name' as researcher_name.
      - 'title' (or 'work_title' if 'title' is missing) as work_title.
      - 'author_name' as authors if no dedicated authors column exists.
      - Constructs 'info' from doi_url, publication_date, and landing_url.
    """
    try:
        df = pd.read_csv(csv_path, sep=",")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    total_records = len(df)
    print(f"Populating research_info table from CSV with {total_records} records...")
    for idx, row in df.iterrows():
        researcher_name = row.get("author_name", "")
        # Prefer "title" column for work_title; fallback to "work_title"
        work_title = row.get("title", "") or row.get("work_title", "")
        authors = row.get("author_name", "")
        doi = row.get("doi_url", "")
        pub_date = row.get("publication_date", "")
        landing_url = row.get("landing_url", "")
        info_parts = []
        if pd.notna(doi) and str(doi).strip() != "":
            info_parts.append(f"DOI: {doi}")
        if pd.notna(pub_date) and str(pub_date).strip() != "":
            info_parts.append(f"Publication Date: {pub_date}")
        if pd.notna(landing_url) and str(landing_url).strip() != "":
            info_parts.append(f"Landing URL: {landing_url}")
        info = " | ".join(info_parts)
        insert_research_info(researcher_name, work_title, authors, info)
        if (idx + 1) % 100 == 0:
            print(f"[{idx+1}/{total_records}] Processed CSV records.")
    print("CSV-based research info population complete.")

def generate_summaries_for_database():
    """
    STEP 2: Generate summaries for all unsummarized works in the database.
    """
    unsummarized_works = fetch_unsummarized_works()
    if not unsummarized_works:
        print("No unsummarized works found.")
        return

    print(f"Found {len(unsummarized_works)} unsummarized works. Generating summaries...")
    for idx, (work_id, full_text) in enumerate(unsummarized_works, start=1):
        try:
            summary = summarize_text(full_text, idx=idx, total=len(unsummarized_works))
            update_summary(work_id, summary)
            print(f"Summary updated for work ID: {work_id}")
        except Exception as e:
            print(f"Error summarizing work ID {work_id}: {e}")
        clear_memory()

def fine_tune_model_on_summaries():
    """
    STEP 3: Fine-tune the T5 model on all summarized data.
    Resumes training from the latest checkpoint if available.
    """
    print("Preparing data for T5 fine-tuning...")
    conn = sqlite3.connect(r"C:\codes\t5-db\researchers.db")
    query = """
        SELECT full_text, summary
        FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """
    papers_df = pd.read_sql_query(query, conn)
    conn.close()

    if papers_df.empty:
        print("No summarized data available for T5 fine-tuning.")
        return

    # Rename 'full_text' column to 'input_text' for the model.
    papers_df = papers_df.rename(columns={"full_text": "input_text"})
    if 'input_text' in papers_df.columns and 'summary' in papers_df.columns:
        print(f"Fine-tuning on {len(papers_df)} summarized entries (T5)...")
        output_dir = fine_tune_t5_on_papers(papers_df, output_model_dir)
        print(f"T5 fine-tuned model saved at: {output_dir}")
    else:
        print("Error: Dataset must contain 'input_text' and 'summary' columns!")
    clear_memory()

def fine_tune_llama_model_on_summaries():
    """
    STEP 4: Fine-tune the LLaMA model on all summarized data.
    Resumes training from the latest checkpoint if available.
    
    Uses the same summarized data as for T5 fine-tuning. Renames columns appropriately:
      - 'full_text' becomes 'input_text'
      - 'summary' becomes 'target_text'
    """
    print("Preparing data for LLaMA fine-tuning...")
    conn = sqlite3.connect(r"C:\codes\t5-db\researchers.db")
    query = """
        SELECT full_text, summary
        FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """
    papers_df = pd.read_sql_query(query, conn)
    conn.close()

    if papers_df.empty:
        print("No summarized data available for LLaMA fine-tuning.")
        return

    # Rename columns for LLaMA: input_text and target_text.
    papers_df = papers_df.rename(columns={"full_text": "input_text", "summary": "target_text"})
    if 'input_text' in papers_df.columns and 'target_text' in papers_df.columns:
        print(f"Fine-tuning LLaMA model on {len(papers_df)} summarized entries...")
        from llama_model import fine_tune_llama_on_papers  # Import here to avoid circular dependency
        output_dir_llama = fine_tune_llama_on_papers(papers_df, FINE_TUNED_MODEL_PATH)
        print(f"LLaMA fine-tuned model saved at: {output_dir_llama}")
    else:
        print("Error: Dataset must contain 'input_text' and 'target_text' columns!")
    clear_memory()

if __name__ == "__main__":
    try:
        print("STEP 1: Populating the database from PDFs (works table)...")
        populate_database_from_pdfs()

        print("STEP 1b: Populating research_info from CSV...")
        populate_research_info_from_csv()

        print("STEP 2: Generating summaries...")
        generate_summaries_for_database()

        print("STEP 3: Fine-tuning the T5 model on summaries...")
        fine_tune_model_on_summaries()

        print("STEP 4: Fine-tuning the LLaMA model on summaries...")
        fine_tune_llama_model_on_summaries()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        close_connection()
        print("Pipeline completed and database connection closed.")
