import pandas as pd
import torch
import gc
import sqlite3
import os
from pdf_pre import extract_text_from_pdf
from database_handler import (
    setup_database, insert_work, fetch_unsummarized_works, update_summary,
    count_entries_in_table, check_missing_files_in_db, close_connection, remove_duplicates
)
from t5_model import summarize_text, fine_tune_t5_on_papers

# Build relative paths for the PDF folder, model output, and DB
script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_folder = os.path.join(script_dir, "download_pdfs")
output_model_dir = os.path.join(script_dir, "fine_tuned_t5")
db_path = os.path.join(script_dir, "researchers.db")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

def clear_memory():
    """Clear GPU and CPU memory."""
    torch.cuda.empty_cache()
    gc.collect()
    print("Cleared memory and cache.")

def populate_database_from_pdfs():
    """Step 1: Process all PDFs in the folder, store text in DB."""
    setup_database()  # Ensure the DB & 'works' table is set up
    remove_duplicates()  # Remove any existing duplicates

    # Get the list of PDF files
    if not os.path.isdir(pdf_folder):
        print(f"PDF folder not found: {pdf_folder}")
        return

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    total_files = len(pdf_files)
    if total_files == 0:
        print("No PDF files found in the directory.")
        return

    print(f"Found {total_files} PDF files in the folder.")

    # Check for missing files in the database
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

    print(f"Database populated with all PDFs. Total entries in database: {count_entries_in_table()}")

def generate_summaries_for_database():
    """Step 2: Summarize all works in the database."""
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
    """Step 3: Fine-tune T5 on all summarized data."""
    print("Preparing data for fine-tuning...")
    conn = sqlite3.connect(db_path)
    query = """
        SELECT full_text, summary
        FROM works
        WHERE summary_status = 'summarized' AND progress = 1
    """
    papers_df = pd.read_sql_query(query, conn)
    conn.close()

    if papers_df.empty:
        print("No summarized data available for fine-tuning.")
        return

    # Ensure correct columns exist before fine-tuning
    if 'full_text' in papers_df.columns and 'summary' in papers_df.columns:
        papers_df = papers_df.rename(columns={"full_text": "input_text", "summary": "summary"})
        print(f"Fine-tuning on {len(papers_df)} summarized entries...")
        output_dir = fine_tune_t5_on_papers(papers_df, output_model_dir)
        print(f"Fine-tuned model saved at: {output_dir}")
    else:
        print("Error: Dataset must contain 'full_text' and 'summary' columns!")

    clear_memory()

if __name__ == "__main__":
    try:
        print("STEP 1: Populating the database from PDFs...")
        populate_database_from_pdfs()

        print("STEP 2: Generating summaries...")
        generate_summaries_for_database()

        print("STEP 3: Fine-tuning the model on summaries...")
        fine_tune_model_on_summaries()

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        close_connection()
        print("Pipeline completed and database connection closed.")
