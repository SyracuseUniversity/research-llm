"""
run_pipeline_after_download.py  –  Execute the pipeline from “ingest full text” onward,
assuming all PDFs have already been downloaded into download_pdfs/.
"""

import os
import glob
import sqlite3
import torch

# Paths (same unified DB as before)
UNIFIED_DB = r"C:\codes\t5-db\researchers_all.db"
PDF_DOWNLOAD_DIR = r"C:\codes\t5-db\download_pdfs"

# Ensure the unified DB exists
if not os.path.exists(UNIFIED_DB):
    raise FileNotFoundError(f"Unified database not found: {UNIFIED_DB}")

print("Using existing database at:", UNIFIED_DB)

# Step 1 (skipped): Download PDFs

print("\n=== Step 2: Ingest full text into 'works' ===")
from ingest_pdf_fulltext import main as ingest_fulltext
ingest_fulltext()

print("\n=== Step 3: Ingest PDF metadata into 'research_info' ===")
from pdf_pre import extract_research_info_from_pdf
def ingest_pdf_metadata():
    conn = sqlite3.connect(UNIFIED_DB)
    cur = conn.cursor()

    insert_sql = """
    INSERT OR IGNORE INTO research_info
      (researcher_name, work_title, authors, info)
    VALUES (?, ?, ?, ?);
    """

    pdf_files = glob.glob(os.path.join(PDF_DOWNLOAD_DIR, "*.pdf"))
    total = len(pdf_files)
    print(f"Found {total} PDFs for metadata ingestion.")

    for idx, pdf_path in enumerate(pdf_files, start=1):
        try:
            meta = extract_research_info_from_pdf(pdf_path)
        except Exception as e:
            print(f"Failed to extract from {pdf_path}: {e}")
            meta = None

        if not meta:
            continue

        cur.execute(
            insert_sql,
            (meta["researcher_name"], meta["work_title"], meta["authors"], meta["info"])
        )

        if idx % 100 == 0 or idx == total:
            print(f"  → {idx}/{total} PDF metadata inserted")

    conn.commit()
    conn.close()
    print("PDF metadata ingestion complete.")

ingest_pdf_metadata()

print("\n=== Step 4: Ingest CSV metadata into 'research_info' ===")
from csv_handler import populate_research_info_from_csv
populate_research_info_from_csv()

print("\n=== Step 5: Clean & normalize `research_info` ===")
from clean_db import main as clean_db_main
clean_db_main()

print("\n=== Step 6: Summarize works with T5 ===")
from summarize_works import main as summarize_works
summarize_works()

print("\n=== Step 7: Generate QA pairs for LLaMA ===")
from llama_data_formatter import generate_qa_pairs
generate_qa_pairs()

print("\n=== Step 8: Fine-tune LLaMA (QLoRA) ===")
import pandas as pd
import fine_tune_llama_rag

QA_PICKLE = r"C:\codes\t5-db\qa_augmented_metadata_training.pkl"
LLAMA_OUTPUT_DIR = os.path.join(".", "models", "llama_rag_qlora")
os.makedirs(LLAMA_OUTPUT_DIR, exist_ok=True)

fine_tune_llama_rag.QA_DATASET_PATH = QA_PICKLE
fine_tune_llama_rag.OUTPUT_PATH = LLAMA_OUTPUT_DIR
df_qas = pd.read_pickle(QA_PICKLE)
fine_tune_llama_rag.fine_tune_llama_on_papers(df_qas)

print("\n=== Step 9: Ingest into ChromaDB ===")
from migrate_to_chromadb import migrate_metadata, migrate_summaries
migrate_metadata()
migrate_summaries()

print("\nPipeline complete (starting after PDF downloads).")
