"""
migrate_to_chromadb.py  –  NULL-safe ingestion of metadata and summaries into ChromaDB.
"""

import os
import sqlite3
import pandas as pd
from chromadb import Client
from chromadb.config import Settings

DB_PATH = r"C:\codes\t5-db\researchers_all.db"
CHROMA_DIR = os.path.join(".", "chroma_storage")
os.makedirs(CHROMA_DIR, exist_ok=True)

# Instantiate Chroma client using keyword‐only Settings
client = Client(
    settings=Settings(
        persist_directory=CHROMA_DIR
    )
)


def _safe(x):
    """
    Convert None or NaN to empty string, otherwise cast to str.
    """
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def migrate_metadata():
    """
    Read from research_info and add each row as a document in 'paper_metadata'.
    ID format: "meta_<id>".
    """
    print("Ingesting metadata")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, researcher_name, work_title, authors, info FROM research_info",
            conn
        )

    if df.empty:
        print("No metadata rows found.")
        return

    col = client.get_or_create_collection("paper_metadata")
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        title = _safe(getattr(row, "work_title", ""))
        authors = _safe(getattr(row, "authors", ""))
        researcher = _safe(getattr(row, "researcher_name", ""))
        info = _safe(getattr(row, "info", ""))

        doc = (
            "This is metadata for a research paper.\n\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Researcher: {researcher}\n"
            f"Info: {info}"
        )
        col.add(
            documents=[doc],
            metadatas=[{"type": "metadata", "title": title}],
            ids=[f"meta_{getattr(row, 'id')}"],
        )

        if idx % 500 == 0 or idx == total:
            print(f"  → {idx}/{total} metadata rows ingested")

    print(f"Ingested {total} metadata rows")


def migrate_summaries():
    """
    Read all summarized works and add each summary as a document in 'paper_summaries'.
    ID format: "sum_<id>".
    """
    print("Ingesting summaries")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT id, file_name, summary FROM works "
            "WHERE summary_status='summarized' AND progress=1",
            conn
        )

    if df.empty:
        print("No summary rows found.")
        return

    col = client.get_or_create_collection("paper_summaries")
    total = len(df)
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        summary = _safe(getattr(row, "summary", ""))
        fname = _safe(getattr(row, "file_name", ""))

        col.add(
            documents=[summary],
            metadatas=[{"type": "summary", "file": fname}],
            ids=[f"sum_{getattr(row, 'id')}"],
        )

        if idx % 500 == 0 or idx == total:
            print(f"  → {idx}/{total} summaries ingested")

    print(f"Ingested {total} summaries")


if __name__ == "__main__":
    migrate_metadata()
    migrate_summaries()
    print("Chroma ingestion complete")
