"""
summarize_works.py  –  Run T5 summarization on each unsummarized 'works' row,
store summary into works.summary, and mark as summarized.
"""

import torch
import sqlite3
from model import summarize_text
from database_handler import fetch_unsummarized_works, update_summary

DB_PATH = r"C:\codes\t5-db\researchers_all.db"


def main(limit: int | None = None):
    unsummarized = fetch_unsummarized_works(limit=limit, db_path=DB_PATH)
    total = len(unsummarized)
    if total == 0:
        print("No unsummarized works found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Summarizing {total} works using T5 on {device}…")

    for idx, (work_id, full_text) in enumerate(unsummarized, start=1):
        try:
            summary = summarize_text(full_text, idx=idx, total=total)
            update_summary(work_id, summary, db_path=DB_PATH)
        except Exception as e:
            print(f"Failed to summarize work_id={work_id}: {e}")

        if idx % 50 == 0 or idx == total:
            print(f"  → {idx}/{total} works summarized")

    print("All works summarized.")


if __name__ == "__main__":
    main()
