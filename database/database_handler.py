"""
database_handler.py

SQLite helpers for works summarization in syr_research_all.db.
"""

import sqlite3
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

DB_PATH_DEFAULT = r"C:\codes\t5-db\syr_research_all.db"


def fetch_unsummarized_works(
    limit: Optional[int] = None,
    db_path: str = DB_PATH_DEFAULT,
) -> List[Tuple[int, int, str]]:
    """
    Returns list of (work_id, paper_id, full_text) where summary_status='unsummarized' and progress=0.
    """
    query = """
    SELECT id, paper_id, full_text
    FROM works
    WHERE summary_status = 'unsummarized'
      AND progress = 0
      AND full_text IS NOT NULL
      AND TRIM(full_text) <> ''
    ORDER BY id
    """
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(query, params)
        return [(int(r[0]), int(r[1]), r[2] or "") for r in cur.fetchall()]


def update_summary(
    work_id: int,
    summary: str,
    db_path: str = DB_PATH_DEFAULT,
) -> None:
    """
    Update works row by works.id.
    """
    sql = """
    UPDATE works
    SET summary = ?, summary_status = 'summarized', progress = 1
    WHERE id = ?
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")
        cur.execute(sql, (summary or "", int(work_id)))
        if cur.rowcount == 0:
            logging.warning("No row updated for work_id=%s", work_id)
        conn.commit()


def close_connection() -> None:
    """
    No-op retained for compatibility with your existing imports.
    """
    return
