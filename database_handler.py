"""
database_handler.py  –  SQLite helper for works: fetch unsummarized & update.
"""

import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def fetch_unsummarized_works(
    limit: int | None = None,
    db_path: str = r"C:\codes\t5-db\researchers_all.db"
) -> list[tuple[int, str]]:
    """
    Return a list of (id, full_text) from works WHERE summary_status='unsummarized' AND progress=0.
    """
    query = "SELECT id, full_text FROM works WHERE summary_status = 'unsummarized' AND progress = 0"
    params = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()


def update_summary(
    work_id: int,
    summary: str,
    db_path: str = r"C:\codes\t5-db\researchers_all.db"
) -> None:
    """
    Update the works row: set summary, summary_status='summarized', progress=1 WHERE id = work_id.
    """
    sql = """
        UPDATE works
        SET summary = ?, summary_status = 'summarized', progress = 1
        WHERE id = ?
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(sql, (summary, work_id))
        if cur.rowcount == 0:
            logging.warning("No row updated for work_id=%s; maybe it does not exist.", work_id)
        conn.commit()


def close_connection():
    """No-op: each function uses its own connection."""
    pass
