"""
ingest_pdf_fulltext.py

Extract full text from each downloaded PDF and insert into `works` in syr_research_all.db.
"""

import os
import glob
import sqlite3

from pdf_pre import extract_raw_text_from_pdf, clean_text

DB_PATH = r"C:\codes\t5-db\syr_research_all.db"
PDF_DOWNLOAD_DIR = r"C:\codes\t5-db\download_pdfs"


def _ensure_pragmas(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")


def _work_exists(cur: sqlite3.Cursor, file_name: str) -> bool:
    cur.execute("SELECT 1 FROM works WHERE file_name = ? LIMIT 1;", (file_name,))
    return cur.fetchone() is not None


def _create_paper(cur: sqlite3.Cursor) -> int:
    """
    Create a bare minimum papers row and return paper_id.
    Assumes papers.paper_id is INTEGER PRIMARY KEY (autoincrement behavior in SQLite).
    """
    cur.execute(
        """
        INSERT INTO papers (title, authors, publication_date, doi, arxiv_id)
        VALUES ('', '', '', '', '')
        """
    )
    return int(cur.lastrowid)


def _insert_work(cur: sqlite3.Cursor, paper_id: int, file_name: str, full_text: str) -> None:
    cur.execute(
        """
        INSERT INTO works (paper_id, file_name, full_text, summary, summary_status, progress)
        VALUES (?, ?, ?, '', 'unsummarized', 0)
        """,
        (paper_id, file_name, full_text),
    )


def _has_column(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table});")
    return any(r[1] == col for r in cur.fetchall())


def _try_insert_research_info_stub(cur: sqlite3.Cursor, paper_id: int, file_name: str) -> None:
    """
    Optional convenience.
    db_repair_enrich.py already ensures missing research_info rows for works,
    but inserting a stub early helps keep everything aligned.
    """
    if not _has_column(cur, "research_info", "paper_id"):
        return

    cols = set()
    cur.execute("PRAGMA table_info(research_info);")
    for r in cur.fetchall():
        cols.add(r[1])

    if "paper_id" not in cols:
        return

    work_title = ""
    try:
        work_title = os.path.splitext(file_name)[0]
    except Exception:
        work_title = ""

    if "topics_status" in cols:
        cur.execute(
            """
            INSERT OR IGNORE INTO research_info
                (paper_id, work_title, authors, doi, publication_date, researcher_name, info, topics_status)
            VALUES (?, ?, '', '', '', '', '', 'untagged')
            """,
            (paper_id, work_title),
        )
    else:
        cur.execute(
            """
            INSERT OR IGNORE INTO research_info
                (paper_id, work_title, authors, doi, publication_date, researcher_name, info)
            VALUES (?, ?, '', '', '', '', '')
            """,
            (paper_id, work_title),
        )


def main() -> None:
    if not os.path.isdir(PDF_DOWNLOAD_DIR):
        print(f"No download directory found: {PDF_DOWNLOAD_DIR}")
        return

    if not os.path.isfile(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    pdf_files = sorted(glob.glob(os.path.join(PDF_DOWNLOAD_DIR, "*.pdf")))
    total = len(pdf_files)
    print(f"Found {total} PDFs. Ingesting full text into works in syr_research_all.db")

    conn = sqlite3.connect(DB_PATH)
    try:
        _ensure_pragmas(conn)
        cur = conn.cursor()

        done = 0
        for idx, pdf_path in enumerate(pdf_files, start=1):
            file_name = os.path.basename(pdf_path)

            try:
                if _work_exists(cur, file_name):
                    continue
            except Exception as e:
                print(f"ERROR checking existence for {file_name}: {e}")
                continue

            try:
                raw_text = extract_raw_text_from_pdf(pdf_path)
                if not raw_text or not raw_text.strip():
                    print(f"Skipping empty PDF: {file_name}")
                    continue
                cleaned = clean_text(raw_text)
                if not cleaned.strip():
                    print(f"Skipping cleaned empty PDF: {file_name}")
                    continue
            except Exception as e:
                print(f"Failed to extract from {pdf_path}: {e}")
                continue

            try:
                paper_id = _create_paper(cur)
                _insert_work(cur, paper_id, file_name, cleaned)
                _try_insert_research_info_stub(cur, paper_id, file_name)
                done += 1
            except sqlite3.IntegrityError:
                print(f"Skipped duplicate insertion for: {file_name}")
                continue
            except Exception as e:
                print(f"ERROR inserting {file_name}: {e}")
                continue

            if idx % 50 == 0 or idx == total:
                print(f"  -> processed {idx}/{total} PDFs, inserted {done} new works")

        conn.commit()
        print(f"Done. Inserted {done} new works rows.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
