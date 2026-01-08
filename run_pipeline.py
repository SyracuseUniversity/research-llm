"""
run_pipeline.py

Execute pipeline on syr_research_all.db starting after PDFs are already downloaded.

Order
1 ingest_pdf_fulltext
2 csv_handler
3 db_repair_enrich
4 topic_tag_openalex
5 summarize_works

No naming changes, no download changes.
"""

import os
import argparse
import sqlite3

DB_PATH = r"C:\codes\t5-db\syr_research_all.db"


def _assert_db_exists() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")


def _quick_db_ping() -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        cur.fetchone()
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Optional limit for heavy steps")
    ap.add_argument("--skip-csv", action="store_true")
    ap.add_argument("--skip-repair", action="store_true")
    ap.add_argument("--skip-topics", action="store_true")
    ap.add_argument("--skip-summarize", action="store_true")
    ap.add_argument("--repair-t5", action="store_true", help="Enable Phase D T5 in db_repair_enrich.py")
    args = ap.parse_args()

    _assert_db_exists()
    _quick_db_ping()

    print("Using database at:", DB_PATH)

    print()
    print("Step 1 ingest full text into works")
    from ingest_pdf_fulltext import main as ingest_fulltext
    ingest_fulltext()

    if not args.skip_csv:
        print()
        print("Step 2 ingest CSV metadata into papers and research_info")
        from csv_handler import populate_research_info_from_csv
        populate_research_info_from_csv()

    if not args.skip_repair:
        print()
        print("Step 3 repair and enrich database")
        import subprocess
        import sys

        cmd = [sys.executable, "db_repair_enrich.py", "--db", DB_PATH]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.repair_t5:
            cmd += ["--t5"]
        subprocess.check_call(cmd)

    if not args.skip_topics:
        print()
        print("Step 4 tag topics with OpenAlex model")
        import subprocess
        import sys

        cmd = [sys.executable, "topic_tag_openalex.py", "--db", DB_PATH]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        subprocess.check_call(cmd)

    if not args.skip_summarize:
        print()
        print("Step 5 summarize works with T5")
        from summarize_works import main as summarize_main
        summarize_main(limit=args.limit)

    print()
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
