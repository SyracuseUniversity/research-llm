
"""
csv_handler.py  –  Combine various source CSVs, standardize columns, and populate research_info table.
"""

import os
import sqlite3
import pandas as pd
import logging

from database_handler import close_connection

DB_PATH = r"C:\codes\t5-db\researchers_all.db"

# ─────────────────────────────────────────────────────────────────────────────
# Match exactly the files you have in Downloads\Application:
CSV_PATHS = [
    r"C:\Users\arapte\Downloads\Application\author_works.csv",
    r"C:\Users\arapte\Downloads\Application\cleaned_author_works.csv",
    r"C:\Users\arapte\Downloads\Application\filtered_author_works.csv",
    r"C:\Users\arapte\Downloads\Application\merged_cleaned.csv",
    r"C:\Users\arapte\Downloads\Application\syracuse_university_orcid_data.csv"
]
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def combine_csvs() -> pd.DataFrame:
    """
    1. Load only existing CSVs from CSV_PATHS.
    2. Lowercase and dedupe columns, then standardize key columns.
    3. Concatenate, filter rows with at least one PDF URL, and return.
    """
    existing_paths = [path for path in CSV_PATHS if os.path.isfile(path)]
    if not existing_paths:
        return pd.DataFrame()

    dfs = []
    for path in existing_paths:
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception as e:
            logging.warning("Error reading %s: %s", path, e)
            continue

        # 1) Lowercase column names
        df.columns = [col.strip().lower() for col in df.columns]

        # 2) Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # 3) Standardize key columns
        cols_lc = set(df.columns)

        if "author_name" in cols_lc and "title" in cols_lc:
            df = df.rename(columns={"author_name": "researcher_name", "title": "work_title"})
            if "authors" not in df.columns:
                df["authors"] = df["researcher_name"]

        elif "display_name" in cols_lc and "authors" in cols_lc and "title" in cols_lc:
            df = df.rename(columns={"title": "work_title"})
            df["researcher_name"] = df["authors"].apply(
                lambda x: x.split(",")[0].strip() if isinstance(x, str) else ""
            )

        elif "full_name" in cols_lc and "work_title" in cols_lc:
            df = df.rename(columns={"full_name": "researcher_name"})
            if "authors" not in df.columns:
                df["authors"] = df["researcher_name"]

        else:
            if "author" in cols_lc and "researcher_name" not in df.columns:
                df = df.rename(columns={"author": "researcher_name"})
            if "title" in cols_lc and "work_title" not in df.columns:
                df = df.rename(columns={"title": "work_title"})
            if "authors" not in df.columns:
                df["authors"] = df.get("researcher_name", "")

        # 4) Ensure optional columns
        for opt in ("doi_url", "publication_date", "landing_url"):
            if opt not in df.columns:
                df[opt] = ""

        # 5) Ensure PDF URL columns
        for pdf_col in ("pdf_url", "final_work_url"):
            if pdf_col not in df.columns:
                df[pdf_col] = ""

        # 6) Lowercase again (in case renaming added uppercase)
        df.columns = [c.lower() for c in df.columns]

        # 7) Drop duplicates again
        df = df.loc[:, ~df.columns.duplicated()]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Filter to rows with at least one PDF link
    mask_pdf = combined["pdf_url"].str.strip().ne("") | combined["final_work_url"].str.strip().ne("")
    combined = combined.loc[mask_pdf].copy()

    # Drop rows missing essential fields
    combined = combined.loc[
        combined["researcher_name"].str.strip().ne("") &
        combined["work_title"].str.strip().ne("")
    ]

    return combined


def get_research_info_columns() -> list[str]:
    """
    Return the column names in the research_info table.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(research_info);")
    cols = [row[1] for row in cur.fetchall()]
    conn.close()
    return cols


def safe_str(val) -> str:
    """
    Convert a value to a trimmed string, or return "" if None/NaN.
    """
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    return str(val).strip()


def populate_research_info_from_csv():
    """
    1. Combine CSVs into a DataFrame.
    2. If empty, skip.
    3. Dynamically inspect which columns exist in research_info.
    4. Build an INSERT statement accordingly.
    5. Insert each row’s values.
    """
    combined_df = combine_csvs()
    if combined_df.empty:
        logging.info("No CSV-based metadata to insert; skipping.")
        return

    existing_cols = get_research_info_columns()
    desired_cols = ["researcher_name", "work_title", "authors", "doi", "publication_date", "info"]
    to_insert = [col for col in desired_cols if col in existing_cols]

    # Must have these three at minimum:
    must_have = {"researcher_name", "work_title", "authors"}
    if not must_have.issubset(set(to_insert)):
        missing = must_have - set(to_insert)
        logging.error(
            "Cannot insert CSV data; missing required column(s) in research_info: %s",
            missing
        )
        return

    cols_clause = ", ".join(to_insert)
    placeholders = ", ".join("?" for _ in to_insert)
    insert_sql = f"INSERT INTO research_info ({cols_clause}) VALUES ({placeholders});"

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    total = len(combined_df)
    logging.info("Inserting %d rows into research_info from CSVs…", total)

    try:
        for idx, row in enumerate(combined_df.itertuples(index=False), start=1):
            values = []
            for col in to_insert:
                if col == "researcher_name":
                    values.append(safe_str(getattr(row, "researcher_name", "")))
                elif col == "work_title":
                    values.append(safe_str(getattr(row, "work_title", "")))
                elif col == "authors":
                    values.append(safe_str(getattr(row, "authors", "")))
                elif col == "doi":
                    # DataFrame has “doi_url”
                    values.append(safe_str(getattr(row, "doi_url", "")))
                elif col == "publication_date":
                    values.append(safe_str(getattr(row, "publication_date", "")))
                elif col == "info":
                    # Build combined “info” from doi_url/publication_date/landing_url
                    parts = []
                    d = safe_str(getattr(row, "doi_url", ""))
                    p = safe_str(getattr(row, "publication_date", ""))
                    l = safe_str(getattr(row, "landing_url", ""))
                    if d:
                        parts.append(f"DOI: {d}")
                    if p:
                        parts.append(f"Publication Date: {p}")
                    if l:
                        parts.append(f"Landing URL: {l}")
                    values.append(" | ".join(parts))
                else:
                    values.append("")

            cur.execute(insert_sql, tuple(values))

            if idx % 500 == 0 or idx == total:
                logging.info("  → %d/%d CSV rows inserted", idx, total)

        conn.commit()
        logging.info("✅ research_info populated from CSV.")
    except Exception as e:
        conn.rollback()
        logging.error("Error inserting CSV rows: %s", e)
        raise
    finally:
        conn.close()
        close_connection()


if __name__ == "__main__":
    populate_research_info_from_csv()
