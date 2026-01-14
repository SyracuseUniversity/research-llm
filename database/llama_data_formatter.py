"""
llama_data_formatter.py  –  Build QA pairs from research_info, output a pickled DataFrame.
"""

import os
import sqlite3
import pandas as pd

DB_PATH = r"C:\codes\t5-db\researchers_all.db"
OUT_PATH = r"C:\codes\t5-db\qa_augmented_metadata_training.pkl"
INFO_MAX_CHARS = 512


def generate_qa_pairs():
    """
    1. Read researcher_name, work_title, authors, info from research_info.
    2. For each row, build four QA pairs.
    3. Save a DataFrame of {'input_text','target_text'} as a pickle.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            "SELECT researcher_name, work_title, authors, info FROM research_info",
            conn
        )

    if df.empty:
        print("⚠️ No rows found in research_info. Exiting.")
        return

    examples = []
    total = len(df)
    print(f"Found {total} rows. Generating QA pairs…")

    for idx, row in enumerate(df.itertuples(index=False), start=1):
        title = getattr(row, "work_title", "") or ""
        author = getattr(row, "researcher_name", "") or ""
        authors = getattr(row, "authors", "") or ""
        info = getattr(row, "info", "") or ""

        if not title.strip() or not author.strip():
            continue

        info_snip = info[:INFO_MAX_CHARS].rstrip() + ("…" if len(info) > INFO_MAX_CHARS else "")

        context = (
            "You are a Syracuse University research assistant.\n\n"
            "Context:\n"
            f"Title: {title}\n"
            f"Authors: {authors}\n"
            f"Researcher: {author}\n"
            f"Info: {info_snip}"
        )

        answer1 = f"{title} is a research work authored by {authors} at Syracuse University."
        if info_snip:
            answer1 += f" {info_snip}"

        qas = [
            (
                f"What research is {author} doing at Syracuse?",
                answer1
            ),
            (
                f"Who authored '{title}'?",
                f"The paper '{title}' was authored by {authors} at Syracuse University."
            ),
            (
                f"When was '{title}' published?",
                # Look for "Publication Date" in info
                next((part.strip() for part in info.split("|") if "Publication Date" in part), 
                     "The publication date is not available.")
            ),
            (
                f"Summarize the paper '{title}'.",
                answer1
            )
        ]

        for question, answer in qas:
            prompt = (
                f"{context}\n\n"
                f"Question: {question}\n"
                "Answer: "
            )
            examples.append({
                "input_text": prompt,
                "target_text": answer
            })

        if idx % 1000 == 0 or idx == total:
            print(f"  → {idx}/{total} rows processed, {len(examples)} examples generated")

    if not examples:
        print("⚠️ No QA examples generated.")
        return

    df_out = pd.DataFrame(examples)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df_out.to_pickle(OUT_PATH)
    print(f"✅ Created {len(df_out)} QA examples → {OUT_PATH}")


if __name__ == "__main__":
    generate_qa_pairs()
