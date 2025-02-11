import sqlite3
import os

# Build a path to 'researchers.db' in the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(script_dir, "researchers.db")

# Connect to SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def setup_database():
    """Create the 'works' table if it doesn't exist."""
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS works (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT UNIQUE,  -- Ensure file_name is unique
            full_text TEXT NOT NULL,
            summary TEXT,
            summary_status TEXT DEFAULT 'unsummarized',
            progress INTEGER DEFAULT 0
        )
    """)
    conn.commit()

def remove_duplicates():
    """Remove duplicate entries in the 'works' table."""
    cursor.execute("""
        DELETE FROM works
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM works
            GROUP BY file_name
        )
    """)
    conn.commit()
    print("Duplicates removed from works table.")

def insert_work(file_name, full_text, summary=None, summary_status="unsummarized", progress=0):
    """Insert a new work into the database, ensuring no duplicates."""
    try:
        cursor.execute("""
            INSERT INTO works (file_name, full_text, summary, summary_status, progress)
            VALUES (?, ?, ?, ?, ?)
        """, (file_name, full_text, summary, summary_status, progress))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"Skipping duplicate file: {file_name}")

def fetch_unsummarized_works(limit=None):
    """Fetch all unsummarized works from the database."""
    query = """
        SELECT id, full_text FROM works
        WHERE summary_status = 'unsummarized' AND progress = 0
    """
    if limit:
        query += " LIMIT ?"
        cursor.execute(query, (limit,))
    else:
        cursor.execute(query)
    return cursor.fetchall()

def update_summary(work_id, summary):
    """Update the summary for a specific work."""
    cursor.execute("""
        UPDATE works
        SET summary = ?, summary_status = 'summarized', progress = 1
        WHERE id = ?
    """, (summary, work_id))
    conn.commit()

def count_entries_in_table():
    """Count the total number of entries in the database."""
    cursor.execute("SELECT COUNT(*) FROM works")
    return cursor.fetchone()[0]

def check_missing_files_in_db(pdf_files):
    """Check which files in the folder are missing in the database."""
    cursor.execute("SELECT file_name FROM works")
    db_files = {row[0] for row in cursor.fetchall()}
    return set(pdf_files) - db_files

def close_connection():
    """Close the database connection."""
    conn.close()
