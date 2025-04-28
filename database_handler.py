# import sqlite3

# # Adjust this path if needed
# db_path = r"C:\codes\t5-db\researchers.db"

# # Connect to SQLite database
# conn = sqlite3.connect(db_path)
# cursor = conn.cursor()

# # Create necessary tables
# def setup_database():
#     """Create the 'works' table if it doesn't exist."""
#     cursor.execute("""
#         CREATE TABLE IF NOT EXISTS works (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             file_name TEXT UNIQUE,  -- Ensure file_name is unique
#             full_text TEXT NOT NULL,
#             summary TEXT,
#             summary_status TEXT DEFAULT 'unsummarized',
#             progress INTEGER DEFAULT 0
#         )
#     """)
#     conn.commit()

# def remove_duplicates():
#     """Remove duplicate entries in the 'works' table."""
#     cursor.execute("""
#         DELETE FROM works
#         WHERE id NOT IN (
#             SELECT MIN(id)
#             FROM works
#             GROUP BY file_name
#         )
#     """)
#     conn.commit()
#     print("Duplicates removed from works table.")

# def insert_work(file_name, full_text, summary=None, summary_status="unsummarized", progress=0):
#     """Insert a new work into the database, ensuring no duplicates."""
#     try:
#         cursor.execute("""
#             INSERT INTO works (file_name, full_text, summary, summary_status, progress)
#             VALUES (?, ?, ?, ?, ?)
#         """, (file_name, full_text, summary, summary_status, progress))
#         conn.commit()
#     except sqlite3.IntegrityError:
#         print(f"Skipping duplicate file: {file_name}")

# def fetch_unsummarized_works(limit=None):
#     """Fetch all unsummarized works from the database."""
#     query = """
#         SELECT id, full_text FROM works
#         WHERE summary_status = 'unsummarized' AND progress = 0
#     """
#     if limit:
#         query += " LIMIT ?"
#         cursor.execute(query, (limit,))
#     else:
#         cursor.execute(query)
#     return cursor.fetchall()

# def update_summary(work_id, summary):
#     """Update the summary for a specific work."""
#     cursor.execute("""
#         UPDATE works
#         SET summary = ?, summary_status = 'summarized', progress = 1
#         WHERE id = ?
#     """, (summary, work_id))
#     conn.commit()

# def count_entries_in_table():
#     """Count the total number of entries in the database."""
#     cursor.execute("SELECT COUNT(*) FROM works")
#     return cursor.fetchone()[0]

# def check_missing_files_in_db(pdf_files):
#     """Check which files in the folder are missing in the database."""
#     cursor.execute("SELECT file_name FROM works")
#     db_files = {row[0] for row in cursor.fetchall()}
#     return set(pdf_files) - db_files

# def close_connection():
#     """Close the database connection."""
#     conn.close()

import sqlite3

# Adjust this path if needed
db_path = r"C:\codes\t5-db\researchers.db"

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

def count_entries_in_table(table_name="works"):
    """Count the total number of entries in the specified table (default: 'works')."""
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    return cursor.fetchone()[0]

def check_missing_files_in_db(pdf_files):
    """Check which files in the folder are missing in the database."""
    cursor.execute("SELECT file_name FROM works")
    db_files = {row[0] for row in cursor.fetchall()}
    return set(pdf_files) - db_files

# NEW: Additional table for storing researcher information (without affiliation)
def setup_research_info_table():
    """
    Create the 'research_info' table if it doesn't exist.
    This table stores:
      - researcher_name: Name of the primary researcher (extracted from authors).
      - work_title: Title of the research work.
      - authors: List of authors.
      - info: Additional correct information.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            researcher_name TEXT,
            work_title TEXT,
            authors TEXT,
            info TEXT
        )
    """)
    conn.commit()
    print("research_info table setup completed.")

def insert_research_info(researcher_name, work_title, authors, info):
    """
    Insert a new record into the 'research_info' table.
    """
    try:
        cursor.execute("""
            INSERT INTO research_info (researcher_name, work_title, authors, info)
            VALUES (?, ?, ?, ?)
        """, (researcher_name, work_title, authors, info))
        conn.commit()
    except sqlite3.IntegrityError as e:
        print(f"Error inserting research info: {e}")

def fetch_research_info():
    """
    Fetch all records from the 'research_info' table.
    Returns a list of tuples.
    """
    cursor.execute("SELECT researcher_name, work_title, authors, info FROM research_info")
    return cursor.fetchall()

def close_connection():
    """Close the database connection."""
    conn.close()
