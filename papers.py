import sqlite3
import pandas as pd

# Database and Table Initialization
def initialize_db():
    conn = sqlite3.connect("papers.db")  # Create SQLite database
    cursor = conn.cursor()

    # Create table if not exists (remove published_date column)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        title TEXT NOT NULL,
        url TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

# Load Data from CSV
def load_data_from_csv(csv_file):
    conn = sqlite3.connect("papers.db")
    cursor = conn.cursor()

    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    # Insert data into the SQLite database
    for _, row in df.iterrows():
        cursor.execute("""
        INSERT INTO papers (name, title, url)
        VALUES (?, ?, ?)
        """, (row["Name"], row["Title"], row["URL"]))

    conn.commit()
    conn.close()

# Retrieve Title by Name
def get_title_by_name(name):
    conn = sqlite3.connect("papers.db")
    cursor = conn.cursor()

    cursor.execute("SELECT title FROM papers WHERE name = ?", (name,))
    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]
    else:
        return f"No title found for name: {name}"

# Update a Record
def update_record(name, new_title):
    conn = sqlite3.connect("papers.db")
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE papers
    SET title = ?
    WHERE name = ?
    """, (new_title, name))

    conn.commit()
    conn.close()

# Delete a Record
def delete_record(name):
    conn = sqlite3.connect("papers.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM papers WHERE name = ?", (name,))
    conn.commit()
    conn.close()

# Main Execution
if __name__ == "__main__":
    # Initialize the database and table
    initialize_db()

    # Load data from a CSV file (replace 'papers.csv' with your actual CSV file path)
    csv_file = "newpapers.csv"
    load_data_from_csv(csv_file)

    # Example: Retrieve a paper title by name
    print(get_title_by_name("Alice"))  # Replace "Alice" with a name from your data

    # Example: Update a record
    update_record("Alice", "Updated Title")

    # Example: Delete a record
    delete_record("Bob")  # Replace "Bob" with a name from your data
