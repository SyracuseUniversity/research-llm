"""
neo_ingest_abstracts.py
Builds a simplified Neo4j graph from abstracts_only.db:
  (Paper {doi, title, year})
  (Source {name})
  (Year {value})
Relationships:
  (Source)-[:PUBLISHED]->(Paper)
  (Year)-[:CONTAINS]->(Paper)
"""

import sqlite3
from neo4j import GraphDatabase
import config_full as config

DB_PATH = r"C:\codes\t5-db\abstracts_only.db"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

def read_rows():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT doi, title, abstract, source, year FROM abstracts_only WHERE abstract IS NOT NULL AND abstract != ''")
    rows = cur.fetchall()
    conn.close()
    return rows

def ensure_schema():
    with driver.session(database=config.NEO4J_DB) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (y:Year) REQUIRE y.value IS UNIQUE")

UPSERT = """
UNWIND $rows AS row
MERGE (p:Paper {doi: row.doi})
  SET p.title = row.title,
      p.year  = row.year

WITH p, row
WHERE row.source IS NOT NULL AND row.source <> ""
MERGE (s:Source {name: row.source})
MERGE (s)-[:PUBLISHED]->(p)

WITH p, row
WHERE row.year IS NOT NULL AND row.year <> ""
MERGE (y:Year {value: row.year})
MERGE (y)-[:CONTAINS]->(p)
"""

def ingest(rows, batch_size=200):
    batches = [rows[i:i+batch_size] for i in range(0, len(rows), batch_size)]
    with driver.session(database=config.NEO4J_DB) as s:
        for i, batch in enumerate(batches, 1):
            s.run(UPSERT, rows=[
                {"doi": r[0], "title": r[1], "abstract": r[2], "source": r[3], "year": r[4]} for r in batch
            ])
            print(f"✅ Batch {i}/{len(batches)} committed.")

def main():
    print("— Neo4j Abstracts Ingest —")
    rows = read_rows()
    print(f"Rows prepared: {len(rows)}")
    if not rows:
        print("No rows found in abstracts_only.db")
        return
    ensure_schema()
    ingest(rows)
    print("✅ Done ingesting abstracts into Neo4j.")

if __name__ == "__main__":
    main()
