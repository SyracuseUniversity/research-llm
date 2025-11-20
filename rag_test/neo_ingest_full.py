#neo_ingest_full.py
"""
Neo4j CE ingest from researchers_fixed.db (no APOC).

Nodes:
  (Paper {paper_id, title, title_short, info, year, doi_link})
  (Author {name})
  (Researcher {name})
Rels:
  (Author)-[:AUTHORED]->(Paper)
  (Paper)-[:HAS_AUTHOR]->(Author)
  (Researcher)-[:WROTE]->(Paper)
  (Paper)-[:HAS_RESEARCHER]->(Researcher)
  (Author)-[:COAUTHORED_WITH]-(Author)  # undirected
"""

import json, math, re, sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

import config_full as config

DATE_RE    = re.compile(r"\b(19|20)\d{2}\b")
PUBDATE_RE = re.compile(r"\b(?:Publication\s*Date|Date)\s*:\s*([0-9]{4})", re.I)
DOI_RE     = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
URL_RE     = re.compile(r"https?://\S+")
SEP_RX     = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

def safe_str(x: Any) -> str:
    if x is None: return ""
    if isinstance(x, float):
        try:
            if math.isnan(x): return ""
        except Exception: return ""
    return str(x).strip()

def extract_year(info: str) -> Optional[int]:
    s = safe_str(info)
    if not s: return None
    m = PUBDATE_RE.search(s)
    if m:
        try: return int(m.group(1))
        except Exception: pass
    m2 = DATE_RE.search(s)
    return int(m2.group(0)) if m2 else None

def extract_doi_link(info: str, doi: str) -> Optional[str]:
    doi = safe_str(doi)
    if doi:
        return f"https://doi.org/{doi}" if doi.lower().startswith("10.") else doi
    s = safe_str(info)
    if not s: return None
    m = DOI_RE.search(s)
    if m: return f"https://doi.org/{m.group(0)}"
    m2 = URL_RE.search(s)
    return m2.group(0) if m2 else None

def _dedupe(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        k = it.lower()
        if k and k not in seen:
            seen.add(k); out.append(it)
    return out

def parse_authors(cell: Any) -> List[str]:
    s = safe_str(cell)
    if not s: return []
    try:
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            v = json.loads(s)
            if isinstance(v, list): return _dedupe([safe_str(x) for x in v if safe_str(x)])
            if isinstance(v, dict) and isinstance(v.get("authors"), list):
                return _dedupe([safe_str(x) for x in v["authors"] if safe_str(x)])
    except Exception:
        pass
    return _dedupe([p for p in SEP_RX.split(s) if p])

def read_rows() -> List[Dict[str, Any]]:
    db_path = config.SQLITE_DB_FULL
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(research_info);")
    cols = {r[1] for r in cur.fetchall()}
    has_doi, has_pub = "doi" in cols, "publication_date" in cols

    sql = "SELECT id, researcher_name, work_title, authors, info"
    sql += ", doi" if has_doi else ", '' AS doi"
    sql += ", publication_date" if has_pub else ", '' AS publication_date"
    sql += " FROM research_info"
    cur.execute(sql)
    raw = cur.fetchall()
    conn.close()

    rows: List[Dict[str, Any]] = []
    for pid, rname, title, authors, info, doi, pubdate in raw:
        title = safe_str(title)
        if not title: continue
        authors_list = parse_authors(authors)
        primary = safe_str(rname) or (authors_list[0] if authors_list else "")
        title_short = title[:2048]

        rows.append({
            "paper_id": str(pid),
            "title": title,
            "title_short": title_short,
            "info": safe_str(info),
            "primary_author": primary,
            "co_authors": [a for a in authors_list if a and a.lower() != primary.lower()],
            "year": extract_year(safe_str(pubdate) or safe_str(info)),
            "doi_link": extract_doi_link(info, doi),
        })
    return rows

def ensure_schema():
    with driver.session(database=config.NEO4J_DB) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")

UPSERT = """
UNWIND $rows AS row
MERGE (p:Paper {paper_id: row.paper_id})
  SET p.title       = row.title,
      p.title_short = row.title_short,
      p.info        = row.info,
      p.year        = row.year,
      p.doi_link    = row.doi_link

WITH p, row
WHERE row.primary_author IS NOT NULL AND row.primary_author <> ""
MERGE (r:Researcher {name: row.primary_author})
MERGE (a_primary:Author {name: row.primary_author})
MERGE (r)-[:WROTE]->(p)
MERGE (a_primary)-[:AUTHORED]->(p)
MERGE (p)-[:HAS_RESEARCHER]->(r)
MERGE (p)-[:HAS_AUTHOR]->(a_primary)

WITH p, row
UNWIND row.co_authors AS coName
WITH p, trim(coName) AS cname
WHERE cname <> ""
MERGE (a:Author {name: cname})
MERGE (a)-[:AUTHORED]->(p)
MERGE (p)-[:HAS_AUTHOR]->(a)
"""

COAUTHORS = """
MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
      (p)<-[:HAS_AUTHOR]-(a2:Author)
WHERE id(a1) < id(a2)
MERGE (a1)-[:COAUTHORED_WITH]-(a2)
"""

def ingest(rows: List[Dict[str, Any]], batch_size: int = 200, workers: int = 8):
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

    def worker(chunk):
        with driver.session(database=config.NEO4J_DB) as s:
            s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
        return len(chunk)

    done, total = 0, len(rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, b) for b in batches]
        for f in as_completed(futures):
            done += f.result()
            print(f"Committed {done}/{total} rows")

def build_coauthor_edges():
    with driver.session(database=config.NEO4J_DB) as s:
        s.run(COAUTHORS)
    print("Co-author edges built.")

if __name__ == "__main__":
    print("— Neo4j CE Ingest —")
    print(f"SQLite : {config.SQLITE_DB_FULL}")
    print(f"Neo4j  : {config.NEO4J_URI} / db={config.NEO4J_DB}")

    try:
        ensure_schema()
    except ClientError as e:
        print("Schema setup failed:", e); raise

    rows = read_rows()
    print(f"Rows prepared: {len(rows)}")
    if not rows:
        print("No rows found. Check your SQLite database and table names.")
    else:
        ingest(rows, batch_size=200, workers=8)
        build_coauthor_edges()
        print("✅ Done ingesting into Neo4j CE.")
