# neo_ingest.py
"""
Neo4j CE ingest for syr_research_all.db

Fixes your current failure mode:
Neo.ClientError.Security.Unauthorized

This script will
1) read Neo4j credentials from environment variables first
2) fall back to config_full.py values
3) do a quick auth check and print a clear message before running schema or ingest

Set these in your terminal before running if needed:

Windows PowerShell
  $env:NEO4J_URI="bolt://localhost:7687"
  $env:NEO4J_USER="neo4j"
  $env:NEO4J_PASS="your_password"
  $env:NEO4J_DB="syr-rag-one"

Windows CMD
  set NEO4J_URI=bolt://localhost:7687
  set NEO4J_USER=neo4j
  set NEO4J_PASS=your_password
  set NEO4J_DB=syr-rag-one
"""

import json
import math
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError, AuthError

import config_full as config


DATE_RE = re.compile(r"\b(19|20)\d{2}\b")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
URL_RE = re.compile(r"https?://\S+")
SEP_RX = re.compile(r"\s*(?:;|,|\band\b|&|\|)\s*", re.I)

BATCH_SIZE = int(getattr(config, "NEO_BATCH_SIZE", 500))
WORKERS = int(getattr(config, "NEO_WORKERS", 6))


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default


def _cfg(name: str, default: str) -> str:
    return str(getattr(config, name, default) or default)


SQLITE_DB = _cfg("SQLITE_DB_FULL", _cfg("SQLITE_DB", ""))

NEO4J_URI = _env("NEO4J_URI", _cfg("NEO4J_URI", "bolt://localhost:7687"))
NEO4J_USER = _env("NEO4J_USER", _cfg("NEO4J_USER", "neo4j"))
NEO4J_PASS = _env("NEO4J_PASS", _cfg("NEO4J_PASS", "neo4j"))
NEO4J_DB = _env("NEO4J_DB", _cfg("NEO4J_DB", "neo4j"))


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        try:
            if math.isnan(x):
                return ""
        except Exception:
            return ""
    return str(x).strip()


def detect_works_fulltext_column(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(works);")
    cols = [r[1] for r in cur.fetchall()]
    if "full_text" in cols:
        return "full_text"
    if "fultext" in cols:
        return "fultext"
    if "fulltext" in cols:
        return "fulltext"
    raise RuntimeError("Could not find works fulltext column. Expected one of: full_text, fultext, fulltext")


def dedupe(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        k = it.lower().strip()
        if not k:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(it.strip())
    return out


def parse_authors(cell: Any) -> List[str]:
    s = safe_str(cell)
    if not s:
        return []

    try:
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            v = json.loads(s)
            if isinstance(v, list):
                return dedupe([safe_str(x) for x in v if safe_str(x)])
            if isinstance(v, dict):
                a = v.get("authors")
                if isinstance(a, list):
                    return dedupe([safe_str(x) for x in a if safe_str(x)])
    except Exception:
        pass

    parts = [p.strip() for p in SEP_RX.split(s) if p and p.strip()]
    return dedupe(parts)


def extract_year(*candidates: Any) -> Optional[int]:
    for c in candidates:
        s = safe_str(c)
        if not s:
            continue
        m = DATE_RE.search(s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                continue
    return None


def doi_link_from(doi: Any, info: Any = None) -> str:
    d = safe_str(doi)
    if d:
        if d.lower().startswith("10."):
            return "https://doi.org/" + d
        if d.lower().startswith("http://") or d.lower().startswith("https://"):
            return d
        m = DOI_RE.search(d)
        if m:
            return "https://doi.org/" + m.group(0)
        return d

    s = safe_str(info)
    if s:
        m = DOI_RE.search(s)
        if m:
            return "https://doi.org/" + m.group(0)
        m2 = URL_RE.search(s)
        if m2:
            return m2.group(0)

    return ""


def read_rows(db_path: str) -> Tuple[List[Dict[str, Any]], str]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    works_fulltext_col = detect_works_fulltext_column(conn)

    cur.execute("PRAGMA table_info(papers);")
    papers_cols = {r[1] for r in cur.fetchall()}
    need = {"paper_id", "doi", "arxiv_id", "title", "authors", "publication_date"}
    missing = [c for c in need if c not in papers_cols]
    if missing:
        conn.close()
        raise RuntimeError("papers table missing required columns: " + ", ".join(missing))

    cur.execute("PRAGMA table_info(research_info);")
    ri_cols = {r[1] for r in cur.fetchall()}
    required_ri = {"paper_id", "researcher_name", "primary_topic", "info"}
    missing_ri = [c for c in required_ri if c not in ri_cols]
    if missing_ri:
        conn.close()
        raise RuntimeError("research_info table missing required columns: " + ", ".join(missing_ri))

    sql = f"""
        SELECT
            p.paper_id,
            p.doi,
            p.arxiv_id,
            p.title,
            p.authors,
            p.publication_date,
            r.researcher_name,
            r.primary_topic,
            r.info,
            w.summary,
            w.{works_fulltext_col}
        FROM papers p
        LEFT JOIN research_info r
          ON p.paper_id = r.paper_id
        LEFT JOIN works w
          ON p.paper_id = w.paper_id
    """
    cur.execute(sql)
    raw = cur.fetchall()
    conn.close()

    by_pid: Dict[str, List[Tuple[Any, ...]]] = {}
    for row in raw:
        pid = safe_str(row[0])
        if not pid:
            continue
        by_pid.setdefault(pid, []).append(row)

    rows: List[Dict[str, Any]] = []

    for pid, items in by_pid.items():
        best = None
        best_score = -1

        for it in items:
            (
                paper_id,
                doi,
                arxiv_id,
                title,
                authors,
                publication_date,
                researcher_name,
                primary_topic,
                info,
                summary,
                fulltext,
            ) = it

            t = safe_str(title)
            if not t:
                continue

            ft_len = len(safe_str(fulltext))
            s_len = len(safe_str(summary))
            score = ft_len + s_len

            if score > best_score:
                best_score = score
                best = it

        if best is None:
            continue

        (
            paper_id,
            doi,
            arxiv_id,
            title,
            authors,
            publication_date,
            researcher_name,
            primary_topic,
            info,
            summary,
            fulltext,
        ) = best

        authors_list = parse_authors(authors)
        researcher = safe_str(researcher_name)
        if not researcher and authors_list:
            researcher = authors_list[0]

        year = extract_year(publication_date, info)
        doi_link = doi_link_from(doi, info)

        rows.append(
            {
                "paper_id": safe_str(paper_id),
                "title": safe_str(title),
                "title_short": safe_str(title)[:2048],
                "doi": safe_str(doi),
                "doi_link": doi_link,
                "arxiv_id": safe_str(arxiv_id),
                "publication_date": safe_str(publication_date),
                "year": year,
                "primary_topic": safe_str(primary_topic),
                "researcher": researcher,
                "authors": dedupe([a for a in authors_list if a]),
            }
        )

    return rows, works_fulltext_col


def connect_driver_or_fail():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
    try:
        driver.verify_connectivity()
        with driver.session(database=NEO4J_DB) as s:
            s.run("RETURN 1").consume()
    except AuthError as e:
        driver.close()
        raise AuthError(
            "Neo4j auth failed. Fix username or password.\n"
            "Either update config_full.py NEO4J_USER and NEO4J_PASS\n"
            "or set environment variables NEO4J_USER and NEO4J_PASS before running.\n"
            f"Details: {e}"
        )
    except ClientError as e:
        driver.close()
        raise ClientError(
            "Neo4j connection established but database access failed.\n"
            "If you are using Neo4j Desktop, ensure the DB name exists and is started.\n"
            "If your Neo4j version does not support multi database, set NEO4J_DB=neo4j.\n"
            f"Details: {e}"
        )
    return driver


def ensure_schema(driver) -> None:
    with driver.session(database=NEO4J_DB) as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Researcher) REQUIRE r.name IS UNIQUE")
        s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.year)")
        s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.doi)")
        s.run("CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.arxiv_id)")


UPSERT = """
UNWIND $rows AS row

MERGE (p:Paper {paper_id: row.paper_id})
SET  p.title            = row.title,
     p.title_short      = row.title_short,
     p.doi              = row.doi,
     p.doi_link         = row.doi_link,
     p.arxiv_id         = row.arxiv_id,
     p.publication_date = row.publication_date,
     p.year             = row.year,
     p.primary_topic    = row.primary_topic

WITH p, row
WHERE row.researcher IS NOT NULL AND row.researcher <> ""
MERGE (r:Researcher {name: row.researcher})
MERGE (r)-[:WROTE]->(p)
MERGE (p)-[:HAS_RESEARCHER]->(r)

WITH p, row
UNWIND row.authors AS aname
WITH p, trim(aname) AS name
WHERE name <> ""
MERGE (a:Author {name: name})
MERGE (a)-[:AUTHORED]->(p)
MERGE (p)-[:HAS_AUTHOR]->(a)
"""


COAUTHORS = """
MATCH (p:Paper)<-[:HAS_AUTHOR]-(a1:Author),
      (p)<-[:HAS_AUTHOR]-(a2:Author)
WHERE id(a1) < id(a2)
MERGE (a1)-[:COAUTHORED_WITH]-(a2)
"""


def ingest_rows(driver, rows: List[Dict[str, Any]], batch_size: int, workers: int) -> None:
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]

    def worker(chunk: List[Dict[str, Any]]) -> int:
        with driver.session(database=NEO4J_DB) as s:
            s.execute_write(lambda tx: tx.run(UPSERT, rows=chunk))
        return len(chunk)

    done = 0
    total = len(rows)
    if total == 0:
        print("No rows to ingest")
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, b) for b in batches]
        for f in as_completed(futures):
            done += f.result()
            print(f"Committed {done}/{total}")


def build_coauthor_edges(driver) -> None:
    with driver.session(database=NEO4J_DB) as s:
        s.execute_write(lambda tx: tx.run(COAUTHORS))
    print("Coauthor edges built")


def main() -> None:
    if not SQLITE_DB:
        raise RuntimeError("SQLITE_DB_FULL not set in config_full.py")

    print("Neo4j ingest starting")
    print("SQLite:", SQLITE_DB)
    print("Neo4j :", NEO4J_URI, "db=", NEO4J_DB)
    print("User  :", NEO4J_USER)
    print("Batch:", BATCH_SIZE, "Workers:", WORKERS)

    rows, works_fulltext_col = read_rows(SQLITE_DB)
    print("Prepared papers:", len(rows))
    print("Works fulltext column:", works_fulltext_col)

    driver = connect_driver_or_fail()

    try:
        ensure_schema(driver)
    except Exception:
        driver.close()
        raise

    ingest_rows(driver, rows, batch_size=BATCH_SIZE, workers=WORKERS)
    build_coauthor_edges(driver)

    driver.close()
    print("Done")


if __name__ == "__main__":
    main()
