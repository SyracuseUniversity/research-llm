from typing import List, Dict
import re
from neo4j import GraphDatabase
import config_full as config

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[A-Za-z0-9]+")

def _norm(s: str) -> str:
    return (_WS.sub(" ", s or "")).strip()

def _keywords(q: str, min_len: int = 3, max_kw: int = 6) -> List[str]:
    toks = [t.lower() for t in _ALNUM.findall(q or "")]
    seen, out = set(), []
    for t in toks:
        if len(t) >= min_len and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_kw:
            break
    return out or ([_norm(q).lower()] if q else [])

def _score_row(row: Dict, kws: List[str]) -> int:
    hay = " ".join([
        str(row.get("title") or ""),
        " ".join(row.get("authors") or []),
        str(row.get("researcher") or "")
    ]).lower()
    return sum(1 for kw in kws if kw in hay)

def _run_query(cypher: str, **params) -> List[Dict]:
    with driver.session(database=config.NEO4J_DB) as s:
        return [dict(r) for r in s.run(cypher, **params)]

CY_SEARCH = """
WITH $kws AS kws
MATCH (p:Paper)
OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
WITH p, collect(DISTINCT a.name) AS authors, r.name AS researcher, kws
WHERE ANY(kw IN kws WHERE
      toLower(p.title) CONTAINS kw
   OR ANY(an IN authors WHERE toLower(coalesce(an,'')) CONTAINS kw)
   OR toLower(coalesce(researcher,'')) CONTAINS kw)
RETURN p.paper_id AS paper_id,
       p.title AS title,
       p.year AS year,
       p.doi_link AS doi_link,
       authors AS authors,
       researcher AS researcher
LIMIT $hard_limit
"""

def query_graph(question: str, k: int = 10) -> List[Dict]:
    kws = _keywords(question)
    rows = _run_query(CY_SEARCH, kws=kws, hard_limit=max(k*5, 50))
    for r in rows:
        r["_score"] = _score_row(r, kws)
    rows.sort(key=lambda x: (-x["_score"], x.get("year") or -9999, x.get("title","")))
    return rows[:k]

CY_BY_AUTHOR = """
MATCH (a:Author {name: $name})-[:AUTHORED]->(p:Paper)
OPTIONAL MATCH (p)-[:HAS_RESEARCHER]->(r:Researcher)
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
ORDER BY coalesce(p.year, -9999) DESC
LIMIT $k
"""

def get_papers_by_author(name: str, k: int = 25) -> List[Dict]:
    return _run_query(CY_BY_AUTHOR, name=name, k=k)

CY_BY_RESEARCHER = """
MATCH (r:Researcher {name: $name})-[:WROTE]->(p:Paper)
OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       p.doi_link AS doi_link, collect(DISTINCT a.name) AS authors, r.name AS researcher
ORDER BY coalesce(p.year, -9999) DESC
LIMIT $k
"""

def get_papers_by_researcher(name: str, k: int = 25) -> List[Dict]:
    return _run_query(CY_BY_RESEARCHER, name=name, k=k)
