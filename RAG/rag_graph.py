# rag_graph.py
import re
from typing import Any, Dict, List

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ClientError

import config_graph as gcfg


_STOP = {
    "a","an","the","and","or","of","to","in","on","for","with","by","from","at","as","is","are","was","were",
    "be","been","being","this","that","these","those","it","its","we","you","i","they","them","their","our",
    "paper","papers","research","author","authors","researcher","researchers","syracuse","university"
}
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{2,}")


def _tokenize(q: str) -> List[str]:
    if not q:
        return []
    words = [w.lower() for w in _WORD.findall(q)]
    out: List[str] = []
    for w in words:
        if w in _STOP:
            continue
        if len(w) < 3:
            continue
        out.append(w)

    seen = set()
    dedup: List[str] = []
    for w in out:
        if w in seen:
            continue
        seen.add(w)
        dedup.append(w)
    return dedup[:12]


CYPHER = """
WITH $terms AS terms, $k AS k

MATCH (p:Paper)
WHERE
  ANY(t IN terms WHERE toLower(p.title) CONTAINS t)
  OR ANY(t IN terms WHERE toLower(p.primary_topic) CONTAINS t)

OPTIONAL MATCH (p)<-[:WROTE]-(r:Researcher)
OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)

WITH
  p,
  collect(DISTINCT r.name)[0..5] AS researchers,
  collect(DISTINCT a.name)[0..10] AS authors,
  size([t IN terms WHERE toLower(p.title) CONTAINS t]) +
  size([t IN terms WHERE toLower(p.primary_topic) CONTAINS t]) AS score

RETURN
  p.paper_id AS paper_id,
  p.title AS title,
  p.year AS year,
  p.doi AS doi,
  p.doi_link AS doi_link,
  p.arxiv_id AS arxiv_id,
  p.primary_topic AS primary_topic,
  researchers AS researchers,
  authors AS authors,
  score AS score
ORDER BY score DESC, year DESC
LIMIT k
"""


def _connect():
    return GraphDatabase.driver(gcfg.NEO4J_URI, auth=(gcfg.NEO4J_USER, gcfg.NEO4J_PASS))


def graph_retrieve(question: str, top_k: int = None) -> Dict[str, Any]:
    terms = _tokenize(question)
    k = int(top_k or gcfg.GRAPH_TOP_K)
    params = {"terms": terms, "k": k}

    if not terms:
        return {"hits": [], "cypher": CYPHER, "params": params}

    driver = _connect()
    try:
        driver.verify_connectivity()
        with driver.session(database=gcfg.NEO4J_DB) as s:
            rows = list(s.run(CYPHER, **params))
    except AuthError as e:
        return {"hits": [], "cypher": CYPHER, "params": params, "error": f"Neo4j auth failed: {e}"}
    except ClientError as e:
        return {"hits": [], "cypher": CYPHER, "params": params, "error": f"Neo4j query failed: {e}"}
    except Exception as e:
        return {"hits": [], "cypher": CYPHER, "params": params, "error": str(e)}
    finally:
        try:
            driver.close()
        except Exception:
            pass

    hits: List[Dict[str, Any]] = []
    for r in rows:
        hits.append(
            {
                "paper_id": r.get("paper_id") or "",
                "title": r.get("title") or "",
                "year": r.get("year"),
                "doi": r.get("doi") or "",
                "doi_link": r.get("doi_link") or "",
                "arxiv_id": r.get("arxiv_id") or "",
                "primary_topic": r.get("primary_topic") or "",
                "researchers": r.get("researchers") or [],
                "authors": r.get("authors") or [],
                "score": r.get("score") or 0,
            }
        )

    return {"hits": hits, "cypher": CYPHER, "params": params}
