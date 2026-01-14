# graph_retriever.py
from typing import List, Dict, Optional
import re
from neo4j import GraphDatabase
import config_full as config

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))
DB_NAME = config.NEO4J_DB

_ALNUM = re.compile(r"[A-Za-z0-9]+")

def _safe_str(v): return str(v).strip() if v else ""

def detect_researcher_name(query: str) -> Optional[str]:
    """Return the exact researcher node name if the query includes one."""
    q = (query or "").lower()
    try:
        with driver.session(database=DB_NAME) as s:
            results = s.run("MATCH (r:Researcher) RETURN r.name AS name")
            for rec in results:
                name = rec["name"]
                if name and name.lower() in q:
                    return name
    except Exception:
        pass
    return None

def get_papers_strict_researcher(name: str, limit=50) -> List[Dict]:
    """
    Only the papers where the named Researcher appears (first/corresponding/any author),
    but do not expand into other authors' networks.
    """
    cy = """
    MATCH (r:Researcher {name: $name})-[:WROTE|HAS_RESEARCHER]->(p:Paper)
    OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)
    RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
           p.doi_link AS doi, collect(DISTINCT a.name) AS authors, r.name AS researcher
    ORDER BY coalesce(p.year, -9999) DESC
    LIMIT $limit
    """
    with driver.session(database=DB_NAME) as s:
        return [dict(r) for r in s.run(cy, name=name, limit=limit)]
