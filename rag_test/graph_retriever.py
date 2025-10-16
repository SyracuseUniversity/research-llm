# # # from kg_retriever import query_graph, get_papers_by_author, get_papers_by_researcher

# # # def search_graph(query: str, k: int = 10):
# # #     return query_graph(query, k=k)

# # # def papers_by_author(name: str, k: int = 25):
# # #     return get_papers_by_author(name, k=k)

# # # def papers_by_researcher(name: str, k: int = 25):
# # #     return get_papers_by_researcher(name, k=k)


# # # graph_retriever.py
# # from neo4j import GraphDatabase
# # import config_full as config
# # import re

# # # ------------------ Neo4j setup ------------------
# # driver = GraphDatabase.driver(
# #     config.NEO4J_URI,
# #     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# # )
# # DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # # ------------------ helpers ------------------
# # def _safe_str(v):
# #     return str(v).strip() if v else ""

# # def _normalize(txt: str):
# #     return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())

# # def _similarity(a, b):
# #     a, b = set(_normalize(a).split()), set(_normalize(b).split())
# #     if not a or not b:
# #         return 0
# #     return len(a & b) / len(a | b)

# # # ------------------ main graph search ------------------
# # def search_graph(query_text: str, k: int = 5):
# #     """
# #     Retrieve related graph nodes based on fuzzy match against title, researcher, or authors.
# #     Expands context with RELATED_TO / CITED_BY links.
# #     """
# #     q_norm = _normalize(query_text)
# #     if not q_norm:
# #         return []

# #     cypher = """
# #     MATCH (p:Paper)
# #     OPTIONAL MATCH (p)-[:AUTHORED_BY]->(r:Researcher)
# #     OPTIONAL MATCH (p)-[:CITED_BY|:RELATED_TO]->(related)
# #     RETURN p.title AS title,
# #            p.year AS year,
# #            p.doi AS doi,
# #            p.info AS info,
# #            r.name AS researcher,
# #            COLLECT(DISTINCT r.name) AS authors,
# #            COLLECT(DISTINCT related.title) AS related_papers
# #     """

# #     results = []
# #     try:
# #         with driver.session(database=DB_NAME) as session:
# #             data = session.run(cypher)
# #             for row in data:
# #                 title = _safe_str(row["title"])
# #                 researcher = _safe_str(row["researcher"])
# #                 authors = row["authors"] or []
# #                 info = _safe_str(row["info"])
# #                 doi = _safe_str(row["doi"])
# #                 year = _safe_str(row["year"])
# #                 related = row["related_papers"] or []

# #                 score = max(
# #                     _similarity(q_norm, title),
# #                     _similarity(q_norm, researcher),
# #                     *[_similarity(q_norm, a) for a in authors]
# #                 )

# #                 if score > 0.1:
# #                     results.append({
# #                         "title": title,
# #                         "researcher": researcher,
# #                         "authors": authors,
# #                         "year": year,
# #                         "doi": doi,
# #                         "info": info,
# #                         "related": related,
# #                         "score": score
# #                     })
# #         results.sort(key=lambda x: x["score"], reverse=True)
# #         return results[:k]

# #     except Exception as e:
# #         print(f"❌ Neo4j query error: {e}")
# #         return []

# # if __name__ == "__main__":
# #     q = "Jeffrey Saltz"
# #     hits = search_graph(q, k=10)
# #     for h in hits:
# #         print(f"{h['score']:.2f} | {h['title']} — {h['researcher']}")

# # graph_retriever.py
# from neo4j import GraphDatabase
# import config_full as config
# import re

# driver = GraphDatabase.driver(
#     config.NEO4J_URI,
#     auth=(config.NEO4J_USER, config.NEO4J_PASS)
# )
# DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# # ------------------ helpers ------------------
# def _safe_str(v): return str(v).strip() if v else ""
# def _normalize(txt: str): return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())
# def _similarity(a, b):
#     a, b = set(_normalize(a).split()), set(_normalize(b).split())
#     if not a or not b: return 0
#     return len(a & b) / len(a | b)

# # ------------------ graph retrieval ------------------
# def expand_graph_context(researcher=None, title=None, doi=None, hops: int = 1):
#     """
#     Pull a small subgraph around the node(s) identified by researcher/title/doi.
#     Fixes:
#       - Use p.doi_link (your ingest writes this), not p.doi
#       - Drop unknown rels (RELATED_TO, CITED_BY)
#       - Parameterized WHERE to avoid injection
#       - Accept either DOI URL or suffix in 'doi'
#     """
#     params = {}
#     where = []
#     match = ["MATCH (p:Paper)"]

#     if researcher:
#         # Prefer WROTE; HAS_RESEARCHER also exists in your ingest
#         match = ["MATCH (r:Researcher {name: $researcher})-[:WROTE|HAS_RESEARCHER]->(p:Paper)"]
#         params["researcher"] = researcher
#     elif title:
#         where.append("toLower(p.title) CONTAINS toLower($title)")
#         params["title"] = title
#     elif doi:
#         # If doi starts with http, compare full; else match by doi suffix
#         where.append("""
#             CASE
#               WHEN $doi STARTS WITH 'http' THEN toLower(p.doi_link) = toLower($doi)
#               ELSE toLower(p.doi_link) ENDS WITH toLower($doi)
#             END
#         """)
#         params["doi"] = doi
#     else:
#         return []

#     rels = "OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)"

#     cypher = f"""
#     {' '.join(match)}
#     {'WHERE ' + ' AND '.join(where) if where else ''}
#     {rels}
#     RETURN DISTINCT
#         p.title AS title,
#         p.year  AS year,
#         p.doi_link AS doi,
#         collect(DISTINCT a.name) AS authors,
#         [] AS related,
#         p.info AS info,
#         $researcher AS researcher
#     LIMIT 25
#     """

#     # Ensure the parameter exists for RETURN even if not filtering by researcher
#     if "researcher" not in params:
#         params["researcher"] = researcher or ""

#     try:
#         with driver.session(database=DB_NAME) as s:
#             rows = [dict(r) for r in s.run(cypher, **params)]
#         return rows
#     except Exception as e:
#         print("Graph expansion error:", e)
#         return []

# def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
#     q_norm = _normalize(query_text)
#     out = []
#     for row in candidates:
#         score = (
#             w_r * _similarity(q_norm, row.get("researcher", "")) +
#             w_t * _similarity(q_norm, row.get("title", "")) +
#             w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / (len(row.get("authors", [])) or 1) +
#             w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / (len(row.get("related", [])) or 1)
#         )
#         row["score"] = round(score, 3)
#         out.append(row)
#     out.sort(key=lambda x: x["score"], reverse=True)
#     return out

# def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
#     """
#     Restrict Neo4j exploration to nodes surfaced by Chroma metadata.
#     """
#     all_rows = []
#     for meta in chroma_metas:
#         rname = _safe_str(meta.get("researcher"))
#         title = _safe_str(meta.get("title"))
#         doi = _safe_str(meta.get("doi"))
#         rows = expand_graph_context(rname, title, doi)
#         all_rows.extend(rows)
#     ranked = weighted_search(query_text, all_rows)
#     return ranked[:k]


# graph_retriever.py
from typing import List, Dict
import re
from neo4j import GraphDatabase
import config_full as config

# ------------------ connection ------------------
driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USER, config.NEO4J_PASS)
)
DB_NAME = getattr(config, "NEO4J_DB", "neo4j")

# ------------------ helpers ------------------
_WS = re.compile(r"\s+")
_ALNUM = re.compile(r"[A-Za-z0-9]+")

def _safe_str(v):
    return str(v).strip() if v else ""

def _normalize(txt: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", _safe_str(txt).lower())

def _similarity(a, b):
    a, b = set(_normalize(a).split()), set(_normalize(b).split())
    if not a or not b:
        return 0
    return len(a & b) / len(a | b)

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
    with driver.session(database=DB_NAME) as s:
        return [dict(r) for r in s.run(cypher, **params)]

# ------------------ graph retrieval ------------------
def expand_graph_context(researcher=None, title=None, doi=None, hops: int = 1):
    """
    Pull a small subgraph around the node(s) identified by researcher/title/doi.

    Fixes:
      - Use p.doi_link (ingest writes this), not p.doi
      - Drop unknown rels (RELATED_TO, CITED_BY)
      - Parameterized WHERE to avoid injection
      - Accept either DOI URL or suffix in 'doi'
      - Return paper_id so the UI can render exactly what was retrieved
    """
    params = {}
    where = []
    match = ["MATCH (p:Paper)"]

    if researcher:
        match = ["MATCH (r:Researcher {name: $researcher})-[:WROTE|HAS_RESEARCHER]->(p:Paper)"]
        params["researcher"] = researcher
    elif title:
        where.append("toLower(p.title) CONTAINS toLower($title)")
        params["title"] = title
    elif doi:
        where.append("""
            CASE
              WHEN $doi STARTS WITH 'http' THEN toLower(p.doi_link) = toLower($doi)
              ELSE toLower(p.doi_link) ENDS WITH toLower($doi)
            END
        """)
        params["doi"] = doi
    else:
        return []

    rels = "OPTIONAL MATCH (p)<-[:HAS_AUTHOR]-(a:Author)"

    cypher = f"""
    {' '.join(match)}
    {'WHERE ' + ' AND '.join(where) if where else ''}
    {rels}
    RETURN DISTINCT
        p.paper_id AS paper_id,
        p.title    AS title,
        p.year     AS year,
        p.doi_link AS doi,
        collect(DISTINCT a.name) AS authors,
        [] AS related,
        p.info AS info,
        $researcher AS researcher
    LIMIT 25
    """

    if "researcher" not in params:
        params["researcher"] = researcher or ""

    try:
        with driver.session(database=DB_NAME) as s:
            rows = [dict(r) for r in s.run(cypher, **params)]
        return rows
    except Exception as e:
        print("Graph expansion error:", e)
        return []

def weighted_search(query_text: str, candidates, w_r=0.6, w_t=0.25, w_a=0.1, w_rel=0.05):
    q_norm = _normalize(query_text)
    out = []
    for row in candidates:
        score = (
            w_r * _similarity(q_norm, row.get("researcher", "")) +
            w_t * _similarity(q_norm, row.get("title", "")) +
            w_a * sum(_similarity(q_norm, a) for a in row.get("authors", [])) / (len(row.get("authors", [])) or 1) +
            w_rel * sum(_similarity(q_norm, r) for r in row.get("related", [])) / (len(row.get("related", [])) or 1)
        )
        row["score"] = round(score, 3)
        out.append(row)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def search_graph_from_chroma_meta(query_text, chroma_metas, k=8):
    """
    Restrict Neo4j exploration to nodes surfaced by Chroma metadata.
    """
    all_rows = []
    for meta in chroma_metas:
        rname = _safe_str(meta.get("researcher"))
        title = _safe_str(meta.get("title"))
        doi = _safe_str(meta.get("doi"))
        rows = expand_graph_context(rname, title, doi)
        all_rows.extend(rows)
    ranked = weighted_search(query_text, all_rows)
    return ranked[:k]

# ------------------ additional queries (keyword-based) ------------------
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
