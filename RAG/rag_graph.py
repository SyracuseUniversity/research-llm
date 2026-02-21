# rag_graph.py
import re
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return s


_SPLIT_RE = re.compile(r"\s*(?:,|;|\band\b|\|)\s*", re.IGNORECASE)


def _split_authors(s: str, limit: int = 25) -> List[str]:
    s = _safe_str(s)
    if not s:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(s) if p.strip()]
    out: List[str] = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def paper_docs_to_graph_hits(paper_docs: List[Document], max_papers: int = 40) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    seen_pid = set()

    for d in paper_docs:
        meta = d.metadata or {}
        pid = _safe_str(meta.get("paper_id"))
        if not pid:
            continue
        if pid in seen_pid:
            continue
        seen_pid.add(pid)

        hits.append(
            {
                "paper_id": pid,
                "title": _safe_str(meta.get("title")),
                "researcher": _safe_str(meta.get("researcher")),
                "authors": _safe_str(meta.get("authors")),
                "doi": _safe_str(meta.get("doi")),
                "year": _safe_str(meta.get("year")) or _safe_str(meta.get("publication_date")),
                "primary_topic": _safe_str(meta.get("primary_topic")),
            }
        )

        if len(hits) >= max_papers:
            break

    return hits


def build_graph_from_hits(
    hits: List[Dict[str, Any]],
    height: int = 650,
    include_topics: bool = True,
    include_authors: bool = True,
    max_authors_per_paper: int = 12,
) -> Dict[str, Any]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    def add_node(node_id: str, label: str, ntype: str) -> None:
        nodes.append({"id": node_id, "label": label, "type": ntype})

    def add_edge(src: str, tgt: str, etype: str) -> None:
        edges.append({"source": src, "target": tgt, "type": etype})

    seen_nodes = set()
    seen_edges = set()

    def ensure_node(node_id: str, label: str, ntype: str) -> None:
        key = (node_id, ntype)
        if key in seen_nodes:
            return
        seen_nodes.add(key)
        add_node(node_id, label, ntype)

    def ensure_edge(src: str, tgt: str, etype: str) -> None:
        key = (src, tgt, etype)
        if key in seen_edges:
            return
        seen_edges.add(key)
        add_edge(src, tgt, etype)

    for h in hits:
        pid = _safe_str(h.get("paper_id"))
        if not pid:
            continue

        p_title = _safe_str(h.get("title")) or f"paper {pid}"
        paper_node = f"paper:{pid}"
        ensure_node(paper_node, p_title, "paper")

        researcher = _safe_str(h.get("researcher"))
        if researcher:
            r_node = f"researcher:{researcher.lower()}"
            ensure_node(r_node, researcher, "researcher")
            ensure_edge(r_node, paper_node, "WROTE")

        if include_authors:
            authors = _split_authors(_safe_str(h.get("authors")), limit=max_authors_per_paper)
            for a in authors:
                a_node = f"author:{a.lower()}"
                ensure_node(a_node, a, "author")
                ensure_edge(a_node, paper_node, "AUTHORED")

        if include_topics:
            topic = _safe_str(h.get("primary_topic"))
            if topic and topic.lower() != "n/a":
                t_node = f"topic:{topic.lower()}"
                ensure_node(t_node, topic, "topic")
                ensure_edge(paper_node, t_node, "HAS_TOPIC")

    return {"nodes": nodes, "edges": edges, "height": height}


def graph_retrieve_from_paper_docs(paper_docs: List[Document], height: int = 650) -> Dict[str, Any]:
    hits = paper_docs_to_graph_hits(paper_docs)
    graph = build_graph_from_hits(hits, height=height)
    return {"hits": hits, "graph": graph, "error": ""}
