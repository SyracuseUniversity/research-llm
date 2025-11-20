#graph_visualizer.py
from typing import Dict, Optional, List, Tuple
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config
import config_full as config
import re

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

def wrap_label(text: str, width: int = 22, max_lines: int = 3) -> str:
    if not text: return "Unknown"
    words = re.findall(r"\S+", str(text))
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) <= width:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines: break
    if cur and len(lines) < max_lines: lines.append(cur)
    return "\n".join(lines)

def _make_node(nid: str, label: str, nlabel: str) -> Node:
    color = {"Author": "#9B59B6", "Researcher": "#1E90FF", "Paper": "#FF6F61"}.get(nlabel, "#888888")
    size = {"Researcher": 40, "Author": 26, "Paper": 22}.get(nlabel, 18)
    return Node(id=nid, label=wrap_label(label), title=f"{nlabel}: {label}",
                color=color, size=size, shape="dot")

def fetch_graph_data(cypher_query: str, params: Optional[Dict] = None) -> Tuple[List[Node], List[Edge]]:
    with driver.session(database=config.NEO4J_DB) as session:
        results = session.run(cypher_query, **(params or {}))
        nodes, edges = {}, []

        def add_path(path):
            for node in path.nodes:
                nid = str(node.id)
                nlabel = list(node.labels)[0] if node.labels else "Unknown"
                name = node.get("name") or node.get("title") or f"{nlabel}-{nid}"
                if nid not in nodes:
                    nodes[nid] = _make_node(nid, name, nlabel)
            for rel in path.relationships:
                src = str(rel.start_node.id)
                tgt = str(rel.end_node.id)
                key = f"{src}-{tgt}-{rel.type}"
                if key not in {f"{e.source}-{e.target}-{e.label}" for e in edges}:
                    edges.append(Edge(source=src, target=tgt, label="", color="#777777", width=2, smooth=True))

        for record in results:
            for v in record.values():
                if hasattr(v, "nodes"):
                    add_path(v)
                elif isinstance(v, list):
                    for p in v:
                        if hasattr(p, "nodes"):
                            add_path(p)

        return list(nodes.values()), edges

def render_graph_from_hits(graph_hits: List[Dict], height: int = 700):
    if not graph_hits:
        return "No graph data found."

    pids = [g.get("paper_id") for g in graph_hits if g.get("paper_id")]
    titles = [g.get("title") for g in graph_hits if g.get("title")]
    researchers = list({g.get("researcher") for g in graph_hits if g.get("researcher")})

    cypher_query, params = "", {}
    if pids:
        cypher_query = """
        UNWIND $pids AS pid
        MATCH p0=(pa:Paper {paper_id: pid})
        OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
        OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
        OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
        RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
        """
        params = {"pids": pids}
    elif titles:
        cypher_query = """
        UNWIND $titles AS t
        MATCH p0=(pa:Paper)
        WHERE toLower(pa.title) = toLower(t)
        OPTIONAL MATCH p1=(r:Researcher)-[:WROTE]->(pa)
        OPTIONAL MATCH p2=(pa)-[:HAS_RESEARCHER]->(r)
        OPTIONAL MATCH p3=(pa)-[:HAS_AUTHOR]->(a:Author)
        RETURN collect(p0)+collect(p1)+collect(p2)+collect(p3) AS paths
        """
        params = {"titles": titles}
    elif researchers:
        cypher_query = """
        UNWIND $names AS name
        MATCH p=(r:Researcher {name:name})-[:WROTE|HAS_RESEARCHER]->(pa:Paper)
        OPTIONAL MATCH a=(pa)-[:HAS_AUTHOR]->(au:Author)
        RETURN collect(p)+collect(a) AS paths
        """
        params = {"names": researchers}
    else:
        return "No identifiable Paper or Researcher data to visualize."

    nodes, edges = fetch_graph_data(cypher_query, params=params)
    if not nodes: return "No graph nodes found for retrieved context."

    cfg = Config(width="100%", height=height, directed=False, physics=True, hierarchical=False,
                 nodeHighlightBehavior=True, highlightColor="#FFD54A",
                 **{"backgroundColor":"#FFFFFF","linkDistance":200,"repulsion":400,
                    "centralGravity":0.25,"springLength":180,"springConstant":0.03,"damping":0.9})
    return agraph(nodes=nodes, edges=edges, config=cfg), cypher_query, params
