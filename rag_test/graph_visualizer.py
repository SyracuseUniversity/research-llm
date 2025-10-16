# # # # graph_visualizer.py
# # # from streamlit_agraph import agraph, Node, Edge, Config

# # # def render_graph(graph_hits):
# # #     """
# # #     Build and render an interactive graph view using streamlit_agraph.
# # #     graph_hits: list of dicts [{title, researcher, authors, related, score}, ...]
# # #     """

# # #     if not graph_hits:
# # #         return "No graph data available."

# # #     nodes, edges, added = [], [], set()

# # #     for g in graph_hits:
# # #         title = g.get("title", "Untitled")
# # #         researcher = g.get("researcher", "Unknown")
# # #         authors = g.get("authors", []) or []
# # #         related = g.get("related", []) or []

# # #         # --- main nodes ---
# # #         if researcher not in added:
# # #             nodes.append(Node(id=researcher, label=researcher,
# # #                               size=35, color="#1E90FF", shape="star"))
# # #             added.add(researcher)

# # #         if title not in added:
# # #             nodes.append(Node(id=title, label=title,
# # #                               size=25, color="#FFD700"))
# # #             added.add(title)

# # #         # --- relationships ---
# # #         edges.append(Edge(source=researcher, target=title,
# # #                           label="authored", color="#FFB86C"))

# # #         for a in authors:
# # #             if a not in added:
# # #                 nodes.append(Node(id=a, label=a,
# # #                                   size=20, color="#BD93F9"))
# # #                 added.add(a)
# # #             edges.append(Edge(source=a, target=title,
# # #                               label="co-author", color="#6272A4"))

# # #         for r in related[:5]:
# # #             if r not in added:
# # #                 nodes.append(Node(id=r, label=r,
# # #                                   size=18, color="#FF79C6"))
# # #                 added.add(r)
# # #             edges.append(Edge(source=title, target=r,
# # #                               label="related", color="#FF5555"))

# # #     config = Config(
# # #         width="100%",
# # #         height=650,
# # #         directed=True,
# # #         physics=True,
# # #         hierarchical=False,
# # #         nodeHighlightBehavior=True,
# # #         highlightColor="#A6E22E",
# # #         collapsible=True,
# # #         link={'color': '#AAAAAA', 'labelProperty': 'label'}
# # #     )

# # #     return agraph(nodes=nodes, edges=edges, config=config)

# # # graph_visualizer.py
# # from neo4j import GraphDatabase
# # from streamlit_agraph import agraph, Node, Edge, Config
# # import config_full as config

# # driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # def fetch_graph_data(cypher_query):
# #     with driver.session(database=config.NEO4J_DB) as session:
# #         results = session.run(cypher_query)
# #         nodes, edges = {}, []

# #         for record in results:
# #             for element in record.values():
# #                 if isinstance(element, dict):  # skip metadata
# #                     continue
# #                 if hasattr(element, 'nodes'):
# #                     for node in element.nodes:
# #                         nid = str(node.id)
# #                         nlabel = list(node.labels)[0] if node.labels else "Unknown"
# #                         nname = node.get("name") or node.get("title") or f"{nlabel}-{nid}"

# #                         if nid not in nodes:
# #                             color = (
# #                                 "#C792EA" if nlabel == "Author"
# #                                 else "#00BFFF" if nlabel == "Researcher"
# #                                 else "#FF6347" if nlabel == "Paper"
# #                                 else "#A9A9A9"
# #                             )
# #                             size = 25 if nlabel == "Researcher" else 20 if nlabel == "Author" else 15
# #                             nodes[nid] = Node(
# #                                 id=nid, label=nname, title=f"{nlabel}: {nname}", color=color, size=size
# #                             )

# #                     for rel in element.relationships:
# #                         src = str(rel.start_node.id)
# #                         tgt = str(rel.end_node.id)
# #                         edges.append(
# #                             Edge(source=src, target=tgt, label=rel.type, color="#F1FA8C", width=2)
# #                         )

# #         return list(nodes.values()), edges


# # def render_graph(cypher_query: str, height=600):
# #     nodes, edges = fetch_graph_data(cypher_query)
# #     if not nodes:
# #         return "No graph data found for this query."

# #     config = Config(
# #         width="100%",
# #         height=height,
# #         directed=True,
# #         physics=True,
# #         hierarchical=False,
# #         nodeHighlightBehavior=True,
# #         highlightColor="#F1FA8C",
# #     )

# #     return agraph(nodes=nodes, edges=edges, config=config)

# # graph_visualizer.py
# from neo4j import GraphDatabase
# from streamlit_agraph import agraph, Node, Edge, Config
# import config_full as config
# import re
# from typing import Dict, Optional, List, Tuple

# driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# # ---------------- utils ----------------

# def wrap_label(text: str, width: int = 24, max_lines: int = 3) -> str:
#     """Soft-wrap labels to multiple lines to reduce overlap."""
#     if not text:
#         return "Unknown"
#     words = re.findall(r"\S+", str(text))
#     lines, cur = [], ""
#     for w in words:
#         nxt = (cur + " " + w).strip()
#         if len(nxt) <= width:
#             cur = nxt
#         else:
#             if cur:
#                 lines.append(cur)
#             cur = w
#             if len(lines) >= max_lines:
#                 break
#     if cur and len(lines) < max_lines:
#         lines.append(cur)
#     return "\n".join(lines) if lines else "Unknown"

# def _make_node(nid: str, label: str, nlabel: str) -> Node:
#     color = {"Author": "#C792EA", "Researcher": "#1E90FF", "Paper": "#FF6347"}.get(nlabel, "#A9A9A9")
#     size = {"Researcher": 38, "Author": 24, "Paper": 20}.get(nlabel, 18)
#     return Node(
#         id=nid,
#         label=wrap_label(label),
#         title=f"{nlabel}: {label}",
#         color=color,
#         size=size,
#         shape="dot",
#     )

# # --------------- fetch & build ---------------

# def fetch_graph_data(cypher_query: str, params: Optional[Dict] = None) -> Tuple[List[Node], List[Edge]]:
#     with driver.session(database=config.NEO4J_DB) as session:
#         results = session.run(cypher_query, **(params or {}))

#         nodes: Dict[str, Node] = {}
#         edges: List[Edge] = []

#         def add_path(path):
#             # Nodes
#             for node in path.nodes:
#                 nid = str(node.id)
#                 nlabel = list(node.labels)[0] if node.labels else "Unknown"
#                 name = node.get("name") or node.get("title") or f"{nlabel}-{nid}"
#                 if nid not in nodes:
#                     nodes[nid] = _make_node(nid, name, nlabel)
#             # Edges
#             for rel in path.relationships:
#                 src = str(rel.start_node.id)
#                 tgt = str(rel.end_node.id)
#                 rel_type = rel.type
#                 edge_id = f"{src}-{tgt}-{rel_type}"
#                 seen = {
#                     f"{getattr(e, 'source', '')}-"
#                     f"{(getattr(e, 'target', None) or getattr(e, 'to', ''))}-"
#                     f"{getattr(e, 'label', '')}"
#                     for e in edges
#                 }
#                 if edge_id not in seen:
#                     edges.append(
#                         Edge(
#                             source=src,
#                             target=tgt,   # required by current streamlit_agraph version
#                             label="",
#                             color="#E6E6E6",
#                             width=2,
#                             smooth=True,
#                         )
#                     )

#         for record in results:
#             for val in record.values():
#                 # Single Path
#                 if hasattr(val, "nodes") and hasattr(val, "relationships"):
#                     add_path(val)
#                 # List/collection of Paths
#                 elif isinstance(val, (list, tuple)):
#                     for maybe_path in val:
#                         if hasattr(maybe_path, "nodes") and hasattr(maybe_path, "relationships"):
#                             add_path(maybe_path)

#         return list(nodes.values()), edges

# # --------------- render ---------------

# def render_graph(cypher_query: str, params: Optional[Dict] = None, height: int = 700):
#     nodes, edges = fetch_graph_data(cypher_query, params=params)
#     if not nodes:
#         return "No graph data found for this query."

#     cfg = Config(
#         width="100%",
#         height=height,
#         directed=False,
#         physics=True,
#         hierarchical=False,
#         nodeHighlightBehavior=True,
#         highlightColor="#F1FA8C",
#         **{
#             "linkDistance": 180,
#             "repulsion": 420,
#             "centralGravity": 0.25,
#             "springLength": 200,
#             "springConstant": 0.025,
#             "damping": 0.88,
#         },
#     )
#     return agraph(nodes=nodes, edges=edges, config=cfg)

# graph_visualizer.py
from typing import Dict, Optional, List, Tuple
from neo4j import GraphDatabase
from streamlit_agraph import agraph, Node, Edge, Config
import config_full as config
import re

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))


# ---------------- utils ----------------
def wrap_label(text: str, width: int = 22, max_lines: int = 3) -> str:
    """Word-wrap labels so long titles don't overflow."""
    if not text:
        return "Unknown"
    words = re.findall(r"\S+", str(text))
    lines, cur = [], ""
    for w in words:
        nxt = (cur + " " + w).strip()
        if len(nxt) <= width:
            cur = nxt
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return "\n".join(lines)


def _make_node(nid: str, label: str, nlabel: str) -> Node:
    color = {"Author": "#9B59B6", "Researcher": "#1E90FF", "Paper": "#FF6F61"}.get(nlabel, "#888888")
    size = {"Researcher": 40, "Author": 26, "Paper": 22}.get(nlabel, 18)
    return Node(
        id=nid,
        label=wrap_label(label),
        title=f"{nlabel}: {label}",
        color=color,
        size=size,
        shape="dot",
    )


# ---------------- core graph fetch ----------------
def fetch_graph_data(cypher_query: str, params: Optional[Dict] = None) -> Tuple[List[Node], List[Edge]]:
    """Executes a Cypher query and converts paths to Streamlit-agraph nodes/edges."""
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
                rel_type = rel.type
                edge_id = f"{src}-{tgt}-{rel_type}"
                if edge_id not in {f"{e.source}-{e.target}-{e.label}" for e in edges}:
                    edges.append(
                        Edge(
                            source=src,
                            target=tgt,
                            label="",
                            color="#777777",
                            width=2,
                            smooth=True,
                        )
                    )

        for record in results:
            for v in record.values():
                if hasattr(v, "nodes"):
                    add_path(v)
                elif isinstance(v, list):
                    for p in v:
                        if hasattr(p, "nodes"):
                            add_path(p)

        return list(nodes.values()), edges


# ---------------- render ----------------
def render_graph_from_hits(graph_hits: List[Dict], height: int = 700):
    """
    Build a subgraph visualization directly from the retrieved graph_hits
    (same context as used in hybrid_langchain_retriever).
    """
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
    if not nodes:
        return "No graph nodes found for retrieved context."

    cfg = Config(
        width="100%",
        height=height,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#FFD54A",
        **{
            "backgroundColor": "#FFFFFF",  # white background
            "linkDistance": 200,
            "repulsion": 400,
            "centralGravity": 0.25,
            "springLength": 180,
            "springConstant": 0.03,
            "damping": 0.9,
        },
    )

    return agraph(nodes=nodes, edges=edges, config=cfg), cypher_query, params
