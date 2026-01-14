# graph_visualizer.py
from typing import Any, Dict, List, Tuple

def render_graph_from_hits(hits: List[Dict[str, Any]], height: int = 650) -> Tuple[Any, str, Dict[str, Any]]:
    nodes: List[Dict[str, str]] = []
    edges: List[Dict[str, str]] = []

    node_seen = set()

    def add_node(nid: str, label: str, kind: str) -> None:
        if not nid or nid in node_seen:
            return
        node_seen.add(nid)
        nodes.append({"id": nid, "label": label, "type": kind})

    def add_edge(src: str, dst: str, rel: str) -> None:
        if not src or not dst:
            return
        edges.append({"source": src, "target": dst, "type": rel})

    for h in hits or []:
        pid = str(h.get("paper_id") or "").strip()
        title = str(h.get("title") or "").strip()
        pnode = f"paper:{pid}" if pid else ""
        add_node(pnode, title[:80] if title else pid, "paper")

        topic = str(h.get("primary_topic") or "").strip()
        if topic:
            tnode = f"topic:{topic.lower()}"
            add_node(tnode, topic[:80], "topic")
            add_edge(tnode, pnode, "ABOUT")

        for r in (h.get("researchers") or [])[:5]:
            rn = str(r or "").strip()
            if not rn:
                continue
            rnode = f"researcher:{rn.lower()}"
            add_node(rnode, rn[:80], "researcher")
            add_edge(rnode, pnode, "WROTE")

        for a in (h.get("authors") or [])[:10]:
            an = str(a or "").strip()
            if not an:
                continue
            anode = f"author:{an.lower()}"
            add_node(anode, an[:80], "author")
            add_edge(anode, pnode, "AUTHORED")

    graph_output = {"nodes": nodes, "edges": edges, "height": height}
    cypher_query = ""
    params: Dict[str, Any] = {}
    return graph_output, cypher_query, params
