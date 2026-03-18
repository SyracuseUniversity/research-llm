# config_graph.py
import os

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASS = _env("NEO4J_PASS", "neo4j")
NEO4J_DB = _env("NEO4J_DB", "neo4j")

GRAPH_TOP_K = int(_env("GRAPH_TOP_K", "25"))
