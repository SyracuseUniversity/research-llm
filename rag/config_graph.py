#config_graph.py
import os

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASS = _env("NEO4J_PASS", "password")

# Default Neo4j DB — used as fallback if DatabaseManager is not available.
_NEO4J_DB_DEFAULT = _env("NEO4J_DB", "syr-rag-one")

def get_neo4j_db() -> str:
    """Return the Neo4j database name for the currently active dataset.

    Tries to read from the global EngineManager's DatabaseManager first,
    so that switching datasets in the UI also switches the graph DB.
    Falls back to the static default if the manager isn't initialised yet.
    """
    try:
        import sys
        mod = sys.modules.get("rag_engine")
        mgr = getattr(mod, "_GLOBAL_MANAGER", None) if mod else None
        if mgr is not None:
            return mgr.dbm.get_active_neo4j_db()
    except Exception:
        pass
    return _NEO4J_DB_DEFAULT

# Keep NEO4J_DB as a module-level string for backward compat with code that
# reads it at import time, but any code that needs the *current* active DB
# should call get_neo4j_db() instead.
NEO4J_DB = _NEO4J_DB_DEFAULT

GRAPH_TOP_K = int(_env("GRAPH_TOP_K", "25"))