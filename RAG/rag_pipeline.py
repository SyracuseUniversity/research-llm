# rag_pipeline.py
from typing import Any, Dict, List

from rag_engine import get_global_manager
from runtime_settings import settings
from rag_graph import graph_retrieve
from conversation_memory import get_cached_answer, set_cached_answer


def _doc_to_source_md(d) -> str:
    meta = d.metadata or {}
    pid = meta.get("paper_id", "")
    title = meta.get("title", "")
    chunk = meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))
    return f"paper_id={pid} chunk={chunk} title={title}\n\n{d.page_content}"


def answer_question(question: str, user_key: str, use_graph: bool = False) -> Dict[str, Any]:
    cached = get_cached_answer(user_key, question)
    if isinstance(cached, dict):
        return cached

    mgr = get_global_manager()
    mgr.switch_mode(settings.active_mode)
    mgr.switch_model(settings.llm_model)

    eng = mgr.get_engine(user_key, mode=mgr.active_mode)
    answer, paper_docs, mem_docs = eng.ask(question)

    sources: List[str] = []
    for d in paper_docs[:12]:
        try:
            sources.append(_doc_to_source_md(d))
        except Exception:
            pass

    out: Dict[str, Any] = {
        "answer": answer,
        "sources": sources,
        "graph_hits": [],
        "graph_cypher": "",
        "graph_params": {},
        "graph_error": "",
    }

    if use_graph:
        g = graph_retrieve(question)
        out["graph_hits"] = g.get("hits", []) or []
        out["graph_cypher"] = g.get("cypher", "") or ""
        out["graph_params"] = g.get("params", {}) or {}
        out["graph_error"] = g.get("error", "") or ""

    set_cached_answer(user_key, question, out)
    return out
