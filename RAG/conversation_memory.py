# conversation_memory.py
import os
import sys
from typing import Dict, Any, Optional
from collections import OrderedDict

_MAX_QA_PER_USER = 10
_QA_CACHE: Dict[str, "OrderedDict[str, Dict[str, Any]]"] = {}
_PIPELINE_CACHE: Dict[str, Dict[str, Any]] = {}


def clear_qa_cache(user_key: str) -> None:
    _QA_CACHE.pop(user_key, None)
    clear_pipeline_cache(user_key)


def get_cached_answer(user_key: str, key: str) -> Optional[Dict[str, Any]]:
    u = _QA_CACHE.get(user_key)
    if not u:
        return None
    return u.get((key or "").strip())


def set_cached_answer(
    user_key: str,
    key: str,
    payload: Dict[str, Any],
) -> None:
    u = _QA_CACHE.setdefault(user_key, OrderedDict())

    cache_key = (key or "").strip()

    if cache_key in u:
        u.move_to_end(cache_key)
    u[cache_key] = payload

    while len(u) > _MAX_QA_PER_USER:
        u.popitem(last=False)


def hard_reset_memory(user_key: str) -> None:
    manager = None
    rag_engine_mod = sys.modules.get("rag_engine")
    if rag_engine_mod is not None:
        manager = getattr(rag_engine_mod, "_GLOBAL_MANAGER", None)

    if manager is not None:
        try:
            manager.reset_session(user_key)
        except Exception:
            pass
    else:
        # Avoid creating EngineManager here; reset lightweight stores directly.
        try:
            from session_store import SessionStore

            state_db = os.getenv("RAG_STATE_DB", "chat_state.sqlite")
            SessionStore(state_db).reset(user_key)
        except Exception:
            pass
        try:
            import chromadb

            mem_dir = os.getenv("RAG_MEMORY_DIR", "chroma_memory")
            client = chromadb.PersistentClient(path=mem_dir)
            col = client.get_collection("memory")
            got = col.get(where={"session_id": user_key}, include=["ids"])
            ids = got.get("ids") or []
            if ids:
                col.delete(ids=ids)
        except Exception:
            pass
    clear_qa_cache(user_key)


def get_pipeline_cache(user_key: str) -> Optional[Dict[str, Any]]:
    return _PIPELINE_CACHE.get(user_key)


def set_pipeline_cache(user_key: str, data: Dict[str, Any]) -> None:
    if data:
        _PIPELINE_CACHE[user_key] = data
    else:
        _PIPELINE_CACHE.pop(user_key, None)


def clear_pipeline_cache(user_key: str) -> None:
    _PIPELINE_CACHE.pop(user_key, None)
