# conversation_memory.py
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_QA_PER_USER = 10
_MAX_USERS = 500

_QA_CACHE: "OrderedDict[str, OrderedDict[str, Dict[str, Any]]]" = OrderedDict()
_PIPELINE_CACHE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _evict_oldest_users() -> None:
    for cache in (_QA_CACHE, _PIPELINE_CACHE):
        while len(cache) > _MAX_USERS:
            cache.popitem(last=False)


def clear_qa_cache(user_key: str) -> None:
    _QA_CACHE.pop(user_key, None)
    clear_pipeline_cache(user_key)


def get_cached_answer(user_key: str, key: str) -> Optional[Dict[str, Any]]:
    u = _QA_CACHE.get(user_key)
    return u.get((key or "").strip()) if u else None


def set_cached_answer(user_key: str, key: str, payload: Dict[str, Any]) -> None:
    u = _QA_CACHE.setdefault(user_key, OrderedDict())
    _QA_CACHE.move_to_end(user_key)
    cache_key = (key or "").strip()
    if cache_key in u:
        u.move_to_end(cache_key)
    u[cache_key] = payload
    while len(u) > _MAX_QA_PER_USER:
        u.popitem(last=False)
    _evict_oldest_users()


def hard_reset_memory(user_key: str) -> None:
    import sys
    mod = sys.modules.get("rag_engine")
    manager = getattr(mod, "_GLOBAL_MANAGER", None) if mod else None
    if manager is not None:
        try:
            manager.reset_session(user_key)
        except Exception:
            logger.warning("Failed to reset session via EngineManager", exc_info=True)
    else:
        # Reset stores directly without creating EngineManager
        try:
            from session_store import SessionStore
            SessionStore(os.getenv("RAG_STATE_DB", "chat_state.sqlite")).reset(user_key)
        except Exception:
            logger.warning("Failed to reset SessionStore directly", exc_info=True)
        try:
            import chromadb
            client = chromadb.PersistentClient(path=os.getenv("RAG_MEMORY_DIR", "chroma_memory"))
            col = client.get_collection("memory")
            ids = ((col.get(where={"session_id": user_key}) or {}).get("ids") or [])
            if ids:
                col.delete(ids=ids)
        except Exception:
            logger.warning("Failed to reset chroma memory directly", exc_info=True)
    clear_qa_cache(user_key)


def get_pipeline_cache(user_key: str) -> Optional[Dict[str, Any]]:
    return _PIPELINE_CACHE.get(user_key)


def set_pipeline_cache(user_key: str, data: Dict[str, Any]) -> None:
    if data:
        _PIPELINE_CACHE[user_key] = data
        _PIPELINE_CACHE.move_to_end(user_key)
        _evict_oldest_users()
    else:
        _PIPELINE_CACHE.pop(user_key, None)


def clear_pipeline_cache(user_key: str) -> None:
    _PIPELINE_CACHE.pop(user_key, None)