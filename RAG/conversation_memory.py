# conversation_memory.py
import json
import os
import threading
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# In-memory QA cache and pipeline cache, keyed by (user_key, cache_key).
# These are process-local and intentionally not persisted to disk — they exist
# only to avoid redundant LLM calls within a single server session.
# ---------------------------------------------------------------------------

_qa_cache: Dict[str, Dict[str, Any]] = {}
_pipeline_cache: Dict[str, Any] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# QA answer cache
# ---------------------------------------------------------------------------

def get_cached_answer(user_key: str, cache_key: str) -> Optional[Dict[str, Any]]:
    """Return a previously cached answer dict, or None if not found."""
    with _lock:
        user_store = _qa_cache.get(user_key)
        if not isinstance(user_store, dict):
            return None
        return user_store.get(cache_key)


def set_cached_answer(user_key: str, cache_key: str, payload: Dict[str, Any]) -> None:
    """Store an answer payload under (user_key, cache_key)."""
    if not user_key or not cache_key or not isinstance(payload, dict):
        return
    with _lock:
        if user_key not in _qa_cache:
            _qa_cache[user_key] = {}
        _qa_cache[user_key][cache_key] = payload


def clear_qa_cache(user_key: str) -> None:
    """Remove all cached answers for a given user/session."""
    with _lock:
        _qa_cache.pop(user_key, None)


# ---------------------------------------------------------------------------
# Pipeline state cache (lightweight — stores retrieval/summary digest only)
# ---------------------------------------------------------------------------

def get_pipeline_cache(user_key: str) -> Optional[Dict[str, Any]]:
    """Return the last pipeline state snapshot for a user, or None."""
    with _lock:
        return _pipeline_cache.get(user_key)


def set_pipeline_cache(user_key: str, payload: Dict[str, Any]) -> None:
    """Overwrite the pipeline state snapshot for a user."""
    if not user_key:
        return
    with _lock:
        _pipeline_cache[user_key] = payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# Hard reset — clears both caches and delegates to the engine manager to wipe
# the vector memory store for the session.
# ---------------------------------------------------------------------------

def hard_reset_memory(user_key: str) -> None:
    """
    Full session reset:
      1. Clears the in-memory QA cache for this user.
      2. Clears the pipeline cache for this user.
      3. Resets the persistent session state and Chroma memory collection via
         the global EngineManager (session_store rows + memory embeddings).
    """
    clear_qa_cache(user_key)
    with _lock:
        _pipeline_cache.pop(user_key, None)

    # Import here to avoid a circular import at module load time.
    try:
        from rag_engine import get_global_manager
        mgr = get_global_manager()
        mgr.reset_session(user_key)
    except Exception:
        pass