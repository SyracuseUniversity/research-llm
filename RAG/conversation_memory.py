# conversation_memory.py
from typing import Dict, Any, Optional

from rag_engine import get_global_manager

_QA_CACHE: Dict[str, Dict[str, Any]] = {}

def clear_qa_cache(user_key: str) -> None:
    if user_key in _QA_CACHE:
        del _QA_CACHE[user_key]

def get_cached_answer(user_key: str, question: str) -> Optional[Dict[str, Any]]:
    u = _QA_CACHE.get(user_key) or {}
    return u.get(question)

def set_cached_answer(user_key: str, question: str, payload: Dict[str, Any]) -> None:
    _QA_CACHE.setdefault(user_key, {})[question] = payload

def hard_reset_memory(user_key: str) -> None:
    mgr = get_global_manager()
    eng = mgr.get_engine(user_key, mode=mgr.active_mode)
    eng.reset_all_for_session()
    clear_qa_cache(user_key)
