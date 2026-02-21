# cache_manager.py
import gc
import hashlib
import os
import shutil
from typing import Any, Dict, Iterable, List

CACHE_DIR = os.getenv("RAG_CACHE_DIR", "cache")
CACHE_KEY_VERSION = os.getenv("RAG_CACHE_VERSION", "v3")


def _norm_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def short_hash(value: Any, *, length: int = 12) -> str:
    data = _norm_text(value).encode("utf-8", errors="ignore")
    return hashlib.sha1(data).hexdigest()[: max(6, int(length))]


def state_signature_from_state(state: Dict[str, Any]) -> str:
    turns = state.get("turns", []) if isinstance(state, dict) else []
    extra = state.get("extra_state", {}) if isinstance(state, dict) else {}
    anchor_value = ""
    if isinstance(state, dict) and isinstance(state.get("anchor"), dict):
        anchor_value = str(state.get("anchor", {}).get("value", "") or "")
    elif isinstance(extra, dict) and isinstance(extra.get("anchor"), dict):
        anchor_value = str(extra.get("anchor", {}).get("value", "") or "")
    rolling_summary = ""
    if isinstance(state, dict):
        rolling_summary = str(state.get("rolling_summary", "") or "")
    if (not rolling_summary) and isinstance(extra, dict):
        rolling_summary = str(extra.get("rolling_summary", "") or extra.get("last_focus", "") or "")
    summary_sig = short_hash(rolling_summary, length=10)
    return f"t{max(0, len(turns or []))}|a{short_hash(anchor_value, length=10)}|s{summary_sig}"


def build_cache_key(
    *,
    user_key: str,
    resolved_text: str,
    effective_mode: str,
    state_signature: str,
) -> str:
    u = _norm_text(user_key)
    r = _norm_text(resolved_text)
    m = _norm_text(effective_mode).lower() or "default"
    s = _norm_text(state_signature)
    return f"{CACHE_KEY_VERSION}|{u}|{m}|{s}|{short_hash(r, length=16)}"


def should_cache_turn(*, retrieval_text: str, rewrite_blocked: bool) -> bool:
    if bool(rewrite_blocked):
        return False
    return bool(_norm_text(retrieval_text))


def _doc_identity(meta: Dict[str, Any]) -> str:
    pid = _norm_text(meta.get("paper_id", ""))
    chunk = _norm_text(meta.get("chunk", meta.get("chunk_id", meta.get("id", ""))))
    if pid or chunk:
        return f"{pid}::{chunk}".strip(":")
    title = _norm_text(meta.get("title", ""))
    return title


def retrieval_cache_summary(
    docs: Iterable[Any],
    *,
    retrieval_text: str,
    limit_ids: int = 12,
) -> Dict[str, Any]:
    docs_list = list(docs or [])
    ids: List[str] = []
    for d in docs_list:
        meta = getattr(d, "metadata", {}) or {}
        doc_id = _doc_identity(meta)
        if not doc_id:
            continue
        if doc_id in ids:
            continue
        ids.append(doc_id)
        if len(ids) >= max(1, int(limit_ids)):
            break

    id_blob = "|".join(ids)
    return {
        "retrieval_text_hash": short_hash(retrieval_text, length=12),
        "retrieval_text_chars": len(_norm_text(retrieval_text)),
        "doc_count": len(docs_list),
        "doc_ids": ids,
        "doc_ids_hash": short_hash(id_blob, length=12),
    }


def clear_cache() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        except Exception:
            pass
