# cache_manager.py
import gc
import hashlib
import os
import shutil
import time
from typing import Any, Dict, Iterable, List, Optional

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


def _gpu_release() -> None:
    """Release GPU memory if torch/CUDA is available."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _entry_age_seconds(path: str) -> float:
    """Return seconds since the file/dir was last modified, or 0 on error."""
    try:
        return time.time() - os.path.getmtime(path)
    except Exception:
        return 0.0


def clear_cache(
    *,
    user_key: Optional[str] = None,
    mode: Optional[str] = None,
    older_than_seconds: Optional[float] = None,
    version: Optional[str] = None,
) -> Dict[str, int]:
    """
    Targeted cache invalidation.

    Parameters
    ----------
    user_key : str, optional
        Delete only entries whose filename contains this user key hash.
    mode : str, optional
        Delete only entries whose filename contains this mode string.
    older_than_seconds : float, optional
        Delete only entries older than this many seconds.
    version : str, optional
        Delete only entries whose filename starts with this version prefix.
        When no filters are provided at all, defaults to removing entries from
        older versions only (files that do NOT start with CACHE_KEY_VERSION).

    Returns
    -------
    dict with "deleted" and "skipped" counts.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    _gpu_release()

    filters = []

    if user_key is not None:
        h = short_hash(user_key, length=12)
        filters.append(lambda name, _p, h=h: h in name)

    if mode is not None:
        m = _norm_text(mode).lower()
        filters.append(lambda name, _p, m=m: m in name)

    if older_than_seconds is not None:
        limit = float(older_than_seconds)
        filters.append(lambda _n, p, limit=limit: _entry_age_seconds(p) > limit)

    if version is not None:
        v = str(version)
        filters.append(lambda name, _p, v=v: name.startswith(v))
    elif not filters:
        cur = CACHE_KEY_VERSION
        filters.append(lambda name, _p, cur=cur: not name.startswith(cur))

    deleted = skipped = 0
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        if all(f(name, p) for f in filters):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    os.remove(p)
                deleted += 1
            except Exception:
                skipped += 1
        else:
            skipped += 1

    return {"deleted": deleted, "skipped": skipped}


def clear_cache_all() -> Dict[str, int]:
    """Unconditionally remove every entry in the cache directory."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    _gpu_release()

    deleted = skipped = 0
    for name in os.listdir(CACHE_DIR):
        p = os.path.join(CACHE_DIR, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
            deleted += 1
        except Exception:
            skipped += 1
    return {"deleted": deleted, "skipped": skipped}