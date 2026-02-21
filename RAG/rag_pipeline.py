# rag_pipeline.py
import json
import os
import re
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from rag_engine import (
    get_global_manager,
    _invoke_with_timeout,
    _strip_prompt_leak,
    _get_stopword_set,
    _classify_generic_intent,
    _has_explicit_entity_signal,
    _normalize_anchor,
    _anchor_support_ratio,
    _is_placeholder_anchor_value,
)
from runtime_settings import settings
from rag_graph import graph_retrieve_from_paper_docs
from conversation_memory import (
    get_cached_answer,
    set_cached_answer,
    get_pipeline_cache,
    set_pipeline_cache,
)
from cache_manager import (
    build_cache_key as build_pipeline_cache_key,
    state_signature_from_state,
    retrieval_cache_summary,
    should_cache_turn,
    short_hash,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return int(default)


PIPELINE_CFG = {
    "cache_version": os.getenv("RAG_CACHE_VERSION", "v2"),
    "max_docs_after_filter": _env_int(
        "RAG_MAX_DOCS_AFTER_FILTER",
        30,
    ),
    "fallback_max_items": _env_int(
        "RAG_FALLBACK_MAX_ITEMS",
        8,
    ),
    "recent_turns_in_prompt": _env_int(
        "RAG_RECENT_TURNS_IN_PROMPT",
        int(getattr(settings, "recent_turns_in_prompt", 4)),
    ),
    "qa_cache_enable": os.getenv(
        "RAG_QA_CACHE_ENABLE",
        "1" if bool(getattr(settings, "qa_cache_enable", False)) else "0",
    ).strip().lower() in {"1", "true", "yes", "y", "on"},
    "empty_question_answer": os.getenv("RAG_EMPTY_QUESTION_ANSWER", "Please enter a question."),
    "llm_no_answer": os.getenv(
        "RAG_LLM_NO_ANSWER",
        "I could not generate an answer from the retrieved context.",
    ),
    "fallback_relevant_papers_prefix": os.getenv(
        "RAG_FALLBACK_PAPERS_PREFIX",
        "I couldn't generate a full answer, but these papers look most relevant:",
    ),
    "prompt_prefix": os.getenv(
        "RAG_PROMPT_PREFIX",
        (
            "Use only the provided Syracuse corpus summary, recent turns, and retrieved context.\n"
            "Ground every material claim in retrieved evidence.\n"
            "If evidence is missing, weak, or conflicting, say so clearly and ask a focused clarification question.\n"
            "Do not fabricate affiliations, awards, memberships, journal claims, or metadata.\n"
            "Style rules:\n"
            "- Write the answer directly.\n"
            "- Do not start with \"Summary:\".\n"
            "- Do not include phrases like \"No further analysis is required\" or \"No additional retrieval is required\".\n"
            "- Do not repeat the question.\n"
            "- Do not narrate your process.\n"
            "- Do not include section headers unless the user asks for them.\n"
            "Detected intents: default, comparison, time_range, list.\n"
        ),
    ),
    "prompt_mid": os.getenv("RAG_PROMPT_MID", "\n\nQUESTION:\n"),
    "prompt_suffix": os.getenv("RAG_PROMPT_SUFFIX", "\n\nRespond with the final user-facing answer only."),
}


def _doc_to_source_md(d) -> str:
    meta = d.metadata or {}
    title = meta.get("title", "")
    authors = meta.get("authors", "")
    year = meta.get("year", meta.get("publication_date", ""))
    snippet = re.sub(r"\s+", " ", str(d.page_content or "")).strip()
    if len(snippet) > 280:
        snippet = snippet[:280].rstrip() + "..."
    return f"title={title} | authors={authors} | year={year}\n{snippet}"


def _doc_to_json(d, text_limit: Optional[int] = None) -> Dict[str, Any]:
    meta = d.metadata or {}
    chunk = meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))
    text = str(d.page_content or "")
    if text_limit is not None:
        if text_limit <= 0:
            text = ""
        elif len(text) > text_limit:
            text = text[:text_limit].rstrip() + "..."
    return {
        "paper_id": str(meta.get("paper_id", "")),
        "title": str(meta.get("title", "")),
        "authors": str(meta.get("authors", "")),
        "doi": str(meta.get("doi", "")),
        "chunk": str(chunk),
        "year": str(meta.get("year", meta.get("publication_date", ""))),
        "primary_topic": str(meta.get("primary_topic", "")),
        "text": text,
    }


def _doc_to_ref(d: Any) -> Dict[str, str]:
    meta = getattr(d, "metadata", {}) or {}
    chunk = meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))
    return {
        "paper_id": str(meta.get("paper_id", "")),
        "chunk": str(chunk),
        "title": str(meta.get("title", "")),
    }


def _query_tokens_for_relevance(question: str) -> List[str]:
    q = (question or "").strip().lower()
    if not q:
        return []
    stopset = _get_stopword_set()
    min_len = int(getattr(settings, "retrieval_keyword_min_term_len", 3))
    toks = re.findall(r"[a-z0-9\-]{2,}", q)
    out: List[str] = []
    seen = set()
    for t in toks:
        if len(t) < max(1, min_len):
            continue
        if stopset and t in stopset:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _token_in_hay(token: str, hay: str) -> bool:
    if not token or not hay:
        return False
    return re.search(rf"\b{re.escape(token)}\b", hay) is not None


def _dedupe_docs(docs: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        key = str(meta.get("paper_id", "")) + "::" + str(meta.get("chunk", meta.get("chunk_id", meta.get("id", ""))))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _doc_haystack(d: Any) -> str:
    meta = getattr(d, "metadata", {}) or {}
    meta_parts: List[str] = []
    for value in meta.values():
        if isinstance(value, (list, dict, tuple, set)):
            continue
        meta_parts.append(str(value or ""))
    meta_text = " ".join(meta_parts)
    page_text = str(getattr(d, "page_content", "") or "")
    return (meta_text + " " + page_text).lower()


def _filter_noisy_docs(docs: List[Any], question: str) -> List[Any]:
    deduped = _dedupe_docs(docs)
    if not deduped:
        return []

    tokens = _query_tokens_for_relevance(question)
    if not tokens:
        return deduped[: int(PIPELINE_CFG["max_docs_after_filter"])]

    kept: List[Any] = []
    for d in deduped:
        hay = _doc_haystack(d)
        if any(_token_in_hay(tok, hay) for tok in tokens):
            kept.append(d)

    filtered = kept if kept else deduped
    return filtered[: int(PIPELINE_CFG["max_docs_after_filter"])]


def _normalize_meta_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _metadata_key_allowed(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k:
        return False
    if k in {"chunk", "chunk_id", "id", "doc_id", "source", "path", "url"}:
        return False
    if k.endswith("_id") and k not in {"paper_id"}:
        return False
    return True


def _metadata_value_allowed(value: str) -> bool:
    v = _normalize_meta_value(value)
    if not v or len(v) < 3:
        return False
    if _is_placeholder_anchor_value(v):
        return False
    if re.fullmatch(r"[0-9\W]+", v):
        return False
    if re.fullmatch(r"(19|20)\d{2}", v):
        return False
    return True


def _iter_doc_metadata_key_values(d: Any) -> List[Tuple[str, str, str]]:
    meta = getattr(d, "metadata", {}) or {}
    out: List[Tuple[str, str, str]] = []
    for key, raw in meta.items():
        k = str(key or "").strip()
        if not k:
            continue
        if not _metadata_key_allowed(k):
            continue
        if isinstance(raw, (list, dict, tuple, set)):
            continue
        raw_text = re.sub(r"\s+", " ", str(raw or "").strip())
        v = _normalize_meta_value(raw_text)
        if not raw_text or not _metadata_value_allowed(raw_text):
            continue
        out.append((k, v, raw_text))
    return out


def _dominant_metadata_filter_from_docs(
    docs: List[Any],
    question: str,
    *,
    majority_ratio: float = 0.6,
    min_count: int = 3,
) -> Dict[str, Any]:
    _ = question
    result: Dict[str, Any] = {
        "dominant": False,
        "key": "",
        "value": "",
        "count": 0,
        "ratio": 0.0,
        "confidence": 0.0,
        "confidence_floor": float(getattr(settings, "dominant_min_confidence", 0.72)),
        "n_docs": len(docs or []),
        "runner_up_count": 0,
        "filter": {},
    }
    if not docs:
        return result
    n_docs = len(docs)
    counts: Dict[Tuple[str, str], int] = {}
    exemplars: Dict[Tuple[str, str], str] = {}
    for d in docs:
        seen_pairs = set()
        for key, norm_val, raw_val in _iter_doc_metadata_key_values(d):
            pair = (key, norm_val)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            counts[pair] = counts.get(pair, 0) + 1
            exemplars.setdefault(pair, raw_val)
    if not counts:
        return result

    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    (best_key, best_norm_val), best_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
    ratio = float(best_count) / max(1, n_docs)
    margin = float(best_count - runner_up_count) / max(1, n_docs)
    confidence = max(0.0, min(1.0, (0.7 * ratio) + (0.3 * max(0.0, margin))))
    result.update(
        {
            "key": best_key,
            "value": exemplars.get((best_key, best_norm_val), best_norm_val),
            "count": int(best_count),
            "ratio": ratio,
            "confidence": confidence,
            "runner_up_count": int(runner_up_count),
        }
    )
    conf_floor = float(getattr(settings, "dominant_min_confidence", 0.72))
    result["confidence_floor"] = conf_floor
    if best_count < max(1, min_count):
        return result
    if ratio < float(majority_ratio):
        return result
    if confidence < conf_floor:
        return result
    if _is_placeholder_anchor_value(str(result.get("value", "") or "")):
        return result
    result["dominant"] = True
    result["filter"] = {"key": result["key"], "value": result["value"]}
    return result


def _insufficient_context_answer(question: str, intent: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    _ = intent
    if q:
        return (
            f"I couldn't find enough matching evidence for \"{q}\" in the retrieved papers. "
            "Please clarify the entity, topic, or time range you want."
        )
    return "I couldn't find enough matching evidence in the retrieved papers. Please clarify the entity or topic you want."


def _uncertain_retrieval_answer(question: str, *, anchor_value: str = "", reason: str = "") -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    reason_text = (reason or "").strip().lower()
    if anchor and reason_text == "inconsistent":
        return (
            f"I'm not confident the retrieved evidence matches the current focus on \"{anchor}\". "
            f"Could you confirm whether you want to continue with \"{anchor}\" or switch topics for \"{q}\"?"
        )
    if q:
        return (
            f"I'm not confident there is enough consistent evidence yet for \"{q}\". "
            "Please add a specific entity, paper title, or year so I can answer accurately."
        )
    return "I'm not confident there is enough consistent evidence yet. Please add a specific entity, paper title, or year."


def _normalize_for_similarity(text: str) -> str:
    lowered = re.sub(r"\s+", " ", str(text or "").strip().lower())
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered


def _strip_leading_answer_labels(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(3):
        updated = re.sub(
            r"^\s*(summary|final summary|answer|response|final response)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _is_process_note_paragraph(paragraph: str) -> bool:
    lower = re.sub(r"\s+", " ", str(paragraph or "").strip().lower())
    if not lower.startswith("note:"):
        return False
    process_terms = (
        "pipeline",
        "retrieval",
        "context",
        "analysis",
        "synthesis",
        "confidence",
        "metadata",
        "prompt",
        "cache",
        "debug",
    )
    return any(term in lower for term in process_terms)


def _sanitize_user_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    blocked_markers = (
        "detected intent",
        "retrieval count",
        "pipeline",
        "chroma",
        "metadata filter",
        "session_id",
        "turn_count",
    )
    boilerplate_phrases = (
        "no further analysis is required",
        "no additional retrieval is required",
        "no further synthesis is required",
        "no additional analysis is required",
        "no further retrieval is required",
    )
    kept: List[str] = []
    for line in raw.splitlines():
        ll = line.strip().lower()
        if any(marker in ll for marker in blocked_markers):
            continue
        if any(phrase in ll for phrase in boilerplate_phrases):
            continue
        kept.append(line)
    cleaned = _strip_leading_answer_labels("\n".join(kept).strip())
    if not cleaned:
        cleaned = _strip_leading_answer_labels(raw)
    if not cleaned:
        return raw

    chunks = [c.strip() for c in re.split(r"\n\s*\n+", cleaned) if c and c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in cleaned.splitlines() if c and c.strip()]

    result: List[str] = []
    seen_norms: List[str] = []
    for chunk in chunks:
        para = _strip_leading_answer_labels(re.sub(r"\s+", " ", chunk).strip())
        if not para:
            continue
        lower = para.lower()
        if lower.startswith("the retrieved context"):
            continue
        if lower.startswith("confidence level"):
            continue
        if _is_process_note_paragraph(para):
            continue
        norm = _normalize_for_similarity(para)
        if not norm:
            continue
        duplicate = False
        for prev in seen_norms:
            if norm == prev:
                duplicate = True
                break
            if SequenceMatcher(None, norm, prev).ratio() >= 0.93:
                duplicate = True
                break
        if duplicate:
            continue
        seen_norms.append(norm)
        result.append(para)

    final_text = "\n\n".join(result).strip()
    return final_text if final_text else cleaned


def _build_anchor_from_dominance(dominance: Dict[str, Any]) -> Dict[str, Any]:
    if not dominance.get("dominant"):
        return {}
    key = re.sub(r"\s+", " ", str(dominance.get("key", "") or "").strip().lower())
    value = re.sub(r"\s+", " ", str(dominance.get("value", "") or "").strip())
    if not key or not value or _is_placeholder_anchor_value(value):
        return {}
    confidence = float(dominance.get("confidence", 0.0) or 0.0)
    conf_floor = float(getattr(settings, "dominant_min_confidence", 0.72))
    if confidence < conf_floor:
        return {}
    return {
        "type": key,
        "value": value,
        "source": f"dominant_metadata:{key}",
        "confidence": max(0.0, min(1.0, confidence)),
    }


def _choose_anchor_update(
    *,
    current_anchor: Dict[str, Any],
    candidate_anchor: Dict[str, Any],
    dominance: Dict[str, Any],
    question: str,
) -> Tuple[Dict[str, Any], str]:
    anchor_now = _normalize_anchor(current_anchor)
    candidate = _normalize_anchor(candidate_anchor)
    if not candidate:
        return anchor_now, "kept_no_dominance" if anchor_now else "none"
    if not anchor_now:
        return candidate, "set_from_dominance"

    same_type = _normalize_meta_value(anchor_now.get("type", "")) == _normalize_meta_value(candidate.get("type", ""))
    same_value = _normalize_meta_value(anchor_now.get("value", "")) == _normalize_meta_value(candidate.get("value", ""))
    if same_type and same_value:
        anchor_now["confidence"] = max(
            float(anchor_now.get("confidence", 0.0) or 0.0),
            float(candidate.get("confidence", 0.0) or 0.0),
        )
        anchor_now["source"] = candidate.get("source", anchor_now.get("source", "retrieval"))
        return anchor_now, "kept_reinforced"

    replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
    explicit_signal = _has_explicit_entity_signal(question)
    strong_ratio = float(dominance.get("ratio", 0.0) or 0.0) >= max(
        replace_conf,
        float(getattr(settings, "dominant_majority_ratio", 0.6)) + 0.15,
    )
    if float(candidate.get("confidence", 0.0) or 0.0) >= replace_conf and (explicit_signal or strong_ratio):
        return candidate, "replaced_with_strong_evidence"
    return anchor_now, "kept_ambiguous_switch_requires_confirmation"


def _retrieval_confidence_label(*, docs_count: int, anchor_consistent: bool) -> str:
    min_docs = max(1, int(getattr(settings, "retrieval_weak_min_docs", 3)))
    if docs_count < min_docs:
        return "weak"
    if not anchor_consistent:
        return "inconsistent"
    if docs_count >= max(8, min_docs * 2):
        return "high"
    if docs_count >= max(4, min_docs + 1):
        return "medium"
    return "low"


def _build_compact_context(
    docs: List[Any],
    max_docs: Optional[int] = None,
    text_limit: Optional[int] = None,
) -> str:
    if max_docs is None:
        max_docs = int(getattr(settings, "prompt_max_docs", 24))
    if text_limit is None:
        text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
    blocks: List[str] = []
    for d in docs[:max_docs]:
        j = _doc_to_json(d, text_limit=text_limit)
        blocks.append(
            "\n".join(
                [
                    f"Title: {j.get('title', '')}",
                    f"Authors: {j.get('authors', '')}",
                    f"Year: {j.get('year', '')}",
                    f"Snippet: {j.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _clip_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 320) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if not compact:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p and p.strip()]
    if parts:
        compact = " ".join(parts[: max(1, max_sentences)]).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _clean_assistant_turn_for_prompt(text: str) -> str:
    blocked_fragments = (
        "summary:",
        "no further analysis is required",
        "no additional retrieval is required",
        "no further synthesis is required",
        "the retrieved context",
        "confidence level",
    )
    kept_lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line or "").strip()
        if not line:
            continue
        lower = line.lower()
        if any(fragment in lower for fragment in blocked_fragments):
            continue
        kept_lines.append(line)
    compact = _strip_leading_answer_labels(" ".join(kept_lines).strip())
    return _clip_sentences(compact, max_sentences=2, max_chars=360)


def _extract_summary_sections_text(summary_text: str) -> Dict[str, List[str]]:
    aliases = {
        "focus": "Current focus",
        "current focus": "Current focus",
        "entities": "Core entities",
        "entities discussed": "Core entities",
        "core entities": "Core entities",
        "findings": "Key themes",
        "key findings so far": "Key themes",
        "key themes": "Key themes",
        "constraints": "Constraints",
        "open questions": "Open questions",
    }
    sections = {
        "Current focus": [],
        "Core entities": [],
        "Key themes": [],
        "Constraints": [],
        "Open questions": [],
    }
    current_key: Optional[str] = None
    for raw_line in str(summary_text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            canonical = aliases.get((m.group(1) or "").strip().lower())
            current_key = canonical
            if canonical:
                rest = (m.group(2) or "").strip().lstrip("- ").strip()
                if rest:
                    sections[canonical].append(rest)
            continue
        if current_key:
            sections[current_key].append(line.lstrip("- ").strip())
    return sections


def _rolling_summary_has_boilerplate(summary_text: str) -> bool:
    lower = str(summary_text or "").lower()
    markers = (
        "summary:",
        "no further analysis is required",
        "no additional retrieval is required",
        "no further synthesis is required",
        "the retrieved context",
        "confidence level",
    )
    return any(marker in lower for marker in markers)


def _format_summary_for_prompt(sections: Dict[str, List[str]], ordered_keys: List[str]) -> str:
    lines: List[str] = []
    for key in ordered_keys:
        vals = [
            re.sub(r"\s+", " ", str(v or "").strip())
            for v in (sections.get(key) or [])
            if str(v or "").strip() and str(v or "").strip() != "(none)"
        ]
        if not vals:
            continue
        lines.append(f"{key}: {' | '.join(vals)}")
    return "\n".join(lines).strip()


def _rolling_summary_for_prompt(summary_text: str) -> str:
    raw = str(summary_text or "").strip()
    if not raw:
        return ""
    sections = _extract_summary_sections_text(raw)
    if _rolling_summary_has_boilerplate(raw):
        filtered = _format_summary_for_prompt(
            sections,
            ["Current focus", "Core entities"],
        )
        return filtered
    return raw


def _build_recent_turns_context(state: Dict[str, Any], max_turns: int) -> str:
    recent_turns = state.get("recent_turns")
    if isinstance(recent_turns, list) and recent_turns:
        turns = list(recent_turns)
    else:
        turns = list(state.get("turns", []) or [])
    if max_turns <= 0 or not turns:
        return ""
    tail = turns[-max_turns:]
    rows: List[str] = []
    for t in tail:
        role = str(t.get("role", "") or "").strip().lower()
        text = str(t.get("text", "") or "").strip()
        if not role or not text:
            continue
        if role == "assistant":
            text = _clean_assistant_turn_for_prompt(text)
        else:
            text = _clip_sentences(text, max_sentences=2, max_chars=420)
        if not text:
            continue
        prefix = "User" if role == "user" else "Assistant"
        rows.append(f"{prefix}: {text}")
    return "\n".join(rows).strip()


def _last_user_turn_text_from_state(state: Dict[str, Any]) -> str:
    turns = list(state.get("turns", []) or [])
    for t in reversed(turns):
        role = str((t or {}).get("role", "") or "").strip().lower()
        text = str((t or {}).get("text", "") or "").strip()
        if role == "user" and text:
            return text
    return ""


def _fallback_retrieval_text(
    *,
    user_question: str,
    resolved_text: str,
    previous_user_text: str,
) -> str:
    q = re.sub(r"\s+", " ", (user_question or "").strip())
    resolved = re.sub(r"\s+", " ", (resolved_text or "").strip())
    prev = re.sub(r"\s+", " ", (previous_user_text or "").strip())
    if resolved:
        return resolved
    if not q:
        return ""
    if prev and prev.lower() != q.lower():
        return f"{prev} {q}".strip()
    return q


def _title_anchor(docs: List[Any]) -> str:
    for d in docs:
        title = str((getattr(d, "metadata", {}) or {}).get("title", "") or "").strip()
        if title and title.lower() != "untitled":
            return title
    return ""


def _year_range_from_docs(docs: List[Any]) -> str:
    years: List[int] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        raw = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
        m = re.search(r"\b(19|20)\d{2}\b", raw)
        if not m:
            continue
        try:
            years.append(int(m.group(0)))
        except Exception:
            continue
    if not years:
        return "Not clearly visible in the retrieved papers."
    lo = min(years)
    hi = max(years)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def _extract_key_themes_from_docs(docs: List[Any], max_items: int = 4) -> List[str]:
    topic_counter: Counter = Counter()
    token_counter: Counter = Counter()
    stopset = _get_stopword_set()
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        topic = str(meta.get("primary_topic", "") or "").strip()
        if topic and topic.lower() not in {"n/a", "na", "none", "unknown"}:
            topic_counter[topic] += 1
        title = str(meta.get("title", "") or "").lower()
        for tok in re.findall(r"[a-z0-9\-]{4,}", title):
            if stopset and tok in stopset:
                continue
            token_counter[tok] += 1
    themes: List[str] = [k for k, _v in topic_counter.most_common(max_items)]
    for tok, _v in token_counter.most_common(max_items * 2):
        if tok in themes:
            continue
        themes.append(tok)
        if len(themes) >= max_items:
            break
    return themes[:max_items]


def _representative_work_lines(docs: List[Any], max_items: int = 4) -> List[str]:
    lines: List[str] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        title = str(meta.get("title", "") or "").strip()
        if not title or title.lower() == "untitled":
            continue
        year = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
        lines.append(f"- {title} ({year})" if year else f"- {title}")
        if len(lines) >= max_items:
            break
    return lines


def _structured_general_fallback(question: str, docs: List[Any], intent: str) -> Optional[str]:
    if not docs:
        return None
    top_docs = list(docs[:12])
    themes = _extract_key_themes_from_docs(top_docs, max_items=4)
    works = _representative_work_lines(top_docs, max_items=4)
    time_range = _year_range_from_docs(top_docs)
    anchor = _title_anchor(top_docs)
    intent_key = (intent or "default").strip().lower()
    lead = "The retrieved papers provide the best-supported answer available from current evidence."
    if intent_key == "comparison":
        lead = "The retrieved papers support a comparison across related themes."
    elif intent_key == "list":
        lead = "The retrieved papers support a concise evidence-grounded list."

    lines: List[str] = [lead]
    if anchor:
        lines.append(f"Representative evidence includes papers such as \"{anchor}\".")

    if intent_key == "time_range":
        lines.append("The studied time period is not explicitly stated in the retrieved excerpts.")
        if time_range and "not clearly visible" not in time_range.lower():
            lines.append(f"Publication years in the retrieved papers: {time_range}.")
    else:
        if themes:
            lines.append("Main themes in the retrieved set include " + "; ".join(themes) + ".")
        if time_range and "not clearly visible" not in time_range.lower():
            lines.append(f"Publication years represented in retrieved papers: {time_range}.")

    work_items = [w.lstrip("- ").strip() for w in works if w and w.strip()]
    if not work_items and anchor:
        work_items = [anchor]
    if work_items:
        lines.append("Representative papers: " + "; ".join(work_items) + ".")

    return " ".join(lines).strip()


def _fallback_answer_from_docs(question: str, docs: List[Any]) -> Optional[str]:
    _ = question
    if not docs:
        return None
    lines: List[str] = []
    max_items = max(1, int(PIPELINE_CFG["fallback_max_items"]))
    for d in docs[:max_items]:
        meta = getattr(d, "metadata", {}) or {}
        title = str(meta.get("title", "") or "").strip()
        if not title or title.lower() == "untitled":
            continue
        year = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
        lines.append(f"{title} ({year})" if year else title)
    if lines:
        return PIPELINE_CFG["fallback_relevant_papers_prefix"] + "\n" + "\n".join(f"- {l}" for l in lines)
    return None


def answer_question(
    question: str,
    user_key: str,
    use_graph: Optional[bool] = None,
    stateless: Optional[bool] = None,
) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {
            "answer": PIPELINE_CFG["empty_question_answer"],
            "sources": [],
            "graph_hits": [],
            "graph_graph": {},
            "graph_error": "",
        }
    t0_total = time.perf_counter()
    generation_time_ms = 0.0
    answer_llm_calls = 0

    stateless = bool(stateless) if stateless is not None else bool(getattr(settings, "stateless_default", False))
    use_graph_flag = settings.use_graph if use_graph is None else bool(use_graph)
    previous_pipeline = get_pipeline_cache(user_key)
    mgr = get_global_manager()
    requested_mode = str(getattr(settings, "active_mode", "") or "").strip()
    effective_mode = mgr.dbm.resolve_mode(requested_mode)
    answer_model_key = str(getattr(settings, "answer_model_key", "") or settings.llm_model)

    pre_state = {"rolling_summary": "", "turns": []} if stateless else mgr.store.load(user_key)
    state_for_cache = pre_state if isinstance(pre_state, dict) else {}
    cache_state_sig = state_signature_from_state(state_for_cache)
    cache_lookup_key = build_pipeline_cache_key(
        user_key=user_key,
        resolved_text=q,
        effective_mode=effective_mode,
        state_signature=cache_state_sig,
    )
    if (not stateless) and PIPELINE_CFG["qa_cache_enable"]:
        cached = get_cached_answer(user_key, cache_lookup_key)
        if isinstance(cached, dict):
            return cached

    mgr.switch_mode(effective_mode)
    mgr.switch_answer_model(answer_model_key)

    eng = mgr.get_engine(user_key, mode=effective_mode, stateless=stateless)
    context = eng.prepare_context(q, stateless=stateless)
    context.raw_question = q

    paper_docs_raw = list(context.paper_docs or [])
    resolved_q = (context.rewritten_question or "").strip()
    prev_user_text = _last_user_turn_text_from_state(pre_state)
    resolved_q = _fallback_retrieval_text(
        user_question=q,
        resolved_text=resolved_q,
        previous_user_text=prev_user_text,
    )
    detected_intent = str(getattr(context, "detected_intent", "") or _classify_generic_intent(q))
    retrieval_q = resolved_q or q
    raw_retrieval_count = len(paper_docs_raw)

    first_pass_docs = _filter_noisy_docs(paper_docs_raw, retrieval_q)
    dominance = _dominant_metadata_filter_from_docs(
        first_pass_docs,
        resolved_q or q,
        majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
        min_count=int(getattr(settings, "dominant_min_count", 3)),
    )
    dominant_filter = dict(dominance.get("filter", {}) or {})
    dominance_conf = float(dominance.get("confidence", 0.0) or 0.0)
    dominant_filter_usable = bool(dominance.get("dominant")) and bool(dominant_filter)
    dominant_filter_usable = bool(
        dominant_filter_usable
        and dominance_conf >= float(getattr(settings, "dominant_replace_confidence", 0.82))
    )
    if dominant_filter_usable and _is_placeholder_anchor_value(str(dominant_filter.get("value", "") or "")):
        dominant_filter_usable = False
    paper_docs = list(first_pass_docs)
    dominant_second_pass_count = 0
    fallback_unfiltered_count = 0
    if dominant_filter_usable:
        where_filter = {str(dominant_filter.get("key", "")): str(dominant_filter.get("value", ""))}
        second_pass_docs = eng.retrieve_papers(
            retrieval_q,
            int((context.budgets or {}).get("BUDGET_PAPERS", getattr(settings, "budget_papers", 0))),
            query_embedding=None,
            where_filter=where_filter,
            raw_question=q,
        )
        second_pass_docs = _filter_noisy_docs(second_pass_docs, retrieval_q)
        dominant_second_pass_count = len(second_pass_docs)
        paper_docs = _dedupe_docs(paper_docs + second_pass_docs)
    post_filter_count = len(paper_docs)
    mem_docs = context.mem_docs or []

    anchor_before = _normalize_anchor(getattr(context, "anchor", {}) or getattr(eng, "anchor", {}))
    candidate_anchor = _build_anchor_from_dominance(dominance)
    anchor_after, anchor_action = _choose_anchor_update(
        current_anchor=anchor_before,
        candidate_anchor=candidate_anchor,
        dominance=dominance,
        question=q,
    )
    eng.anchor = dict(anchor_after or {})
    eng.anchor_last_action = anchor_action
    context.anchor = dict(anchor_after or {})
    context.paper_docs = list(paper_docs)

    anchor_value = str(anchor_after.get("value", "") or "").strip()
    anchor_support_ratio = _anchor_support_ratio(anchor_value, paper_docs) if anchor_value else 1.0
    anchor_consistent = (
        (not anchor_value)
        or (anchor_support_ratio >= float(getattr(settings, "anchor_consistency_min_ratio", 0.45)))
    )
    retrieval_confidence = _retrieval_confidence_label(
        docs_count=post_filter_count,
        anchor_consistent=anchor_consistent,
    )
    eng.last_retrieval_confidence = retrieval_confidence
    eng.last_anchor_support_ratio = float(anchor_support_ratio)
    weak_or_inconsistent = retrieval_confidence in {"weak", "inconsistent"}

    prompt_max_docs = int(getattr(settings, "prompt_max_docs", 24))
    prompt_text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
    if weak_or_inconsistent:
        prompt_max_docs = min(
            prompt_max_docs,
            max(2, int(getattr(settings, "low_conf_prompt_max_docs", 8))),
        )
        prompt_text_limit = min(
            prompt_text_limit,
            max(160, int(getattr(settings, "low_conf_prompt_doc_text_limit", 420))),
        )

    papers_ctx = (
        _build_compact_context(
            paper_docs,
            max_docs=prompt_max_docs,
            text_limit=prompt_text_limit,
        )
        if paper_docs
        else ""
    )

    user_query_json = {
        "text": q,
        "resolved_text": resolved_q,
        "standalone_question": resolved_q,
        "detected_intent": detected_intent,
        "retrieval_text": retrieval_q,
        "dominance": dominance,
        "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dominant_filter_usable,
        "anchor_before": anchor_before,
        "anchor_after": anchor_after,
        "anchor_action": anchor_action,
        "rewrite_blocked": bool(getattr(eng, "last_rewrite_blocked", False)),
        "rewrite_anchor_valid": bool(getattr(eng, "last_rewrite_anchor_valid", False)),
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "stateless": stateless,
        "timestamp": datetime.utcnow().isoformat(),
    }

    rolling_summary_text = pre_state.get("rolling_summary", "") or ""
    pre_turns_count = len(pre_state.get("turns", []) or [])
    pre_summary_len = len(rolling_summary_text)
    rolling_summary_json = {
        "summary": rolling_summary_text,
        "turns": len(pre_state.get("turns", []) or []),
    }
    recent_turns_ctx = _build_recent_turns_context(
        pre_state,
        max_turns=min(6, max(2, int(PIPELINE_CFG["recent_turns_in_prompt"]))),
    )

    chroma_doc_refs = [_doc_to_ref(d) for d in paper_docs[:12]]
    memory_doc_refs = [_doc_to_ref(d) for d in mem_docs[:8]]
    retrieval_summary_cache = retrieval_cache_summary(
        paper_docs,
        retrieval_text=retrieval_q,
        limit_ids=12,
    )

    chroma_json = {
        "count": len(paper_docs),
        "retrieval_count_raw": raw_retrieval_count,
        "first_pass_count": len(first_pass_docs),
        "dominant_second_pass_count": dominant_second_pass_count,
        "fallback_unfiltered_count": fallback_unfiltered_count,
        "post_filter_count": post_filter_count,
        "dominance": dominance,
        "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dominant_filter_usable,
        "anchor_support_ratio": round(float(anchor_support_ratio), 4),
        "anchor_consistent": bool(anchor_consistent),
        "retrieval_confidence": retrieval_confidence,
        "doc_refs": chroma_doc_refs,
        "retrieval_digest": retrieval_summary_cache,
    }
    conversation_memory_json = {
        "count": len(mem_docs),
        "doc_refs": memory_doc_refs,
    }

    cache_json = {
        "previous_pipeline_present": bool(previous_pipeline),
        "previous_pipeline_sig": short_hash(
            json.dumps(
                (previous_pipeline or {}).get("session_state", {}),
                ensure_ascii=False,
                sort_keys=True,
            )
            if isinstance(previous_pipeline, dict)
            else "",
            length=10,
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }

    prompt_sections: List[str] = [PIPELINE_CFG["prompt_prefix"].rstrip()]
    rolling_summary_prompt = _rolling_summary_for_prompt(rolling_summary_text)
    if rolling_summary_prompt:
        prompt_sections.append("ROLLING SUMMARY:\n" + rolling_summary_prompt)
    if recent_turns_ctx:
        prompt_sections.append("RECENT TURNS:\n" + recent_turns_ctx)
    context_blob = papers_ctx.strip()
    if context_blob:
        prompt_sections.append("RETRIEVED CONTEXT:\n" + context_blob)

    question_for_answer = resolved_q or q
    style_hint = "Give a direct evidence-grounded answer in plain prose."
    if detected_intent == "comparison":
        style_hint = "Compare the relevant items directly and keep claims grounded in retrieved evidence."
    elif detected_intent == "time_range":
        style_hint = (
            "If asked about the studied time period, report it only when explicitly stated in retrieved context. "
            "If the studied interval is not stated, say that directly and optionally provide publication years as a separate fallback. "
            "Do not infer studied period from publication years."
        )
    elif detected_intent == "list":
        style_hint = "Provide a concise list with short evidence-backed descriptors."
    prompt_sections.append("ANSWER POLICY:\n" + style_hint)
    prompt = (
        "\n\n".join(prompt_sections)
        + PIPELINE_CFG["prompt_mid"]
        + question_for_answer
        + PIPELINE_CFG["prompt_suffix"]
    )

    if not paper_docs:
        answer_text = _insufficient_context_answer(q, detected_intent)
    elif weak_or_inconsistent:
        answer_text = _uncertain_retrieval_answer(
            q,
            anchor_value=anchor_value,
            reason=retrieval_confidence,
        )
    else:
        t0_generation = time.perf_counter()
        answer_lock_ctx = nullcontext()
        if hasattr(mgr, "answer_generation_lock") and not bool(getattr(settings, "allow_utility_concurrency", False)):
            answer_lock_ctx = mgr.answer_generation_lock
        try:
            with answer_lock_ctx:
                answer_llm_calls += 1
                raw_answer = _invoke_with_timeout(
                    mgr.answer_runtime.llm,
                    prompt,
                    int(getattr(settings, "llm_timeout_s", 40)),
                )
        except Exception:
            raw_answer = ""
        generation_time_ms = (time.perf_counter() - t0_generation) * 1000.0
        raw_text = str(raw_answer or "")
        answer_text = _strip_prompt_leak(raw_text).strip()
        if not answer_text:
            for marker in ("ANSWER:", "FINAL RESPONSE:", "RESPONSE:"):
                if marker in raw_text:
                    answer_text = raw_text.split(marker)[-1].strip()
                    break
            if (not answer_text) and raw_text:
                answer_text = raw_text.strip()
        if not answer_text:
            answer_text = PIPELINE_CFG["llm_no_answer"]

    if paper_docs and (not weak_or_inconsistent):
        cleaned_answer = (answer_text or "").strip()
        needs_fallback = (
            not cleaned_answer
            or cleaned_answer == PIPELINE_CFG["llm_no_answer"]
            or len(cleaned_answer) < 24
        )
        if needs_fallback:
            fallback = _structured_general_fallback(q, paper_docs, detected_intent)
            if not fallback:
                fallback = _fallback_answer_from_docs(q, paper_docs)
            if fallback:
                answer_text = fallback

    answer_text = _sanitize_user_answer(answer_text).strip()
    if not answer_text:
        answer_text = PIPELINE_CFG["llm_no_answer"]

    eng.last_answer_llm_calls = int(answer_llm_calls)

    eng.finalize_turn(context, answer_text, no_results=not paper_docs)

    post_state = {"rolling_summary": rolling_summary_text, "turns": pre_state.get("turns", []) or []} if stateless else mgr.store.load(user_key)
    post_turns = post_state.get("turns", []) or []
    post_summary = post_state.get("rolling_summary", "") or ""
    post_extra = post_state.get("extra_state", {}) if isinstance(post_state.get("extra_state"), dict) else {}
    post_anchor = _normalize_anchor(post_state.get("anchor") or post_extra.get("anchor") or anchor_after)
    post_anchor_action = str(post_state.get("anchor_last_action") or post_extra.get("anchor_last_action") or anchor_action)
    post_summary_updated = bool(
        post_state.get("summary_updated")
        if "summary_updated" in post_state
        else post_extra.get("summary_updated", getattr(eng, "last_summary_updated", False))
    )
    post_retrieval_confidence = str(
        post_state.get("retrieval_confidence")
        or post_extra.get("retrieval_confidence")
        or retrieval_confidence
    )
    post_anchor_ratio = float(
        post_extra.get("anchor_support_ratio", getattr(eng, "last_anchor_support_ratio", anchor_support_ratio))
        or 0.0
    )
    rolling_summary_json = {
        "summary": post_summary,
        "turns": len(post_turns),
        "updated": post_summary_updated,
    }
    session_state_json = {
        "session_id": user_key,
        "pre_turn_count": pre_turns_count,
        "turn_count": len(post_turns),
        "turn_count_delta": len(post_turns) - pre_turns_count,
        "pre_summary_len": pre_summary_len,
        "summary_len": len(post_summary),
        "summary_len_delta": len(post_summary) - pre_summary_len,
        "summary_non_empty": bool(post_summary.strip()),
        "turns_json_chars": len(json.dumps(post_turns, ensure_ascii=False)),
        "detected_intent": detected_intent,
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "anchor": post_anchor,
        "anchor_action": post_anchor_action,
        "summary_updated": post_summary_updated,
        "retrieval_confidence": post_retrieval_confidence,
        "anchor_support_ratio": round(post_anchor_ratio, 4),
    }

    sources: List[str] = []
    for d in paper_docs:
        try:
            sources.append(_doc_to_source_md(d))
        except Exception:
            pass

    timing_json = {
        "rewrite_ms": round(float(getattr(eng, "last_rewrite_time_ms", 0.0) or 0.0), 2),
        "retrieval_total_ms": round(float(getattr(eng, "last_retrieval_time_ms", 0.0) or 0.0), 2),
        "generation_ms": round(float(generation_time_ms or 0.0), 2),
        "total_ms": round((time.perf_counter() - t0_total) * 1000.0, 2),
    }
    llm_calls_json = {
        "answer_llm_calls": int(getattr(eng, "last_answer_llm_calls", answer_llm_calls) or 0),
        "utility_llm_calls": int(getattr(eng, "last_utility_llm_calls", 0) or 0),
    }
    timing_json["llm_calls"] = llm_calls_json
    session_state_json["timing_ms"] = timing_json
    session_state_json["llm_calls"] = llm_calls_json

    combined_json = {
        "user_query": user_query_json,
        "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_summary_json,
        "conversation_memory": conversation_memory_json,
        "cache": cache_json,
        "session_state": session_state_json,
        "context_size_chars": len(context_blob),
        "timing_ms": timing_json,
        "llm_calls": llm_calls_json,
    }

    cacheable_turn = should_cache_turn(
        retrieval_text=retrieval_q,
        rewrite_blocked=bool(getattr(eng, "last_rewrite_blocked", False)),
    )

    pipeline_cache_payload = {
        "user_query": {
            "text": user_query_json.get("text", ""),
            "resolved_text": user_query_json.get("resolved_text", ""),
            "retrieval_text": user_query_json.get("retrieval_text", ""),
            "detected_intent": user_query_json.get("detected_intent", ""),
            "effective_mode": user_query_json.get("effective_mode", ""),
        },
        "chroma_retrieval": {
            "count": chroma_json.get("count", 0),
            "retrieval_count_raw": chroma_json.get("retrieval_count_raw", 0),
            "post_filter_count": chroma_json.get("post_filter_count", 0),
            "retrieval_confidence": chroma_json.get("retrieval_confidence", ""),
            "retrieval_digest": chroma_json.get("retrieval_digest", {}),
        },
        "rolling_summary": rolling_summary_json,
        "conversation_memory": {"count": conversation_memory_json.get("count", 0)},
        "cache": cache_json,
        "session_state": {
            "turn_count": session_state_json.get("turn_count", 0),
            "summary_len": session_state_json.get("summary_len", 0),
            "effective_mode": session_state_json.get("effective_mode", ""),
            "llm_calls": llm_calls_json,
        },
        "timing_ms": timing_json,
    }
    set_pipeline_cache(user_key, pipeline_cache_payload if cacheable_turn else {})

    if getattr(settings, "debug_rag", False):
        try:
            print("\n[PIPELINE_JSON]")
            print(json.dumps(combined_json, ensure_ascii=False, indent=2))
        except Exception:
            pass

    out: Dict[str, Any] = {
        "answer": answer_text,
        "sources": sources,
        "graph_hits": [],
        "graph_graph": {},
        "graph_error": "",
        "user_query": user_query_json,
        "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_summary_json,
        "conversation_memory": conversation_memory_json,
        "cache": cache_json,
        "session_state": session_state_json,
        "timing_ms": timing_json,
        "llm_calls": llm_calls_json,
        "pipeline_json": combined_json,
    }

    if use_graph_flag:
        g = graph_retrieve_from_paper_docs(paper_docs, height=650)
        out["graph_hits"] = g.get("hits", []) or []
        out["graph_graph"] = g.get("graph", {}) or {}
        out["graph_error"] = g.get("error", "") or ""

    if (not stateless) and PIPELINE_CFG["qa_cache_enable"] and (not use_graph_flag):
        cache_write_key = build_pipeline_cache_key(
            user_key=user_key,
            resolved_text=resolved_q or q,
            effective_mode=effective_mode,
            state_signature=cache_state_sig,
        )
        if cacheable_turn:
            cached_payload: Dict[str, Any] = {
                "answer": out.get("answer", ""),
                "sources": list(out.get("sources", []) or [])[:10],
                "graph_hits": [],
                "graph_graph": {},
                "graph_error": "",
                "user_query": pipeline_cache_payload.get("user_query", {}),
                "chroma_retrieval": pipeline_cache_payload.get("chroma_retrieval", {}),
                "rolling_summary": rolling_summary_json,
                "conversation_memory": {"count": conversation_memory_json.get("count", 0)},
                "cache": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "retrieval_digest": retrieval_summary_cache,
                    "state_sig": cache_state_sig,
                },
                "session_state": pipeline_cache_payload.get("session_state", {}),
                "timing_ms": timing_json,
                "llm_calls": llm_calls_json,
            }
            set_cached_answer(user_key, cache_write_key, cached_payload)

    return out
