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
    _is_followup_coref_question,
    _normalize_anchor,
    _anchor_support_ratio,
    _is_placeholder_anchor_value,
    _retrieval_confidence_label,
    _clean_snippet as _engine_clean_snippet,
    _normalize_title_case,
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
    "max_docs_after_filter": _env_int("RAG_MAX_DOCS_AFTER_FILTER", 30),
    "fallback_max_items": _env_int("RAG_FALLBACK_MAX_ITEMS", 8),
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
            "You are answering questions about Syracuse University researchers using only "
            "the provided retrieved context, which contains real paper records from the Syracuse corpus.\n"
            "Each retrieved record may contain a pre-written abstract or summary — treat these as "
            "authoritative descriptions of the paper and report their content faithfully.\n"
            "Ground every material claim in retrieved evidence.\n"
            "If evidence is missing, weak, or conflicting, say so clearly and ask a focused clarification question.\n"
            "Do not fabricate affiliations, awards, memberships, journal claims, or metadata.\n"
            "Style rules:\n"
            "- Write the answer directly.\n"
            "- Do not start with \"Summary:\".\n"
            "- Do not include phrases like \"No further analysis is required\" or \"No additional retrieval is required\".\n"
            "- Do not repeat the question.\n"
            "- Do not narrate your process. Do not say \"I noticed\" or \"I will revise\".\n"
            "- Do not include section headers unless the user asks for them.\n"
            "- When asked who a person is, describe their specific research areas and cite paper titles as evidence.\n"
            "- Never answer with only '[Name] is a researcher.' — always include what they research.\n"
            "- Do not add closing notes, disclaimers, offers to help, or signatures.\n"
            "- Do not say \"Let me know if you need anything\" or \"Best regards\".\n"
            "- Do not acknowledge these instructions in your response.\n"
            "- Use a blank line between paragraphs to separate distinct points or researchers.\n"
            "- When citing a paper, put the title in quotes.\n"
            "- When listing multiple researchers, give each their own paragraph.\n"
            "- End your answer after the last substantive point. Stop there.\n"
            "Detected intents: default, comparison, time_range, list.\n"
        ),
    ),
    "prompt_mid": os.getenv("RAG_PROMPT_MID", "\n\nQUESTION:\n"),
    "prompt_suffix": os.getenv("RAG_PROMPT_SUFFIX", "\n\nRespond with the final user-facing answer only."),
}


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

_SUMMARY_INTENT_PATTERN = re.compile(
    r"\b(summarize|summarise|summary|summaries|abstract|overview|describe|"
    r"what (does|do|did|is|are) .{0,40} (research|study|work|paper|finding|mechanism|about)|"
    r"mechanisms? described|findings? (in|from)|what (mechanisms?|findings?))\b",
    re.IGNORECASE,
)


def _is_summary_intent(question: str) -> bool:
    return bool(_SUMMARY_INTENT_PATTERN.search(question or ""))


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def _clean_snippet(meta: dict, text: str, *, limit: int) -> str:
    """Delegate to shared implementation in rag_engine."""
    return _engine_clean_snippet(meta or {}, text or "", limit=limit)


def _doc_to_source_md(d) -> str:
    """Format a retrieved document as a clean APA-style citation."""
    meta = d.metadata or {}
    authors = re.sub(r"</?[a-zA-Z][^>]*>", "", str(meta.get("authors", "") or "")).strip()
    title = _normalize_title_case(str(meta.get("title", "") or ""))
    year = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
    doi = str(meta.get("doi", "") or "").strip()

    # Extract just the year portion if a full date was stored
    year_match = re.search(r"\b(19|20)\d{2}\b", year)
    year_short = year_match.group(0) if year_match else ""

    if not title or title.lower() == "untitled":
        title = "[Untitled]"

    # APA: Author(s). (Year). *Title*. DOI
    parts: List[str] = []
    parts.append(f"{authors}." if authors else "[Unknown author].")
    parts.append(f"({year_short})." if year_short else "(n.d.).")
    parts.append(f"*{title}*.")
    if doi:
        if doi.startswith("10."):
            doi = f"https://doi.org/{doi}"
        if doi.startswith("http"):
            parts.append(doi)

    return " ".join(parts)


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
    # Strip institutional terms that aren't in paper content
    q = re.sub(
        r"\b(syracuse\s+university|syracuse|university|faculty|professor|"
        r"researcher|department|college|school|campus|institute)\b",
        "", q, flags=re.IGNORECASE,
    )
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return []
    stopset = _get_stopword_set()
    min_len = int(getattr(settings, "retrieval_keyword_min_term_len", 3))
    out: List[str] = []
    seen = set()
    for t in re.findall(r"[a-z0-9\-]{2,}", q):
        if len(t) < max(1, min_len) or (stopset and t in stopset) or t in seen:
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
        key = str(meta.get("paper_id", "")) + "::" + str(
            meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _doc_haystack(d: Any) -> str:
    meta = getattr(d, "metadata", {}) or {}
    meta_parts = [str(v or "") for v in meta.values() if not isinstance(v, (list, dict, tuple, set))]
    return (" ".join(meta_parts) + " " + str(getattr(d, "page_content", "") or "")).lower()


def _filter_noisy_docs(docs: List[Any], question: str) -> List[Any]:
    deduped = _dedupe_docs(docs)
    if not deduped:
        return []
    # Use the cleaned token list (institutional terms already stripped)
    tokens = _query_tokens_for_relevance(question)
    if not tokens:
        return deduped[: int(PIPELINE_CFG["max_docs_after_filter"])]
    kept = [d for d in deduped if any(_token_in_hay(tok, _doc_haystack(d)) for tok in tokens)]
    return (kept if kept else deduped)[: int(PIPELINE_CFG["max_docs_after_filter"])]


def _normalize_meta_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _metadata_key_allowed(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k or k in {"chunk", "chunk_id", "id", "doc_id", "source", "path", "url"}:
        return False
    if k.endswith("_id") and k not in {"paper_id"}:
        return False
    return True


def _metadata_value_allowed(value: str) -> bool:
    v = _normalize_meta_value(value)
    if not v or len(v) < 3 or _is_placeholder_anchor_value(v):
        return False
    if re.fullmatch(r"[0-9\W]+", v) or re.fullmatch(r"(19|20)\d{2}", v):
        return False
    return True


def _iter_doc_metadata_key_values(d: Any) -> List[Tuple[str, str, str]]:
    meta = getattr(d, "metadata", {}) or {}
    out: List[Tuple[str, str, str]] = []
    for key, raw in meta.items():
        k = str(key or "").strip()
        if not k or not _metadata_key_allowed(k) or isinstance(raw, (list, dict, tuple, set)):
            continue
        raw_text = re.sub(r"\s+", " ", str(raw or "").strip())
        v = _normalize_meta_value(raw_text)
        if raw_text and _metadata_value_allowed(raw_text):
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
        "dominant": False, "key": "", "value": "", "count": 0,
        "ratio": 0.0, "confidence": 0.0,
        "confidence_floor": float(getattr(settings, "dominant_min_confidence", 0.72)),
        "n_docs": len(docs or []), "runner_up_count": 0, "filter": {},
    }
    if not docs:
        return result
    n_docs = len(docs)
    counts: Dict[Tuple[str, str], int] = {}
    exemplars: Dict[Tuple[str, str], str] = {}
    for d in docs:
        seen_pairs: set = set()
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
    runner_up_count = next(
        (cnt for (_k, nv), cnt in ranked[1:] if nv != best_norm_val), 0
    )
    ratio = float(best_count) / max(1, n_docs)
    margin = float(best_count - runner_up_count) / max(1, n_docs)
    confidence = max(0.0, min(1.0, (0.85 * ratio) + (0.15 * max(0.0, margin))))
    if ratio >= 0.8 and best_count >= max(4, min_count):
        confidence = max(confidence, min(0.98, ratio))

    result.update({
        "key": best_key,
        "value": exemplars.get((best_key, best_norm_val), best_norm_val),
        "count": int(best_count),
        "ratio": ratio,
        "confidence": confidence,
        "runner_up_count": int(runner_up_count),
    })
    conf_floor = float(getattr(settings, "dominant_min_confidence", 0.72))
    result["confidence_floor"] = conf_floor

    if (best_count >= max(1, min_count)
            and ratio >= float(majority_ratio)
            and confidence >= conf_floor
            and not _is_placeholder_anchor_value(str(result.get("value", "") or ""))):
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
    return "I couldn't find enough matching evidence. Please clarify the entity or topic you want."


def _uncertain_retrieval_answer(question: str, *, anchor_value: str = "", reason: str = "") -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    if anchor and (reason or "").strip().lower() == "inconsistent":
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
    return re.sub(r"[^a-z0-9\s]", "", lowered)


def _dedupe_repeated_sentences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    kept = []
    last_norms: List[str] = []
    for p in re.split(r"(?<=[.!?])\s+", raw):
        sent = p.strip()
        if not sent:
            continue
        norm = _normalize_for_similarity(sent)
        if norm and norm in last_norms:
            continue
        kept.append(sent)
        last_norms = (last_norms + [norm])[-6:]
    return " ".join(kept).strip()


def _strip_leading_answer_labels(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(3):
        updated = re.sub(
            r"^\s*(summary|final summary|answer|response|final response|the final answer is)\s*:\s*",
            "", cleaned, flags=re.IGNORECASE,
        ).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _is_process_note_paragraph(paragraph: str) -> bool:
    lower = re.sub(r"\s+", " ", str(paragraph or "").strip().lower())
    if not lower.startswith("note:"):
        return False
    return any(t in lower for t in ("pipeline", "retrieval", "context", "analysis", "synthesis",
                                     "confidence", "metadata", "prompt", "cache", "debug",
                                     "reformatted", "revised", "reformat", "response format",
                                     "required format", "requested format"))


def _is_assistant_closure_sentence(sentence: str) -> bool:
    lower = re.sub(r"\s+", " ", str(sentence or "").strip().lower())
    if re.match(r"^note:\s.*(format|revised|reformatted|response|answer)", lower):
        return True
    if re.match(r"^(also,?\s*)?(i can help|let me know|best regards|please let me know)", lower):
        return True
    if re.match(r"^(however,?\s*)?i noticed that the response", lower):
        return True
    if re.match(r"^i will revise the response", lower):
        return True
    return any(m in lower for m in (
        "please let me know if you need", "if you need any further assistance",
        "i can help with anything else", "let me know if you need anything else",
        "let me know if you have any further requests", "i have revised the answer",
        "i've revised the response", "i made some minor changes",
        "please provide the next question", "i am ready to help",
        "i'm ready to help", "i am ready when you are",
        "let me know if you would like me to proceed",
        "let me know if you'd like me to proceed",
        "i'm here to assist you", "i'm here to help",
        "best regards", "[assistant]",
        "i can help with the next question",
        "i will revise the response to only include",
        "i noticed that the response contains some extraneous",
        "to only include information present in the retrieved",
    ))


def _remove_closure_sentences(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    kept = [p for p in parts if not _is_assistant_closure_sentence(p)]
    return " ".join(kept).strip() if kept else ""


def _trim_incomplete_tail_sentence(text: str) -> str:
    raw = str(text or "").strip()
    if not raw or re.search(r"[.!?\"')\]]\s*$", raw):
        return raw
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", raw) if p.strip()]
    if len(parts) <= 1:
        return raw
    tail = parts[-1]
    if re.match(r"^[-*]|\d+\.", tail):
        return raw
    if len(re.findall(r"[A-Za-z0-9]+", tail)) <= 14:
        trimmed = " ".join(parts[:-1]).strip()
        return trimmed if trimmed else raw
    return raw


def _strip_self_referential_notes(text: str) -> str:
    """Remove trailing 'Note: ...' sentences that reference formatting/structure."""
    _SELF_REF_TERMS = (
        "reformatted",
        "required response format",
        "response format",
        "answer format",
        "the answer has been",
        "this answer has been",
    )
    lines = text.splitlines()
    kept = []
    for line in lines:
        lower = line.strip().lower()
        if lower.startswith("note:") and any(t in lower for t in _SELF_REF_TERMS):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _sanitize_user_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    blocked_markers = (
        "detected intent", "retrieval count", "pipeline", "chroma",
        "metadata filter", "session_id", "turn_count",
        "please provide the answer in the requested format",
        "note: the user-facing answer", "the final answer is",
    )
    boilerplate_phrases = (
        "no further analysis is required", "no additional retrieval is required",
        "no further synthesis is required", "no additional analysis is required",
        "no further retrieval is required",
        "please let me know if you would like me to revise",
        "i have revised the answer", "i've revised the response",
        "i made some minor changes", "let me know if you have any further requests",
        "please let me know if you need any further assistance",
        "please let me know if you need anything else",
        "if you need any further assistance", "i can help with anything else",
        "let me know if you need anything else", "please provide the next question",
        "i am ready to help", "i'm ready to help", "i am ready when you are",
        "the original question was not provided", "i will follow the provided style rules",
        "session diagnostics", "llm calls this turn", "retrieved sources",
        "note: the answer has been reformatted",
        "has been reformatted to",
        "to better match the required response format",
        "to match the required response format",
        "reformatted to better match",
        # Self-revision patterns where LLM critiques then repeats itself
        "however, i noticed that the response contains",
        "i will revise the response to only include",
        "to only include information present in the retrieved",
        "let me know if you'd like me to proceed",
        "let me know if you would like me to proceed",
        "i can help with the next question",
        "i'm here to assist you", "i'm here to help you",
        "best regards",
    )
    kept = [
        line for line in raw.splitlines()
        if not any(m in line.strip().lower() for m in blocked_markers)
        and not any(p in line.strip().lower() for p in boilerplate_phrases)
    ]
    cleaned = _strip_leading_answer_labels("\n".join(kept).strip()) or _strip_leading_answer_labels(raw)
    if not cleaned:
        return raw

    chunks = [c.strip() for c in re.split(r"\n\s*\n+", cleaned) if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in cleaned.splitlines() if c.strip()]

    result: List[str] = []
    seen_norms: List[str] = []
    for chunk in chunks:
        para = _strip_leading_answer_labels(re.sub(r"\s+", " ", chunk).strip())
        if not para:
            continue
        lower = para.lower()
        if lower.startswith("the retrieved context") or lower.startswith("confidence level"):
            continue
        if _is_process_note_paragraph(para):
            continue
        norm = _normalize_for_similarity(para)
        if not norm:
            continue
        if any(norm == p or SequenceMatcher(None, norm, p).ratio() >= 0.80 for p in seen_norms):
            continue
        seen_norms.append(norm)
        result.append(para)

    final_text = _strip_self_referential_notes(
        _trim_incomplete_tail_sentence(
            _remove_closure_sentences(
                _dedupe_repeated_sentences("\n\n".join(result).strip())
            )
        )
    )
    if final_text:
        return final_text
    return _strip_self_referential_notes(
        _trim_incomplete_tail_sentence(_remove_closure_sentences(cleaned))
    ) or cleaned


def sanitize_answer_for_display(text: str) -> str:
    cleaned = _sanitize_user_answer(text)
    # Strip residual HTML/XML tags from paper metadata (e.g. <scp>, <i>, <b>)
    cleaned = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned)
    # Collapse runs of whitespace but preserve intentional paragraph breaks
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    # If the LLM already produced paragraph breaks, respect them
    if "\n\n" in cleaned:
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # --- Force paragraph breaks since LLM produced a wall of text ---

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) <= 3:
        return cleaned.strip()

    # Patterns that signal a new topic/researcher paragraph should start
    _NEW_BLOCK_PATTERNS = re.compile(
        r"^("
        # Name + verb: "Shahar Sukenik has...", "Carlos A. Castañeda's..."
        r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:'s)?\s+"
        # Transition words
        r"|Additionally[,\s]"
        r"|Furthermore[,\s]"
        r"|Moreover[,\s]"
        r"|Another\s+researcher"
        r"|In\s+addition[,\s]"
        r"|Separately[,\s]"
        r"|Meanwhile[,\s]"
        r"|In\s+contrast[,\s]"
        r"|His\s+research\s+also"
        r"|Her\s+research\s+also"
        r"|Their\s+research\s+also"
        r"|While\s+(?:his|her|their|none|not)"
        r"|Although\s"
        r")",
        re.IGNORECASE,
    )

    paragraphs: List[str] = []
    current: List[str] = []

    for sent in sentences:
        # Start a new paragraph if sentence matches a block-start pattern
        # AND we already have at least 2 sentences in the current paragraph
        if current and len(current) >= 2 and _NEW_BLOCK_PATTERNS.match(sent):
            paragraphs.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
            # Also break every 4 sentences as a fallback for readability
            if len(current) >= 4:
                paragraphs.append(" ".join(current))
                current = []

    if current:
        paragraphs.append(" ".join(current))

    cleaned = "\n\n".join(paragraphs)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_anchor_from_dominance(dominance: Dict[str, Any]) -> Dict[str, Any]:
    if not dominance.get("dominant"):
        return {}
    key = re.sub(r"\s+", " ", str(dominance.get("key", "") or "").strip().lower())
    value = re.sub(r"\s+", " ", str(dominance.get("value", "") or "").strip())
    if not key or not value or _is_placeholder_anchor_value(value):
        return {}
    confidence = float(dominance.get("confidence", 0.0) or 0.0)
    if confidence < float(getattr(settings, "dominant_min_confidence", 0.72)):
        return {}
    return {
        "type": key, "value": value,
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
        replace_conf, float(getattr(settings, "dominant_majority_ratio", 0.6)) + 0.15,
    )
    if float(candidate.get("confidence", 0.0) or 0.0) >= replace_conf and (explicit_signal or strong_ratio):
        return candidate, "replaced_with_strong_evidence"
    return anchor_now, "kept_ambiguous_switch_requires_confirmation"


def _build_compact_context(
    docs: List[Any],
    max_docs: Optional[int] = None,
    text_limit: Optional[int] = None,
) -> str:
    def _clean_for_llm(s: str) -> str:
        """Strip HTML and lowercase for cleaner LLM input."""
        return re.sub(r"</?[a-zA-Z][^>]*>", "", str(s or "")).lower().strip()

    if max_docs is None:
        max_docs = int(getattr(settings, "prompt_max_docs", 24))
    if text_limit is None:
        text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
    blocks: List[str] = []
    for d in docs[:max_docs]:
        j = _doc_to_json(d, text_limit=text_limit)
        blocks.append("\n".join([
            f"title: {_clean_for_llm(j.get('title', ''))}",
            f"authors: {_clean_for_llm(j.get('authors', ''))}",
            f"year: {j.get('year', '')}",
            f"snippet: {_clean_for_llm(j.get('text', ''))}",
        ]))
    return "\n\n".join(blocks)


def _extract_answer_text(raw_answer: Any) -> str:
    raw_text = str(raw_answer or "")
    answer_text = _strip_prompt_leak(raw_text).strip()
    if answer_text:
        return answer_text
    for marker in ("ANSWER:", "FINAL RESPONSE:", "RESPONSE:"):
        if marker in raw_text:
            cand = raw_text.split(marker)[-1].strip()
            if cand:
                return cand
    return raw_text.strip()


def _runtime_prompt_token_budget(runtime: Any, reserved_new_tokens: int) -> int:
    if runtime is None:
        return 0
    max_ctx = 0
    model = getattr(runtime, "model", None)
    tokenizer = getattr(runtime, "tokenizer", None)
    try:
        max_ctx = int(getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0)
    except Exception:
        max_ctx = 0
    if max_ctx <= 0:
        try:
            max_ctx = int(getattr(tokenizer, "model_max_length", 0) or 0)
        except Exception:
            max_ctx = 0
    if max_ctx <= 0 or max_ctx > 65536:
        max_ctx = 4096
    return max(256, max_ctx - max(64, int(reserved_new_tokens) + 24))


def _compose_answer_prompt(
    *,
    base_sections: List[str],
    style_hint: str,
    question_for_answer: str,
    context_blob: str,
) -> str:
    prompt_sections = list(base_sections)
    context = (context_blob or "").strip()
    if context:
        prompt_sections.append("RETRIEVED CONTEXT:\n" + context)
    prompt_sections.append("ANSWER POLICY:\n" + style_hint)
    return (
        "\n\n".join(prompt_sections)
        + PIPELINE_CFG["prompt_mid"]
        + (question_for_answer or "")
        + PIPELINE_CFG["prompt_suffix"]
    )


def _fit_prompt_to_budget(
    *,
    runtime: Any,
    docs: List[Any],
    base_sections: List[str],
    style_hint: str,
    question_for_answer: str,
    max_docs: int,
    text_limit: int,
    min_docs: int = 1,
    min_text_limit: int = 120,
) -> Tuple[str, int, int]:
    docs_cap = max(1, int(max_docs))
    text_cap = max(96, int(text_limit))
    min_docs = max(1, int(min_docs))
    min_text_limit = max(96, int(min_text_limit))
    token_budget = _runtime_prompt_token_budget(
        runtime, reserved_new_tokens=int(getattr(settings, "answer_max_new_tokens", 384)),
    )

    if not docs:
        return _compose_answer_prompt(
            base_sections=base_sections, style_hint=style_hint,
            question_for_answer=question_for_answer, context_blob="",
        ), docs_cap, text_cap

    shrink_docs_next = True
    prompt = _compose_answer_prompt(
        base_sections=base_sections, style_hint=style_hint,
        question_for_answer=question_for_answer,
        context_blob=_build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap),
    )
    for _ in range(28):
        if token_budget <= 0 or runtime is None:
            return prompt, docs_cap, text_cap
        try:
            tok_count = int(runtime.count_tokens(prompt))
        except Exception:
            return prompt, docs_cap, text_cap
        if tok_count <= token_budget:
            return prompt, docs_cap, text_cap

        prev_pair = (docs_cap, text_cap)
        if shrink_docs_next and docs_cap > min_docs:
            docs_cap = max(min_docs, docs_cap - max(1, docs_cap // 4))
        elif text_cap > min_text_limit:
            text_cap = max(min_text_limit, text_cap - max(24, text_cap // 5))
        elif docs_cap > 1:
            docs_cap -= 1
        elif text_cap > 96:
            text_cap = max(96, text_cap - 16)
        else:
            return prompt, docs_cap, text_cap

        shrink_docs_next = not shrink_docs_next
        if (docs_cap, text_cap) == prev_pair:
            return prompt, docs_cap, text_cap
        prompt = _compose_answer_prompt(
            base_sections=base_sections, style_hint=style_hint,
            question_for_answer=question_for_answer,
            context_blob=_build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap),
        )

    return prompt, docs_cap, text_cap


def _clip_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 320) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if not compact:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p.strip()]
    if parts:
        compact = " ".join(parts[: max(1, max_sentences)]).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact


def _clean_assistant_turn_for_prompt(text: str) -> str:
    blocked = (
        "summary:", "no further analysis is required",
        "no additional retrieval is required", "no further synthesis is required",
        "the retrieved context", "confidence level",
        "please let me know if you need", "if you need any further assistance",
        "i can help with anything else", "let me know if you need anything else",
        "i have revised the answer", "i've revised the response",
    )
    kept = [
        line for line in str(text or "").splitlines()
        if not any(b in re.sub(r"\s+", " ", line or "").strip().lower() for b in blocked)
    ]
    return _clip_sentences(_strip_leading_answer_labels(" ".join(kept).strip()), max_sentences=2, max_chars=360)


def _extract_summary_sections_text(summary_text: str) -> Dict[str, List[str]]:
    aliases = {
        "focus": "Current focus", "current focus": "Current focus",
        "entities": "Core entities", "entities discussed": "Core entities",
        "core entities": "Core entities", "findings": "Key themes",
        "key findings so far": "Key themes", "key themes": "Key themes",
        "constraints": "Constraints", "open questions": "Open questions",
    }
    sections: Dict[str, List[str]] = {k: [] for k in aliases.values()}
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
    return any(m in lower for m in (
        "summary:", "no further analysis is required", "no additional retrieval is required",
        "no further synthesis is required", "the retrieved context", "confidence level",
    ))


def _format_summary_for_prompt(sections: Dict[str, List[str]], ordered_keys: List[str]) -> str:
    lines: List[str] = []
    for key in ordered_keys:
        vals = [
            re.sub(r"\s+", " ", str(v or "").strip())
            for v in (sections.get(key) or [])
            if str(v or "").strip() and str(v or "").strip() != "(none)"
        ]
        if vals:
            lines.append(f"{key}: {' | '.join(vals)}")
    return "\n".join(lines).strip()


def _rolling_summary_for_prompt(summary_text: str) -> str:
    raw = str(summary_text or "").strip()
    if not raw:
        return ""
    sections = _extract_summary_sections_text(raw)
    if _rolling_summary_has_boilerplate(raw):
        return _format_summary_for_prompt(sections, ["Current focus", "Core entities"])
    return raw


def _build_recent_turns_context(state: Dict[str, Any], max_turns: int) -> str:
    turns = list(state.get("recent_turns") or state.get("turns", []) or [])
    if max_turns <= 0 or not turns:
        return ""
    rows: List[str] = []
    for t in turns[-max_turns:]:
        role = str(t.get("role", "") or "").strip().lower()
        text = str(t.get("text", "") or "").strip()
        if not role or not text:
            continue
        text = _clean_assistant_turn_for_prompt(text) if role == "assistant" else _clip_sentences(text, max_sentences=2, max_chars=420)
        if text:
            rows.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(rows).strip()


def _last_user_turn_text_from_state(state: Dict[str, Any]) -> str:
    for t in reversed(list(state.get("turns", []) or [])):
        role = str((t or {}).get("role", "") or "").strip().lower()
        text = str((t or {}).get("text", "") or "").strip()
        if role == "user" and text:
            return text
    return ""


def _fallback_retrieval_text(*, user_question: str, resolved_text: str, previous_user_text: str) -> str:
    """Return the best retrieval query text.

    Previously this prepended the previous user turn for referential queries,
    but that created garbage composite queries like "now list the faculty
    Summarize the mechanisms described in their papers".  The rolling summary
    already captures prior context, so we just use the resolved text or the
    raw question as-is.
    """
    q = re.sub(r"\s+", " ", (user_question or "").strip())
    resolved = re.sub(r"\s+", " ", (resolved_text or "").strip())
    return resolved or q or ""


def _year_range_from_docs(docs: List[Any]) -> str:
    years: List[int] = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        raw = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
        m = re.search(r"\b(19|20)\d{2}\b", raw)
        if m:
            try:
                years.append(int(m.group(0)))
            except Exception:
                pass
    if not years:
        return "Not clearly visible in the retrieved papers."
    lo, hi = min(years), max(years)
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


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
    works = _representative_work_lines(top_docs, max_items=4)
    time_range = _year_range_from_docs(top_docs)
    intent_key = (intent or "default").strip().lower()
    lead = "I couldn't generate a complete narrative answer, but these retrieved papers are the strongest evidence."
    if intent_key == "comparison":
        lead = "I couldn't generate a full comparison, but these retrieved papers are most relevant to compare."
    elif intent_key == "list":
        lead = "I couldn't generate a full list answer, but these retrieved papers are most relevant."

    lines: List[str] = [lead]
    if time_range and "not clearly visible" not in time_range.lower():
        lines.append(f"Publication years represented: {time_range}.")
    work_items = [w.lstrip("- ").strip() for w in works if w.strip()]
    if work_items:
        lines.append("Most relevant papers: " + "; ".join(work_items) + ".")
    return " ".join(lines).strip()


def _fallback_answer_from_docs(question: str, docs: List[Any]) -> Optional[str]:
    _ = question
    if not docs:
        return None
    max_items = max(1, int(PIPELINE_CFG["fallback_max_items"]))
    lines: List[str] = []
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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
            "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": "",
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
        user_key=user_key, resolved_text=q,
        effective_mode=effective_mode, state_signature=cache_state_sig,
    )
    if (not stateless) and PIPELINE_CFG["qa_cache_enable"]:
        cached = get_cached_answer(user_key, cache_lookup_key)
        if isinstance(cached, dict):
            return cached

    mgr.switch_mode(effective_mode)
    mgr.switch_answer_model(answer_model_key)


    # Acquire the generation lock for the full engine-stateful section
    # (prepare_context through finalize_turn) to prevent concurrent
    # requests for the same session from racing on Engine mutable state.
    engine_lock_ctx = nullcontext()
    if hasattr(mgr, "answer_generation_lock") and not bool(getattr(settings, "allow_utility_concurrency", False)):
        engine_lock_ctx = mgr.answer_generation_lock

    with engine_lock_ctx:
        eng = mgr.get_engine(user_key, mode=effective_mode, stateless=stateless)
        context = eng.prepare_context(q, stateless=stateless)
        context.raw_question = q

        paper_docs_raw = list(context.paper_docs or [])
        resolved_q = (context.rewritten_question or "").strip()
        prev_user_text = _last_user_turn_text_from_state(pre_state)
        resolved_q = _fallback_retrieval_text(
            user_question=q, resolved_text=resolved_q, previous_user_text=prev_user_text,
        )
        summary_intent = _is_summary_intent(q)
        detected_intent = str(getattr(context, "detected_intent", "") or _classify_generic_intent(q))
        retrieval_q = resolved_q or q
        raw_retrieval_count = len(paper_docs_raw)

        first_pass_docs = _filter_noisy_docs(paper_docs_raw, retrieval_q)
        dominance = _dominant_metadata_filter_from_docs(
            first_pass_docs, resolved_q or q,
            majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
            min_count=int(getattr(settings, "dominant_min_count", 3)),
        )
        dominant_filter = dict(dominance.get("filter", {}) or {})
        dominance_conf = float(dominance.get("confidence", 0.0) or 0.0)
        dominance_ratio = float(dominance.get("ratio", 0.0) or 0.0)
        replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
        majority_ratio = float(getattr(settings, "dominant_majority_ratio", 0.6))
        strong_ratio_floor = min(0.95, max(replace_conf + 0.02, majority_ratio + 0.22))
        dominant_filter_usable = (
            bool(dominance.get("dominant")) and bool(dominant_filter)
            and (dominance_conf >= replace_conf or dominance_ratio >= strong_ratio_floor)
            and not _is_placeholder_anchor_value(str(dominant_filter.get("value", "") or ""))
        )

        paper_docs = list(first_pass_docs)
        dominant_second_pass_count = 0
        if dominant_filter_usable:
            where_filter = {str(dominant_filter.get("key", "")): str(dominant_filter.get("value", ""))}
            second_pass_docs = eng.retrieve_papers(
                retrieval_q,
                int((context.budgets or {}).get("BUDGET_PAPERS", getattr(settings, "budget_papers", 0))),
                query_embedding=None, where_filter=where_filter, raw_question=q,
            )
            second_pass_docs = _filter_noisy_docs(second_pass_docs, retrieval_q)
            dominant_second_pass_count = len(second_pass_docs)
            paper_docs = _dedupe_docs(paper_docs + second_pass_docs)

        # If dominance didn't trigger but the query has a person name,
        # try a second pass using the 'researcher' metadata field which
        # stores full names (avoiding "D. Brown" ambiguity from 'authors').
        if not dominant_filter_usable and _has_explicit_entity_signal(q):
            from rag_engine import _extract_person_name
            person_name = _extract_person_name(q)
            if person_name and len(person_name) > 3:
                for field_name in ("researcher",):
                    researcher_docs = eng.retrieve_papers(
                        retrieval_q,
                        int((context.budgets or {}).get("BUDGET_PAPERS", getattr(settings, "budget_papers", 0))),
                        query_embedding=None, where_filter={field_name: person_name}, raw_question=q,
                    )
                    if researcher_docs:
                        researcher_docs = _filter_noisy_docs(researcher_docs, retrieval_q)
                        if researcher_docs:
                            paper_docs = _dedupe_docs(paper_docs + researcher_docs)
                            break

        post_filter_count = len(paper_docs)
        mem_docs = context.mem_docs or []

        anchor_before = _normalize_anchor(getattr(context, "anchor", {}) or getattr(eng, "anchor", {}))
        candidate_anchor = _build_anchor_from_dominance(dominance)
        anchor_after, anchor_action = _choose_anchor_update(
            current_anchor=anchor_before, candidate_anchor=candidate_anchor,
            dominance=dominance, question=q,
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
            docs_count=post_filter_count, anchor_consistent=anchor_consistent,
        )
        eng.last_retrieval_confidence = retrieval_confidence
        eng.last_anchor_support_ratio = float(anchor_support_ratio)

        prompt_max_docs = int(getattr(settings, "prompt_max_docs", 24))
        prompt_text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
        if retrieval_confidence in {"weak", "inconsistent"}:
            prompt_max_docs = min(prompt_max_docs, max(2, int(getattr(settings, "low_conf_prompt_max_docs", 8))))
            prompt_text_limit = min(prompt_text_limit, max(160, int(getattr(settings, "low_conf_prompt_doc_text_limit", 420))))

        # ── Prompt assembly ────────────────────────────────────────────────────────
        rolling_summary_text = pre_state.get("rolling_summary", "") or ""
        pre_turns_count = len(pre_state.get("turns", []) or [])
        pre_summary_len = len(rolling_summary_text)
        recent_turns_ctx = _build_recent_turns_context(
            pre_state, max_turns=min(6, max(2, int(PIPELINE_CFG["recent_turns_in_prompt"]))),
        )

        prompt_base_sections: List[str] = [PIPELINE_CFG["prompt_prefix"].rstrip()]
        rolling_summary_prompt = _rolling_summary_for_prompt(rolling_summary_text)
        if rolling_summary_prompt:
            prompt_base_sections.append("ROLLING SUMMARY:\n" + rolling_summary_prompt)
        if recent_turns_ctx:
            prompt_base_sections.append("RECENT TURNS:\n" + recent_turns_ctx)

        question_for_answer = (resolved_q or q).lower()

        if summary_intent:
            style_hint = (
                "The retrieved context already contains pre-written paper summaries. "
                "Extract and report the key mechanisms, findings, and methods described "
                "in those summaries directly — do not invent or paraphrase beyond what "
                "is stated. Attribute each point to its paper title. "
                "Write at least one paragraph per relevant paper or research theme."
            )
        elif detected_intent == "comparison":
            style_hint = "Compare the relevant items directly and keep claims grounded in retrieved evidence."
        elif detected_intent == "time_range":
            style_hint = (
                "If asked about the studied time period, report it only when explicitly stated in retrieved context. "
                "If not stated, say that directly and optionally provide publication years as a separate fallback. "
                "Do not infer studied period from publication years."
            )
        elif detected_intent == "list":
            style_hint = "Provide a concise list with short evidence-backed descriptors."
        else:
            style_hint = (
                "Give a direct evidence-grounded answer in plain prose. "
                "You MUST write at least 3 full sentences. "
                "Identify specific research topics, methods, or findings from the papers. "
                "Never respond with only '[Name] is a researcher' — always elaborate."
            )

        prompt, used_prompt_docs, used_prompt_text_limit = _fit_prompt_to_budget(
            runtime=getattr(mgr, "answer_runtime", None),
            docs=paper_docs, base_sections=prompt_base_sections,
            style_hint=style_hint, question_for_answer=question_for_answer,
            max_docs=prompt_max_docs, text_limit=prompt_text_limit,
            min_docs=1, min_text_limit=120,
        )

        # ── Answer generation ──────────────────────────────────────────────────────
        if not paper_docs:
            answer_text = _insufficient_context_answer(q, detected_intent)
        else:
            # Always attempt LLM generation when we have at least 1 paper.
            # Even weak/inconsistent retrieval can produce useful partial answers.
            if retrieval_confidence == "inconsistent":
                prompt_sections_for_call = [s for s in prompt_base_sections if not s.startswith("ROLLING SUMMARY")]
                prompt, used_prompt_docs, used_prompt_text_limit = _fit_prompt_to_budget(
                    runtime=getattr(mgr, "answer_runtime", None),
                    docs=paper_docs, base_sections=prompt_sections_for_call,
                    style_hint=style_hint, question_for_answer=question_for_answer,
                    max_docs=prompt_max_docs, text_limit=prompt_text_limit,
                    min_docs=1, min_text_limit=120,
                )

            t0_gen = time.perf_counter()
            try:
                answer_llm_calls += 1
                raw_answer = _invoke_with_timeout(
                    mgr.answer_runtime.llm, prompt, int(getattr(settings, "llm_timeout_s", 40)),
                )
            except Exception:
                raw_answer = ""
            generation_time_ms = (time.perf_counter() - t0_gen) * 1000.0
            answer_text = _extract_answer_text(raw_answer)

            if not answer_text:
                retry_prompt, _rd, _rl = _fit_prompt_to_budget(
                    runtime=getattr(mgr, "answer_runtime", None),
                    docs=paper_docs,
                    base_sections=prompt_sections_for_call if retrieval_confidence == "inconsistent" else prompt_base_sections,
                    style_hint=style_hint, question_for_answer=question_for_answer,
                    max_docs=max(1, min(6, max(1, used_prompt_docs // 2))),
                    text_limit=max(140, min(320, int(max(120, used_prompt_text_limit) * 0.7))),
                    min_docs=1, min_text_limit=96,
                )
                t0_retry = time.perf_counter()
                try:
                    answer_llm_calls += 1
                    raw_retry = _invoke_with_timeout(
                        mgr.answer_runtime.llm, retry_prompt, int(getattr(settings, "llm_timeout_s", 40)),
                    )
                except Exception:
                    raw_retry = ""
                generation_time_ms += (time.perf_counter() - t0_retry) * 1000.0
                answer_text = _extract_answer_text(raw_retry)

            if not answer_text:
                answer_text = PIPELINE_CFG["llm_no_answer"]

        if paper_docs and retrieval_confidence not in {"weak", "inconsistent"}:
            cleaned = (answer_text or "").strip()
            if not cleaned or cleaned == PIPELINE_CFG["llm_no_answer"] or len(cleaned) < 24:
                fallback = _structured_general_fallback(q, paper_docs, detected_intent)
                if not fallback:
                    fallback = _fallback_answer_from_docs(q, paper_docs)
                if fallback:
                    answer_text = fallback

        answer_text = _sanitize_user_answer(answer_text).strip() or PIPELINE_CFG["llm_no_answer"]
        eng.last_answer_llm_calls = int(answer_llm_calls)
        eng.finalize_turn(context, answer_text, no_results=not paper_docs)

    # ── Post-turn state ────────────────────────────────────────────────────────
    post_state = (
        {"rolling_summary": rolling_summary_text, "turns": pre_state.get("turns", []) or []}
        if stateless else mgr.store.load(user_key)
    )
    post_turns = post_state.get("turns", []) or []
    post_summary = post_state.get("rolling_summary", "") or ""
    post_extra = post_state.get("extra_state", {}) if isinstance(post_state.get("extra_state"), dict) else {}
    post_anchor = _normalize_anchor(post_state.get("anchor") or post_extra.get("anchor") or anchor_after)
    post_anchor_action = str(post_state.get("anchor_last_action") or post_extra.get("anchor_last_action") or anchor_action)
    post_summary_updated = bool(
        post_state.get("summary_updated") if "summary_updated" in post_state
        else post_extra.get("summary_updated", getattr(eng, "last_summary_updated", False))
    )
    post_retrieval_confidence = str(
        post_state.get("retrieval_confidence") or post_extra.get("retrieval_confidence") or retrieval_confidence
    )
    post_anchor_ratio = float(
        post_extra.get("anchor_support_ratio", getattr(eng, "last_anchor_support_ratio", anchor_support_ratio)) or 0.0
    )

    rolling_summary_json = {"summary": post_summary, "turns": len(post_turns), "updated": post_summary_updated}
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
    seen_titles: set = set()
    for d in paper_docs:
        try:
            meta = getattr(d, "metadata", {}) or {}
            title_key = re.sub(r"\s+", " ", str(meta.get("title", "") or "").strip().lower())
            if title_key and title_key != "untitled" and title_key in seen_titles:
                continue
            if title_key and title_key != "untitled":
                seen_titles.add(title_key)
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

    chroma_doc_refs = [_doc_to_ref(d) for d in paper_docs[:12]]
    memory_doc_refs = [_doc_to_ref(d) for d in mem_docs[:8]]
    retrieval_summary_cache = retrieval_cache_summary(paper_docs, retrieval_text=retrieval_q, limit_ids=12)

    chroma_json = {
        "count": len(paper_docs),
        "retrieval_count_raw": raw_retrieval_count,
        "first_pass_count": len(first_pass_docs),
        "dominant_second_pass_count": dominant_second_pass_count,
        "fallback_unfiltered_count": 0,
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
    user_query_json = {
        "text": q, "resolved_text": resolved_q, "standalone_question": resolved_q,
        "detected_intent": detected_intent, "retrieval_text": retrieval_q,
        "dominance": dominance, "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dominant_filter_usable,
        "anchor_before": anchor_before, "anchor_after": anchor_after, "anchor_action": anchor_action,
        "rewrite_blocked": bool(getattr(eng, "last_rewrite_blocked", False)),
        "rewrite_anchor_valid": bool(getattr(eng, "last_rewrite_anchor_valid", False)),
        "requested_mode": requested_mode, "effective_mode": effective_mode,
        "stateless": stateless, "timestamp": datetime.utcnow().isoformat(),
    }
    cache_json = {
        "previous_pipeline_present": bool(previous_pipeline),
        "previous_pipeline_sig": short_hash(
            json.dumps((previous_pipeline or {}).get("session_state", {}), ensure_ascii=False, sort_keys=True)
            if isinstance(previous_pipeline, dict) else "",
            length=10,
        ),
        "timestamp": datetime.utcnow().isoformat(),
    }

    combined_json = {
        "user_query": user_query_json,
        "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_summary_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": memory_doc_refs},
        "cache": cache_json,
        "session_state": session_state_json,
        "context_size_chars": len(_build_compact_context(paper_docs, max_docs=prompt_max_docs, text_limit=prompt_text_limit)),
        "timing_ms": timing_json,
        "llm_calls": llm_calls_json,
    }

    if getattr(settings, "debug_rag", False):
        try:
            print("\n[PIPELINE_JSON]")
            print(json.dumps(combined_json, ensure_ascii=False, indent=2))
        except Exception:
            pass

    cacheable_turn = should_cache_turn(
        retrieval_text=retrieval_q,
        rewrite_blocked=bool(getattr(eng, "last_rewrite_blocked", False)),
    )
    pipeline_cache_payload = {
        "user_query": {k: user_query_json[k] for k in ("text", "resolved_text", "retrieval_text", "detected_intent", "effective_mode")},
        "chroma_retrieval": {k: chroma_json[k] for k in ("count", "retrieval_count_raw", "post_filter_count", "retrieval_confidence", "retrieval_digest")},
        "rolling_summary": rolling_summary_json,
        "conversation_memory": {"count": len(mem_docs)},
        "cache": cache_json,
        "session_state": {
            "turn_count": session_state_json["turn_count"],
            "summary_len": session_state_json["summary_len"],
            "effective_mode": session_state_json["effective_mode"],
            "llm_calls": llm_calls_json,
        },
        "timing_ms": timing_json,
    }
    set_pipeline_cache(user_key, pipeline_cache_payload if cacheable_turn else {})

    out: Dict[str, Any] = {
        "answer": answer_text,
        "sources": sources,
        "graph_hits": [], "graph_graph": {}, "graph_error": "",
        "user_query": user_query_json,
        "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_summary_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": memory_doc_refs},
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

    if (not stateless) and PIPELINE_CFG["qa_cache_enable"] and (not use_graph_flag) and cacheable_turn:
        cache_write_key = build_pipeline_cache_key(
            user_key=user_key, resolved_text=resolved_q or q,
            effective_mode=effective_mode, state_signature=cache_state_sig,
        )
        set_cached_answer(user_key, cache_write_key, {
            "answer": out.get("answer", ""),
            "sources": list(out.get("sources", []) or [])[:10],
            "graph_hits": [], "graph_graph": {}, "graph_error": "",
            "user_query": pipeline_cache_payload["user_query"],
            "chroma_retrieval": pipeline_cache_payload["chroma_retrieval"],
            "rolling_summary": rolling_summary_json,
            "conversation_memory": {"count": len(mem_docs)},
            "cache": {"timestamp": datetime.utcnow().isoformat(), "retrieval_digest": retrieval_summary_cache, "state_sig": cache_state_sig},
            "session_state": pipeline_cache_payload["session_state"],
            "timing_ms": timing_json,
            "llm_calls": llm_calls_json,
        })

    return out