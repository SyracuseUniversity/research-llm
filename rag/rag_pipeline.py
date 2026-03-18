# rag_pipeline.py
import json
import logging
import os
import re
import time
from collections import Counter
from contextlib import nullcontext
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from rag_engine import get_global_manager, _invoke_with_timeout, _extract_person_name
from rag_utils import (
    norm_text, clean_html, normalize_title_case, collapse_whitespace,
    tokenize_words, token_in_hay, get_stopword_set, is_generic_query_token,
    is_followup_coref_question, strip_corpus_noise_terms, dedupe_docs, doc_haystack,
    clean_snippet, build_compact_context, dedupe_ci, is_placeholder_anchor_value,
    normalize_anchor, anchor_in_text, anchor_is_stable, anchor_support_ratio,
    retrieval_confidence_label, classify_generic_intent, strip_prompt_leak,
    looks_like_person_candidate, strip_possessive, has_explicit_entity_signal,
    short_hash, utcnow_iso, query_tokens_for_relevance,
    is_meta_command, anchor_query_overlap,
)
from runtime_settings import settings
from rag_graph import graph_retrieve_from_paper_docs
from conversation_memory import (
    get_cached_answer, set_cached_answer, get_pipeline_cache, set_pipeline_cache,
)
from cache_manager import (
    build_cache_key as build_pipeline_cache_key,
    state_signature_from_state, retrieval_cache_summary, should_cache_turn,
    CACHE_KEY_VERSION as _CACHE_KEY_VERSION,
)

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try: return int(str(os.getenv(name, str(default))).strip())
    except Exception: return int(default)


PIPELINE_CFG = {
    "cache_version": os.getenv("RAG_CACHE_VERSION", _CACHE_KEY_VERSION),
    "max_docs_after_filter": _env_int("RAG_MAX_DOCS_AFTER_FILTER", 30),
    "fallback_max_items": _env_int("RAG_FALLBACK_MAX_ITEMS", 8),
    "recent_turns_in_prompt": _env_int("RAG_RECENT_TURNS_IN_PROMPT",
        int(getattr(settings, "recent_turns_in_prompt", 4))),
    "qa_cache_enable": os.getenv("RAG_QA_CACHE_ENABLE",
        "1" if bool(getattr(settings, "qa_cache_enable", False)) else "0"
    ).strip().lower() in {"1", "true", "yes", "y", "on"},
    "empty_question_answer": os.getenv("RAG_EMPTY_QUESTION_ANSWER", "Please enter a question."),
    "llm_no_answer": os.getenv("RAG_LLM_NO_ANSWER",
        "I could not generate an answer from the retrieved context."),
    "fallback_relevant_papers_prefix": os.getenv("RAG_FALLBACK_PAPERS_PREFIX",
        "I couldn't generate a full answer, but these papers look most relevant:"),
    "prompt_prefix": os.getenv("RAG_PROMPT_PREFIX", (
        "You are answering questions about Syracuse University researchers using only "
        "the provided retrieved context, which contains real paper records from the Syracuse corpus.\n"
        "Each retrieved record may contain a pre-written abstract or summary — treat these as "
        "authoritative descriptions of the paper and report their content faithfully.\n"
        "Ground every material claim in retrieved evidence.\n"
        "If evidence is missing, weak, or conflicting, say so clearly and ask a focused clarification question.\n"
        "Do not fabricate affiliations, awards, memberships, journal claims, or metadata.\n"
        "CRITICAL: Only attribute a paper to a researcher if the paper's researcher or authors field "
        "explicitly names them. Do not assume a paper belongs to someone just because they were "
        "mentioned previously in conversation.\n"
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
        "Detected intents: default, comparison, time_range, list.\n")),
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

def _doc_to_source_md(d) -> str:
    meta = d.metadata or {}
    authors = re.sub(r"</?[a-zA-Z][^>]*>", "", str(meta.get("authors", "") or "")).strip()
    title = normalize_title_case(str(meta.get("title", "") or ""))
    year = str(meta.get("year", meta.get("publication_date", "")) or "").strip()
    year_match = re.search(r"\b(19|20)\d{2}\b", year)
    year_short = year_match.group(0) if year_match else ""
    doi = str(meta.get("doi", "") or "").strip()

    if not title or title.lower() == "untitled":
        title = "[Untitled]"

    parts = [f"{authors}." if authors else "[Unknown author].",
             f"({year_short})." if year_short else "(n.d.).",
             f"*{title}*."]
    if doi:
        if doi.startswith("10."): doi = f"https://doi.org/{doi}"
        if doi.startswith("http"): parts.append(doi)
    return " ".join(parts)


def _doc_to_ref(d) -> Dict[str, str]:
    meta = getattr(d, "metadata", {}) or {}
    return {
        "paper_id": str(meta.get("paper_id", "")),
        "chunk": str(meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))),
        "title": str(meta.get("title", "")),
    }


def _filter_noisy_docs(docs, question: str) -> list:
    deduped = dedupe_docs(docs)
    if not deduped:
        return []
    tokens = query_tokens_for_relevance(question)
    if not tokens:
        return deduped[:int(PIPELINE_CFG["max_docs_after_filter"])]
    # Score docs by how many query tokens they match and sort by relevance,
    # but keep single-token threshold so we don't drop docs where the name
    # format differs (e.g. "D. Brown" vs query "Duncan Brown").
    scored = []
    for d in deduped:
        hay = doc_haystack(d)
        match_count = sum(1 for tok in tokens if token_in_hay(tok, hay))
        if match_count >= 1:
            scored.append((d, match_count))
    scored.sort(key=lambda x: -x[1])
    kept = [d for d, _ in scored]
    return (kept if kept else deduped)[:int(PIPELINE_CFG["max_docs_after_filter"])]


def _normalize_meta_value(value) -> str:
    return norm_text(str(value or ""))


def _metadata_key_allowed(key: str) -> bool:
    k = (key or "").strip().lower()
    if not k or k in {"chunk", "chunk_id", "id", "doc_id", "source", "path", "url"}:
        return False
    return not (k.endswith("_id") and k != "paper_id")


def _metadata_value_allowed(value: str) -> bool:
    v = _normalize_meta_value(value)
    if not v or len(v) < 3 or is_placeholder_anchor_value(v):
        return False
    return not (re.fullmatch(r"[0-9\W]+", v) or re.fullmatch(r"(19|20)\d{2}", v))


def _iter_doc_metadata_key_values(d) -> List[Tuple[str, str, str]]:
    meta = getattr(d, "metadata", {}) or {}
    out = []
    for key, raw in meta.items():
        k = str(key or "").strip()
        if not k or not _metadata_key_allowed(k) or isinstance(raw, (list, dict, tuple, set)):
            continue
        raw_text = re.sub(r"\s+", " ", str(raw or "").strip())
        v = _normalize_meta_value(raw_text)
        if raw_text and _metadata_value_allowed(raw_text):
            out.append((k, v, raw_text))
    return out


def _split_author_names(raw_authors: str) -> List[str]:
    if not raw_authors:
        return []
    return [n for p in re.split(r"\s*[;,]\s*|\s+and\s+", raw_authors)
            if (n := re.sub(r"\s+", " ", p.strip())) and len(n) >= 2]


def _person_name_signatures(name: str) -> Dict[str, str]:
    toks = [t for t in re.findall(r"[A-Za-z]+", str(name or "").strip()) if t]
    if len(toks) < 2:
        return {}
    first, last = toks[0], toks[-1]
    return {
        "first": first.lower(),
        "last": last.lower(),
        "last_name": last,
        "full_name": f"{first} {last}",
        "initial_last_dot": f"{first[0]}. {last}",
        "initial_last_plain": f"{first[0]} {last}",
    }


def _name_match_strength(text: str, sig: Dict[str, str]) -> int:
    if not text or not sig:
        return 0
    first = str(sig.get("first", "") or "").strip()
    last = str(sig.get("last", "") or "").strip()
    if not first or not last:
        return 0
    hay = re.sub(r"[,\(\)\[\]]+", " ", _normalize_meta_value(text))
    hay = re.sub(r"\s+", " ", hay).strip()
    if not hay:
        return 0
    if (re.search(rf"\b{re.escape(first)}\s+{re.escape(last)}\b", hay)
        or re.search(rf"\b{re.escape(last)}\s+{re.escape(first)}\b", hay)):
        return 3
    if (re.search(rf"\b{re.escape(first[0])}\.?\s+{re.escape(last)}\b", hay)
        or re.search(rf"\b{re.escape(last)}\s+{re.escape(first[0])}\.?\b", hay)):
        return 2
    if re.search(rf"\b{re.escape(last)}\b", hay):
        return 1
    return 0


def _doc_person_match_score(d, person_name: str) -> float:
    sig = _person_name_signatures(person_name)
    if not sig:
        return 0.0
    meta = getattr(d, "metadata", {}) or {}
    researcher_text = str(meta.get("researcher", "") or "")
    authors_text = str(meta.get("authors", "") or "")
    researcher_strength = _name_match_strength(researcher_text, sig)
    author_strength = 0
    for author in _split_author_names(authors_text):
        author_strength = max(author_strength, _name_match_strength(author, sig))
    if author_strength == 0 and authors_text:
        author_strength = _name_match_strength(authors_text, sig)

    if researcher_strength >= 3:
        return 4.0
    if author_strength >= 3:
        return 3.0
    if researcher_strength == 2:
        return 2.0
    if author_strength == 2:
        return 1.5
    if researcher_strength == 1:
        return 0.75
    if author_strength == 1:
        return 0.5
    return 0.0


def _rank_docs_for_person(docs, person_name: str) -> List[Tuple[Any, float]]:
    ranked: List[Tuple[Any, float]] = []
    for d in docs or []:
        ranked.append((d, _doc_person_match_score(d, person_name)))

    def _sort_key(item: Tuple[Any, float]) -> Tuple[float, str, str]:
        doc, score = item
        meta = getattr(doc, "metadata", {}) or {}
        paper_id = str(meta.get("paper_id", "") or "")
        chunk = str(meta.get("chunk", meta.get("chunk_id", meta.get("id", ""))) or "")
        return (-float(score), paper_id, chunk)

    ranked.sort(key=_sort_key)
    return ranked


def _select_docs_for_person(ranked_docs: List[Tuple[Any, float]], *,
                            max_docs: int) -> Tuple[List[Any], Dict[str, int]]:
    # Score 2.0 = initial + last-name match on researcher field (e.g. "D. Brown"
    # matching "Duncan Brown").  This is strong evidence in academic metadata
    # where initial-format names are the norm.
    strong_docs = [d for d, score in ranked_docs if score >= 2.0]
    matched_docs = [d for d, score in ranked_docs if score >= 1.0]
    weak_docs = [d for d, score in ranked_docs if score < 1.0]
    limit = max(4, int(max_docs or 0))

    if strong_docs:
        keep = strong_docs[:limit]
        if len(keep) < limit:
            keep.extend(matched_docs[:max(0, limit - len(keep))])
        if len(keep) < limit:
            keep.extend(weak_docs[:max(0, limit - len(keep))])
    elif matched_docs:
        keep = matched_docs[:limit]
        if len(keep) < limit:
            keep.extend(weak_docs[:max(0, limit - len(keep))])
    else:
        keep = [d for d, _ in ranked_docs[:limit]]

    return dedupe_docs(keep), {
        "strong": len(strong_docs),
        "matched": len(matched_docs),
        "total": len(ranked_docs),
    }


_CONFIDENCE_ORDER = ("weak", "low", "medium", "high")


def _downgrade_confidence(label: str, *, steps: int = 1) -> str:
    key = str(label or "").strip().lower()
    if key not in _CONFIDENCE_ORDER:
        return key or "weak"
    idx = _CONFIDENCE_ORDER.index(key)
    return _CONFIDENCE_ORDER[max(0, idx - max(1, int(steps)))]


def _downshift_confidence_for_person_support(label: str, *,
                                             strong_count: int, matched_count: int,
                                             total_count: int) -> str:
    if total_count <= 0:
        return label
    if matched_count <= 0:
        return "weak"

    out = str(label or "weak").strip().lower()
    strong_ratio = float(strong_count) / max(1, int(total_count))
    matched_ratio = float(matched_count) / max(1, int(total_count))

    # Only downgrade when evidence is genuinely weak — not when names are
    # stored in initial format (D. Brown) which gives score 2.0 = strong.
    if out == "high" and (strong_ratio < 0.3 or matched_ratio < 0.5):
        out = "medium"
    if out in {"high", "medium"} and strong_ratio < 0.1 and matched_ratio < 0.3:
        out = _downgrade_confidence(out, steps=1)
    if out in {"medium", "low"} and matched_ratio < 0.15:
        out = _downgrade_confidence(out, steps=1)
    return out


def _dominant_metadata_filter_from_docs(docs, question: str, *,
                                        majority_ratio: float = 0.6, min_count: int = 3) -> Dict[str, Any]:
    result = {
        "dominant": False, "key": "", "value": "", "count": 0, "ratio": 0.0,
        "confidence": 0.0, "confidence_floor": float(getattr(settings, "dominant_min_confidence", 0.72)),
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
            if key.lower() == "authors":
                for name in _split_author_names(raw_val):
                    nv = _normalize_meta_value(name)
                    if not _metadata_value_allowed(name):
                        continue
                    pair = (key, nv)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        counts[pair] = counts.get(pair, 0) + 1
                        exemplars.setdefault(pair, name)
            else:
                pair = (key, norm_val)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    counts[pair] = counts.get(pair, 0) + 1
                    exemplars.setdefault(pair, raw_val)
    if not counts:
        return result

    def _sort_key(item):
        (k, nv), cnt = item
        # Only 'researcher' is a reliable key for second-pass filtering.
        # 'title' dominant hits are almost always corpus noise ("Untitled") or
        # coincidental shared titles — never use title as a filter pivot.
        ALLOWED_FILTER_KEYS = {"researcher"}
        return (-cnt,
                0 if k.lower() in ALLOWED_FILTER_KEYS else 2,
                0 if k.lower() == "researcher" else 1,
                k, nv)

    ranked = sorted(counts.items(), key=_sort_key)
    (best_key, best_nv), best_count = ranked[0]
    runner_up = next((cnt for (_, nv), cnt in ranked[1:] if nv != best_nv), 0)
    ratio = float(best_count) / max(1, n_docs)
    margin = float(best_count - runner_up) / max(1, n_docs)
    confidence = max(0.0, min(1.0, 0.85 * ratio + 0.15 * max(0.0, margin)))
    if ratio >= 0.8 and best_count >= max(4, min_count):
        confidence = max(confidence, min(0.98, ratio))

    result.update({
        "key": best_key, "value": exemplars.get((best_key, best_nv), best_nv),
        "count": int(best_count), "ratio": ratio, "confidence": confidence,
        "runner_up_count": int(runner_up),
    })
    conf_floor = float(getattr(settings, "dominant_min_confidence", 0.72))
    result["confidence_floor"] = conf_floor

    _DOMINANT_ALLOWED_KEYS = {"researcher"}
    if (best_count >= max(1, min_count) and ratio >= float(majority_ratio)
        and confidence >= conf_floor
        and best_key.lower() in _DOMINANT_ALLOWED_KEYS
        and not is_placeholder_anchor_value(str(result.get("value", "") or ""))):
        result["dominant"] = True
        result["filter"] = {"key": result["key"], "value": result["value"]}
    return result


# ---------------------------------------------------------------------------
# Answer helpers
# ---------------------------------------------------------------------------

def _insufficient_context_answer(question: str, intent: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    if q:
        return (f'I couldn\'t find enough matching evidence for "{q}" in the retrieved papers. '
                "Please clarify the entity, topic, or time range you want.")
    return "I couldn't find enough matching evidence. Please clarify the entity or topic you want."


def _uncertain_retrieval_answer(question: str, *, anchor_value: str = "", reason: str = "") -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    if anchor and reason.strip().lower() == "inconsistent":
        return (f'I\'m not confident the retrieved evidence matches the current focus on "{anchor}". '
                f'Could you confirm whether you want to continue with "{anchor}" or switch topics for "{q}"?')
    if q:
        return (f'I\'m not confident there is enough consistent evidence yet for "{q}". '
                "Please add a specific entity, paper title, or year so I can answer accurately.")
    return "I'm not confident there is enough consistent evidence yet. Please add a specific entity, paper title, or year."


# ---------------------------------------------------------------------------
# Answer sanitization
# ---------------------------------------------------------------------------

_BLOCKED_MARKERS = (
    "detected intent", "retrieval count", "pipeline", "chroma",
    "metadata filter", "session_id", "turn_count",
    "please provide the answer in the requested format",
    "note: the user-facing answer", "the final answer is",
)

_BOILERPLATE_PHRASES = (
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
    "note: the answer has been reformatted", "has been reformatted to",
    "to better match the required response format",
    "to match the required response format", "reformatted to better match",
    "however, i noticed that the response contains",
    "i will revise the response to only include",
    "to only include information present in the retrieved",
    "let me know if you'd like me to proceed",
    "let me know if you would like me to proceed",
    "i can help with the next question",
    "i'm here to assist you", "i'm here to help you", "best regards",
)

_SELF_REF_TERMS = ("reformatted", "required response format", "response format",
                    "answer format", "the answer has been", "this answer has been")


def _normalize_for_similarity(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", re.sub(r"\s+", " ", str(text or "").strip().lower()))


def _strip_leading_answer_labels(text: str) -> str:
    cleaned = str(text or "").strip()
    for _ in range(3):
        updated = re.sub(r"^\s*(summary|final summary|answer|response|final response|the final answer is)\s*:\s*",
                          "", cleaned, flags=re.IGNORECASE).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _is_closure_or_process(text: str) -> bool:
    lower = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if lower.startswith("note:") and any(t in lower for t in (
        "pipeline", "retrieval", "context", "analysis", "synthesis", "confidence",
        "metadata", "prompt", "cache", "debug", "reformatted", "revised", "reformat",
        "response format", "required format", "requested format")):
        return True
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


def _sanitize_user_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw

    kept = [line for line in raw.splitlines()
            if not any(m in line.strip().lower() for m in _BLOCKED_MARKERS)
            and not any(p in line.strip().lower() for p in _BOILERPLATE_PHRASES)]
    cleaned = _strip_leading_answer_labels("\n".join(kept).strip()) or _strip_leading_answer_labels(raw)
    if not cleaned:
        return raw

    chunks = [c.strip() for c in re.split(r"\n\s*\n+", cleaned) if c.strip()]
    if len(chunks) <= 1:
        chunks = [c.strip() for c in cleaned.splitlines() if c.strip()]

    result, seen_norms = [], []
    for chunk in chunks:
        para = _strip_leading_answer_labels(re.sub(r"\s+", " ", chunk).strip())
        if not para:
            continue
        lower = para.lower()
        if lower.startswith("the retrieved context") or lower.startswith("confidence level"):
            continue
        if _is_closure_or_process(para):
            continue
        norm = _normalize_for_similarity(para)
        if not norm or any(norm == p or SequenceMatcher(None, norm, p).ratio() >= 0.80 for p in seen_norms):
            continue
        seen_norms.append(norm)
        result.append(para)

    # Post-processing pipeline
    text_out = "\n\n".join(result).strip()
    # Dedupe repeated sentences
    if text_out:
        kept_sents, last_norms = [], []
        for p in re.split(r"(?<=[.!?])\s+", text_out):
            sent = p.strip()
            if not sent:
                continue
            n = _normalize_for_similarity(sent)
            if n and n in last_norms:
                continue
            kept_sents.append(sent)
            last_norms = (last_norms + [n])[-6:]
        text_out = " ".join(kept_sents).strip()
    # Remove closure sentences
    if text_out:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text_out) if p.strip()]
        text_out = " ".join(p for p in parts if not _is_closure_or_process(p)).strip()
    # Trim incomplete tail
    if text_out and not re.search(r'[.!?"\')\\]]\s*$', text_out):
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text_out) if p.strip()]
        if len(parts) > 1:
            tail = parts[-1]
            if not re.match(r"^[-*]|\d+\.", tail) and len(re.findall(r"[A-Za-z0-9]+", tail)) <= 14:
                text_out = " ".join(parts[:-1]).strip() or text_out
    # Strip self-referential notes
    if text_out:
        text_out = "\n".join(ln for ln in text_out.splitlines()
                             if not (ln.strip().lower().startswith("note:")
                                     and any(t in ln.strip().lower() for t in _SELF_REF_TERMS))).strip()
    return text_out or _strip_leading_answer_labels(cleaned) or cleaned


def sanitize_answer_for_display(text: str) -> str:
    cleaned = _sanitize_user_answer(text)
    cleaned = re.sub(r"</?[a-zA-Z][^>]*>", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)

    if "\n\n" in cleaned:
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if len(sentences) <= 3:
        return cleaned.strip()

    _NEW_BLOCK = re.compile(
        r"^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:'s)?\s+"
        r"|Additionally[,\s]|Furthermore[,\s]|Moreover[,\s]|Another\s+researcher"
        r"|In\s+addition[,\s]|Separately[,\s]|Meanwhile[,\s]|In\s+contrast[,\s]"
        r"|His\s+research\s+also|Her\s+research\s+also|Their\s+research\s+also"
        r"|While\s+(?:his|her|their|none|not)|Although\s)", re.IGNORECASE)

    paragraphs, current = [], []
    for sent in sentences:
        if current and len(current) >= 2 and _NEW_BLOCK.match(sent):
            paragraphs.append(" ".join(current))
            current = [sent]
        else:
            current.append(sent)
            if len(current) >= 4:
                paragraphs.append(" ".join(current))
                current = []
    if current:
        paragraphs.append(" ".join(current))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(paragraphs)).strip()


# ---------------------------------------------------------------------------
# Anchor management
# ---------------------------------------------------------------------------

def _build_anchor_from_dominance(dominance) -> Dict[str, Any]:
    if not dominance.get("dominant"):
        return {}
    key = re.sub(r"\s+", " ", str(dominance.get("key", "") or "").strip().lower())
    value = re.sub(r"\s+", " ", str(dominance.get("value", "") or "").strip())
    if not key or not value or is_placeholder_anchor_value(value):
        return {}
    confidence = float(dominance.get("confidence", 0.0) or 0.0)
    if confidence < float(getattr(settings, "dominant_min_confidence", 0.72)):
        return {}
    return {"type": key, "value": value, "source": f"dominant_metadata:{key}",
            "confidence": max(0.0, min(1.0, confidence))}


def _choose_anchor_update(*, current_anchor, candidate_anchor, dominance, question,
                          resolved_question=""):
    anchor_now = normalize_anchor(current_anchor)
    candidate = normalize_anchor(candidate_anchor)

    # Fix #7: If there's no new candidate and the current anchor has zero
    # token overlap with the query, clear it instead of keeping stale context.
    # BUT: if the query contains pronouns/follow-up phrases ("he", "his",
    # "tell me more"), keep the anchor — the user is referring to it.
    if anchor_now and not candidate:
        anchor_value = str(anchor_now.get("value", "") or "").strip()
        if anchor_value and not anchor_query_overlap(anchor_value, question):
            if is_followup_coref_question(question):
                return anchor_now, "kept_followup_coref"
            return {}, "cleared_no_query_overlap"
        return anchor_now, "kept_no_dominance"

    if not candidate:
        return {}, "none"
    if not anchor_now:
        # --- Fix: Before accepting a brand-new anchor from dominance, verify
        # it has overlap with either the raw or resolved question.  This prevents
        # e.g. Samuel Johnson becoming the anchor when the user asked about
        # Duncan Brown and the resolved query says "Duncan Brown". ---
        candidate_value = str(candidate.get("value", "") or "").strip()
        if candidate_value:
            has_raw_overlap = anchor_query_overlap(candidate_value, question)
            has_resolved_overlap = (bool(resolved_question)
                                    and anchor_query_overlap(candidate_value, resolved_question))
            if not has_raw_overlap and not has_resolved_overlap:
                # The candidate has no overlap with what the user asked about —
                # it came from noisy retrieval results.  Don't set it.
                return {}, "blocked_no_query_overlap"
        return candidate, "set_from_dominance"

    same_type = _normalize_meta_value(anchor_now.get("type", "")) == _normalize_meta_value(candidate.get("type", ""))
    same_value = _normalize_meta_value(anchor_now.get("value", "")) == _normalize_meta_value(candidate.get("value", ""))
    if same_type and same_value:
        anchor_now["confidence"] = max(float(anchor_now.get("confidence", 0) or 0),
                                       float(candidate.get("confidence", 0) or 0))
        anchor_now["source"] = candidate.get("source", anchor_now.get("source", "retrieval"))
        return anchor_now, "kept_reinforced"

    # --- Fix 4 (revised): Before replacing, verify the new candidate has token
    # overlap with the raw OR resolved user query.  A high-confidence anchor on
    # unrelated docs (e.g. Samuel Johnson when user asked about Duncan Brown)
    # must NOT silently replace the current anchor. ---
    candidate_value = str(candidate.get("value", "") or "").strip()
    if candidate_value:
        has_raw_overlap = anchor_query_overlap(candidate_value, question)
        has_resolved_overlap = (bool(resolved_question)
                                and anchor_query_overlap(candidate_value, resolved_question))
        if not has_raw_overlap and not has_resolved_overlap:
            # The candidate anchor has zero overlap with what the user actually asked.
            if is_followup_coref_question(question):
                return anchor_now, "kept_candidate_no_query_overlap"
            return anchor_now, "kept_candidate_no_query_overlap"

    replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
    strong_ratio = float(dominance.get("ratio", 0) or 0) >= max(
        replace_conf, float(getattr(settings, "dominant_majority_ratio", 0.6)) + 0.15)
    if float(candidate.get("confidence", 0) or 0) >= replace_conf and (
        has_explicit_entity_signal(question) or has_explicit_entity_signal(resolved_question or "")
        or strong_ratio):
        return candidate, "replaced_with_strong_evidence"
    return anchor_now, "kept_ambiguous_switch_requires_confirmation"


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _extract_answer_text(raw_answer) -> str:
    raw_text = str(raw_answer or "")
    answer_text = strip_prompt_leak(raw_text).strip()
    if answer_text:
        return answer_text
    for marker in ("ANSWER:", "FINAL RESPONSE:", "RESPONSE:"):
        if marker in raw_text:
            cand = raw_text.split(marker)[-1].strip()
            if cand:
                return cand
    return raw_text.strip()


def _runtime_prompt_token_budget(runtime, reserved_new_tokens: int) -> int:
    if runtime is None:
        return 0
    max_ctx = 0
    for source in [getattr(getattr(runtime, "model", None), "config", None),
                   getattr(runtime, "tokenizer", None)]:
        if source and max_ctx <= 0:
            try: max_ctx = int(getattr(source, "max_position_embeddings", 0) or
                               getattr(source, "model_max_length", 0) or 0)
            except Exception: pass
    if max_ctx <= 0 or max_ctx > 65536:
        max_ctx = 4096
    return max(256, max_ctx - max(64, int(reserved_new_tokens) + 24))


def _compose_answer_prompt(*, base_sections, style_hint, question_for_answer, context_blob):
    parts = list(base_sections)
    ctx = (context_blob or "").strip()
    if ctx:
        parts.append("RETRIEVED CONTEXT:\n" + ctx)
    parts.append("ANSWER POLICY:\n" + style_hint)
    return ("\n\n".join(parts) + PIPELINE_CFG["prompt_mid"]
            + (question_for_answer or "") + PIPELINE_CFG["prompt_suffix"])


def _fit_prompt_to_budget(*, runtime, docs, base_sections, style_hint,
                          question_for_answer, max_docs, text_limit,
                          min_docs=1, min_text_limit=120):
    docs_cap = max(1, int(max_docs))
    text_cap = max(96, int(text_limit))
    min_docs = max(1, int(min_docs))
    min_text_limit = max(96, int(min_text_limit))
    budget = _runtime_prompt_token_budget(
        runtime, int(getattr(settings, "answer_max_new_tokens", 384)))

    if not docs:
        return _compose_answer_prompt(base_sections=base_sections, style_hint=style_hint,
                                      question_for_answer=question_for_answer, context_blob=""), docs_cap, text_cap

    prompt = _compose_answer_prompt(
        base_sections=base_sections, style_hint=style_hint,
        question_for_answer=question_for_answer,
        context_blob=build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap))

    # Phase 1: Shrink text_limit first (preserves doc count which is more
    # important for answer quality than per-doc verbosity).
    # Phase 2: Then alternate shrinking docs and text.
    phase = 1
    for _ in range(36):
        if budget <= 0 or runtime is None:
            return prompt, docs_cap, text_cap
        try:
            if int(runtime.count_tokens(prompt)) <= budget:
                return prompt, docs_cap, text_cap
        except Exception:
            return prompt, docs_cap, text_cap

        prev = (docs_cap, text_cap)
        if phase == 1:
            # Phase 1: only shrink text until we hit the floor
            if text_cap > min_text_limit:
                text_cap = max(min_text_limit, text_cap - max(20, text_cap // 6))
            else:
                phase = 2  # switch to alternating
        if phase == 2:
            # Phase 2: alternate between shrinking docs and text
            if docs_cap > min_docs and text_cap <= min_text_limit:
                docs_cap = max(min_docs, docs_cap - max(1, docs_cap // 5))
            elif text_cap > min_text_limit:
                text_cap = max(min_text_limit, text_cap - max(16, text_cap // 6))
            elif docs_cap > min_docs:
                docs_cap = max(min_docs, docs_cap - 1)
            elif docs_cap > 1:
                docs_cap -= 1
            elif text_cap > 96:
                text_cap = max(96, text_cap - 16)
            else:
                return prompt, docs_cap, text_cap

        if (docs_cap, text_cap) == prev:
            return prompt, docs_cap, text_cap
        prompt = _compose_answer_prompt(
            base_sections=base_sections, style_hint=style_hint,
            question_for_answer=question_for_answer,
            context_blob=build_compact_context(docs, max_docs=docs_cap, text_limit=text_cap))
    return prompt, docs_cap, text_cap


def _clip_sentences(text: str, *, max_sentences: int = 2, max_chars: int = 320) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if not compact:
        return ""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", compact) if p.strip()]
    if parts:
        compact = " ".join(parts[:max(1, max_sentences)]).strip()
    return compact[:max_chars].rstrip() + "..." if len(compact) > max_chars else compact


def _clean_assistant_turn_for_prompt(text: str) -> str:
    blocked = ("summary:", "no further analysis is required", "no additional retrieval is required",
               "no further synthesis is required", "the retrieved context", "confidence level",
               "please let me know if you need", "if you need any further assistance",
               "i can help with anything else", "let me know if you need anything else",
               "i have revised the answer", "i've revised the response")
    kept = [line for line in str(text or "").splitlines()
            if not any(b in re.sub(r"\s+", " ", line or "").strip().lower() for b in blocked)]
    return _clip_sentences(_strip_leading_answer_labels(" ".join(kept).strip()),
                           max_sentences=2, max_chars=360)


def _rolling_summary_for_prompt(summary_text: str) -> str:
    raw = str(summary_text or "").strip()
    if not raw:
        return ""
    aliases = {"focus": "Current focus", "current focus": "Current focus",
               "researcher mentions": "Researcher mentions", "researchers": "Researcher mentions",
               "entities": "Core entities", "entities discussed": "Core entities",
               "core entities": "Core entities", "findings": "Key themes",
               "key findings so far": "Key themes", "key themes": "Key themes",
               "constraints": "Constraints", "open questions": "Open questions"}
    sections: Dict[str, List[str]] = {k: [] for k in set(aliases.values())}
    current_key = None
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            current_key = aliases.get((m.group(1) or "").strip().lower())
            if current_key:
                rest = (m.group(2) or "").strip().lstrip("- ").strip()
                if rest:
                    sections[current_key].append(rest)
            continue
        if current_key:
            sections[current_key].append(line.lstrip("- ").strip())

    has_boilerplate = any(m in raw.lower() for m in (
        "summary:", "no further analysis is required", "no additional retrieval is required",
        "no further synthesis is required", "the retrieved context", "confidence level"))
    if has_boilerplate:
        keys = ["Current focus", "Core entities"]
    else:
        return raw
    lines = []
    for key in keys:
        vals = [re.sub(r"\s+", " ", str(v or "").strip()) for v in (sections.get(key) or [])
                if str(v or "").strip() and str(v or "").strip() != "(none)"]
        if vals:
            lines.append(f"{key}: {' | '.join(vals)}")
    return "\n".join(lines).strip()


def _build_recent_turns_context(state, max_turns: int) -> str:
    turns = list(state.get("recent_turns") or state.get("turns", []) or [])
    if max_turns <= 0 or not turns:
        return ""
    rows = []
    for t in turns[-max_turns:]:
        role = str(t.get("role", "") or "").strip().lower()
        text = str(t.get("text", "") or "").strip()
        if not role or not text:
            continue
        text = (_clean_assistant_turn_for_prompt(text) if role == "assistant"
                else _clip_sentences(text, max_sentences=2, max_chars=420))
        if text:
            rows.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
    return "\n".join(rows).strip()


def _fallback_answer_from_docs(question: str, docs, intent: str = "default") -> Optional[str]:
    if not docs:
        return None
    top = list(docs[:12])
    years = []
    for d in top:
        m = re.search(r"\b(19|20)\d{2}\b", str((getattr(d, "metadata", {}) or {}).get(
            "year", getattr(d, "metadata", {}).get("publication_date", "")) or ""))
        if m:
            try: years.append(int(m.group(0)))
            except Exception: pass

    works = []
    for d in top[:4]:
        title = str((getattr(d, "metadata", {}) or {}).get("title", "") or "").strip()
        if not title or title.lower() == "untitled":
            continue
        year = str((getattr(d, "metadata", {}) or {}).get("year", "") or "").strip()
        works.append(f"{title} ({year})" if year else title)

    leads = {"comparison": "I couldn't generate a full comparison, but these retrieved papers are most relevant to compare.",
             "list": "I couldn't generate a full list answer, but these retrieved papers are most relevant."}
    lead = leads.get(intent, "I couldn't generate a complete narrative answer, but these retrieved papers are the strongest evidence.")

    lines = [lead]
    if years:
        lo, hi = min(years), max(years)
        lines.append(f"Publication years represented: {lo}" + (f" to {hi}" if lo != hi else "") + ".")
    if works:
        lines.append("Most relevant papers: " + "; ".join(works) + ".")
    return " ".join(lines).strip() if len(lines) > 1 else None


def _clean_title_for_answer(title: str) -> str:
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", str(title or "")).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or text.lower() == "untitled":
        return ""
    return normalize_title_case(text)


def _supported_researcher_evidence(docs, *, max_researchers: int = 6,
                                   max_titles_per_researcher: int = 3) -> List[Tuple[str, Dict[str, Any]]]:
    evidence: Dict[str, Dict[str, Any]] = {}
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        researcher = re.sub(r"\s+", " ", str(meta.get("researcher", "") or "").strip())
        if not researcher or not looks_like_person_candidate(researcher):
            continue
        row = evidence.setdefault(researcher, {"count": 0, "titles": []})
        row["count"] += 1
        title = _clean_title_for_answer(meta.get("title", ""))
        if title and title not in row["titles"]:
            row["titles"].append(title)
    ranked = sorted(evidence.items(), key=lambda item: (-int(item[1].get("count", 0)), item[0]))
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, payload in ranked[:max(1, max_researchers)]:
        out.append((name, {
            "count": int(payload.get("count", 0) or 0),
            "titles": list(payload.get("titles", [])[:max(1, max_titles_per_researcher)]),
        }))
    return out


def _extract_person_like_spans(text: str, *, max_items: int = 24) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    pattern = re.compile(r"\b(?:[A-Z][A-Za-z'\-]*\.?\s+){1,3}[A-Z][A-Za-z'\-]+\b")
    out: List[str] = []
    seen = set()
    for span in pattern.findall(raw):
        candidate = re.sub(r"\s+", " ", span.strip())
        if not candidate or not looks_like_person_candidate(candidate):
            continue
        key = norm_text(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
        if len(out) >= max(1, max_items):
            break
    return out


def _answer_mentions_unsupported_researcher(answer: str, docs) -> bool:
    answer_people = _extract_person_like_spans(answer)
    if not answer_people:
        return False

    # Collect signatures from doc metadata (both directions)
    doc_sigs: List[Dict[str, str]] = []
    doc_names: List[str] = []
    seen_names = set()
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        candidates = []
        researcher = re.sub(r"\s+", " ", str(meta.get("researcher", "") or "").strip())
        if researcher:
            candidates.append(researcher)
        authors = str(meta.get("authors", "") or "")
        candidates.extend(_split_author_names(authors))
        for candidate in candidates:
            person = re.sub(r"\s+", " ", str(candidate or "").strip())
            if not person or not looks_like_person_candidate(person):
                continue
            key = norm_text(person)
            if key in seen_names:
                continue
            seen_names.add(key)
            sig = _person_name_signatures(person)
            if sig:
                doc_sigs.append(sig)
                doc_names.append(person)

    if not doc_sigs:
        return False

    for person in answer_people:
        # Standard check: answer person against doc sigs
        if any(_name_match_strength(person, sig) >= 2 for sig in doc_sigs):
            continue
        # Reverse check: doc names against answer person's sig.
        # This catches "Duncan Brown" (answer) vs "D. Brown" (doc):
        # _name_match_strength("D. Brown", sig_for_Duncan_Brown) gives 2
        # because D matches first initial of Duncan.
        answer_sig = _person_name_signatures(person)
        if answer_sig and any(_name_match_strength(doc_name, answer_sig) >= 2 for doc_name in doc_names):
            continue
        # Last-name-only fallback: if both share the same last name,
        # don't flag as unsupported (common in academic metadata).
        person_last = (answer_sig.get("last", "") or "").lower() if answer_sig else ""
        if person_last and any((sig.get("last", "") or "").lower() == person_last for sig in doc_sigs):
            continue
        return True
    return False


def _build_researcher_extract_answer(docs, *, max_researchers: int = 5) -> Optional[str]:
    evidence = _supported_researcher_evidence(docs, max_researchers=max_researchers,
                                              max_titles_per_researcher=5)
    if not evidence:
        return None
    intro = "Based on the retrieved papers, the following researchers are directly supported by the evidence:"
    paragraphs = [intro]
    for name, payload in evidence:
        titles = list(payload.get("titles", []) or [])
        count = int(payload.get("count", 0) or 0)
        if titles:
            title_text = "; ".join(f'"{t}"' for t in titles[:5])
            paragraphs.append(f"{name}: supported by {title_text}.")
        else:
            paragraphs.append(f"{name}: supported by {count} retrieved paper(s).")
    return "\n\n".join(paragraphs).strip()


def _collect_doc_titles(docs) -> set:
    """Collect normalized titles from retrieved documents for validation."""
    titles = set()
    for d in docs or []:
        meta = getattr(d, "metadata", {}) or {}
        raw_title = re.sub(r"</?[a-zA-Z][^>]*>", "", str(meta.get("title", "") or "")).strip()
        raw_title = re.sub(r"\s+", " ", raw_title).strip()
        if raw_title and raw_title.lower() not in {"untitled", "unknown", "n/a"}:
            titles.add(raw_title.lower())
    return titles


def _strip_hallucinated_citations(answer: str, docs) -> str:
    """Remove sentences containing fabricated paper titles/citations not found
    in retrieved documents.  Uses fuzzy matching (SequenceMatcher ratio >= 0.6)
    to account for minor formatting differences."""
    if not answer or not docs:
        return answer

    doc_titles = _collect_doc_titles(docs)
    if not doc_titles:
        return answer

    # Find all quoted strings in the answer that look like paper titles
    quoted_pattern = re.compile(r'"([^"]{15,200})"')
    # Also find citation-like patterns: Author et al., "Title", Journal...
    citation_pattern = re.compile(
        r'[A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*["\u201c]([^"\u201d]{15,200})["\u201d]')

    lines = answer.split("\n")
    cleaned_lines = []
    for line in lines:
        # Check each quoted title in this line against doc titles
        has_fabricated = False
        for pattern in [quoted_pattern, citation_pattern]:
            for match in pattern.finditer(line):
                cited_title = re.sub(r"\s+", " ", match.group(1)).strip().lower()
                if not cited_title:
                    continue
                # Check if this title matches any retrieved doc title
                matched = False
                for doc_title in doc_titles:
                    if cited_title in doc_title or doc_title in cited_title:
                        matched = True
                        break
                    ratio = SequenceMatcher(None, cited_title, doc_title).ratio()
                    if ratio >= 0.6:
                        matched = True
                        break
                if not matched:
                    has_fabricated = True
                    break
            if has_fabricated:
                break

        if not has_fabricated:
            cleaned_lines.append(line)
        # else: drop the line with fabricated citation

    result = "\n".join(cleaned_lines).strip()
    return result if result else answer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, user_key: str,
                    use_graph: Optional[bool] = None, stateless: Optional[bool] = None) -> Dict[str, Any]:
    q = (question or "").strip()
    if not q:
        return {"answer": PIPELINE_CFG["empty_question_answer"],
                "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": ""}

    # Fix #4: Intercept meta-commands before any retrieval or generation
    if is_meta_command(q):
        mgr = get_global_manager()
        stateless_flag = bool(stateless) if stateless is not None else bool(getattr(settings, "stateless_default", False))
        if not stateless_flag:
            try:
                store = mgr.store
                state = store.load(user_key)
                turns = state.get("turns", []) or []
                turns.append({"role": "user", "text": q})
                turns.append({"role": "assistant", "text": "Topic cleared."})
                store.save(user_key, "", turns, extra_state={
                    "last_focus": "", "last_topic": "",
                    "anchor": {}, "anchor_last_action": "meta_command_reset",
                    "summary_updated": False, "retrieval_confidence": "",
                })
            except Exception:
                pass
        return {"answer": "Topic cleared. What would you like to explore next?",
                "sources": [], "graph_hits": [], "graph_graph": {}, "graph_error": "",
                "timing_ms": {"total_ms": 0.0}, "llm_calls": {"answer_llm_calls": 0, "utility_llm_calls": 0}}

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
    cache_state_sig = state_signature_from_state(pre_state if isinstance(pre_state, dict) else {})
    cache_key = build_pipeline_cache_key(user_key=user_key, resolved_text=q,
                                         effective_mode=effective_mode, state_signature=cache_state_sig)
    if (not stateless) and PIPELINE_CFG["qa_cache_enable"]:
        cached = get_cached_answer(user_key, cache_key)
        if isinstance(cached, dict):
            return cached

    mgr.switch_mode(effective_mode)
    mgr.switch_answer_model(answer_model_key)

    engine_lock_ctx = nullcontext()
    if hasattr(mgr, "answer_generation_lock") and not bool(getattr(settings, "allow_utility_concurrency", False)):
        engine_lock_ctx = mgr.answer_generation_lock

    with engine_lock_ctx:
        eng = mgr.get_engine(user_key, mode=effective_mode, stateless=stateless)
        context = eng.prepare_context(q, stateless=stateless)
        context.raw_question = q

        paper_docs_raw = list(context.paper_docs or [])
        resolved_q = (context.rewritten_question or "").strip()
        prev_user_text = ""
        for t in reversed(list(pre_state.get("turns", []) or [])):
            if str((t or {}).get("role", "") or "").strip().lower() == "user":
                prev_user_text = str((t or {}).get("text", "") or "").strip()
                break
        resolved_q = re.sub(r"\s+", " ", (resolved_q or q)).strip()

        summary_intent = _is_summary_intent(q)
        detected_intent = str(getattr(context, "detected_intent", "") or classify_generic_intent(q))
        retrieval_q = resolved_q or q
        raw_retrieval_count = len(paper_docs_raw)

        first_pass_docs = _filter_noisy_docs(paper_docs_raw, retrieval_q)
        dominance = _dominant_metadata_filter_from_docs(
            first_pass_docs, resolved_q or q,
            majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
            min_count=int(getattr(settings, "dominant_min_count", 3)))
        dominant_filter = dict(dominance.get("filter", {}) or {})
        dom_conf = float(dominance.get("confidence", 0) or 0)
        dom_ratio = float(dominance.get("ratio", 0) or 0)
        replace_conf = float(getattr(settings, "dominant_replace_confidence", 0.82))
        maj_ratio = float(getattr(settings, "dominant_majority_ratio", 0.6))
        # --- Bug 6 fix: Relax dom_usable gate.  The previous strong_floor was
        # too high (0.84) and sometimes blocked targeted second-pass retrieval
        # even when dominance was clearly established.  Now: if the dominance
        # detector already confirmed dominant=True (ratio >= majority_ratio AND
        # confidence >= conf_floor), that's sufficient for a second-pass query. ---
        dom_usable = (bool(dominance.get("dominant")) and bool(dominant_filter)
                      and not is_placeholder_anchor_value(str(dominant_filter.get("value", "") or "")))

        paper_docs = list(first_pass_docs)
        dom_second_count = 0
        budget_papers = int((context.budgets or {}).get("BUDGET_PAPERS", getattr(settings, "budget_papers", 0)))
        explicit_person_name = ""
        person_support = {"strong": 0, "matched": 0, "total": 0}

        if dom_usable:
            where = {str(dominant_filter.get("key", "")): str(dominant_filter.get("value", ""))}
            second = _filter_noisy_docs(
                eng.retrieve_papers(retrieval_q, budget_papers, query_embedding=None,
                                    where_filter=where, raw_question=q), retrieval_q)
            dom_second_count = len(second)
            paper_docs = dedupe_docs(paper_docs + second)

        if not dom_usable and has_explicit_entity_signal(q):
            person_name = strip_possessive(_extract_person_name(q) or "")
            candidate_support = {"strong": 0, "matched": 0, "total": 0}
            if person_name and len(person_name) > 3 and looks_like_person_candidate(person_name):
                candidate_ranked = _rank_docs_for_person(first_pass_docs, person_name)
                _, candidate_support = _select_docs_for_person(candidate_ranked, max_docs=4)
            if (person_name and len(person_name) > 3 and looks_like_person_candidate(person_name)
                and candidate_support["matched"] > 0):
                explicit_person_name = person_name
                sig = _person_name_signatures(person_name)
                targeted_docs = []
                seen_variants = set()
                for variant in [sig.get("full_name", ""), sig.get("initial_last_dot", ""),
                                sig.get("initial_last_plain", "")]:
                    v = re.sub(r"\s+", " ", str(variant or "").strip())
                    if not v:
                        continue
                    v_key = v.lower()
                    if v_key in seen_variants:
                        continue
                    seen_variants.add(v_key)
                    docs_full = eng.retrieve_papers(
                        retrieval_q, budget_papers, query_embedding=None,
                        where_filter={"researcher": v}, raw_question=q)
                    if docs_full:
                        targeted_docs.extend(_filter_noisy_docs(docs_full, retrieval_q))

                merged_docs = dedupe_docs(paper_docs + targeted_docs)
                ranked_docs = _rank_docs_for_person(merged_docs, person_name)
                strong_docs = [d for d, score in ranked_docs if score >= 3.0]
                matched_docs = [d for d, score in ranked_docs if score >= 1.0]

                # Last-name retrieval is allowed only as low-priority backfill.
                last_name = str(sig.get("last_name", "") or "").strip()
                if len(strong_docs) < 3 and last_name:
                    docs_last = eng.retrieve_papers(
                        retrieval_q, budget_papers, query_embedding=None,
                        where_filter={"researcher": last_name}, raw_question=q)
                    if docs_last:
                        merged_docs = dedupe_docs(merged_docs + _filter_noisy_docs(docs_last, retrieval_q))
                        ranked_docs = _rank_docs_for_person(merged_docs, person_name)
                        strong_docs = [d for d, score in ranked_docs if score >= 3.0]
                        matched_docs = [d for d, score in ranked_docs if score >= 1.0]

                selected_docs, person_support = _select_docs_for_person(
                    ranked_docs,
                    max_docs=min(max(6, budget_papers or 12), 12),
                )
                if selected_docs:
                    paper_docs = dedupe_docs(selected_docs)

                # Re-compute dominance after person-aware reranking.
                dominance = _dominant_metadata_filter_from_docs(
                    paper_docs, resolved_q or q,
                    majority_ratio=float(getattr(settings, "dominant_majority_ratio", 0.6)),
                    min_count=int(getattr(settings, "dominant_min_count", 3)))
                dominant_filter = dict(dominance.get("filter", {}) or {})

        post_filter_count = len(paper_docs)
        mem_docs = context.mem_docs or []

        anchor_before = normalize_anchor(getattr(context, "anchor", {}) or getattr(eng, "anchor", {}))
        candidate_anchor = _build_anchor_from_dominance(dominance)

        if explicit_person_name and person_support["matched"] > 0:
            explicit_conf = min(
                0.9,
                max(
                    0.58,
                    float(person_support["matched"]) / max(1, int(person_support["total"])),
                ),
            )
            candidate_anchor = {
                "type": "researcher",
                "value": explicit_person_name,
                "source": "explicit_entity_signal",
                "confidence": explicit_conf,
            }

        # If dominance didn't produce an anchor candidate but we have a named
        # person in the query, build an anchor from that so follow-up pronouns
        # ("his", "her") resolve correctly.
        if not candidate_anchor and not dom_usable:
            person_name = strip_possessive(_extract_person_name(q) or "")
            if person_name and len(person_name) > 3 and looks_like_person_candidate(person_name):
                ratio = anchor_support_ratio(person_name, paper_docs)
                if ratio >= 0.1:
                    candidate_anchor = {
                        "type": "researcher", "value": person_name,
                        "source": "explicit_entity_signal",
                        "confidence": min(0.85, max(0.6, ratio)),
                    }

        anchor_after, anchor_action = _choose_anchor_update(
            current_anchor=anchor_before, candidate_anchor=candidate_anchor,
            dominance=dominance, question=q, resolved_question=resolved_q)

        # Clear anchor when the new query has zero keyword overlap with the
        # anchor value — catches topic shifts like going from "William Gearty"
        # to "neurological injury" without explicit shift phrases.
        if anchor_after and anchor_after.get("value"):
            anchor_val = str(anchor_after.get("value", "") or "").strip()
            if anchor_val and not anchor_in_text(anchor_val, q):
                a_toks = {t for t in tokenize_words(anchor_val) if len(t) >= 3 and not is_generic_query_token(t)}
                q_toks = {t for t in tokenize_words(q) if len(t) >= 3}
                if a_toks and not (a_toks & q_toks) and not is_followup_coref_question(q):
                    anchor_after = {}
                    anchor_action = "cleared_no_query_overlap"

        eng.anchor = dict(anchor_after or {})
        eng.anchor_last_action = anchor_action
        context.anchor = dict(anchor_after or {})
        context.paper_docs = list(paper_docs)

        anchor_value = str(anchor_after.get("value", "") or "").strip()
        anc_ratio = anchor_support_ratio(anchor_value, paper_docs) if anchor_value else 1.0

        # If the anchor is present but almost entirely absent from the retrieved docs
        # (ratio < 0.15) and this is not a follow-up coref question, the anchor is
        # stale — clear it now so retrieval_confidence isn't dragged to "inconsistent".
        # Exception: if dominance already confirmed the anchor via a targeted second pass
        # (dom_second_count > 0) or the query is a summary/follow-up of the anchor entity,
        # keep it even with a low first-pass ratio.
        anchor_confirmed_by_dominance = (
            dom_second_count > 0
            and anchor_value
            and _normalize_meta_value(dominant_filter.get("value", "")) == _normalize_meta_value(anchor_value)
        )
        if (anchor_value and anc_ratio < 0.15
                and not is_followup_coref_question(q)
                and not anchor_confirmed_by_dominance):
            anchor_after = {}
            anchor_action = "cleared_low_support_ratio"
            anchor_value = ""
            anc_ratio = 1.0
            eng.anchor = {}
            eng.anchor_last_action = anchor_action
            context.anchor = {}

        anchor_consistent = (not anchor_value) or (anc_ratio >= float(
            getattr(settings, "anchor_consistency_min_ratio", 0.45)))
        retrieval_confidence = retrieval_confidence_label(
            docs_count=post_filter_count, anchor_consistent=anchor_consistent)
        if explicit_person_name and person_support["total"] > 0:
            retrieval_confidence = _downshift_confidence_for_person_support(
                retrieval_confidence,
                strong_count=int(person_support["strong"]),
                matched_count=int(person_support["matched"]),
                total_count=int(person_support["total"]),
            )
        eng.last_retrieval_confidence = retrieval_confidence
        eng.last_anchor_support_ratio = float(anc_ratio)

        prompt_max_docs = int(getattr(settings, "prompt_max_docs", 24))
        prompt_text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
        if retrieval_confidence in {"weak", "inconsistent"}:
            prompt_max_docs = min(prompt_max_docs, max(2, int(getattr(settings, "low_conf_prompt_max_docs", 8))))
            prompt_text_limit = min(prompt_text_limit, max(160, int(getattr(settings, "low_conf_prompt_doc_text_limit", 420))))

        # Prompt assembly
        rolling_summary_text = pre_state.get("rolling_summary", "") or ""
        pre_turns_count = len(pre_state.get("turns", []) or [])
        # Only include recent turns context when the topic is continuous.
        # When the anchor was cleared due to a topic shift, recent turns from a different
        # topic will cause the LLM to hallucinate or open with the wrong subject.
        topic_shifted = anchor_action in {
            "cleared_no_query_overlap", "cleared_low_support_ratio",
        } or (anchor_action == "none" and not anchor_before and not anchor_after)
        recent_turns_ctx = ""
        if not topic_shifted:
            recent_turns_ctx = _build_recent_turns_context(
                pre_state, max_turns=min(6, max(2, int(PIPELINE_CFG["recent_turns_in_prompt"]))))

        # Retrieved context goes first so the LLM grounds on actual paper data.
        # Rolling summary is NOT included — the LLM should answer from docs only,
        # not from conversation history that can leak stale/wrong researcher names.
        prompt_base = [PIPELINE_CFG["prompt_prefix"].rstrip()]
        if recent_turns_ctx:
            prompt_base.append("RECENT TURNS (for coreference only, not evidence):\n" + recent_turns_ctx)
        if mem_docs:
            mem_lines = [f"- {str(getattr(md, 'page_content', '') or '').strip()[:200]}"
                         for md in mem_docs[:6]
                         if str(getattr(md, "page_content", "") or "").strip()]
            if mem_lines:
                prompt_base.append("CONVERSATION MEMORY:\n" + "\n".join(mem_lines))

        question_for_answer = (resolved_q or q).lower()

        style_hints = {
            "summary": ("The retrieved context contains paper records with a 'summary' field. "
                        "Use ONLY the content from these summary fields to answer. "
                        "Extract and report the key mechanisms, findings, and methods described "
                        "in those summaries directly — do not invent or paraphrase beyond what "
                        "is stated. Attribute each point to its paper title and researcher. "
                        "If a paper has no summary field, describe it based on its title only. "
                        "Write at least one detailed paragraph per relevant paper."),
            "comparison": "Compare the relevant items directly and keep claims grounded in retrieved evidence.",
            "time_range": ("If asked about the studied time period, report it only when explicitly stated in retrieved context. "
                           "If not stated, say that directly and optionally provide publication years as a separate fallback. "
                           "Do not infer studied period from publication years."),
            "list": "Provide a concise list with short evidence-backed descriptors.",
        }
        if summary_intent:
            style_hint = style_hints["summary"]
        elif detected_intent in style_hints:
            style_hint = style_hints[detected_intent]
        else:
            style_hint = ("Give a direct evidence-grounded answer in plain prose. "
                          "Base your answer STRICTLY on the retrieved paper records provided below — "
                          "do NOT use outside knowledge or information not present in the retrieved records. "
                          "Each retrieved record lists a researcher, title, and often a summary field. "
                          "Read the summary fields and use them as your primary source of information. "
                          "You MUST write at least 4-5 detailed sentences. "
                          "For each researcher found in the retrieved records, give them their own paragraph: "
                          "name the researcher, describe their specific research topics drawn from the summaries, "
                          "and cite at least one paper title as supporting evidence. "
                          "Never respond with only '[Name] is a researcher' — always elaborate with what they study. "
                          "Never fabricate facts not present in the retrieved records.")

        prompt, used_docs, used_text_limit = _fit_prompt_to_budget(
            runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
            base_sections=prompt_base, style_hint=style_hint,
            question_for_answer=question_for_answer,
            max_docs=prompt_max_docs, text_limit=prompt_text_limit)

        # Answer generation
        if not paper_docs:
            answer_text = _insufficient_context_answer(q, detected_intent)
        else:
            # LLM call
            prompt_for_call = prompt

            # Guard against hallucination: when retrieval is weak/low AND the
            # anchor was cleared (topic shift), inject a strict constraint to
            # prevent fabricating connections to previous conversation topics.
            low_evidence_guard = (
                retrieval_confidence in {"weak", "low"}
                and anchor_action in {"cleared_no_query_overlap", "cleared_weak_retrieval", "none"}
                and post_filter_count <= 4
            )

            if retrieval_confidence == "inconsistent" or low_evidence_guard:
                base_no_summary = [s for s in prompt_base if not s.startswith("ROLLING SUMMARY")]
                if retrieval_confidence == "inconsistent":
                    # For inconsistent retrievals, use a strict guard that prevents
                    # the LLM from drawing on prior conversation topics or outside knowledge.
                    inconsistent_guard_style = (
                        "CRITICAL: The retrieved evidence is inconsistent — the papers do not "
                        "all match the current question. "
                        "You MUST ONLY report what is EXPLICITLY stated in the retrieved paper records. "
                        "Do NOT draw on outside knowledge, training data, or prior conversation topics. "
                        "List the researchers found in the retrieved records along with their paper titles. "
                        "For each researcher, briefly describe what their retrieved papers are about. "
                        "If a researcher's paper is off-topic, still list it — do not invent relevance. "
                        "Do NOT fabricate facts, affiliations, or locations not in the records."
                    )
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=inconsistent_guard_style,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)
                elif low_evidence_guard:
                    guard_style = (
                        "CRITICAL: Very few papers were retrieved and evidence is weak. "
                        "Only describe what is EXPLICITLY stated in the retrieved papers. "
                        "Do NOT speculate, infer connections, or relate this topic to "
                        "previous conversation topics. If the papers don't clearly address "
                        "the question, say so directly and list the papers that were found. "
                        "Do NOT fabricate relevance between unrelated research areas."
                    )
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=guard_style,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)
                else:
                    prompt_for_call, used_docs, used_text_limit = _fit_prompt_to_budget(
                        runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                        base_sections=base_no_summary, style_hint=style_hint,
                        question_for_answer=question_for_answer,
                        max_docs=prompt_max_docs, text_limit=prompt_text_limit)

            t0_gen = time.perf_counter()
            try:
                answer_llm_calls += 1
                raw_answer = _invoke_with_timeout(mgr.answer_runtime.llm, prompt_for_call,
                                                  int(getattr(settings, "llm_timeout_s", 40)))
            except Exception:
                raw_answer = ""
            generation_time_ms = (time.perf_counter() - t0_gen) * 1000.0
            answer_text = _extract_answer_text(raw_answer)

            if not answer_text:
                retry_base = base_no_summary if (retrieval_confidence == "inconsistent" or low_evidence_guard) else prompt_base
                retry_prompt, _, _ = _fit_prompt_to_budget(
                    runtime=getattr(mgr, "answer_runtime", None), docs=paper_docs,
                    base_sections=retry_base, style_hint=style_hint,
                    question_for_answer=question_for_answer,
                    max_docs=max(1, min(6, used_docs // 2)),
                    text_limit=max(140, min(320, int(max(120, used_text_limit) * 0.7))))
                t0_r = time.perf_counter()
                try:
                    answer_llm_calls += 1
                    raw_retry = _invoke_with_timeout(mgr.answer_runtime.llm, retry_prompt,
                                                     int(getattr(settings, "llm_timeout_s", 40)))
                except Exception:
                    raw_retry = ""
                generation_time_ms += (time.perf_counter() - t0_r) * 1000.0
                answer_text = _extract_answer_text(raw_retry)

            if not answer_text:
                answer_text = PIPELINE_CFG["llm_no_answer"]

        if paper_docs:
            cleaned = (answer_text or "").strip()
            weak_or_inconsistent = retrieval_confidence in {"weak", "inconsistent"}
            if (not cleaned
                or cleaned == PIPELINE_CFG["llm_no_answer"]
                or len(cleaned) < 24
                or (weak_or_inconsistent and len(cleaned) < 120)):
                answer_text = _fallback_answer_from_docs(q, paper_docs, detected_intent) or answer_text

            researcher_evidence = _supported_researcher_evidence(paper_docs)
            explicit_person_query = bool(strip_possessive(_extract_person_name(q) or ""))

            # --- Bug 5 fix: Strip fabricated citations before grounding checks.
            # The 3B model sometimes invents plausible-looking paper titles, DOIs,
            # and journal references that don't exist in retrieved docs.  Remove
            # sentences containing such fabricated citations. ---
            answer_text = _strip_hallucinated_citations(answer_text, paper_docs)

            # Grounding check: only replace with extract answer when hallucination is
            # clearly present AND retrieval confidence is low/weak.
            # Do NOT replace on medium/high confidence — the LLM likely answered correctly
            # and the name-match heuristic has false positives (initials vs full names, etc.)
            has_unsupported = _answer_mentions_unsupported_researcher(answer_text, paper_docs)

            # Only trigger the sterile extract on weak/low confidence where hallucination risk is high
            if has_unsupported and retrieval_confidence in {"weak", "low"}:
                safe_answer = _build_researcher_extract_answer(paper_docs)
                if safe_answer:
                    answer_text = safe_answer
                else:
                    answer_text = _fallback_answer_from_docs(q, paper_docs, detected_intent) or answer_text
            elif (not explicit_person_query and len(researcher_evidence) >= 2
                  and has_unsupported and retrieval_confidence == "inconsistent"):
                answer_text = _build_researcher_extract_answer(paper_docs) or answer_text

        answer_text = _sanitize_user_answer(answer_text).strip() or PIPELINE_CFG["llm_no_answer"]
        eng.last_answer_llm_calls = int(answer_llm_calls)
        eng.finalize_turn(context, answer_text, no_results=not paper_docs)

    return _build_output(
        answer_text=answer_text, paper_docs=paper_docs, mem_docs=mem_docs,
        eng=eng, mgr=mgr, q=q, resolved_q=resolved_q, retrieval_q=retrieval_q,
        detected_intent=detected_intent, effective_mode=effective_mode,
        requested_mode=requested_mode, retrieval_confidence=retrieval_confidence,
        dominance=dominance, dominant_filter=dominant_filter, dom_usable=dom_usable,
        anchor_before=anchor_before, anchor_after=anchor_after, anchor_action=anchor_action,
        anc_ratio=anc_ratio, anchor_consistent=anchor_consistent,
        raw_retrieval_count=raw_retrieval_count, first_pass_docs=first_pass_docs,
        dom_second_count=dom_second_count, post_filter_count=post_filter_count,
        rolling_summary_text=rolling_summary_text, pre_turns_count=pre_turns_count,
        stateless=stateless, t0_total=t0_total, generation_time_ms=generation_time_ms,
        answer_llm_calls=answer_llm_calls, user_key=user_key, use_graph_flag=use_graph_flag,
        previous_pipeline=previous_pipeline, pre_state=pre_state,
        cache_state_sig=cache_state_sig, prompt_max_docs=prompt_max_docs,
        prompt_text_limit=prompt_text_limit)


def _build_output(*, answer_text, paper_docs, mem_docs, eng, mgr, q, resolved_q,
                  retrieval_q, detected_intent, effective_mode, requested_mode,
                  retrieval_confidence, dominance, dominant_filter, dom_usable,
                  anchor_before, anchor_after, anchor_action, anc_ratio, anchor_consistent,
                  raw_retrieval_count, first_pass_docs, dom_second_count, post_filter_count,
                  rolling_summary_text, pre_turns_count, stateless, t0_total,
                  generation_time_ms, answer_llm_calls, user_key, use_graph_flag,
                  previous_pipeline, pre_state, cache_state_sig,
                  prompt_max_docs, prompt_text_limit) -> Dict[str, Any]:

    post_state = ({"rolling_summary": rolling_summary_text, "turns": pre_state.get("turns", []) or []}
                  if stateless else mgr.store.load(user_key))
    post_turns = post_state.get("turns", []) or []
    post_summary = post_state.get("rolling_summary", "") or ""
    post_extra = post_state.get("extra_state", {}) if isinstance(post_state.get("extra_state"), dict) else {}
    post_anchor = normalize_anchor(post_state.get("anchor") or post_extra.get("anchor") or anchor_after)
    post_anchor_action = str(post_state.get("anchor_last_action") or post_extra.get("anchor_last_action") or anchor_action)
    post_summary_updated = bool(post_state.get("summary_updated") if "summary_updated" in post_state
                                else post_extra.get("summary_updated", getattr(eng, "last_summary_updated", False)))
    post_conf = str(post_state.get("retrieval_confidence") or post_extra.get("retrieval_confidence") or retrieval_confidence)
    post_anc_ratio = float(post_extra.get("anchor_support_ratio",
                           getattr(eng, "last_anchor_support_ratio", anc_ratio)) or 0.0)

    # Sources
    sources, seen_titles = [], set()
    for d in paper_docs:
        try:
            meta = getattr(d, "metadata", {}) or {}
            tk = re.sub(r"\s+", " ", str(meta.get("title", "") or "").strip().lower())
            if tk and tk != "untitled" and tk in seen_titles:
                continue
            if tk and tk != "untitled":
                seen_titles.add(tk)
            sources.append(_doc_to_source_md(d))
        except Exception:
            pass

    timing = {
        "rewrite_ms": round(float(getattr(eng, "last_rewrite_time_ms", 0) or 0), 2),
        "retrieval_total_ms": round(float(getattr(eng, "last_retrieval_time_ms", 0) or 0), 2),
        "generation_ms": round(float(generation_time_ms or 0), 2),
        "total_ms": round((time.perf_counter() - t0_total) * 1000.0, 2),
    }
    llm_calls = {
        "answer_llm_calls": int(getattr(eng, "last_answer_llm_calls", answer_llm_calls) or 0),
        "utility_llm_calls": int(getattr(eng, "last_utility_llm_calls", 0) or 0),
    }
    timing["llm_calls"] = llm_calls

    session_state = {
        "session_id": user_key, "pre_turn_count": pre_turns_count,
        "turn_count": len(post_turns), "turn_count_delta": len(post_turns) - pre_turns_count,
        "pre_summary_len": len(rolling_summary_text), "summary_len": len(post_summary),
        "summary_len_delta": len(post_summary) - len(rolling_summary_text),
        "summary_non_empty": bool(post_summary.strip()),
        "turns_json_chars": len(json.dumps(post_turns, ensure_ascii=False)),
        "detected_intent": detected_intent, "requested_mode": requested_mode,
        "effective_mode": effective_mode, "anchor": post_anchor,
        "anchor_action": post_anchor_action, "summary_updated": post_summary_updated,
        "retrieval_confidence": post_conf, "anchor_support_ratio": round(post_anc_ratio, 4),
        "timing_ms": timing, "llm_calls": llm_calls,
    }

    chroma_refs = [_doc_to_ref(d) for d in paper_docs[:12]]
    mem_refs = [_doc_to_ref(d) for d in mem_docs[:8]]
    ret_digest = retrieval_cache_summary(paper_docs, retrieval_text=retrieval_q, limit_ids=12)

    chroma_json = {
        "count": len(paper_docs), "retrieval_count_raw": raw_retrieval_count,
        "first_pass_count": len(first_pass_docs), "dominant_second_pass_count": dom_second_count,
        "fallback_unfiltered_count": 0, "post_filter_count": post_filter_count,
        "dominance": dominance, "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dom_usable,
        "anchor_support_ratio": round(float(anc_ratio), 4),
        "anchor_consistent": bool(anchor_consistent), "retrieval_confidence": retrieval_confidence,
        "doc_refs": chroma_refs, "retrieval_digest": ret_digest,
    }

    user_query_json = {
        "text": q, "resolved_text": resolved_q, "standalone_question": resolved_q,
        "detected_intent": detected_intent, "retrieval_text": retrieval_q,
        "dominance": dominance, "dominant_metadata_filter": dominant_filter,
        "dominant_filter_usable": dom_usable,
        "anchor_before": anchor_before, "anchor_after": anchor_after, "anchor_action": anchor_action,
        "rewrite_blocked": bool(getattr(eng, "last_rewrite_blocked", False)),
        "rewrite_anchor_valid": bool(getattr(eng, "last_rewrite_anchor_valid", False)),
        "requested_mode": requested_mode, "effective_mode": effective_mode,
        "stateless": stateless, "timestamp": utcnow_iso(),
    }

    rolling_json = {"summary": post_summary, "turns": len(post_turns), "updated": post_summary_updated}
    cache_json = {
        "previous_pipeline_present": bool(previous_pipeline),
        "previous_pipeline_sig": short_hash(
            json.dumps((previous_pipeline or {}).get("session_state", {}), ensure_ascii=False, sort_keys=True)
            if isinstance(previous_pipeline, dict) else "", length=10),
        "timestamp": utcnow_iso(),
    }

    combined = {
        "user_query": user_query_json, "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": mem_refs},
        "cache": cache_json, "session_state": session_state,
        "context_size_chars": len(build_compact_context(paper_docs, max_docs=prompt_max_docs,
                                                        text_limit=prompt_text_limit)),
        "timing_ms": timing, "llm_calls": llm_calls,
    }

    if getattr(settings, "debug_rag", False):
        try:
            print("\n[PIPELINE_JSON]")
            print(json.dumps(combined, ensure_ascii=False, indent=2))
        except Exception:
            pass

    cacheable = should_cache_turn(retrieval_text=retrieval_q,
                                  rewrite_blocked=bool(getattr(eng, "last_rewrite_blocked", False)))
    pipeline_cache = {
        "user_query": {k: user_query_json[k] for k in
                       ("text", "resolved_text", "retrieval_text", "detected_intent", "effective_mode")},
        "chroma_retrieval": {k: chroma_json[k] for k in
                             ("count", "retrieval_count_raw", "post_filter_count", "retrieval_confidence", "retrieval_digest")},
        "rolling_summary": rolling_json,
        "conversation_memory": {"count": len(mem_docs)}, "cache": cache_json,
        "session_state": {"turn_count": session_state["turn_count"], "summary_len": session_state["summary_len"],
                          "effective_mode": effective_mode, "llm_calls": llm_calls},
        "timing_ms": timing,
    }
    set_pipeline_cache(user_key, pipeline_cache if cacheable else {})

    out = {
        "answer": answer_text, "sources": sources,
        "graph_hits": [], "graph_graph": {}, "graph_error": "",
        "user_query": user_query_json, "chroma_retrieval": chroma_json,
        "rolling_summary": rolling_json,
        "conversation_memory": {"count": len(mem_docs), "doc_refs": mem_refs},
        "cache": cache_json, "session_state": session_state,
        "timing_ms": timing, "llm_calls": llm_calls, "pipeline_json": combined,
    }

    if use_graph_flag:
        g = graph_retrieve_from_paper_docs(paper_docs, height=650)
        out["graph_hits"] = g.get("hits", []) or []
        out["graph_graph"] = g.get("graph", {}) or {}
        out["graph_error"] = g.get("error", "") or ""

    if (not stateless) and PIPELINE_CFG["qa_cache_enable"] and (not use_graph_flag) and cacheable:
        cache_write_key = build_pipeline_cache_key(
            user_key=user_key, resolved_text=resolved_q or q,
            effective_mode=effective_mode, state_signature=cache_state_sig)
        set_cached_answer(user_key, cache_write_key, {
            "answer": out["answer"], "sources": list(out.get("sources", []))[:10],
            "graph_hits": [], "graph_graph": {}, "graph_error": "",
            "user_query": pipeline_cache["user_query"],
            "chroma_retrieval": pipeline_cache["chroma_retrieval"],
            "rolling_summary": rolling_json,
            "conversation_memory": {"count": len(mem_docs)},
            "cache": {"timestamp": utcnow_iso(), "retrieval_digest": ret_digest, "state_sig": cache_state_sig},
            "session_state": pipeline_cache["session_state"],
            "timing_ms": timing, "llm_calls": llm_calls,
        })

    return out