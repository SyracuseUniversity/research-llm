import os
import re
import uuid
import json
import gc
import time
import threading
import queue
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional, Set

import psutil
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
try:
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tree import Tree
except Exception:
    nltk = None
    ne_chunk = None
    pos_tag = None
    word_tokenize = None
    nltk_stopwords = None
    Tree = None

import chromadb

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

import config_full as config
from session_store import SessionStore
from database_manager import DatabaseManager
from runtime_settings import settings


MEMORY_DIR = os.getenv("RAG_MEMORY_DIR", "chroma_memory")
CACHE_DIR = os.getenv("RAG_CACHE_DIR", "cache")
STATE_DB = os.getenv("RAG_STATE_DB", "chat_state.sqlite")
_EMBED_DEVICE = os.getenv("RAG_EMBED_DEVICE", "").strip().lower()
if _EMBED_DEVICE not in {"cuda", "cpu"}:
    _EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _make_local_chroma_client(persist_dir: str) -> chromadb.Client:
    """
    Create a PersistentClient for the given directory, creating the directory
    if it does not already exist.

    Args:
        persist_dir: Path to the Chroma persistence directory.

    Returns:
        A chromadb.PersistentClient bound to persist_dir.

    Raises:
        chromadb exceptions if the client cannot be initialised (e.g. corrupt
        database or incompatible settings from a previous client instance).

    Note:
        PersistentClient is used consistently across the app to avoid
        "already exists with different settings" collisions that occur when
        mixing EphemeralClient and PersistentClient against the same directory.
    """
    _ensure_dir(persist_dir)
    return chromadb.PersistentClient(path=persist_dir)


def _dbg(title: str, obj: Any = None, limit: int = 2000) -> None:
    if not getattr(settings, "debug_rag", False):
        return
    print(f"\n{title}")
    if obj is None:
        return
    s = obj if isinstance(obj, str) else repr(obj)
    if limit and len(s) > limit:
        s = s[:limit] + "\n...truncated..."
    print(s)


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class EngineContext:
    raw_question: str
    rewritten_question: str
    detected_intent: str
    paper_docs: List[Document]
    mem_docs: List[Document]
    stateless: bool
    user_turns: int
    allow_prev_context: bool
    allow_summary: bool
    budgets: Dict[str, int]
    anchor: Dict[str, Any]


def available_ram_mb() -> int:
    return int(psutil.virtual_memory().available / (1024 * 1024))


def available_vram_mb() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free_b, _total_b = torch.cuda.mem_get_info()
        return int(free_b / (1024 * 1024))
    except Exception:
        return 0


def dynamic_budgets() -> Dict[str, int]:
    ram = available_ram_mb()
    vram = available_vram_mb()

    base_memory = int(getattr(settings, "budget_memory", 700))
    base_papers = int(getattr(settings, "budget_papers", 3200))
    base_trigger = int(getattr(settings, "trigger_tokens", 7200))

    pressure = 0
    if ram < 2000:
        pressure += 2
    elif ram < 4000:
        pressure += 1

    if torch.cuda.is_available():
        if vram < 1500:
            pressure += 2
        elif vram < 3000:
            pressure += 1

    if pressure >= 3:
        return {
            "BUDGET_MEMORY": max(300, int(base_memory * 0.65)),
            "BUDGET_PAPERS": max(1200, int(base_papers * 0.7)),
            "TRIGGER": max(3000, int(base_trigger * 0.88)),
        }
    if pressure == 2:
        return {
            "BUDGET_MEMORY": max(350, int(base_memory * 0.78)),
            "BUDGET_PAPERS": max(1600, int(base_papers * 0.84)),
            "TRIGGER": max(3500, int(base_trigger * 0.92)),
        }
    if pressure == 1:
        return {
            "BUDGET_MEMORY": max(450, int(base_memory * 0.9)),
            "BUDGET_PAPERS": max(2000, int(base_papers * 0.92)),
            "TRIGGER": max(4200, int(base_trigger * 0.97)),
        }
    return {
        "BUDGET_MEMORY": base_memory,
        "BUDGET_PAPERS": base_papers,
        "TRIGGER": base_trigger,
    }


def _norm_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _no_results_summary_line(question: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    if not q:
        return "No results for the last query."
    return f"No results for: {q}"


_SUMMARY_SECTIONS: Tuple[str, ...] = (
    "Current focus",
    "Core entities",
    "Key themes",
    "Constraints",
    "Open questions",
)

_SUMMARY_ALIASES: Dict[str, str] = {
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


def _summary_template_empty() -> str:
    blocks: List[str] = []
    for sec in _SUMMARY_SECTIONS:
        if sec == "Constraints":
            blocks.append("Constraints: Use only retrieved Syracuse corpus context.")
        else:
            blocks.append(f"{sec}: (none)")
    return "\n".join(blocks)


def _extract_summary_sections(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {sec: [] for sec in _SUMMARY_SECTIONS}
    current: Optional[str] = None
    for raw_line in (text or "").splitlines():
        line = (raw_line or "").strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z ]+):\s*(.*)$", line)
        if m:
            raw_key = (m.group(1) or "").strip().lower()
            canonical = _SUMMARY_ALIASES.get(raw_key)
            if canonical:
                current = canonical
                rest = (m.group(2) or "").strip()
                if rest:
                    out[canonical].append(rest.lstrip("- ").strip())
                continue
            current = None
            continue
        if current is not None:
            out[current].append(line.lstrip("- ").strip())
    return out


def _format_summary_sections(sections: Dict[str, List[str]]) -> str:
    blocks: List[str] = []
    for sec in _SUMMARY_SECTIONS:
        vals = [
            re.sub(r"\s+", " ", v.strip())
            for v in (sections.get(sec) or [])
            if v and v.strip()
        ]
        if not vals:
            vals = ["(none)"]
        blocks.append(f"{sec}: {' | '.join(vals)}")
    return "\n".join(blocks).strip()


def _clean_answer_for_summary_signal(text: str) -> str:
    blocked_fragments = (
        "no further analysis is required",
        "no additional retrieval is required",
        "no further synthesis is required",
        "confidence level",
        "the retrieved context",
    )
    lines: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line or "").strip()
        if not line:
            continue
        lower = line.lower()
        if any(fragment in lower for fragment in blocked_fragments):
            continue
        line = re.sub(r"^(summary|answer|response)\s*:\s*", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        lines.append(line)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _extract_answer_theme_keywords(answer_text: str, *, max_items: int = 6) -> List[str]:
    cleaned = _clean_answer_for_summary_signal(answer_text)
    if not cleaned:
        return []
    stopset = _get_stopword_set()
    counts: Dict[str, int] = {}
    for tok in _tokenize_words(cleaned):
        t = tok.lower().strip()
        if not t or len(t) < 4:
            continue
        if t.isdigit():
            continue
        if stopset and t in stopset:
            continue
        if _is_generic_query_token(t):
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
        if first_sentence:
            return [first_sentence[:180].rstrip()]
        return []
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    items = [token for token, _ in ranked]
    return items[: max(1, max_items)]


def _sanitize_entity_values(values: List[str], *, max_items: int = 8) -> List[str]:
    stopset = _get_stopword_set()
    out: List[str] = []
    seen = set()
    for raw in values or []:
        v = re.sub(r"\s+", " ", str(raw or "").strip())
        if not v:
            continue
        if len(v) < 3:
            continue
        if re.fullmatch(r"[A-Z][a-z]?$", v):
            continue
        toks = _tokenize_words(v)
        if not toks:
            continue
        if len(toks) == 1:
            tok = toks[0]
            if stopset and tok in stopset:
                continue
            if _is_generic_query_token(tok):
                continue
        if len(toks) > 8:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= max(1, max_items):
            break
    return out


def build_rolling_summary(
    previous_summary: str,
    user_question: str,
    retrieval_metadata: str,
    assistant_answer: str,
) -> str:
    base = previous_summary if (previous_summary or "").strip() else _summary_template_empty()
    sections = _extract_summary_sections(base)

    q = re.sub(r"\s+", " ", (user_question or "").strip())
    a = re.sub(r"\s+", " ", (assistant_answer or "").strip())
    raw_meta = re.sub(r"\s+", " ", (retrieval_metadata or "").strip())

    if q:
        sections["Current focus"] = [q[:220]]

    meta_entities: List[str] = []
    for part in re.split(r"\s*\|\s*", raw_meta):
        item = re.sub(r"\s+", " ", (part or "").strip())
        if not item or len(item) < 3:
            continue
        meta_entities.append(item)
    if meta_entities:
        existing = [v for v in sections.get("Core entities", []) if v and v.strip() and v.strip() != "(none)"]
        combined = _dedupe_ci(existing + meta_entities)
        max_items = max(1, int(getattr(settings, "summary_max_items_per_field", 6)))
        # Keep most recent entities (drop oldest if over the limit).
        sections["Core entities"] = combined[-max_items:]

    if a:
        existing_themes = [v for v in sections.get("Key themes", []) if v and v.strip() and v.strip() != "(none)"]
        max_items = max(1, int(getattr(settings, "summary_max_items_per_field", 6)))
        theme_keywords = _extract_answer_theme_keywords(a, max_items=max_items)
        if theme_keywords:
            combined_themes = _dedupe_ci(existing_themes + theme_keywords)
            sections["Key themes"] = combined_themes[-max_items:]

    if not sections.get("Constraints"):
        sections["Constraints"] = ["Use only retrieved Syracuse corpus context."]
    if not sections.get("Open questions"):
        sections["Open questions"] = ["(none)"]

    for sec in _SUMMARY_SECTIONS:
        if sec not in sections:
            sections[sec] = []
        vals = [v for v in sections.get(sec, []) if v and v.strip() and v.strip() != "(none)"]
        max_items = max(1, int(getattr(settings, "summary_max_items_per_field", 6)))
        sections[sec] = vals[-max_items:]

    limit = max(1, int(getattr(settings, "summary_max_chars", 1800)))
    summary = _format_summary_sections(sections)
    trim_order = ("Core entities", "Key themes", "Open questions", "Current focus")
    guard = 0
    while len(summary) > limit and guard < 256:
        guard += 1
        changed = False
        for key in trim_order:
            vals = sections.get(key, []) or []
            if len(vals) > 1:
                sections[key] = vals[1:]
                changed = True
                break
        if not changed:
            break
        summary = _format_summary_sections(sections)

    if len(summary) > limit:
        summary = summary[:limit].rstrip()
    if summary.strip():
        return summary
    if (previous_summary or "").strip():
        return previous_summary
    return _summary_template_empty()


def _invoke_with_timeout(llm: Any, prompt: str, timeout_s: int) -> str:
    if timeout_s <= 0:
        return str(llm.invoke(prompt) or "")
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(llm.invoke, prompt)
    try:
        return str(fut.result(timeout=timeout_s) or "")
    except FuturesTimeout:
        return ""
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _regenerate_rolling_summary(
    *,
    llm: Any,
    old_summary: str,
    new_turns_text: str,
    question: str,
    ner_line: str,
    source_context: str,
    no_results: bool,
    timeout_s: int,
) -> str:
    prompt_turns = (new_turns_text or "").strip()
    if no_results:
        line = _no_results_summary_line(question)
        prompt_turns = (prompt_turns + "\n" + line).strip() if prompt_turns else line

    answer_snippet = ""
    m = re.search(r"ASSISTANT:\s*(.+)", new_turns_text or "", re.IGNORECASE | re.DOTALL)
    if m:
        answer_snippet = re.sub(r"\s+", " ", m.group(1)).strip()
    if not answer_snippet:
        answer_snippet = re.sub(r"\s+", " ", new_turns_text or "").strip()

    if prompt_turns:
        try:
            msgs = SUMMARY_PROMPT.format_messages(
                old_summary=old_summary,
                new_turns=prompt_turns,
            )
            candidate = _invoke_with_timeout(
                llm,
                msgs[0].content + "\n" + msgs[1].content,
                timeout_s,
            ).strip()
            if candidate:
                old_summary = candidate
        except Exception:
            pass
    summary = build_rolling_summary(
        old_summary,
        question,
        " ".join([source_context or "", ner_line or ""]).strip(),
        answer_snippet,
    )
    return summary


def _tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+(?:['\-\.][a-z0-9]+)?", _norm_text(s))


def _dedupe_ci(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        key = re.sub(r"\s+", " ", item.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


_NLTK_BOOTSTRAP_ATTEMPTED = False
_NLTK_STOPWORDS_CACHE: Optional[Set[str]] = None


def _bootstrap_nltk_data() -> None:
    global _NLTK_BOOTSTRAP_ATTEMPTED
    if _NLTK_BOOTSTRAP_ATTEMPTED or nltk is None:
        return
    _NLTK_BOOTSTRAP_ATTEMPTED = True
    for pkg in (
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "stopwords",
    ):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


def _get_stopword_set() -> Set[str]:
    global _NLTK_STOPWORDS_CACHE
    if _NLTK_STOPWORDS_CACHE is not None:
        return _NLTK_STOPWORDS_CACHE

    stopset: Set[str] = set()
    if nltk_stopwords is not None:
        try:
            stopset = set(w.lower() for w in nltk_stopwords.words("english"))
        except LookupError:
            _bootstrap_nltk_data()
            try:
                stopset = set(w.lower() for w in nltk_stopwords.words("english"))
            except Exception:
                stopset = set()
        except Exception:
            stopset = set()

    _NLTK_STOPWORDS_CACHE = stopset
    return _NLTK_STOPWORDS_CACHE


_GENERIC_QUERY_TERMS_CACHE: Optional[Set[str]] = None
_FOLLOWUP_PHRASES_CACHE: Optional[List[str]] = None
_FOLLOWUP_PRONOUN_PATTERN_CACHE: Optional[re.Pattern] = None
_FOLLOWUP_PRONOUN_PATTERN_READY: bool = False


def _split_config_terms(raw: str) -> List[str]:
    if not raw:
        return []
    out: List[str] = []
    for item in re.split(r"[,;\n|]+", raw):
        s = (item or "").strip().lower()
        if s:
            out.append(s)
    return out


def _get_generic_query_terms() -> Set[str]:
    global _GENERIC_QUERY_TERMS_CACHE
    if _GENERIC_QUERY_TERMS_CACHE is not None:
        return _GENERIC_QUERY_TERMS_CACHE
    raw = str(getattr(settings, "generic_query_terms", "") or "")
    _GENERIC_QUERY_TERMS_CACHE = set(_split_config_terms(raw))
    return _GENERIC_QUERY_TERMS_CACHE


def _get_followup_phrases() -> List[str]:
    global _FOLLOWUP_PHRASES_CACHE
    if _FOLLOWUP_PHRASES_CACHE is not None:
        return _FOLLOWUP_PHRASES_CACHE
    raw = str(getattr(settings, "followup_phrases", "") or "")
    _FOLLOWUP_PHRASES_CACHE = _split_config_terms(raw)
    return _FOLLOWUP_PHRASES_CACHE


def _get_followup_pronoun_pattern() -> Optional[re.Pattern]:
    global _FOLLOWUP_PRONOUN_PATTERN_READY, _FOLLOWUP_PRONOUN_PATTERN_CACHE
    if _FOLLOWUP_PRONOUN_PATTERN_READY:
        return _FOLLOWUP_PRONOUN_PATTERN_CACHE
    _FOLLOWUP_PRONOUN_PATTERN_READY = True
    raw = str(getattr(settings, "followup_pronoun_regex", "") or "").strip()
    if not raw:
        _FOLLOWUP_PRONOUN_PATTERN_CACHE = None
        return None
    try:
        _FOLLOWUP_PRONOUN_PATTERN_CACHE = re.compile(raw, re.IGNORECASE)
    except re.error:
        _FOLLOWUP_PRONOUN_PATTERN_CACHE = None
    return _FOLLOWUP_PRONOUN_PATTERN_CACHE


def _is_followup_coref_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    pat = _get_followup_pronoun_pattern()
    if pat is not None and pat.search(q):
        return True
    lowered = q.lower()
    return any(key and key in lowered for key in _get_followup_phrases())


def _is_generic_query_token(token: str) -> bool:
    t = (token or "").strip().lower()
    if not t:
        return True
    min_len = int(getattr(settings, "generic_token_min_len", 3))
    if len(t) < max(1, min_len):
        return True
    if t in _get_generic_query_terms():
        return True
    stopset = _get_stopword_set()
    if stopset and t in stopset:
        return True
    return False


def _looks_like_person_token(token: str) -> bool:
    if not token:
        return False
    return bool(re.match(r"^[A-Z][A-Za-z\-']+$", token) or re.match(r"^[A-Z]\.$", token))


def _extract_entities_regex(raw: str, *, max_items: int = 6) -> Dict[str, List[str]]:
    people: List[str] = []
    entities: List[str] = []

    for quoted in re.findall(r"\"([^\"]{3,120})\"", raw):
        entities.append(quoted.strip())

    span_re = re.compile(r"\b(?:[A-Z][A-Za-z'\-\.]*\s+){1,5}[A-Z][A-Za-z'\-\.]*\b")
    for span in span_re.findall(raw):
        toks = [t for t in re.split(r"\s+", span.strip()) if t]
        if 2 <= len(toks) <= 4 and all(_looks_like_person_token(t) for t in toks):
            people.append(span.strip())
        else:
            entities.append(span.strip())

    for acr in re.findall(r"\b[A-Z]{2,10}\b", raw):
        entities.append(acr)

    people = _dedupe_ci(people)[:max_items]
    entities = _dedupe_ci(entities)[:max_items]

    blocked_tokens = set(_tokenize_words(" ".join(people + entities)))
    tokens = [t for t in _tokenize_words(raw) if len(t) >= 4 and t not in blocked_tokens and not t.isdigit()]
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    topics = [k for k, _v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))][:max_items]

    orgs = [e for e in entities if re.fullmatch(r"[A-Z]{2,10}", e)]
    orgs = _dedupe_ci(orgs)[:max_items]
    return {"people": people, "orgs": orgs, "entities": entities, "topics": topics}


def _extract_entities_nltk(raw: str, *, max_items: int = 6) -> Optional[Dict[str, List[str]]]:
    if nltk is None or word_tokenize is None or pos_tag is None or ne_chunk is None:
        return None

    def _run() -> Dict[str, List[str]]:
        tokens = word_tokenize(raw)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged, binary=False)

        people: List[str] = []
        orgs: List[str] = []
        entities: List[str] = []

        for node in tree:
            if Tree is not None and isinstance(node, Tree):
                label = str(node.label() or "").upper()
                phrase = " ".join(tok for tok, _p in node.leaves()).strip()
                if not phrase:
                    continue
                if label == "PERSON":
                    people.append(phrase)
                elif label in {"ORGANIZATION", "GPE", "FACILITY"}:
                    orgs.append(phrase)
                    entities.append(phrase)
                else:
                    entities.append(phrase)

        for tok, pos in tagged:
            if re.fullmatch(r"[A-Z]{2,10}", tok):
                orgs.append(tok)
                entities.append(tok)
            elif pos == "NNP" and re.fullmatch(r"[A-Z][A-Za-z\-']+", tok):
                entities.append(tok)

        for quoted in re.findall(r"\"([^\"]{3,120})\"", raw):
            entities.append(quoted.strip())

        stopset = _get_stopword_set()
        blocked_tokens = set(_tokenize_words(" ".join(people + orgs + entities)))
        counts: Dict[str, int] = {}
        for tok in tokens:
            if not re.fullmatch(r"[A-Za-z][A-Za-z\-']*", tok):
                continue
            t = tok.lower()
            if len(t) < 4:
                continue
            if t in stopset or t in blocked_tokens:
                continue
            counts[t] = counts.get(t, 0) + 1

        topics = [k for k, _v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))][:max_items]
        return {
            "people": _dedupe_ci(people)[:max_items],
            "orgs": _dedupe_ci(orgs)[:max_items],
            "entities": _dedupe_ci(entities)[:max_items],
            "topics": topics,
        }

    try:
        return _run()
    except LookupError:
        _bootstrap_nltk_data()
        try:
            return _run()
        except Exception:
            return None
    except Exception:
        return None


def _extract_entities_basic(text: str, *, max_items: int = 6) -> Dict[str, List[str]]:
    raw = text or ""
    if not raw.strip():
        return {"people": [], "orgs": [], "entities": [], "topics": []}
    extracted = _extract_entities_nltk(raw, max_items=max_items)
    if extracted is not None:
        out = extracted
    else:
        out = _extract_entities_regex(raw, max_items=max_items)
    cleaned: Dict[str, List[str]] = {}
    cleaned["people"] = _sanitize_entity_values(out.get("people") or [], max_items=max_items)
    cleaned["orgs"] = _sanitize_entity_values(out.get("orgs") or [], max_items=max_items)
    cleaned["entities"] = _sanitize_entity_values(out.get("entities") or [], max_items=max_items)
    topics = [t for t in (out.get("topics") or []) if t and not _is_generic_query_token(t)]
    cleaned["topics"] = _dedupe_ci(topics)[:max_items]
    return cleaned


def _build_ner_context_text(docs: List[Document], max_docs: int = 12) -> str:
    parts: List[str] = []
    for d in docs[:max_docs]:
        meta = d.metadata or {}
        for key in ("title", "primary_topic", "researcher", "authors"):
            val = str(meta.get(key, "") or "").strip()
            if val:
                parts.append(val)
    return " | ".join(parts)


def _summary_query_from_text(summary: str, *, max_chars: int = 320) -> str:
    if not summary:
        return ""
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    if not lines:
        return ""
    tail = " ".join(lines[-3:])
    if len(tail) > max_chars:
        tail = tail[: max_chars - 1].rstrip() + "…"
    return tail


def _has_explicit_entity_signal(question: str, ents: Optional[Dict[str, List[str]]] = None) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if re.search(r"\"[^\"]{3,120}\"", q):
        return True
    data = ents if ents is not None else _extract_entities_basic(q, max_items=4)
    if data.get("people") or data.get("orgs") or data.get("entities"):
        return True
    # Topic-only extraction can be noisy; require a minimum number of
    # non-generic terms before treating the query as having an entity signal.
    topic_tokens = [t for t in data.get("topics", []) if not _is_generic_query_token(t)]
    min_terms = int(getattr(settings, "retrieval_topic_min_terms", 2))
    return len(topic_tokens) >= max(1, min_terms)


def _looks_like_person_candidate(name: str) -> bool:
    if not name:
        return False
    toks = [t for t in re.split(r"\s+", name) if t]
    if len(toks) < 2 or len(toks) > 4:
        return False
    if any(re.search(r"\d", t) for t in toks):
        return False
    if all(len(t.replace(".", "")) <= 1 for t in toks):
        return False
    if not all(re.match(r"^[A-Za-z][A-Za-z\.\-']*$", t) for t in toks):
        return False
    return True


def _extract_person_name(question: str) -> str:
    raw = question or ""
    if not raw.strip():
        return ""

    def pick(ents: Dict[str, List[str]]) -> str:
        for cand in ents.get("people", []) or []:
            if not _looks_like_person_candidate(cand):
                continue
            if len(cand) < 2:
                continue
            return cand
        return ""

    ents = _extract_entities_basic(raw, max_items=4)
    name = pick(ents)
    if name:
        return name

    # Try a title-cased variant to handle lower-case queries.
    title_raw = " ".join(
        w.capitalize() if w.islower() else w for w in re.split(r"\s+", raw) if w
    )
    ents2 = _extract_entities_basic(title_raw, max_items=4)
    return pick(ents2)


def _strip_prompt_leak(answer: str) -> str:
    a = (answer or "").strip()
    if not a:
        return a
    leak_markers = [
        "papers:",
        "paper context:",
        "context:",
        "question:",
        "system:",
        "user:",
        "assistant:",
        "[paper",
        "[mem",
        "[recent",
        "[summary",
    ]
    lower = a.lower()
    if any(m in lower for m in leak_markers):
        lines = [ln.strip() for ln in a.splitlines()]
        kept: List[str] = []
        for ln in lines:
            lnl = ln.lower()
            if any(m in lnl for m in leak_markers):
                continue
            kept.append(ln)
        a = "\n".join([k for k in kept if k]).strip()
    a = re.sub(r"\n{3,}", "\n\n", a).strip()
    return a


def _answer_is_bad(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return True
    return len(_tokenize_words(a)) < 3


def _extract_focus_from_question(question: str) -> str:
    raw = (question or "").strip()
    if not raw:
        return ""
    ents = _extract_entities_basic(raw, max_items=6)
    if ents.get("topics"):
        topic_terms = [t for t in ents["topics"] if not _is_generic_query_token(t)]
        if topic_terms:
            return " ".join(topic_terms[:4])
    if ents.get("entities"):
        vals = [v for v in ents["entities"] if not _is_generic_query_token(v)]
        if vals:
            return " ".join(vals[:3])
    if ents.get("orgs"):
        return " ".join(ents["orgs"][:2])
    return ""


def _is_invalid_focus_value(text: str) -> bool:
    toks = [t for t in _tokenize_words(text) if t]
    if not toks:
        return True
    if max((len(t) for t in toks), default=0) < 3:
        return True
    meaningful = [t for t in toks if not _is_generic_query_token(t)]
    return not meaningful


def _query_is_short_or_pronoun(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _is_followup_coref_question(q):
        return True
    max_followup_words = int(getattr(settings, "followup_query_max_words", 8))
    return len(_tokenize_words(q)) < max(1, max_followup_words)


def _is_placeholder_anchor_value(value: str) -> bool:
    raw = re.sub(r"\s+", " ", str(value or "").strip().lower())
    if not raw:
        return True
    compact = re.sub(r"[^a-z0-9]+", "", raw)
    if not compact:
        return True
    if compact in {
        "na",
        "nslasha",
        "unknown",
        "none",
        "null",
        "nil",
        "empty",
        "unspecified",
        "notavailable",
        "notapplicable",
        "notprovided",
        "tbd",
    }:
        return True
    return False


def _normalize_anchor(anchor: Any) -> Dict[str, Any]:
    if not isinstance(anchor, dict):
        return {}
    value = re.sub(r"\s+", " ", str(anchor.get("value", "") or "").strip())
    if not value or _is_placeholder_anchor_value(value):
        return {}
    a_type = re.sub(r"\s+", " ", str(anchor.get("type", "") or "").strip().lower()) or "metadata"
    source = re.sub(r"\s+", " ", str(anchor.get("source", "") or "").strip()) or "retrieval"
    try:
        confidence = float(anchor.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {"type": a_type, "value": value, "source": source, "confidence": confidence}


def _anchor_in_text(anchor_value: str, text: str) -> bool:
    if _is_placeholder_anchor_value(anchor_value):
        return False
    a = _norm_text(anchor_value)
    t = _norm_text(text)
    if not a or not t:
        return False
    if a in t:
        return True
    a_toks = [tok for tok in _tokenize_words(a) if len(tok) >= 3]
    if len(a_toks) < 2:
        return False
    t_toks = set(_tokenize_words(t))
    return all(tok in t_toks for tok in a_toks)


def _inject_anchor_into_query(question: str, anchor_value: str) -> str:
    q = re.sub(r"\s+", " ", (question or "").strip())
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    if not q or not anchor or _is_placeholder_anchor_value(anchor):
        return q
    if _anchor_in_text(anchor, q):
        return q
    pronoun_re = re.compile(
        r"\b(him|her|them|they|it|this|that|those|these|he|she|his|hers|their|there)\b",
        re.IGNORECASE,
    )
    replaced = pronoun_re.sub(anchor, q, count=1)
    replaced = re.sub(r"\s+", " ", replaced).strip()
    if _anchor_in_text(anchor, replaced):
        return replaced
    stem = q.rstrip(" ?")
    if not stem:
        return anchor
    return f"{stem} for {anchor}?"


def _anchor_support_ratio(anchor_value: str, docs: List[Document]) -> float:
    anchor = re.sub(r"\s+", " ", (anchor_value or "").strip())
    if not anchor or not docs or _is_placeholder_anchor_value(anchor):
        return 0.0
    hits = 0
    for d in docs:
        meta = d.metadata or {}
        meta_parts: List[str] = []
        for v in meta.values():
            if isinstance(v, (list, dict, tuple, set)):
                continue
            meta_parts.append(str(v or ""))
        hay = " ".join(meta_parts + [str(d.page_content or "")])
        if _anchor_in_text(anchor, hay):
            hits += 1
    return float(hits) / max(1, len(docs))


def _anchor_is_stable(anchor: Dict[str, Any]) -> bool:
    data = _normalize_anchor(anchor)
    if not data:
        return False
    min_conf = float(getattr(settings, "anchor_stable_confidence", 0.72))
    return float(data.get("confidence", 0.0) or 0.0) >= min_conf


def _classify_generic_intent(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "default"
    if any(k in q for k in ("compare", "difference", "versus", "vs", "similarity")):
        return "comparison"
    if any(k in q for k in ("time", "period", "year", "range", "when")):
        return "time_range"
    if any(k in q for k in ("list", "which papers", "who are", "what are", "show me")):
        return "list"
    return "default"


class _DirectGenerationLLM:
    def __init__(self, *, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, generation_kwargs: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = dict(generation_kwargs or {})

    def invoke(self, prompt: str) -> str:
        p = str(prompt or "")
        if not p:
            return ""

        max_new_tokens = int(self.generation_kwargs.get("max_new_tokens", 256))
        max_ctx = int(
            getattr(self.model.config, "max_position_embeddings", 0)
            or getattr(self.tokenizer, "model_max_length", 4096)
            or 4096
        )
        max_input_tokens = max(64, max_ctx - max_new_tokens - 16)
        enc = self.tokenizer(
            p,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda:0") for k, v in enc.items()}

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(self.generation_kwargs.get("do_sample", False)),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if gen_kwargs["do_sample"]:
            temp = self.generation_kwargs.get("temperature")
            if temp is not None:
                gen_kwargs["temperature"] = float(temp)
            top_p = self.generation_kwargs.get("top_p")
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        # Use autocast on GPU for throughput; note that determinism depends
        # on the CUDA version and hardware architecture.
        ctx = torch.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
        with torch.no_grad():
            with ctx:
                out = self.model.generate(**enc, **gen_kwargs)

        prompt_len = int(enc["input_ids"].shape[1])
        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


class ModelRuntime:
    def __init__(
        self,
        model_id_or_path: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
    ):
        if getattr(settings, "force_gpu", True) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is False")

        self.model_id_or_path = model_id_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            local_files_only=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )

        if torch.cuda.is_available():
            self.model.to("cuda:0")
        self.model.eval()

        # Keep generation kwargs explicit to avoid ignored-parameter warnings
        # when sampling is disabled.
        self.generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": bool(do_sample),
            "return_full_text": False,
        }
        if do_sample:
            self.generation_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                self.generation_kwargs["top_p"] = float(top_p)

        # Some model configs ship sampling params by default; clear them for greedy decode.
        if not do_sample:
            try:
                self.model.generation_config.temperature = None
                self.model.generation_config.top_p = None
            except Exception:
                pass

        self.llm = _DirectGenerationLLM(
            model=self.model,
            tokenizer=self.tokenizer,
            generation_kwargs=self.generation_kwargs,
        )

    def close(self) -> None:
        try:
            del self.llm
        except Exception:
            pass
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def pack_docs(docs: List[Document], budget: int, count_tokens_fn) -> List[Document]:
    out: List[Document] = []
    total = 0
    for d in docs:
        t = count_tokens_fn(d.page_content)
        if total + t > budget:
            break
        out.append(d)
        total += t
    return out


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(".", 1)[0].strip()
    return cut + "." if cut else text[:limit].strip()


def format_docs_compact(docs: List[Document], *, max_docs: int, text_limit: int) -> str:
    blocks: List[str] = []
    for d in docs[:max_docs]:
        meta = d.metadata or {}
        title = meta.get("title", "")
        authors = meta.get("authors", "")
        year = meta.get("year", meta.get("publication_date", ""))
        text = _truncate_text(d.page_content or "", text_limit)
        blocks.append(
            "\n".join(
                [
                    f"Title: {title}",
                    f"Authors: {authors}",
                    f"Year: {year}",
                    f"Snippet: {text}",
                ]
            )
        )
    return "\n\n".join(blocks)


class _TransformerMeanEmbeddings:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=False,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda:0")
            self.device = "cuda"
        else:
            self.model.to("cpu")
            self.device = "cpu"
        self.model.eval()

    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = [str(t or "") for t in texts[i : i + batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            if self.device == "cuda":
                enc = {k: v.to("cuda:0") for k, v in enc.items()}
            else:
                enc = {k: v.to("cpu") for k, v in enc.items()}

            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                summed = (hidden * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1e-9)
                pooled = summed / denom
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors = pooled.detach().cpu().tolist()
                out.extend(vectors)
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        vecs = self._encode_texts([text])
        return vecs[0] if vecs else []


def build_embeddings() -> Any:
    preferred = _EMBED_DEVICE

    def _make(device: str) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL,
            model_kwargs={
                "device": device,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_kwargs": {
                    "low_cpu_mem_usage": False,
                    "torch_dtype": torch.float32,
                },
            },
            encode_kwargs={"normalize_embeddings": True, "batch_size": 128},
        )

    try:
        return _make(preferred)
    except Exception as exc:
        if preferred != "cuda":
            _dbg("[EMBED] primary init failed; trying transformer mean-pooling fallback", repr(exc))
            try:
                return _TransformerMeanEmbeddings(config.EMBED_MODEL, "cpu")
            except Exception:
                raise
        _dbg("[EMBED] cuda init failed; retrying on cpu", repr(exc))
        try:
            return _make("cpu")
        except Exception as cpu_exc:
            _dbg("[EMBED] cpu sentence-transformers init failed; trying transformer mean-pooling fallback", repr(cpu_exc))
            return _TransformerMeanEmbeddings(config.EMBED_MODEL, "cpu")


def clear_runtime_cache() -> None:
    _ensure_dir(CACHE_DIR)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_llm_path(llm_model_key: str) -> str:
    key = (llm_model_key or "").strip().lower()
    one_b_path = str(getattr(settings, "llama_1b_path", "") or "").strip()
    if key in {"llama-3.2-1b", "llama_1b", "1b"}:
        return one_b_path or config.LLAMA_1B
    if key in {"llama-3.2-3b", "llama_3b", "3b"}:
        return config.LLAMA_3B
    return config.LLAMA_3B


MEMORY_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return JSON only with keys facts, decisions, preferences, tasks. "
            "Each value is a list of {\"text\": str, \"salience\": 1-5}. "
            "No extra keys. No prose.",
        ),
        ("human", "User:\n{user}\n\nAssistant:\n{assistant}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Update the running conversation summary.\n"
            "Return ONLY this template and keep it concise:\n"
            "Current focus:\n- ...\n"
            "Core entities:\n- ...\n"
            "Key themes:\n- ...\n"
            "Constraints:\n- ...\n"
            "Open questions:\n- ...\n"
            "Use factual statements only from user question, retrieved metadata, and final answer. "
            "Do not include raw copied blocks or partial NER fragments.",
        ),
        ("human", "Summary so far:\n{old_summary}\n\nNew turns:\n{new_turns}"),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a standalone question for retrieval.\n"
            "Use rolling summary and recent turns to resolve pronouns and omitted context.\n"
            "If anchor_value is provided, use it to resolve referential language.\n"
            "Keep concrete entities, dates, venues, paper ids, and constraints.\n"
            "Do not answer the question.\n"
            "Return JSON only with key: standalone_question.\n"
            "If no rewrite is needed, standalone_question should equal the user question.\n"
            "For pronoun or referential follow-ups, standalone_question must include anchor_value when available.\n"
            "Example: \"what field does he study\" -> "
            "{\"standalone_question\":\"What field does William Gearty study based on the retrieved papers\"}",
        ),
        (
            "human",
            "Anchor value:\n{anchor_value}\n\nRolling summary:\n{rolling_summary}\n\nRecent turns:\n{recent_turns}\n\nUser question:\n{question}",
        ),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer only using the provided Syracuse corpus papers text.\n"
            "Do not suggest external websites, databases, or sources.\n"
            "Every major claim must be anchored to at least one retrieved paper title.\n"
            "Do not answer with only a list of titles. Synthesize first.\n"
            "If papers were retrieved, produce the best-supported answer. "
            "Use insufficient-information refusal only when zero relevant papers are retrieved.\n"
            "If field or role is not explicitly stated, infer it from repeated terms in titles/summaries and mark it as inferred.\n"
            "Choose output structure by detected intent category (default, comparison, list, time_range).",
        ),
        ("human", "Papers:\n{papers}\n\nQuestion:\n{question}"),
    ]
)


@dataclass
class UtilityJob:
    session_id: str
    user_text: str
    assistant_text: str
    new_turns_text: str
    run_summary: bool
    run_memory_extract: bool
    last_focus: Optional[str] = None
    last_topic: Optional[str] = None
    no_results: bool = False
    ner_line: str = ""
    retrieval_meta_text: str = ""
    turns_already_persisted: bool = False


class UtilityWorker:
    def __init__(
        self,
        *,
        store: SessionStore,
        memory_vs: Chroma,
        runtime: ModelRuntime,
        answer_generation_lock: Optional[threading.Lock] = None,
    ):
        self.store = store
        self.memory_vs = memory_vs
        self.runtime = runtime
        self.answer_generation_lock = answer_generation_lock

        self.q: "queue.Queue[UtilityJob]" = queue.Queue(
            maxsize=int(getattr(settings, "utility_queue_max", 2000))
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="utility-worker", daemon=True)

        self._io_lock = threading.Lock()
        self._memory_add_counter = 0

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit(self, job: UtilityJob) -> None:
        try:
            self.q.put_nowait(job)
        except queue.Full:
            return

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self.q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._process(job)
            except Exception:
                pass
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    pass

    def _prune_memory_if_needed_locked(self, session_id: str) -> None:
        try:
            col = getattr(self.memory_vs, "_collection", None)
            if col is None:
                return
            got = col.get(where={"session_id": session_id}, include=["ids"])
            ids = got.get("ids") or []
            max_n = int(getattr(settings, "memory_max_per_session", 500))
            target = int(getattr(settings, "memory_prune_target", 420))
            if len(ids) <= max_n:
                return
            to_delete = ids[: max(0, len(ids) - target)]
            if to_delete:
                col.delete(ids=to_delete)
        except Exception:
            return

    def _extract_memory_locked(self, session_id: str, user: str, assistant: str) -> None:
        try:
            msgs = MEMORY_EXTRACT_PROMPT.format_messages(user=user, assistant=assistant)
            raw = self.runtime.llm.invoke(msgs[0].content + "\n" + msgs[1].content)
            data = json.loads(raw)
        except Exception:
            return

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        seen_text = set()

        def _compact_memory_text(text: str, limit: int = 220) -> str:
            compact = re.sub(r"\s+", " ", str(text or "").strip())
            if len(compact) <= limit:
                return compact
            return compact[:limit].rsplit(" ", 1)[0].strip()

        def add(key: str) -> None:
            items = data.get(key, [])
            if not isinstance(items, list):
                return
            for obj in items[:20]:
                if not isinstance(obj, dict):
                    continue
                text = _compact_memory_text(obj.get("text") or "")
                if not text:
                    continue
                text_key = text.lower()
                if text_key in seen_text:
                    continue
                seen_text.add(text_key)
                try:
                    sal = int(obj.get("salience") or 3)
                except Exception:
                    sal = 3
                sal = max(1, min(5, sal))
                texts.append(text)
                metas.append(
                    {
                        "type": key,
                        "salience": sal,
                        "session_id": session_id,
                    }
                )

        add("facts")
        add("decisions")
        add("preferences")
        add("tasks")

        if not texts:
            return

        ids = [str(uuid.uuid4()) for _ in texts]
        try:
            self.memory_vs.add_texts(texts=texts, metadatas=metas, ids=ids)
            self._memory_add_counter += len(texts)
        except Exception:
            return

        self._prune_memory_if_needed_locked(session_id)

        if self._memory_add_counter >= int(getattr(settings, "memory_persist_every_n_adds", 25)):
            self._memory_add_counter = 0
            try:
                self.memory_vs.persist()
            except Exception:
                pass

    def _update_summary_locked(
        self,
        session_id: str,
        new_turns_text: str,
        *,
        no_results: bool = False,
        question: str = "",
        ner_line: str = "",
        retrieval_meta_text: str = "",
    ) -> None:
        state = self.store.load(session_id)
        old_summary = state.get("rolling_summary", "") or ""
        turns = state.get("turns", []) or []
        if bool(getattr(settings, "enable_llm_summary_regen", False)):
            rolling_summary = _regenerate_rolling_summary(
                llm=self.runtime.llm,
                old_summary=old_summary,
                new_turns_text=new_turns_text,
                question=question,
                ner_line=ner_line,
                source_context=retrieval_meta_text,
                no_results=no_results,
                timeout_s=int(getattr(settings, "llm_timeout_s", 40)),
            )
        else:
            answer_snippet = ""
            m = re.search(r"ASSISTANT:\\s*(.+)", new_turns_text or "", re.IGNORECASE | re.DOTALL)
            if m:
                answer_snippet = re.sub(r"\s+", " ", m.group(1)).strip()
            if not answer_snippet:
                answer_snippet = re.sub(r"\s+", " ", new_turns_text or "").strip()
            rolling_summary = build_rolling_summary(
                old_summary,
                question,
                " ".join([retrieval_meta_text or "", ner_line or ""]).strip(),
                assistant_answer=answer_snippet,
            )

        self.store.save(session_id, rolling_summary, turns)

    def _append_turns_locked(
        self,
        session_id: str,
        user_text: str,
        assistant_text: str,
        last_focus: Optional[str] = None,
        last_topic: Optional[str] = None,
    ) -> None:
        state = self.store.load(session_id)
        rolling_summary = state.get("rolling_summary", "") or ""
        turns = state.get("turns", []) or []
        turns.append({"role": "user", "text": user_text})
        turns.append({"role": "assistant", "text": assistant_text})
        extra = None
        if last_focus is not None or last_topic is not None:
            extra = {}
            if last_focus is not None:
                extra["last_focus"] = last_focus
            if last_topic is not None:
                extra["last_topic"] = last_topic
        if extra is None:
            extra = {}
        self.store.save(session_id, rolling_summary, turns, extra_state=extra)

    def _process(self, job: UtilityJob) -> None:
        lock_ctx = nullcontext()
        if self.answer_generation_lock is not None and not bool(
            getattr(settings, "allow_utility_concurrency", False)
        ):
            lock_ctx = self.answer_generation_lock
        with lock_ctx:
            with self._io_lock:
                if not job.turns_already_persisted:
                    self._append_turns_locked(
                        job.session_id,
                        job.user_text,
                        job.assistant_text,
                        job.last_focus,
                        job.last_topic,
                    )
                if job.run_memory_extract:
                    self._extract_memory_locked(job.session_id, job.user_text, job.assistant_text)
                if job.run_summary:
                    self._update_summary_locked(
                        job.session_id,
                        job.new_turns_text,
                        no_results=job.no_results,
                        question=job.user_text,
                        ner_line=job.ner_line,
                        retrieval_meta_text=job.retrieval_meta_text,
                    )


class Engine:
    def __init__(
        self,
        *,
        answer_runtime: ModelRuntime,
        utility_runtime: Optional[ModelRuntime],
        papers_vs: Chroma,
        memory_vs: Chroma,
        store: SessionStore,
        session_id: str,
        utility_worker: Optional[UtilityWorker],
        stateless: bool = False,
    ):
        self.answer_runtime = answer_runtime
        self.utility_runtime = utility_runtime
        self.papers_vs = papers_vs
        self.memory_vs = memory_vs
        self.store = store
        self.session_id = session_id
        self.utility_worker = utility_worker
        self.stateless = bool(stateless)

        self.last_focus = ""
        self.last_topic = ""
        self.rolling_summary = ""
        self.anchor: Dict[str, Any] = {}
        self.anchor_last_action = "none"
        self.last_rewrite_referential = False
        self.last_rewrite_anchor_valid = False
        self.last_rewrite_blocked = False
        self.last_summary_updated = False
        self.last_retrieval_confidence = "weak"
        self.last_anchor_support_ratio = 0.0
        self.last_rewrite_time_ms = 0.0
        self.last_retrieval_time_ms = 0.0
        self.last_answer_llm_calls = 0
        self.last_utility_llm_calls = 0
        self.turns: List[Turn] = []
        if not self.stateless:
            state = self.store.load(session_id)
            self.last_focus = (state.get("last_focus", "") or "").strip()
            self.last_topic = (state.get("last_topic", "") or "").strip()
            self.rolling_summary = (state.get("rolling_summary", "") or "").strip()
            extra = state.get("extra_state", {}) if isinstance(state.get("extra_state"), dict) else {}
            self.anchor = _normalize_anchor(extra.get("anchor", {}) if isinstance(extra, dict) else {})
            if self.anchor:
                self.anchor_last_action = str(extra.get("anchor_last_action", "loaded") or "loaded")
            for obj in (state.get("turns", []) or []):
                role = (obj.get("role") or "").strip()
                text = (obj.get("text") or "").strip()
                if role and text:
                    self.turns.append(Turn(role=role, text=text))

    def _user_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    def _recent_turns_text(self, max_turns: int) -> str:
        if max_turns <= 0:
            return ""
        rows: List[str] = []
        for t in self.turns[-max_turns:]:
            role = "User" if t.role == "user" else "Assistant"
            rows.append(f"{role}: {t.text}")
        return "\n".join(rows).strip()

    def _last_user_turn_text(self) -> str:
        for t in reversed(self.turns):
            if t.role == "user" and t.text.strip():
                return t.text.strip()
        return ""

    def _rewrite_query_structured(self, question: str, *, anchor_value: str = "") -> Dict[str, Any]:
        q = (question or "").strip()
        max_recent = int(getattr(settings, "rewrite_max_recent_turns", 3))
        max_recent = max(1, min(3, max_recent))
        timeout_s = int(getattr(settings, "rewrite_timeout_s", 10))
        max_chars = int(getattr(settings, "rewrite_max_chars", 220))
        recent_turns = self._recent_turns_text(max_recent)
        fallback = {
            "standalone_question": q,
        }
        if not bool(getattr(settings, "rewrite_enable", True)):
            return fallback
        runtime = self.utility_runtime
        if runtime is None:
            return fallback
        try:
            msgs = REWRITE_PROMPT.format_messages(
                rolling_summary=self.rolling_summary or "",
                recent_turns=recent_turns or "",
                question=question,
                anchor_value=anchor_value or "(none)",
            )
            raw = _invoke_with_timeout(
                runtime.llm,
                msgs[0].content + "\n" + msgs[1].content,
                timeout_s,
            )
            self.last_utility_llm_calls += 1
            txt = str(raw or "").strip()
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                txt = m.group(0)
            obj = json.loads(txt)
            sq = re.sub(r"\s+", " ", str(obj.get("standalone_question", "") or "")).strip()
            if not sq:
                sq = q
            if len(sq) > max_chars:
                sq = sq[:max_chars].rstrip()
            return {"standalone_question": sq}
        except Exception:
            return fallback

    def maybe_rewrite_query(self, raw_q: str) -> str:
        q = (raw_q or "").strip()
        self.last_rewrite_referential = False
        self.last_rewrite_anchor_valid = False
        self.last_rewrite_blocked = False
        if not q:
            return q

        max_words = max(1, int(getattr(settings, "followup_query_max_words", 8)))
        is_short = len(_tokenize_words(q)) < max_words
        pronoun_pat = _get_followup_pronoun_pattern()
        has_pronoun = pronoun_pat is not None and pronoun_pat.search(q) is not None
        lowered = q.lower()
        has_followup_phrase = any(p and p in lowered for p in _get_followup_phrases())
        is_referential = bool(has_pronoun or has_followup_phrase)
        self.last_rewrite_referential = is_referential

        if not (is_short or has_pronoun or has_followup_phrase):
            return q

        anchor_data = _normalize_anchor(self.anchor)
        anchor_value = str(anchor_data.get("value", "") or "").strip() if _anchor_is_stable(anchor_data) else ""
        rewrite = self._rewrite_query_structured(q, anchor_value=anchor_value)
        standalone = (rewrite.get("standalone_question") or q).strip()
        if is_referential and anchor_value and not _anchor_in_text(anchor_value, standalone):
            standalone = _inject_anchor_into_query(standalone or q, anchor_value).strip()
        if is_referential and anchor_value:
            self.last_rewrite_anchor_valid = _anchor_in_text(anchor_value, standalone)
        return standalone

    def _log_retrieval_state(self, *, where_filter: Optional[Dict[str, Any]]) -> None:
        collection = getattr(self.papers_vs, "_collection", None)
        collection_name = str(getattr(collection, "name", "") or getattr(collection, "_name", "") or "").strip()
        collection_count = -1
        if collection is not None:
            try:
                collection_count = int(collection.count())
            except Exception:
                collection_count = -1
        filter_json = json.dumps(where_filter, ensure_ascii=False) if where_filter else "{}"
        print(f"[RETRIEVE] collection={collection_name or '(unknown)'} count={collection_count} filter={filter_json}")
        if collection_count == 0:
            print(
                "[RETRIEVE] WARNING: collection count is zero. "
                "Check persist directory, collection name, and ingestion/retrieval embedding match."
            )

    def _embed_query_vector(self, query: str) -> Optional[List[float]]:
        q = (query or "").strip()
        if not q:
            return None
        ef = getattr(self.papers_vs, "_embedding_function", None)
        if ef is None:
            ef = getattr(self.papers_vs, "embedding_function", None)
        try:
            if hasattr(ef, "embed_query"):
                vec = ef.embed_query(q)
                if isinstance(vec, list) and vec:
                    return vec
            elif callable(ef):
                vecs = ef([q])
                if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
                    return vecs[0]
        except Exception:
            return None
        return None

    def _vector_retrieve_docs(
        self,
        vs: Any,
        *,
        query_embedding: List[float],
        k: int,
        fetch_k: int,
        lambda_mult: float,
        where_filter: Optional[Dict[str, Any]],
        prefer_mmr: bool,
    ) -> Optional[List[Document]]:
        if not query_embedding:
            return None

        if prefer_mmr and hasattr(vs, "max_marginal_relevance_search_by_vector"):
            method = getattr(vs, "max_marginal_relevance_search_by_vector")
            variants: List[Dict[str, Any]] = [
                {"embedding": query_embedding, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
                {"query_embedding": query_embedding, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
            ]
            if where_filter:
                for base in list(variants):
                    with_filter = dict(base)
                    with_filter["filter"] = where_filter
                    variants.insert(0, with_filter)
            for kwargs in variants:
                try:
                    got = method(**kwargs)
                    if isinstance(got, list):
                        return got
                except Exception:
                    continue

        if hasattr(vs, "similarity_search_by_vector"):
            method = getattr(vs, "similarity_search_by_vector")
            variants = [
                {"embedding": query_embedding, "k": k},
                {"query_embedding": query_embedding, "k": k},
            ]
            if where_filter:
                for base in list(variants):
                    with_filter = dict(base)
                    with_filter["filter"] = where_filter
                    variants.insert(0, with_filter)
            for kwargs in variants:
                try:
                    got = method(**kwargs)
                    if isinstance(got, list):
                        return got
                except Exception:
                    continue
            try:
                got = method(query_embedding, k=k, filter=where_filter) if where_filter else method(query_embedding, k=k)
                if isinstance(got, list):
                    return got
            except Exception:
                pass
        return None

    def _retrieve_once(
        self,
        query: str,
        *,
        query_embedding: Optional[List[float]] = None,
        search_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        k = int(search_k or getattr(settings, "search_k", 30))
        fk = int(fetch_k or getattr(settings, "search_fetch_k", 140))
        if k <= 0 or fk <= 0:
            total = 0
            col = getattr(self.papers_vs, "_collection", None)
            if col is not None:
                try:
                    total = int(col.count())
                except Exception:
                    total = 0
            if total <= 0:
                total = max(k, fk, 10000)
            if k <= 0:
                k = total
            if fk <= 0:
                fk = total
        if fk < (2 * k):
            fk = max(fk, 2 * k)
        lm = float(lambda_mult if lambda_mult is not None else getattr(settings, "mmr_lambda", 0.4))
        lm = max(0.3, min(0.5, lm))
        search_kwargs: Dict[str, Any] = {
            "k": k,
            "fetch_k": fk,
            "lambda_mult": lm,
        }
        if query_embedding:
            vect_docs = self._vector_retrieve_docs(
                self.papers_vs,
                query_embedding=query_embedding,
                k=k,
                fetch_k=fk,
                lambda_mult=lm,
                where_filter=where_filter,
                prefer_mmr=True,
            )
            if isinstance(vect_docs, list):
                return vect_docs
        if where_filter:
            search_kwargs["filter"] = where_filter
        retriever = self.papers_vs.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )
        return retriever.invoke(query)

    def _merge_unique_docs(self, docs_a: List[Document], docs_b: List[Document]) -> List[Document]:
        seen = set()
        merged: List[Document] = []
        for d in (docs_a or []) + (docs_b or []):
            meta = d.metadata or {}
            key = str(meta.get("paper_id", "")) + "::" + str(
                meta.get("chunk", meta.get("chunk_id", meta.get("id", "")))
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
        return merged

    def _rerank_papers_with_llm(self, query: str, docs: List[Document]) -> List[Document]:
        if not bool(getattr(settings, "rerank_enable", True)):
            return docs
        if self.utility_runtime is None:
            return docs
        if not docs:
            return docs

        cand_k = max(1, int(getattr(settings, "rerank_candidate_k", 30)))
        final_k = max(1, int(getattr(settings, "rerank_final_k", 12)))
        timeout_s = max(1, int(getattr(settings, "rerank_timeout_s", 12)))

        candidates = list(docs[:cand_k])
        if len(candidates) <= final_k:
            return candidates

        rows: List[str] = []
        for idx, d in enumerate(candidates):
            meta = d.metadata or {}
            title = str(meta.get("title", "") or "")
            researcher = str(meta.get("researcher", "") or "")
            year = str(meta.get("year", meta.get("publication_date", "")) or "")
            snippet = re.sub(r"\s+", " ", str(d.page_content or "")).strip()
            if len(snippet) > 260:
                snippet = snippet[:260].rstrip() + "..."
            rows.append(
                f"{idx}. title={title} | researcher={researcher} | year={year} | snippet={snippet}"
            )

        prompt = (
            "Select the most relevant chunks for answering the query.\n"
            "Return JSON only as {\"keep\":[idx,...]} with at most "
            f"{final_k} indices.\n"
            "Prioritize direct topical relevance and useful evidence coverage.\n\n"
            f"Query:\n{query}\n\nCandidates:\n" + "\n".join(rows)
        )
        raw = _invoke_with_timeout(self.utility_runtime.llm, prompt, timeout_s)
        self.last_utility_llm_calls += 1
        txt = str(raw or "").strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            txt = m.group(0)

        keep_idx: List[int] = []
        try:
            parsed = json.loads(txt)
            raw_keep = parsed.get("keep", [])
            if isinstance(raw_keep, list):
                for v in raw_keep:
                    try:
                        i = int(v)
                    except Exception:
                        continue
                    if i < 0 or i >= len(candidates):
                        continue
                    if i in keep_idx:
                        continue
                    keep_idx.append(i)
        except Exception:
            keep_idx = []

        selected: List[Document] = [candidates[i] for i in keep_idx[:final_k]]
        if len(selected) < final_k:
            have = {id(d) for d in selected}
            for d in candidates:
                if id(d) in have:
                    continue
                selected.append(d)
                have.add(id(d))
                if len(selected) >= final_k:
                    break
        return selected[:final_k]

    def _explicit_topic_shift(self, question: str) -> bool:
        q = (question or "").strip().lower()
        if not q:
            return False
        cues = (
            "switch topic",
            "different topic",
            "new topic",
            "unrelated",
            "change subject",
            "another area",
            "instead let's discuss",
        )
        return any(c in q for c in cues)

    def _post_filter_retrieved_docs(
        self,
        docs: List[Document],
        *,
        query: str,
    ) -> List[Document]:
        if not docs:
            return docs

        unique = self._merge_unique_docs(docs, [])
        q_tokens = [t for t in _tokenize_words(query) if not _is_generic_query_token(t)]
        if not q_tokens:
            return unique

        pruned: List[Document] = []
        for d in unique:
            meta = d.metadata or {}
            meta_text = " ".join(str(v or "") for v in meta.values())
            hay = (meta_text + " " + str(d.page_content or "")).lower()
            if any(re.search(rf"\b{re.escape(t)}\b", hay) for t in q_tokens[:16]):
                pruned.append(d)
        return pruned if pruned else unique

    def retrieve_papers(
        self,
        query: str,
        budget_papers: int,
        *,
        query_embedding: Optional[List[float]] = None,
        search_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        where_filter: Optional[Dict[str, Any]] = None,
        raw_question: str = "",
    ) -> List[Document]:
        q = (query or "").strip()
        if not q:
            return []
        self._log_retrieval_state(where_filter=where_filter)
        docs = self._retrieve_once(
            q,
            query_embedding=query_embedding,
            search_k=search_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            where_filter=where_filter,
        )
        q2 = ""

        # Dual-query retrieval only for short/coreferential follow-ups.
        is_followup_like = _query_is_short_or_pronoun(raw_question or q)
        anchor_value = str((self.anchor or {}).get("value", "") or "").strip()
        anchor_ratio = _anchor_support_ratio(anchor_value, docs) if anchor_value else 0.0
        min_anchor_ratio = float(getattr(settings, "anchor_consistency_min_ratio", 0.45))
        explicit_entity = _has_explicit_entity_signal(raw_question or q)
        query_anchor_consistent = (
            (not explicit_entity)
            or _anchor_in_text(anchor_value, raw_question or q)
            or _is_followup_coref_question(raw_question or q)
        )
        allow_summary_aug = (
            bool(getattr(settings, "retrieval_dual_query", True))
            and is_followup_like
            and _anchor_is_stable(self.anchor)
            and query_anchor_consistent
            and (anchor_ratio >= min_anchor_ratio)
            and (not where_filter)
        )
        if allow_summary_aug:
            summary_q = _summary_query_from_text(self.rolling_summary)
            if summary_q:
                q2 = f"{q} {summary_q}".strip()
                if q2 and q2 != q:
                    docs2 = self._retrieve_once(
                        q2,
                        query_embedding=query_embedding,
                        search_k=search_k,
                        fetch_k=fetch_k,
                        lambda_mult=lambda_mult,
                        where_filter=where_filter,
                    )
                    docs = self._merge_unique_docs(docs, docs2)

        docs = self._post_filter_retrieved_docs(
            docs,
            query=q,
        )

        if budget_papers <= 0:
            return docs
        return pack_docs(docs, budget_papers, self.answer_runtime.count_tokens)

    def retrieve_memory(
        self,
        query: str,
        budget_memory: int,
        *,
        query_embedding: Optional[List[float]] = None,
    ) -> List[Document]:
        docs: List[Document] = []
        if query_embedding:
            vect_docs = self._vector_retrieve_docs(
                self.memory_vs,
                query_embedding=query_embedding,
                k=12,
                fetch_k=24,
                lambda_mult=0.4,
                where_filter={"session_id": self.session_id},
                prefer_mmr=False,
            )
            if isinstance(vect_docs, list):
                docs = vect_docs
        if not docs:
            retriever = self.memory_vs.as_retriever(
                search_kwargs={
                    "k": 12,
                    "filter": {"session_id": self.session_id},
                }
            )
            docs = retriever.invoke(query)
        return pack_docs(docs, budget_memory, self.answer_runtime.count_tokens)

    def _rewrite_query_if_needed(
        self,
        raw_q: str,
    ) -> Tuple[str, List[Document], str]:
        q = (raw_q or "").strip()
        mem_docs: List[Document] = []
        if not q:
            return q, mem_docs, "default"

        t0_rewrite = time.perf_counter()
        query_for_retrieval = self.maybe_rewrite_query(q)
        self.last_rewrite_time_ms = (time.perf_counter() - t0_rewrite) * 1000.0
        anchor_data = _normalize_anchor(self.anchor)
        anchor_value = str(anchor_data.get("value", "") or "").strip() if _anchor_is_stable(anchor_data) else ""
        referential = _is_followup_coref_question(q)
        if referential and anchor_value and not _anchor_in_text(anchor_value, query_for_retrieval):
            query_for_retrieval = _inject_anchor_into_query(query_for_retrieval or q, anchor_value).strip()
        if referential and (not anchor_value or not _anchor_in_text(anchor_value, query_for_retrieval)):
            prev_user = self._last_user_turn_text()
            fallback_parts: List[str] = []
            if prev_user and _norm_text(prev_user) != _norm_text(q):
                fallback_parts.append(prev_user)
            fallback_parts.append(q)
            query_for_retrieval = " ".join(p for p in fallback_parts if p).strip()
            self.last_rewrite_blocked = True
        if not query_for_retrieval and q:
            query_for_retrieval = q
        detected_intent = _classify_generic_intent(query_for_retrieval or q)
        return query_for_retrieval, mem_docs, detected_intent

    def _persist_light_state(self, *, last_focus: str, last_topic: str) -> None:
        state = self.store.load(self.session_id)
        rolling_summary = state.get("rolling_summary", "") or ""
        turns_json = state.get("turns", []) or []
        state["last_focus"] = last_focus
        state["last_topic"] = last_topic
        self.store.save(
            self.session_id,
            rolling_summary,
            turns_json,
            extra_state={"last_focus": last_focus, "last_topic": last_topic},
        )

    def prepare_context(self, question: str, *, stateless: bool = False) -> EngineContext:
        self.last_answer_llm_calls = 0
        self.last_utility_llm_calls = 0
        budgets = dynamic_budgets()
        budget_memory = budgets["BUDGET_MEMORY"]
        budget_papers = budgets["BUDGET_PAPERS"]
        raw_q = (question or "").strip()
        if (not stateless) and self._explicit_topic_shift(raw_q):
            self.last_focus = ""
            self.last_topic = ""
            try:
                self._persist_light_state(last_focus="", last_topic="")
            except Exception:
                pass
        user_turns = self._user_turn_count()
        allow_prev_context = (user_turns > 0) and (not stateless)
        # Allow using/updating the rolling summary for this turn
        # (disabled in stateless mode for stable multi-turn continuity).
        allow_summary = not stateless

        rewritten_q, focus_mem_docs, detected_intent = self._rewrite_query_if_needed(raw_q)

        search_k = None
        fetch_k = None
        if _query_is_short_or_pronoun(raw_q):
            base_k = int(getattr(settings, "search_k", 30))
            base_fk = int(getattr(settings, "search_fetch_k", 140))
            k_mult = float(
                getattr(
                    settings,
                    "followup_k_mult",
                    2.0,
                )
            )
            fk_mult = float(
                getattr(
                    settings,
                    "followup_fetch_k_mult",
                    2.0,
                )
            )
            search_k = max(base_k, int(base_k * k_mult))
            fetch_k = max(base_fk, int(base_fk * fk_mult))

        paper_docs: List[Document]
        query_embedding: Optional[List[float]] = None
        can_reuse_embedding = bool(
            hasattr(self.papers_vs, "max_marginal_relevance_search_by_vector")
            or hasattr(self.papers_vs, "similarity_search_by_vector")
            or (allow_prev_context and hasattr(self.memory_vs, "similarity_search_by_vector"))
        )
        if rewritten_q.strip() and can_reuse_embedding:
            query_embedding = self._embed_query_vector(rewritten_q)
        if rewritten_q.strip():
            t0_retrieval = time.perf_counter()
            paper_docs = self.retrieve_papers(
                rewritten_q,
                budget_papers,
                query_embedding=query_embedding,
                search_k=search_k,
                fetch_k=fetch_k,
                raw_question=raw_q,
            )
            self.last_retrieval_time_ms = (time.perf_counter() - t0_retrieval) * 1000.0
        else:
            paper_docs = []
            self.last_retrieval_time_ms = 0.0
        if allow_prev_context:
            mem_docs = focus_mem_docs or (
                self.retrieve_memory(
                    rewritten_q,
                    budget_memory,
                    query_embedding=query_embedding,
                )
                if rewritten_q.strip()
                else []
            )
        else:
            mem_docs = []

        return EngineContext(
            raw_question=raw_q,
            rewritten_question=rewritten_q,
            detected_intent=detected_intent,
            paper_docs=paper_docs,
            mem_docs=mem_docs,
            stateless=bool(stateless),
            user_turns=user_turns,
            allow_prev_context=allow_prev_context,
            allow_summary=allow_summary,
            budgets=budgets,
            anchor=_normalize_anchor(self.anchor),
        )

    def _choose_last_focus_from_answer(self, question: str, paper_docs: List[Document]) -> str:
        q = (question or "").strip()
        name = _extract_person_name(q)
        if name and not _is_invalid_focus_value(name):
            return name

        focus_from_question = _extract_focus_from_question(q)
        if focus_from_question and not _is_invalid_focus_value(focus_from_question):
            return focus_from_question

        counts: Dict[str, int] = {}
        for d in paper_docs:
            meta = d.metadata or {}
            r = (meta.get("researcher") or "").strip()
            if r and not _is_invalid_focus_value(r):
                counts[r] = counts.get(r, 0) + 1

        if counts:
            top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            return top

        return ""

    def _choose_last_topic_from_question(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return ""
        if _extract_person_name(q):
            return ""
        topic = _extract_focus_from_question(q)
        if topic and not _is_invalid_focus_value(topic):
            return topic
        return ""

    def finalize_turn(
        self,
        context: EngineContext,
        answer: str,
        *,
        no_results: bool = False,
    ) -> None:
        if context.stateless:
            return
        new_focus = self._choose_last_focus_from_answer(
            context.rewritten_question,
            context.paper_docs,
        )
        if new_focus:
            self.last_focus = new_focus
        new_topic = self._choose_last_topic_from_question(context.rewritten_question)
        if new_topic:
            self.last_topic = new_topic
        new_turns_text = f"USER: {context.raw_question}\nASSISTANT: {answer}\n"
        ner_line = ""
        max_docs = int(getattr(settings, "ner_context_max_docs", 12))
        if self.last_retrieval_confidence in {"weak", "inconsistent"}:
            max_docs = min(max_docs, max(2, int(getattr(settings, "low_conf_ner_context_max_docs", 6))))
        retrieval_meta_text = _build_ner_context_text(context.paper_docs, max_docs=max_docs)
        anchor_value = str((self.anchor or {}).get("value", "") or "").strip()
        min_docs_for_conf = max(1, int(getattr(settings, "retrieval_weak_min_docs", 3)))
        min_anchor_ratio = float(getattr(settings, "anchor_consistency_min_ratio", 0.45))
        anchor_ratio = _anchor_support_ratio(anchor_value, context.paper_docs) if anchor_value else 1.0
        anchor_consistent = (not anchor_value) or (anchor_ratio >= min_anchor_ratio)
        anchor_high_quality = (not anchor_value) or _anchor_is_stable(self.anchor)
        retrieval_weak = len(context.paper_docs) < min_docs_for_conf
        summary_should_update = bool(
            context.allow_summary
            and (not no_results)
            and (not retrieval_weak)
            and anchor_consistent
            and anchor_high_quality
        )
        self.last_summary_updated = summary_should_update
        self.last_anchor_support_ratio = float(anchor_ratio)
        if not context.paper_docs:
            self.last_retrieval_confidence = "weak"
        elif anchor_value and not anchor_consistent:
            self.last_retrieval_confidence = "inconsistent"
        elif len(context.paper_docs) >= max(8, min_docs_for_conf * 2):
            self.last_retrieval_confidence = "high"
        elif len(context.paper_docs) >= max(4, min_docs_for_conf + 1):
            self.last_retrieval_confidence = "medium"
        else:
            self.last_retrieval_confidence = "low"
        retrieval_meta_for_summary = retrieval_meta_text if summary_should_update else ""
        llm_summary_regen = bool(getattr(settings, "enable_llm_summary_regen", False))
        memory_extract_allowed = (
            (not no_results)
            and anchor_consistent
            and (self.last_retrieval_confidence in {"medium", "high"})
        )
        run_memory_extract = bool(
            memory_extract_allowed
            and (
                bool(getattr(settings, "memory_extract_first_turn", True))
                or context.allow_prev_context
            )
        )

        if self.utility_worker is not None and int(
            getattr(settings, "enable_utility_background", 1)
        ) == 1:
            # Persist lightweight context immediately so the next turn can use
            # it even if the background worker has not processed this job yet.
            state = self.store.load(self.session_id)
            rolling_summary = state.get("rolling_summary", "") or ""
            if summary_should_update:
                rolling_summary = build_rolling_summary(
                    rolling_summary,
                    context.raw_question,
                    retrieval_meta_for_summary,
                    answer,
                )
            turns = state.get("turns", []) or []
            turns.append({"role": "user", "text": context.raw_question})
            turns.append({"role": "assistant", "text": answer})
            extra_state = {
                "last_focus": self.last_focus,
                "last_topic": self.last_topic,
                "anchor": _normalize_anchor(self.anchor),
                "anchor_last_action": self.anchor_last_action,
                "summary_updated": self.last_summary_updated,
                "retrieval_confidence": self.last_retrieval_confidence,
                "anchor_support_ratio": self.last_anchor_support_ratio,
                "rewrite_anchor_valid": self.last_rewrite_anchor_valid,
                "rewrite_blocked": self.last_rewrite_blocked,
            }
            self.store.save(self.session_id, rolling_summary, turns, extra_state=extra_state)

            job = UtilityJob(
                session_id=self.session_id,
                user_text=context.raw_question,
                assistant_text=answer,
                new_turns_text=new_turns_text,
                run_summary=bool(summary_should_update and llm_summary_regen),
                run_memory_extract=run_memory_extract,
                last_focus=self.last_focus if new_focus else None,
                last_topic=self.last_topic if new_topic else None,
                no_results=no_results,
                ner_line=ner_line,
                retrieval_meta_text=retrieval_meta_for_summary,
                turns_already_persisted=True,
            )
            self.utility_worker.submit(job)
            return

        # Synchronous fallback keeps state consistent when the worker is disabled.
        state = self.store.load(self.session_id)
        rolling_summary = state.get("rolling_summary", "") or ""
        old_summary = rolling_summary
        turns = state.get("turns", []) or []
        turns.append({"role": "user", "text": context.raw_question})
        turns.append({"role": "assistant", "text": answer})

        if summary_should_update:
            if llm_summary_regen and self.utility_runtime is not None:
                rolling_summary = _regenerate_rolling_summary(
                    llm=self.utility_runtime.llm,
                    old_summary=old_summary,
                    new_turns_text=new_turns_text,
                    question=context.raw_question,
                    ner_line=ner_line,
                    source_context=retrieval_meta_for_summary,
                    no_results=no_results,
                    timeout_s=int(getattr(settings, "llm_timeout_s", 40)),
                )
                self.last_utility_llm_calls += 1
            else:
                rolling_summary = build_rolling_summary(
                    old_summary,
                    context.raw_question,
                    retrieval_meta_for_summary,
                    answer,
                )

        extra = {
            "last_focus": self.last_focus,
            "last_topic": self.last_topic,
            "anchor": _normalize_anchor(self.anchor),
            "anchor_last_action": self.anchor_last_action,
            "summary_updated": self.last_summary_updated,
            "retrieval_confidence": self.last_retrieval_confidence,
            "anchor_support_ratio": self.last_anchor_support_ratio,
            "rewrite_anchor_valid": self.last_rewrite_anchor_valid,
            "rewrite_blocked": self.last_rewrite_blocked,
        }
        self.store.save(self.session_id, rolling_summary, turns, extra_state=extra)

    def ask(self, question: str) -> Tuple[str, List[Document], List[Document]]:
        context = self.prepare_context(question, stateless=False)
        raw_q = context.raw_question
        pap_docs = list(context.paper_docs)
        budgets = context.budgets
        trigger = budgets["TRIGGER"]
        budget_papers = budgets["BUDGET_PAPERS"]

        _dbg("[RAG] raw_question", raw_q)
        _dbg("[RAG] rewritten_question", context.rewritten_question)

        if not pap_docs:
            answer = f"I could not find anything for {raw_q} in the retrieved corpus."
            self.finalize_turn(context, answer, no_results=True)
            return answer, [], []

        max_docs = int(getattr(settings, "prompt_max_docs", 24))
        text_limit = int(getattr(settings, "prompt_doc_text_limit", 800))
        papers_ctx = format_docs_compact(
            pap_docs,
            max_docs=max_docs,
            text_limit=text_limit,
        )

        msgs = ANSWER_PROMPT.format_messages(papers=papers_ctx, question=context.rewritten_question)
        full_prompt = "\n\n".join(m.content for m in msgs)

        tok = self.answer_runtime.count_tokens(full_prompt)
        _dbg("[LLM] token_count", str(tok))
        if tok > trigger:
            pap_docs = pack_docs(
                pap_docs,
                max(600, int(budget_papers * 0.7)),
                self.answer_runtime.count_tokens,
            )
            papers_ctx = format_docs_compact(
                pap_docs,
                max_docs=max_docs,
                text_limit=text_limit,
            )
            msgs = ANSWER_PROMPT.format_messages(
                papers=papers_ctx,
                question=context.rewritten_question,
            )
            full_prompt = "\n\n".join(m.content for m in msgs)

        try:
            self.last_answer_llm_calls += 1
            raw_answer = self.answer_runtime.llm.invoke(full_prompt) or ""
        except Exception:
            raw_answer = ""

        answer = _strip_prompt_leak(str(raw_answer)).strip()

        if _answer_is_bad(answer):
            focus = self._choose_last_focus_from_answer(context.rewritten_question, pap_docs)
            if focus:
                answer = focus
            else:
                answer = "I found related papers, but the answer could not be cleanly extracted."

        self.finalize_turn(context, answer)

        return answer, context.paper_docs, []


class EngineManager:
    def __init__(self) -> None:
        _ensure_dir(CACHE_DIR)
        _ensure_dir(MEMORY_DIR)

        self.dbm = DatabaseManager()
        self.dbm.ensure_dirs_exist()

        self.store = SessionStore(STATE_DB)
        self.embeddings = build_embeddings()

        self.active_mode = self.dbm.resolve_mode(getattr(settings, "active_mode", ""))
        if not self.active_mode:
            raise RuntimeError("No retrieval mode configured")
        self.papers_vs_cache: Dict[str, Chroma] = {}

        self._memory_client = _make_local_chroma_client(MEMORY_DIR)
        self.memory_vs = Chroma(
            collection_name="memory",
            persist_directory=MEMORY_DIR,
            embedding_function=self.embeddings,
            client=self._memory_client,
        )

        self.answer_runtime: Optional[ModelRuntime] = None
        self.utility_runtime: Optional[ModelRuntime] = None
        self.active_answer_model_key: str = ""
        self.active_utility_model_key: str = ""
        self.answer_generation_lock = threading.Lock()

        self.utility_worker: Optional[UtilityWorker] = None

    def get_papers_vs(self, mode: str) -> Chroma:
        m = self.dbm.resolve_mode(mode)
        if not m:
            raise RuntimeError("No database config available")
        if m not in self.papers_vs_cache:
            cfg = self.dbm.get_config(m)
            if cfg is None:
                raise RuntimeError("No database config available")
            _ensure_dir(cfg.chroma_dir)
            client = _make_local_chroma_client(cfg.chroma_dir)
            self.papers_vs_cache[m] = Chroma(
                collection_name=cfg.collection,
                persist_directory=cfg.chroma_dir,
                embedding_function=self.embeddings,
                client=client,
            )
        return self.papers_vs_cache[m]

    def switch_mode(self, mode: str) -> None:
        resolved = self.dbm.resolve_mode(mode)
        if not resolved:
            raise RuntimeError("No retrieval mode configured")
        self.active_mode = resolved
        self.dbm.switch_config(resolved)

    def switch_answer_model(self, llm_model_key: str) -> None:
        key = (llm_model_key or "").strip().lower()
        if key == self.active_answer_model_key and self.answer_runtime is not None:
            return

        if self.answer_runtime is not None:
            try:
                self.answer_runtime.close()
            except Exception:
                pass
            self.answer_runtime = None

        clear_runtime_cache()

        model_path = _resolve_llm_path(key)
        self.answer_runtime = ModelRuntime(
            model_path,
            max_new_tokens=int(getattr(settings, "answer_max_new_tokens", 256)),
            do_sample=False,
            temperature=0.0,
        )
        self.active_answer_model_key = key

    def switch_model(self, llm_model_key: str) -> None:
        self.switch_answer_model(llm_model_key)

    def switch_utility_model(self, llm_model_key: str) -> None:
        key = (llm_model_key or "").strip().lower()
        if key == self.active_utility_model_key and self.utility_runtime is not None:
            return

        if self.utility_runtime is not None:
            try:
                self.utility_runtime.close()
            except Exception:
                pass
            self.utility_runtime = None

        clear_runtime_cache()

        model_path = _resolve_llm_path(key)
        self.utility_runtime = ModelRuntime(
            model_path,
            max_new_tokens=int(getattr(settings, "utility_max_new_tokens", 256)),
            do_sample=False,
            temperature=0.0,
        )
        self.active_utility_model_key = key

    def _ensure_utility_worker(self) -> None:
        if self.utility_worker is not None:
            return

        util_key = getattr(settings, "utility_model_key", "llama-3.2-1b")
        self.switch_utility_model(util_key)

        if self.utility_runtime is None:
            return

        self.utility_worker = UtilityWorker(
            store=self.store,
            memory_vs=self.memory_vs,
            runtime=self.utility_runtime,
            answer_generation_lock=self.answer_generation_lock,
        )
        self.utility_worker.start()

    def get_engine(self, session_id: str, mode: str, *, stateless: bool = False) -> Engine:
        if self.answer_runtime is None:
            self.switch_answer_model(getattr(settings, "answer_model_key", "llama-3.2-3b"))

        if self.utility_runtime is None:
            self.switch_utility_model(getattr(settings, "utility_model_key", "llama-3.2-1b"))

        if int(getattr(settings, "enable_utility_background", 1)) == 1:
            self._ensure_utility_worker()

        papers_vs = self.get_papers_vs(mode)
        return Engine(
            answer_runtime=self.answer_runtime,
            utility_runtime=self.utility_runtime,
            papers_vs=papers_vs,
            memory_vs=self.memory_vs,
            store=self.store,
            session_id=session_id,
            utility_worker=self.utility_worker,
            stateless=stateless,
        )

    def reset_session(self, session_id: str) -> None:
        try:
            self.store.reset(session_id)
        except Exception:
            pass
        try:
            col = getattr(self.memory_vs, "_collection", None)
            if col is not None:
                got = col.get(where={"session_id": session_id}, include=["ids"])
                ids = got.get("ids") or []
                if ids:
                    col.delete(ids=ids)
            try:
                self.memory_vs.persist()
            except Exception:
                pass
        except Exception:
            pass


_GLOBAL_MANAGER: Optional[EngineManager] = None


def get_global_manager() -> EngineManager:
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = EngineManager()
    return _GLOBAL_MANAGER