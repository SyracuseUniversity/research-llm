"""
chunker.py — Chunk normalized works into embeddable pieces.

Chunk types (in priority order):
  title_abstract  — always created
  keywords        — always created if keywords/topics exist
  section         — from Docling structured sections (best)
  fallback_text   — from pdfminer text stored in Docling JSON
  table           — tables extracted by Docling (never split)
  figure_caption  — figure captions from Docling
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _token_est(text: str) -> int:
    return int(len(text.split()) * 1.3)

def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def chunk_with_overlap(text: str) -> List[str]:
    max_tok = config.CHUNK_MAX_TOKENS
    overlap = config.CHUNK_OVERLAP_TOKENS

    sentences = _sentence_split(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    current_tok = 0

    for sent in sentences:
        sent_tok = _token_est(sent)
        if current_tok + sent_tok > max_tok and current:
            chunks.append(" ".join(current))
            keep, keep_tok = [], 0
            for s in reversed(current):
                t = _token_est(s)
                if keep_tok + t > overlap:
                    break
                keep.insert(0, s)
                keep_tok += t
            current, current_tok = keep, keep_tok
        current.append(sent)
        current_tok += sent_tok

    if current:
        chunks.append(" ".join(current))
    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# Docling sections loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_docling_sections(docling_path: str) -> List[Dict]:
    if not docling_path or not Path(docling_path).exists():
        return []
    try:
        data = json.loads(Path(docling_path).read_text(encoding="utf-8"))
        return data.get("sections") or []
    except Exception as e:
        logger.debug("Failed to load Docling JSON %s: %s", docling_path, e)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# Context prefix (prepended to embed_text only, not stored text)
# ═══════════════════════════════════════════════════════════════════════════════

def _context_prefix(work: Dict) -> str:
    parts = []
    if work.get("primary_researcher"):
        parts.append(f"Research by {work['primary_researcher']} at Syracuse University")
    topics = work.get("topics") or []
    if topics and topics[0].get("name"):
        parts.append(f"in {topics[0]['name']}")
    if work.get("publication_year"):
        parts.append(f"({work['publication_year']})")
    prefix = " ".join(parts)
    return f"{prefix}. " if prefix else ""

def _chunk_id(work_id: str, ctype: str, idx: int) -> str:
    return f"{work_id}_{ctype}_{idx}"


# ═══════════════════════════════════════════════════════════════════════════════
# Main chunker
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_work(work: Dict) -> List[Dict]:
    """
    Produce all chunks for a single normalized work.
    Each chunk: {chunk_id, work_id, chunk_type, chunk_index, text, embed_text, metadata}
    """
    chunks: List[Dict] = []
    work_id = work["openalex_id"]
    title   = work.get("title", "")
    prefix  = _context_prefix(work)
    topics  = work.get("topics") or []

    base_meta = {
        "researcher":      work.get("primary_researcher", ""),
        "year":            str(work.get("publication_year") or ""),
        "topic":           topics[0]["name"] if topics else "",
        "doi":             work.get("doi", ""),
        "title":           title,
        "work_type":       work.get("work_type", ""),
        "cited_by_count":  work.get("cited_by_count", 0),
        "fulltext_status": work.get("fulltext_status", "none"),
        "docling_status":  work.get("docling_status", "none"),
        "has_fulltext":    work.get("has_fulltext", False),
        # referenced_works intentionally excluded — stored in Neo4j only
    }
    su = work.get("su_researchers") or []
    if su:
        base_meta["su_researchers"] = " | ".join(su)

    def _add(ctype, idx, text, extra_meta=None):
        m = {**base_meta, "chunk_type": ctype, **(extra_meta or {})}
        chunks.append({
            "chunk_id":    _chunk_id(work_id, ctype, idx),
            "work_id":     work_id,
            "chunk_type":  ctype,
            "chunk_index": idx,
            "text":        text,
            "embed_text":  f"{prefix}{text}",
            "metadata":    m,
        })

    # ── 1. Title + Abstract (always) ──────────────────────────────────────────
    abstract = work.get("abstract", "")
    ta       = f"{title}\n\n{abstract}".strip() if abstract else title
    if title:
        _add("title_abstract", 0, ta)

    # ── 2. Keywords + Topics (always if present) ──────────────────────────────
    keywords = work.get("keywords") or []
    if keywords or topics:
        parts = [title]
        if keywords:
            parts.append("Keywords: " + ", ".join(keywords[:15]))
        if topics:
            tnames = [t["name"] for t in topics[:5] if t.get("name")]
            if tnames:
                parts.append("Research topics: " + ", ".join(tnames))
            fields = list({t["field"] for t in topics[:5] if t.get("field")})
            if fields:
                parts.append("Fields: " + ", ".join(fields))
            domains = list({t["domain"] for t in topics[:5] if t.get("domain")})
            if domains:
                parts.append("Domains: " + ", ".join(domains))
        _add("keywords", 1, ". ".join(parts))

    # ── 3. Docling fulltext sections ──────────────────────────────────────────
    docling_status = work.get("docling_status", "none")
    docling_path   = work.get("docling_path", "")

    if docling_status in ("docling_ok", "fallback_pdf") and docling_path:
        sections = _load_docling_sections(docling_path)
        idx = 100

        for sec in sections:
            heading      = sec.get("heading", "Section")
            body         = sec.get("text", "").strip()
            element_type = sec.get("element_type", "section")
            level        = sec.get("level", 1)

            if not body or len(body) < 30:
                continue

            sec_meta = {"section_heading": heading, "section_level": level}

            if element_type in ("table", "figure_caption"):
                # Never split tables or captions
                ctype = "table" if element_type == "table" else "figure_caption"
                _add(ctype, idx, f"{title} — {heading}: {body}", sec_meta)
                idx += 1
            else:
                # Text sections: split with overlap
                ctype = "section" if docling_status == "docling_ok" else "fallback_text"
                for sub in chunk_with_overlap(body):
                    _add(ctype, idx, f"{title} — {heading}: {sub}", sec_meta)
                    idx += 1

    return chunks