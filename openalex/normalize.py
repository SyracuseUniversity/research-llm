"""
normalize.py — Transform raw OpenAlex JSONL into clean structured records.

Input priority (auto-detected):
  1. data/raw/works_with_docling.jsonl   — has fulltext + docling sections
  2. data/raw/works_with_fulltext.jsonl  — has fulltext, no docling
  3. data/raw/works.jsonl               — raw fetch only

Writes:
  data/raw/normalized_works.jsonl
  data/raw/normalized_authors.jsonl
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

INSTITUTION_ID = config.OPENALEX_INSTITUTION_ID


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _oa_id(url: Optional[str]) -> str:
    if not url:
        return ""
    return url.rsplit("/", 1)[-1].strip()

def _safe(val: Any, default: str = "") -> str:
    if val is None:
        return default
    return str(val).strip() or default

def _year(raw: Any) -> Optional[int]:
    if isinstance(raw, int):
        return raw
    m = re.search(r"\b(19|20)\d{2}\b", str(raw or ""))
    return int(m.group(0)) if m else None

def _clean_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Work normalizer
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_work(raw: Dict) -> Optional[Dict]:
    work_id = _oa_id(raw.get("id"))
    title   = _clean_html(_safe(raw.get("title") or raw.get("display_name")))
    if not work_id or not title:
        return None

    # ── Authors ───────────────────────────────────────────────────────────────
    all_authors: List[Dict] = []
    su_authors:  List[Dict] = []

    for auth in (raw.get("authorships") or []):
        author_obj   = auth.get("author") or {}
        institutions = auth.get("institutions") or []
        name         = _safe(author_obj.get("display_name"))
        if not name:
            continue

        is_su = any(
            _oa_id(inst.get("id")) == INSTITUTION_ID
            or INSTITUTION_ID in [_oa_id(l) for l in (inst.get("lineage") or [])]
            for inst in institutions
        )

        rec = {
            "id":               _oa_id(author_obj.get("id")),
            "name":             name,
            "orcid":            _safe(author_obj.get("orcid")),
            "position":         _safe(auth.get("author_position")),
            "is_corresponding": bool(auth.get("is_corresponding")),
            "is_su":            is_su,
        }
        all_authors.append(rec)
        if is_su:
            su_authors.append(rec)

    primary_researcher  = su_authors[0]["name"] if su_authors else (all_authors[0]["name"] if all_authors else "")
    su_researcher_names = [a["name"] for a in su_authors]

    authors_str = ", ".join(a["name"] for a in all_authors[:10])
    if len(all_authors) > 10:
        authors_str += f" et al. ({len(all_authors)} authors)"

    # ── Topics + Keywords ─────────────────────────────────────────────────────
    topics = []
    for t in (raw.get("topics") or [])[:5]:
        topics.append({
            "id":       _oa_id(t.get("id")),
            "name":     _safe(t.get("display_name")),
            "subfield": _safe((t.get("subfield") or {}).get("display_name")),
            "field":    _safe((t.get("field") or {}).get("display_name")),
            "domain":   _safe((t.get("domain") or {}).get("display_name")),
            "score":    float(t.get("score", 0) or 0),
        })

    keywords = []
    for k in (raw.get("keywords") or []):
        kw = _safe(k.get("display_name") or k.get("keyword"))
        if kw and kw not in keywords:
            keywords.append(kw)

    # ── Misc ──────────────────────────────────────────────────────────────────
    oa         = raw.get("open_access") or {}

    # ── Fulltext + Docling fields ─────────────────────────────────────────────
    fulltext_status = raw.get("fulltext_status", "none")
    docling_status  = raw.get("docling_status",  "none")

    return {
        "openalex_id":        work_id,
        "doi":                _safe(raw.get("doi")),
        "title":              title,
        "abstract":           _safe(raw.get("abstract_text")),
        "publication_year":   _year(raw.get("publication_year")),
        "publication_date":   _safe(raw.get("publication_date")),
        "work_type":          _safe(raw.get("type")),
        "cited_by_count":     int(raw.get("cited_by_count") or 0),
        "oa_status":          _safe(oa.get("oa_status")),
        "oa_url":             _safe(oa.get("oa_url")),
        "primary_researcher": primary_researcher,
        "su_researchers":     su_researcher_names,
        "authors_str":        authors_str,
        "all_authors":        all_authors,
        "topics":             topics,
        "keywords":           keywords,
        # References — kept for Neo4j citation edges, not ingested into Chroma
        "referenced_works":   [r for r in (raw.get("referenced_works") or []) if r],
        # Fulltext
        "fulltext_status":    fulltext_status,
        "fulltext_path":      raw.get("fulltext_path", ""),
        "has_fulltext":       fulltext_status in ("tei_xml", "pdf"),
        # Docling
        "docling_status":     docling_status,
        "docling_path":       raw.get("docling_path", ""),
        "has_docling":        docling_status in ("docling_ok", "fallback_pdf"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Author normalizer
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_author(raw: Dict) -> Optional[Dict]:
    author_id = _oa_id(raw.get("id"))
    name      = _safe(raw.get("display_name"))
    if not author_id or not name:
        return None

    stats = raw.get("summary_stats") or {}
    topics = [
        {"id": _oa_id(t.get("id")), "name": _safe(t.get("display_name")), "count": int(t.get("count", 0) or 0)}
        for t in (raw.get("topics") or [])[:10]
    ]
    affiliations = [
        {"id": _oa_id(i.get("id")), "name": _safe(i.get("display_name")),
         "country": _safe(i.get("country_code")), "type": _safe(i.get("type"))}
        for i in (raw.get("last_known_institutions") or [])
    ]

    return {
        "openalex_id":    author_id,
        "name":           name,
        "orcid":          _safe(raw.get("orcid")),
        "alt_names":      [_safe(n) for n in (raw.get("display_name_alternatives") or []) if _safe(n)],
        "works_count":    int(raw.get("works_count") or 0),
        "cited_by_count": int(raw.get("cited_by_count") or 0),
        "h_index":        int(stats.get("h_index") or 0),
        "i10_index":      int(stats.get("i10_index") or 0),
        "topics":         topics,
        "affiliations":   affiliations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run() -> Dict[str, int]:
    raw_dir = Path(config.RAW_DIR)

    # Auto-select best available input
    candidates = [
        raw_dir / "works_with_docling.jsonl",
        raw_dir / "works_with_fulltext.jsonl",
        raw_dir / "works.jsonl",
    ]
    works_in = next((p for p in candidates if p.exists()), None)
    if works_in is None:
        logger.error("No works input file found — run fetch_and_download first")
        return {"works": 0, "authors": 0}

    label = {
        "works_with_docling.jsonl":  "docling (best)",
        "works_with_fulltext.jsonl": "fulltext only",
        "works.jsonl":               "raw (no fulltext)",
    }.get(works_in.name, works_in.name)
    logger.info("Input: %s [%s]", works_in.name, label)

    works_out   = raw_dir / "normalized_works.jsonl"
    authors_in  = raw_dir / "authors.jsonl"
    authors_out = raw_dir / "normalized_authors.jsonl"

    # ── Works ─────────────────────────────────────────────────────────────────
    works_count = skipped = 0
    seen_ids    = set()
    ft_counts   = {"tei_xml": 0, "pdf": 0, "none": 0}
    dl_counts   = {"docling_ok": 0, "fallback_pdf": 0, "none": 0}

    with open(works_in, encoding="utf-8") as fin, \
         open(works_out, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            norm = normalize_work(raw)
            if norm and norm["openalex_id"] not in seen_ids:
                seen_ids.add(norm["openalex_id"])
                fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                works_count += 1
                ft_counts[norm["fulltext_status"]] = ft_counts.get(norm["fulltext_status"], 0) + 1
                dl_counts[norm["docling_status"]]  = dl_counts.get(norm["docling_status"], 0) + 1
            else:
                skipped += 1

    logger.info(
        "Works: %d normalized (%d skipped) | "
        "Fulltext — pdf:%d tei:%d none:%d | "
        "Docling — ok:%d fallback:%d none:%d",
        works_count, skipped,
        ft_counts["pdf"], ft_counts["tei_xml"], ft_counts["none"],
        dl_counts["docling_ok"], dl_counts["fallback_pdf"], dl_counts["none"],
    )

    # ── Authors ───────────────────────────────────────────────────────────────
    authors_count = 0
    seen_authors  = set()

    if authors_in.exists():
        with open(authors_in, encoding="utf-8") as fin, \
             open(authors_out, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                norm = normalize_author(raw)
                if norm and norm["openalex_id"] not in seen_authors:
                    seen_authors.add(norm["openalex_id"])
                    fout.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    authors_count += 1
        logger.info("Authors: %d normalized", authors_count)

    return {"works": works_count, "authors": authors_count}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    print(run())