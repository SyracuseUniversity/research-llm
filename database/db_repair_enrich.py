# db_repair_enrich.py
import argparse
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import sentencepiece  # noqa: F401
except Exception:
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


DB_PATH_DEFAULT = r"C:\codes\t5-db\syr_research_all.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("db_repair_enrich")

EMAIL_FALLBACK = "example@example.com"
USER_AGENT = f"db-repair-enrich/1.0 (mailto:{EMAIL_FALLBACK})"


MATHML_TAG = re.compile(r"<mml:[^>]+>|</mml:[^>]+>", re.I)
XML_TAG = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

doi_pat = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
doi_url_pat = re.compile(r"https?://(?:dx\.)?doi\.org/(10\.\d{4,9}/[^\s\"\'<>]+)", re.I)

arxiv_url_pat = re.compile(
    r"https?://arxiv\.org/(?:abs|pdf)/"
    r"([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)(?:\.pdf)?",
    re.I,
)
arxiv_pat = re.compile(
    r"\barxiv\s*:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)\b",
    re.I,
)
arxiv_plain_pat = re.compile(r"^(?:[0-9]{4}\.[0-9]{4,5})(?:v\d+)?$", re.I)
arxiv_old_pat = re.compile(r"^(?:[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)

year_pat = re.compile(r"\b(19|20)\d{2}\b")
year_context_pat = re.compile(
    r"\b(?:published|accepted|received|revised|copyright|conference|proceedings)\b.{0,60}\b((?:19|20)\d{2})\b",
    re.I | re.S,
)

email_pat = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)

DUMMY_PREFIX = re.compile(r"^(arxiv|cms\s+paper|european\s+organization|cern|preprint)", re.I)
BAD_TITLE_START = re.compile(r"^(abstract|introduction|keywords|contents|table\s+of\s+contents)$", re.I)
TRAILING_PUNCT_RE = re.compile(r"""[\)\]\}\>\.,;:'"\u2019\u201d]+$""", re.U)

ABSTRACT_MARKERS = re.compile(r"^\s*(abstract|summary|keywords)\s*$", re.I)
SECTION_MARKERS = re.compile(r"^\s*(introduction|contents|table\s+of\s+contents)\s*$", re.I)

JSON_OBJ_RE = re.compile(r"\{.*\}", re.S)


def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\x00", " ")
    s = MATHML_TAG.sub(" ", s)
    s = XML_TAG.sub(" ", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def is_missing(s: Any) -> bool:
    return (s is None) or (not str(s).strip())


def safe_commit(conn: sqlite3.Connection) -> None:
    try:
        conn.commit()
    except Exception:
        pass


def ensure_pragmas(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")
    cur.execute("PRAGMA cache_size=-200000;")
    safe_commit(conn)


def ensure_min_columns(conn: sqlite3.Connection) -> None:
    ensure_pragmas(conn)
    cur = conn.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}

    if "research_info" in tables:
        cur.execute("PRAGMA table_info(research_info);")
        cols = {r[1] for r in cur.fetchall()}

        def add_col(name: str, decl: str) -> None:
            if name in cols:
                return
            cur.execute(f"ALTER TABLE research_info ADD COLUMN {name} {decl};")

        add_col("topics_status", "TEXT DEFAULT 'untagged'")
        add_col("topics_json", "TEXT")
        add_col("primary_topic", "TEXT")
        add_col("subject", "TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_works_paper_id ON works(paper_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_research_info_paper_id ON research_info(paper_id);")
    safe_commit(conn)


def ensure_research_info_rows_for_works(conn: sqlite3.Connection, batch: int = 5000) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT w.paper_id
        FROM works w
        LEFT JOIN research_info ri ON ri.paper_id = w.paper_id
        WHERE ri.paper_id IS NULL
        GROUP BY w.paper_id
        ORDER BY w.paper_id
        """
    )
    missing = [int(r[0]) for r in cur.fetchall()]
    if not missing:
        return 0

    inserted = 0
    for i in tqdm(range(0, len(missing), batch), desc="Ensuring research_info rows", unit="paper"):
        chunk = missing[i : i + batch]
        cur.execute("DROP TABLE IF EXISTS tmp_missing_pids;")
        cur.execute("CREATE TEMP TABLE tmp_missing_pids(paper_id INTEGER PRIMARY KEY);")
        cur.executemany("INSERT OR IGNORE INTO tmp_missing_pids(paper_id) VALUES (?);", [(pid,) for pid in chunk])

        cur.execute(
            """
            INSERT INTO research_info (paper_id, work_title, authors, doi, publication_date, researcher_name, info, topics_status)
            SELECT
                p.paper_id,
                COALESCE(p.title, ''),
                COALESCE(p.authors, ''),
                COALESCE(p.doi, ''),
                COALESCE(p.publication_date, ''),
                '',
                '',
                'untagged'
            FROM papers p
            JOIN tmp_missing_pids t ON t.paper_id = p.paper_id
            """
        )
        inserted += int(cur.rowcount or 0)
        safe_commit(conn)

    return inserted


def counts_snapshot(cur: sqlite3.Cursor) -> Dict[str, int]:
    snap: Dict[str, int] = {}
    pairs = [
        ("SELECT COUNT(*) FROM papers", "papers_total"),
        ("SELECT COUNT(*) FROM works", "works_total"),
        ("SELECT COUNT(*) FROM research_info", "ri_total"),
        ("SELECT COUNT(*) FROM papers WHERE title IS NULL OR TRIM(title)=''", "papers_title_missing"),
        ("SELECT COUNT(*) FROM papers WHERE authors IS NULL OR TRIM(authors)=''", "papers_authors_missing"),
        ("SELECT COUNT(*) FROM papers WHERE publication_date IS NULL OR TRIM(publication_date)=''", "papers_pub_missing"),
        ("SELECT COUNT(*) FROM papers WHERE doi IS NULL OR TRIM(doi)=''", "papers_doi_missing"),
        ("SELECT COUNT(*) FROM papers WHERE arxiv_id IS NULL OR TRIM(arxiv_id)=''", "papers_arxiv_missing"),
        ("SELECT COUNT(*) FROM research_info WHERE work_title IS NULL OR TRIM(work_title)=''", "ri_title_missing"),
        ("SELECT COUNT(*) FROM research_info WHERE authors IS NULL OR TRIM(authors)=''", "ri_authors_missing"),
        ("SELECT COUNT(*) FROM research_info WHERE publication_date IS NULL OR TRIM(publication_date)=''", "ri_pub_missing"),
        ("SELECT COUNT(*) FROM research_info WHERE doi IS NULL OR TRIM(doi)=''", "ri_doi_missing"),
        ("SELECT COUNT(*) FROM works WHERE summary_status='summarized'", "works_summarized"),
    ]
    for q, k in pairs:
        cur.execute(q)
        snap[k] = int(cur.fetchone()[0])
    return snap


def print_snapshot(s: Dict[str, int], label: str) -> None:
    print()
    print(label)
    keys = [
        "papers_total",
        "works_total",
        "ri_total",
        "papers_title_missing",
        "papers_authors_missing",
        "papers_pub_missing",
        "papers_doi_missing",
        "papers_arxiv_missing",
        "ri_title_missing",
        "ri_authors_missing",
        "ri_pub_missing",
        "ri_doi_missing",
        "works_summarized",
    ]
    for k in keys:
        print(f"{k}:", s.get(k, 0))


def _strip_trailing_punct(s: str) -> str:
    s = (s or "").strip()
    while True:
        ns = TRAILING_PUNCT_RE.sub("", s).strip()
        if ns == s:
            return s
        s = ns


def norm_doi(s: Any) -> str:
    s = norm(s)
    if not s:
        return ""
    m = doi_url_pat.search(s)
    if m:
        s = m.group(1)
    s = re.sub(r"^\s*doi\s*:\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^\s*https?://(?:dx\.)?doi\.org/", "", s, flags=re.I).strip()
    s = _strip_trailing_punct(s)
    return s.lower()


def norm_arxiv(s: Any) -> str:
    s = norm(s)
    if not s:
        return ""
    m = arxiv_url_pat.search(s)
    if m:
        s = m.group(1)
    s = re.sub(r"^\s*arxiv\s*:\s*", "", s, flags=re.I).strip()
    s = re.sub(r"^\s*https?://arxiv\.org/(?:abs|pdf)/", "", s, flags=re.I).strip()
    s = re.sub(r"\.pdf\s*$", "", s, flags=re.I).strip()
    s = _strip_trailing_punct(s)
    return s


def is_valid_arxiv_id(s: Any) -> bool:
    a = norm_arxiv(s)
    if not a:
        return False
    return bool(arxiv_plain_pat.fullmatch(a) or arxiv_old_pat.fullmatch(a))


def _extract_arxiv_from_filename(file_name: str) -> str:
    s = norm(file_name)
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1]
    s = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", s)
    s = s.replace("_", " ").replace("-", " ")

    m = arxiv_url_pat.search(s)
    if m:
        a = norm_arxiv(m.group(1))
        return a if is_valid_arxiv_id(a) else ""

    m2 = arxiv_pat.search(s)
    if m2:
        a = norm_arxiv(m2.group(1))
        return a if is_valid_arxiv_id(a) else ""

    a = norm_arxiv(s)
    return a if is_valid_arxiv_id(a) else ""


def _try_set_unique_doi(cur: sqlite3.Cursor, pid: int, doi_val: str) -> None:
    doi_val = norm_doi(doi_val)
    if not doi_val:
        return
    try:
        cur.execute(
            """
            UPDATE papers
            SET doi = ?
            WHERE paper_id = ?
              AND (doi IS NULL OR TRIM(doi) = '')
              AND NOT EXISTS (
                SELECT 1 FROM papers p2
                WHERE LOWER(NULLIF(TRIM(p2.doi), '')) = LOWER(?)
                  AND p2.paper_id <> ?
              )
            """,
            (doi_val, pid, doi_val, pid),
        )
    except sqlite3.IntegrityError:
        return


def _try_set_unique_arxiv(cur: sqlite3.Cursor, pid: int, arxiv_val: str) -> None:
    arxiv_val = norm_arxiv(arxiv_val)
    if not is_valid_arxiv_id(arxiv_val):
        return
    try:
        cur.execute(
            """
            UPDATE papers
            SET arxiv_id = ?
            WHERE paper_id = ?
              AND (arxiv_id IS NULL OR TRIM(arxiv_id) = '')
              AND NOT EXISTS (
                SELECT 1 FROM papers p2
                WHERE NULLIF(TRIM(p2.arxiv_id), '') = ?
                  AND p2.paper_id <> ?
              )
            """,
            (arxiv_val, pid, arxiv_val, pid),
        )
    except sqlite3.IntegrityError:
        return


def phase_a_copy_across(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    ensure_pragmas(conn)

    cur.execute("DROP TABLE IF EXISTS tmp_ri_map;")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_ri_map AS
        SELECT paper_id,
               MAX(NULLIF(TRIM(work_title), '')) AS work_title,
               MAX(NULLIF(TRIM(authors), '')) AS authors,
               MAX(NULLIF(TRIM(publication_date), '')) AS publication_date,
               MAX(NULLIF(TRIM(doi), '')) AS doi
        FROM research_info
        GROUP BY paper_id
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tmp_ri_map_pid ON tmp_ri_map(paper_id);")

    cur.execute(
        """
        UPDATE papers
        SET title = COALESCE(NULLIF(TRIM(title), ''), (SELECT m.work_title FROM tmp_ri_map m WHERE m.paper_id = papers.paper_id)),
            authors = COALESCE(NULLIF(TRIM(authors), ''), (SELECT m.authors FROM tmp_ri_map m WHERE m.paper_id = papers.paper_id)),
            publication_date = COALESCE(NULLIF(TRIM(publication_date), ''), (SELECT m.publication_date FROM tmp_ri_map m WHERE m.paper_id = papers.paper_id)),
            doi = COALESCE(NULLIF(TRIM(doi), ''), (SELECT m.doi FROM tmp_ri_map m WHERE m.paper_id = papers.paper_id))
        """
    )

    cur.execute("DROP TABLE IF EXISTS tmp_p_map;")
    cur.execute(
        """
        CREATE TEMP TABLE tmp_p_map AS
        SELECT paper_id,
               MAX(NULLIF(TRIM(title), '')) AS title,
               MAX(NULLIF(TRIM(authors), '')) AS authors,
               MAX(NULLIF(TRIM(publication_date), '')) AS publication_date,
               MAX(NULLIF(TRIM(doi), '')) AS doi
        FROM papers
        GROUP BY paper_id
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tmp_p_map_pid ON tmp_p_map(paper_id);")

    cur.execute(
        """
        UPDATE research_info
        SET work_title = COALESCE(NULLIF(TRIM(work_title), ''), (SELECT p.title FROM tmp_p_map p WHERE p.paper_id = research_info.paper_id)),
            authors = COALESCE(NULLIF(TRIM(authors), ''), (SELECT p.authors FROM tmp_p_map p WHERE p.paper_id = research_info.paper_id)),
            publication_date = COALESCE(NULLIF(TRIM(publication_date), ''), (SELECT p.publication_date FROM tmp_p_map p WHERE p.paper_id = research_info.paper_id)),
            doi = COALESCE(NULLIF(TRIM(doi), ''), (SELECT p.doi FROM tmp_p_map p WHERE p.paper_id = research_info.paper_id))
        """
    )

    cur.execute(
        """
        SELECT w.paper_id, MAX(NULLIF(TRIM(w.file_name), '')) AS file_name
        FROM works w
        WHERE w.file_name IS NOT NULL AND TRIM(w.file_name) <> ''
        GROUP BY w.paper_id
        """
    )
    for pid, fn in cur.fetchall():
        arx = _extract_arxiv_from_filename(fn or "")
        if arx:
            _try_set_unique_arxiv(cur, int(pid), arx)

    safe_commit(conn)


@dataclass
class ExtractResult:
    paper_id: int
    title: str = ""
    authors: str = ""
    year: str = ""
    doi: str = ""
    arxiv_id: str = ""
    researcher_name: str = ""
    info: str = ""


def looks_dummy_title(t: str) -> bool:
    t = (t or "").strip()
    if (not t) or len(t) > 240:
        return True
    if DUMMY_PREFIX.match(t):
        return True
    if BAD_TITLE_START.match(t):
        return True
    w = t.split()
    if len(w) < 4:
        return True
    if sum(ch.isdigit() for ch in t) > 12:
        return True
    return False


def _clean_candidate_line(l: str) -> str:
    l = norm(l)
    l = l.strip(" \t\r\n")
    l = re.sub(r"^\d+\s*$", "", l).strip()
    l = re.sub(r"^\s*(?:page|pages)\s+\d+\s*(?:of\s+\d+)?\s*$", "", l, flags=re.I).strip()
    l = re.sub(r"\s{2,}", " ", l).strip()
    return l


def _find_line_index(lines: List[str], pattern: re.Pattern) -> int:
    for i, l in enumerate(lines[:600]):
        if pattern.search(l):
            return i
    return -1


def _extract_title_and_authors(lines: List[str]) -> Tuple[str, str]:
    cleaned: List[str] = []
    for l in lines[:800]:
        l2 = _clean_candidate_line(l)
        if not l2:
            continue
        if email_pat.search(l2):
            continue
        low = l2.lower()
        if low.startswith("http://") or low.startswith("https://"):
            continue
        if "doi.org/" in low or "dx.doi.org/" in low or "arxiv.org/" in low:
            continue
        if low.startswith("doi:") or low.startswith("arxiv:") or low.startswith("arxiv :"):
            continue
        cleaned.append(l2)

    if not cleaned:
        return "", ""

    abs_i = _find_line_index(cleaned, ABSTRACT_MARKERS)
    if abs_i == -1:
        abs_i = _find_line_index(cleaned, SECTION_MARKERS)

    window = cleaned[:200]
    if abs_i != -1:
        lo = max(0, abs_i - 20)
        hi = min(len(cleaned), abs_i + 5)
        window = cleaned[lo:hi]

    title = ""
    for l in window[:80]:
        if 10 <= len(l) <= 240 and 4 <= len(l.split()) <= 50 and not looks_dummy_title(l):
            title = l
            break

    if not title:
        for l in cleaned[:120]:
            if 10 <= len(l) <= 240 and 4 <= len(l.split()) <= 50 and not looks_dummy_title(l):
                title = l
                break

    authors = ""

    def is_authorish(s: str) -> bool:
        if len(s) < 6 or len(s) > 280:
            return False
        low = s.lower()
        if ABSTRACT_MARKERS.match(s.strip()) or SECTION_MARKERS.match(s.strip()):
            return False
        if any(x in low for x in ["university", "institute", "department", "laboratory", "collaboration", "faculty", "school of"]):
            return False
        if sum(ch.isdigit() for ch in s) > 6:
            return False
        if ("," in s) or (";" in s) or (" and " in low):
            tokens = [t.strip() for t in re.split(r"[;,]", s) if t.strip()]
            return len(tokens) >= 2
        return False

    if title and title in cleaned:
        idx = cleaned.index(title)
        for l in cleaned[idx + 1 : idx + 60]:
            if is_authorish(l):
                authors = l
                break

    if not authors:
        for l in window[:60]:
            if is_authorish(l):
                authors = l
                break

    return norm(title), norm(authors)


def _pick_year(txt: str) -> str:
    head = txt[:20000]
    m = year_context_pat.search(head)
    if m:
        y = m.group(1)
        if y and year_pat.fullmatch(y):
            return y
    years = [m.group(0) for m in year_pat.finditer(head)]
    years_int: List[int] = []
    for y in years:
        try:
            yi = int(y)
            if 1900 <= yi <= 2099:
                years_int.append(yi)
        except Exception:
            continue
    return str(max(years_int)) if years_int else ""


def _extract_doi(txt: str) -> str:
    slab = txt[:40000]
    m = doi_pat.search(slab)
    if m:
        return norm_doi(m.group(0))
    m2 = doi_url_pat.search(slab)
    if m2:
        return norm_doi(m2.group(1))
    m3 = re.search(r"\bdoi\s*:\s*(10\.\d{4,9}/[^\s\"\'<>]+)", slab, re.I)
    if m3:
        return norm_doi(m3.group(1))
    return ""


def _extract_arxiv(txt: str) -> str:
    slab = txt[:60000]
    m = arxiv_pat.search(slab)
    if m:
        a = norm_arxiv(m.group(1))
        return a if is_valid_arxiv_id(a) else ""
    m2 = arxiv_url_pat.search(slab)
    if m2:
        a = norm_arxiv(m2.group(1))
        return a if is_valid_arxiv_id(a) else ""
    return ""


def parse_from_fulltext(full_text: str, file_name: str = "") -> ExtractResult:
    txt0 = full_text or ""
    if not txt0.strip():
        return ExtractResult(paper_id=-1)

    raw = norm(txt0[:80000])
    lines = [l for l in raw.splitlines() if l and l.strip()]
    if not lines:
        lines = [raw]

    title, authors = _extract_title_and_authors(lines)
    doi = _extract_doi(txt0)
    arxiv_id = _extract_arxiv(txt0)
    if not arxiv_id and file_name:
        arxiv_id = _extract_arxiv_from_filename(file_name)

    year = _pick_year(txt0)

    researcher_name = ""
    if authors:
        first = authors.split(",")[0].strip()
        if 2 <= len(first) <= 80 and not DUMMY_PREFIX.match(first):
            researcher_name = first

    info_parts: List[str] = []
    if doi:
        info_parts.append("DOI: " + doi)
    if arxiv_id:
        info_parts.append("arXiv: " + arxiv_id)
    if year:
        info_parts.append("Date: " + year)
    info = " | ".join(info_parts)

    return ExtractResult(
        paper_id=-1,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        researcher_name=norm(researcher_name),
        info=norm(info),
    )


def _update_papers(cur: sqlite3.Cursor, pid: int, title: str, authors: str, year: str) -> None:
    cur.execute(
        """
        UPDATE papers
        SET title = COALESCE(NULLIF(TRIM(title), ''), ?),
            authors = COALESCE(NULLIF(TRIM(authors), ''), ?),
            publication_date = COALESCE(NULLIF(TRIM(publication_date), ''), ?)
        WHERE paper_id = ?
        """,
        (title or None, authors or None, year or None, pid),
    )


def _update_research_info(
    cur: sqlite3.Cursor,
    pid: int,
    title: str,
    authors: str,
    year: str,
    doi: str,
    researcher_name: str,
    info: str,
) -> None:
    cur.execute(
        """
        UPDATE research_info
        SET work_title = COALESCE(NULLIF(TRIM(work_title), ''), ?),
            authors = COALESCE(NULLIF(TRIM(authors), ''), ?),
            publication_date = COALESCE(NULLIF(TRIM(publication_date), ''), ?),
            doi = COALESCE(NULLIF(TRIM(doi), ''), ?),
            researcher_name = COALESCE(NULLIF(TRIM(researcher_name), ''), ?),
            info = COALESCE(NULLIF(TRIM(info), ''), ?)
        WHERE paper_id = ?
        """,
        (title or None, authors or None, year or None, doi or None, researcher_name or None, info or None, pid),
    )


def phase_b_parse_fulltext(db_path: str, limit: Optional[int], db_batch: int) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        SELECT w.paper_id, COALESCE(w.full_text, ''), COALESCE(w.file_name, ''),
               COALESCE(p.title, ''), COALESCE(p.authors, ''), COALESCE(p.publication_date, ''), COALESCE(p.doi, ''), COALESCE(p.arxiv_id, ''),
               COALESCE(ri.work_title, ''), COALESCE(ri.authors, ''), COALESCE(ri.publication_date, ''), COALESCE(ri.doi, ''), COALESCE(ri.researcher_name, ''), COALESCE(ri.info, '')
        FROM works w
        LEFT JOIN papers p ON p.paper_id = w.paper_id
        LEFT JOIN research_info ri ON ri.paper_id = w.paper_id
        WHERE w.full_text IS NOT NULL AND TRIM(w.full_text) <> ''
        GROUP BY w.paper_id
        ORDER BY w.paper_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    tasks: List[Tuple[int, str, str]] = []
    for (
        paper_id,
        full_text,
        file_name,
        p_title,
        p_authors,
        p_pub,
        p_doi,
        p_arxiv,
        ri_title,
        ri_authors,
        ri_pub,
        ri_doi,
        ri_name,
        ri_info,
    ) in rows:
        need = (
            is_missing(p_title)
            or is_missing(p_authors)
            or is_missing(p_pub)
            or is_missing(p_doi)
            or is_missing(p_arxiv)
            or is_missing(ri_title)
            or is_missing(ri_authors)
            or is_missing(ri_pub)
            or is_missing(ri_doi)
            or is_missing(ri_name)
            or is_missing(ri_info)
        )
        if need:
            tasks.append((int(paper_id), full_text, file_name or ""))

    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        return

    conn2 = sqlite3.connect(db_path)
    cur2 = conn2.cursor()
    ensure_pragmas(conn2)

    pending: List[ExtractResult] = []
    for pid, full_text, file_name in tqdm(tasks, desc="Phase B parse full_text", unit="paper"):
        r = parse_from_fulltext(full_text, file_name=file_name)
        r.paper_id = pid
        pending.append(r)

        if len(pending) >= db_batch:
            _apply_parse_batch(cur2, pending)
            safe_commit(conn2)
            pending.clear()

    if pending:
        _apply_parse_batch(cur2, pending)
        safe_commit(conn2)

    conn2.close()


def _apply_parse_batch(cur: sqlite3.Cursor, batch: List[ExtractResult]) -> None:
    for r in batch:
        if r.paper_id <= 0:
            continue

        title = norm(r.title)
        authors = norm(r.authors)
        year = norm(r.year)
        doi_val = norm_doi(r.doi)
        arx_val = norm_arxiv(r.arxiv_id)

        if title and looks_dummy_title(title):
            title = ""
        if authors and email_pat.search(authors):
            authors = ""
        if year and not re.fullmatch(r"(19|20)\d{2}", year):
            year = ""

        _update_papers(cur, r.paper_id, title, authors, year)

        if doi_val:
            _try_set_unique_doi(cur, r.paper_id, doi_val)
        if arx_val:
            _try_set_unique_arxiv(cur, r.paper_id, arx_val)

        _update_research_info(
            cur,
            r.paper_id,
            title,
            authors,
            year,
            doi_val,
            r.researcher_name,
            r.info,
        )


def _http_get(url: str, timeout: int) -> Optional[requests.Response]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        if r.status_code != 200:
            return None
        return r
    except Exception:
        return None


def crossref_lookup_by_doi(doi: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    doi = norm_doi(doi)
    if not doi:
        return None
    r = _http_get(f"https://api.crossref.org/works/{doi}", timeout)
    if not r:
        return None
    try:
        data = r.json()
    except Exception:
        return None
    msg = data.get("message") or {}

    title = ""
    if isinstance(msg.get("title"), list) and msg["title"]:
        title = msg["title"][0]

    authors = ""
    if isinstance(msg.get("author"), list) and msg.get("author"):
        names = []
        for a in msg["author"][:20]:
            given = a.get("given") or ""
            family = a.get("family") or ""
            nm = (given + " " + family).strip()
            if nm:
                names.append(nm)
        authors = ", ".join(names)

    year = ""
    issued = msg.get("issued", {}).get("date-parts")
    if isinstance(issued, list) and issued and isinstance(issued[0], list) and issued[0]:
        year = str(issued[0][0])

    return {"doi": doi, "title": norm(title), "authors": norm(authors), "year": norm(year)}


def arxiv_lookup(arxiv_id: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    arxiv_id = norm_arxiv(arxiv_id)
    if not is_valid_arxiv_id(arxiv_id):
        return None

    r = _http_get(f"https://export.arxiv.org/api/query?id_list={requests.utils.quote(arxiv_id)}", timeout)
    if not r:
        return None
    txt = r.text or ""

    titles = re.findall(r"<title>(.*?)</title>", txt, re.S)
    title = titles[1] if len(titles) >= 2 else ""

    a_list = re.findall(r"<name>(.*?)</name>", txt, re.S)
    authors = ", ".join([norm(x) for x in a_list[:20] if norm(x)])

    p_m = re.search(r"<published>(\d{4})-", txt)
    year = p_m.group(1) if p_m else ""

    doi = ""
    m_doi = re.search(r"<arxiv:doi>(.*?)</arxiv:doi>", txt, re.S | re.I)
    if m_doi:
        doi = norm_doi(norm(m_doi.group(1)))

    return {"title": norm(title), "authors": norm(authors), "year": norm(year), "doi": doi, "arxiv_id": arxiv_id}


def phase_c_external_enrich(
    db_path: str,
    do_crossref: bool,
    do_arxiv: bool,
    limit: Optional[int],
    db_batch: int,
    timeout: int,
) -> None:
    if not (do_crossref or do_arxiv):
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.paper_id,
               COALESCE(p.title, ''), COALESCE(p.authors, ''), COALESCE(p.publication_date, ''), COALESCE(p.doi, ''), COALESCE(p.arxiv_id, '')
        FROM papers p
        WHERE
            p.title IS NULL OR TRIM(p.title) = ''
            OR p.authors IS NULL OR TRIM(p.authors) = ''
            OR p.publication_date IS NULL OR TRIM(p.publication_date) = ''
            OR p.doi IS NULL OR TRIM(p.doi) = ''
            OR p.arxiv_id IS NULL OR TRIM(p.arxiv_id) = ''
        ORDER BY p.paper_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    tasks: List[Tuple[int, str, str]] = []
    for pid, title, authors, pub, doi, arx in rows:
        pid_i = int(pid)
        doi_n = norm_doi(doi)
        arx_n = norm_arxiv(arx)

        if do_arxiv and arx_n and is_valid_arxiv_id(arx_n):
            tasks.append((pid_i, "arxiv", arx_n))
        if do_crossref and doi_n:
            tasks.append((pid_i, "crossref", doi_n))

    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        return

    conn2 = sqlite3.connect(db_path)
    cur2 = conn2.cursor()
    ensure_pragmas(conn2)

    pending: List[Tuple[int, Dict[str, Any]]] = []
    for pid, kind, key in tqdm(tasks, desc="Phase C external enrich", unit="paper"):
        data = None
        if kind == "crossref":
            data = crossref_lookup_by_doi(key, timeout=timeout)
        elif kind == "arxiv":
            data = arxiv_lookup(key, timeout=timeout)

        if not data:
            continue

        pending.append((pid, data))
        if len(pending) >= db_batch:
            _apply_external_batch(cur2, pending)
            safe_commit(conn2)
            pending.clear()

    if pending:
        _apply_external_batch(cur2, pending)
        safe_commit(conn2)

    conn2.close()


def _apply_external_batch(cur: sqlite3.Cursor, pending: List[Tuple[int, Dict[str, Any]]]) -> None:
    for pid, data in pending:
        title = norm(data.get("title") or "")
        authors = norm(data.get("authors") or "")
        year = norm(data.get("year") or "")

        doi_val = norm_doi(data.get("doi") or "")
        arx_val = norm_arxiv(data.get("arxiv_id") or "")

        if year and not re.fullmatch(r"(19|20)\d{2}", year):
            year = ""

        _update_papers(cur, pid, title, authors, year)

        if doi_val:
            _try_set_unique_doi(cur, pid, doi_val)
        if arx_val:
            _try_set_unique_arxiv(cur, pid, arx_val)

        info_parts: List[str] = []
        if doi_val:
            info_parts.append("DOI: " + doi_val)
        if arx_val:
            info_parts.append("arXiv: " + arx_val)
        if year:
            info_parts.append("Date: " + year)
        info = " | ".join(info_parts)

        researcher_name = ""
        if authors:
            first = authors.split(",")[0].strip()
            if 2 <= len(first) <= 80 and not DUMMY_PREFIX.match(first):
                researcher_name = first

        _update_research_info(cur, pid, title, authors, year, doi_val, researcher_name, info)


def _t5_make_prompt(text: str) -> str:
    txt = norm(text)[:12000]
    return (
        "Extract the paper title, author list, and publication year from the text.\n"
        "Return only strict JSON with keys title, authors, year.\n"
        "If a field is unknown, use an empty string.\n"
        "Text:\n"
        + txt
    )


def _t5_parse_json(gen: str) -> Dict[str, str]:
    s = (gen or "").strip()
    m = JSON_OBJ_RE.search(s)
    if m:
        s = m.group(0).strip()
    try:
        obj = json.loads(s)
        if not isinstance(obj, dict):
            return {}
        return {
            "title": norm(obj.get("title") or ""),
            "authors": norm(obj.get("authors") or ""),
            "year": norm(obj.get("year") or ""),
        }
    except Exception:
        return {}


def phase_d_t5_fill_remaining(
    db_path: str,
    model_name_or_path: str,
    device: str,
    limit: Optional[int],
    db_batch: int,
    max_new_tokens: int,
    temperature: float,
) -> None:
    if torch is None or AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
        print()
        print("T5 is unavailable in this environment. Install transformers, torch, sentencepiece.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT w.paper_id, w.full_text,
               COALESCE(p.title, ''), COALESCE(p.authors, ''), COALESCE(p.publication_date, '')
        FROM works w
        JOIN papers p ON p.paper_id = w.paper_id
        WHERE w.full_text IS NOT NULL AND TRIM(w.full_text) <> ''
          AND (
                (p.title IS NULL OR TRIM(p.title)='')
             OR (p.authors IS NULL OR TRIM(p.authors)='')
             OR (p.publication_date IS NULL OR TRIM(p.publication_date)='')
          )
        ORDER BY w.paper_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return

    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model.eval()

    use_device = device
    if use_device == "auto":
        use_device = "cuda" if (torch.cuda.is_available()) else "cpu"
    model.to(use_device)

    conn2 = sqlite3.connect(db_path)
    cur2 = conn2.cursor()
    ensure_pragmas(conn2)

    pending_updates: List[Tuple[int, str, str, str]] = []

    with torch.no_grad():
        for pid, full_text, cur_title, cur_authors, cur_year in tqdm(rows, desc="Phase D T5 fill", unit="paper"):
            pid_i = int(pid)
            need_title = is_missing(cur_title)
            need_authors = is_missing(cur_authors)
            need_year = is_missing(cur_year)
            if not (need_title or need_authors or need_year):
                continue

            txt = full_text or ""
            if not txt.strip():
                continue

            prompt = _t5_make_prompt(txt)
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
            enc = {k: v.to(use_device) for k, v in enc.items()}

            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True if temperature > 0 else False,
                temperature=temperature if temperature > 0 else 1.0,
                num_beams=1,
            )
            gen = tok.decode(gen_ids[0], skip_special_tokens=True)
            obj = _t5_parse_json(gen)

            title = obj.get("title", "")
            authors = obj.get("authors", "")
            year = obj.get("year", "")

            if title and looks_dummy_title(title):
                title = ""
            if authors and email_pat.search(authors):
                authors = ""
            if year and not re.fullmatch(r"(19|20)\d{2}", year):
                year = ""

            if not need_title:
                title = ""
            if not need_authors:
                authors = ""
            if not need_year:
                year = ""

            if not title and not authors and not year:
                continue

            pending_updates.append((pid_i, title, authors, year))
            if len(pending_updates) >= db_batch:
                _apply_t5_batch(cur2, pending_updates)
                safe_commit(conn2)
                pending_updates.clear()

    if pending_updates:
        _apply_t5_batch(cur2, pending_updates)
        safe_commit(conn2)

    conn2.close()


def _apply_t5_batch(cur: sqlite3.Cursor, batch: List[Tuple[int, str, str, str]]) -> None:
    for pid, title, authors, year in batch:
        _update_papers(cur, pid, title, authors, year)

        rn = ""
        if authors:
            first = authors.split(",")[0].strip()
            if 2 <= len(first) <= 80 and not DUMMY_PREFIX.match(first):
                rn = first

        info_parts: List[str] = []
        if year:
            info_parts.append("Date: " + year)
        if authors:
            info_parts.append("Authors: " + authors)
        if title:
            info_parts.append("Title: " + title)
        info = " | ".join(info_parts)

        _update_research_info(cur, pid, title, authors, year, "", rn, info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH_DEFAULT)

    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--db-batch", type=int, default=2000)

    ap.add_argument("--no-external", action="store_true")
    ap.add_argument("--no-crossref", action="store_true")
    ap.add_argument("--no-arxiv", action="store_true")
    ap.add_argument("--timeout", type=int, default=20)

    ap.add_argument("--t5", action="store_true")
    ap.add_argument("--t5-model", default="google/flan-t5-small")
    ap.add_argument("--t5-device", default="auto")
    ap.add_argument("--t5-max-new-tokens", type=int, default=160)
    ap.add_argument("--t5-temperature", type=float, default=0.0)
    ap.add_argument("--t5-db-batch", type=int, default=300)

    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_min_columns(conn)
    inserted = ensure_research_info_rows_for_works(conn)
    if inserted:
        LOG.info("Inserted %d missing research_info rows for works paper_ids", inserted)

    cur = conn.cursor()
    print_snapshot(counts_snapshot(cur), "Snapshot before")
    print()
    print("Phase A copy across tables")
    phase_a_copy_across(conn)
    print_snapshot(counts_snapshot(cur), "Snapshot after Phase A")
    conn.close()

    print()
    print("Phase B deterministic parse from works.full_text")
    phase_b_parse_fulltext(args.db, limit=args.limit, db_batch=args.db_batch)
    conn2 = sqlite3.connect(args.db)
    cur2 = conn2.cursor()
    print_snapshot(counts_snapshot(cur2), "Snapshot after Phase B")
    conn2.close()

    if not args.no_external:
        do_crossref = not args.no_crossref
        do_arxiv = not args.no_arxiv
        if do_crossref or do_arxiv:
            print()
            print("Phase C external enrichment (Crossref, arXiv)")
            phase_c_external_enrich(
                db_path=args.db,
                do_crossref=do_crossref,
                do_arxiv=do_arxiv,
                limit=args.limit,
                db_batch=args.db_batch,
                timeout=args.timeout,
            )
            conn3 = sqlite3.connect(args.db)
            cur3 = conn3.cursor()
            print_snapshot(counts_snapshot(cur3), "Snapshot after Phase C")
            conn3.close()

    if args.t5:
        print()
        print("Phase D T5 fill remaining (title, authors, year)")
        phase_d_t5_fill_remaining(
            db_path=args.db,
            model_name_or_path=args.t5_model,
            device=args.t5_device,
            limit=args.limit,
            db_batch=args.t5_db_batch,
            max_new_tokens=args.t5_max_new_tokens,
            temperature=args.t5_temperature,
        )
        conn4 = sqlite3.connect(args.db)
        cur4 = conn4.cursor()
        print_snapshot(counts_snapshot(cur4), "Snapshot after Phase D")
        conn4.close()

    print()
    print("Done")


if __name__ == "__main__":
    main()
