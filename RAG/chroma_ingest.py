# chroma_ingest.py
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

import config_full as config


DB_PATH         = config.SQLITE_DB_FULL
CHROMA_DIR      = config.CHROMA_DIR_FULL
COLLECTION_NAME = getattr(config, "CHROMA_COLLECTION_FULL", "papers_all")
EMBED_MODEL     = getattr(config, "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PAPERS_PER_BATCH = int(getattr(config, "PAPERS_PER_BATCH", 200))
CHROMA_MAX_BATCH = 166          # ChromaDB hard limit — do not read from config

# Chunking knobs (match the pipeline's config defaults if you set them there)
CHUNK_MAX_TOKENS     = int(getattr(config, "CHUNK_MAX_TOKENS",     512))
CHUNK_OVERLAP_TOKENS = int(getattr(config, "CHUNK_OVERLAP_TOKENS", 128))

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


# ── Overlapping chunker (mirrors chunker.py from the pipeline) ────────────────

def _token_est(text: str) -> int:
    """Rough word-based token estimate (×1.3 for subword overhead)."""
    return int(len(text.split()) * 1.3)


def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_with_overlap(text: str) -> List[str]:
    """
    Split *text* into token-bounded chunks with sentence-aware overlap.

    Each chunk is at most CHUNK_MAX_TOKENS (estimated) tokens long.
    When a new chunk starts, the tail sentences of the previous chunk
    are carried over up to CHUNK_OVERLAP_TOKENS tokens so that context
    is not lost at boundaries.
    """
    sentences = _sentence_split(text)
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: List[str] = []
    current: List[str] = []
    current_tok = 0

    for sent in sentences:
        sent_tok = _token_est(sent)
        if current_tok + sent_tok > CHUNK_MAX_TOKENS and current:
            chunks.append(" ".join(current))
            # carry-over: keep trailing sentences up to CHUNK_OVERLAP_TOKENS
            keep, keep_tok = [], 0
            for s in reversed(current):
                t = _token_est(s)
                if keep_tok + t > CHUNK_OVERLAP_TOKENS:
                    break
                keep.insert(0, s)
                keep_tok += t
            current, current_tok = keep, keep_tok
        current.append(sent)
        current_tok += sent_tok

    if current:
        chunks.append(" ".join(current))
    return chunks


# ── Metadata / text helpers ───────────────────────────────────────────────────

def safe_meta(val: Any, default: Any = "N/A") -> Any:
    if val is None:
        return default
    if isinstance(val, (int, float, bool)):
        return val
    s = str(val).strip()
    return s if s else default


def _pick_year(pub_date: Any) -> str:
    s = "" if pub_date is None else str(pub_date)
    m = _YEAR_RE.search(s)
    return m.group(0) if m else safe_meta(pub_date)


def _join_nonempty(*parts: Any, sep: str = "\n") -> str:
    out = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            out.append(s)
    return sep.join(out)


# ── SQLite helpers ────────────────────────────────────────────────────────────

def detect_works_fulltext_column(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(works);")
    cols = [r[1] for r in cur.fetchall()]
    for candidate in ("full_text", "fultext", "fulltext"):
        if candidate in cols:
            return candidate
    raise RuntimeError(
        "Could not find works fulltext column. "
        "Expected one of: full_text, fultext, fulltext"
    )


def fetch_rows(conn: sqlite3.Connection, works_fulltext_col: str) -> List[Tuple[Any, ...]]:
    cur = conn.cursor()
    sql = f"""
        SELECT
            r.paper_id,
            r.id,
            r.researcher_name,
            r.work_title,
            r.authors,
            r.info,
            r.doi,
            r.publication_date,
            r.primary_topic,
            w.summary,
            w.{works_fulltext_col}
        FROM research_info r
        LEFT JOIN works w
          ON r.paper_id = w.paper_id
    """
    cur.execute(sql)
    return cur.fetchall()


# ── Document builder ──────────────────────────────────────────────────────────

def build_paper_document(
    row: Tuple[Any, ...],
    works_fulltext_colname: str,
) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    (
        paper_id,
        r_id,
        researcher_name,
        work_title,
        authors,
        info,
        doi,
        publication_date,
        primary_topic,
        summary,
        fulltext,
    ) = row

    title = (work_title or "").strip()
    summ  = (summary   or "").strip()
    ft    = (fulltext   or "").strip()

    if not (title or summ or ft):
        return None

    paper_id_s = str(paper_id).strip() if paper_id is not None else ""
    if not paper_id_s:
        return None

    doc = _join_nonempty(
        f"Paper ID: {safe_meta(paper_id_s, 'Unknown')}",
        f"Researcher: {safe_meta(researcher_name, 'Unknown')}",
        f"Title: {safe_meta(title, 'Untitled')}",
        f"Authors: {safe_meta(authors)}",
        f"Primary Topic: {safe_meta(primary_topic)}",
        f"Info: {safe_meta(info)}",
        f"DOI: {safe_meta(doi)}",
        f"Publication Date: {safe_meta(publication_date)}",
        "",
        f"Summary:\n{safe_meta(summ)}",
        "",
        f"Fulltext ({works_fulltext_colname}):\n{safe_meta(ft)}",
        sep="\n",
    ).strip()

    meta = {
        "paper_id":         safe_meta(paper_id_s),
        "research_info_id": safe_meta(r_id),
        "researcher":       safe_meta(researcher_name, "Unknown"),
        "title":            safe_meta(title, "Untitled"),
        "authors":          safe_meta(authors),
        "doi":              safe_meta(doi),
        "year":             _pick_year(publication_date),
        "publication_date": safe_meta(publication_date),
        "primary_topic":    safe_meta(primary_topic),
    }

    return paper_id_s, doc, meta


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client   = chromadb.PersistentClient(path=CHROMA_DIR)
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedder
    )

    conn               = sqlite3.connect(DB_PATH)
    works_fulltext_col = detect_works_fulltext_column(conn)
    rows               = fetch_rows(conn, works_fulltext_col)
    conn.close()

    # Group rows by paper_id
    by_paper: Dict[str, List[Tuple[Any, ...]]] = {}
    for row in rows:
        pid = row[0]
        if pid is None:
            continue
        pid_s = str(pid).strip()
        if pid_s:
            by_paper.setdefault(pid_s, []).append(row)

    paper_ids   = list(by_paper.keys())
    total_start = time.time()

    for batch_idx in tqdm(
        range(0, len(paper_ids), PAPERS_PER_BATCH),
        desc="Ingesting papers",
        unit="batch",
    ):
        ids_batch = paper_ids[batch_idx : batch_idx + PAPERS_PER_BATCH]

        docs:  List[str]            = []
        ids:   List[str]            = []
        metas: List[Dict[str, Any]] = []

        for pid in ids_batch:
            candidates = by_paper.get(pid) or []

            # Pick the candidate row with the most text
            best: Optional[Tuple[str, Dict]] = None
            best_score = -1
            for row in candidates:
                built = build_paper_document(row, works_fulltext_col)
                if not built:
                    continue
                _, doc_text, meta = built
                if len(doc_text) > best_score:
                    best_score = len(doc_text)
                    best = (doc_text, meta)

            if not best:
                continue

            doc_text, meta = best

            # ── Sentence-aware overlapping chunks ────────────────────────────
            chunks = chunk_with_overlap(doc_text)
            if not chunks:
                continue

            for part_i, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                ids.append(f"{pid}_part{part_i}")
                m = dict(meta)
                m["chunk"]        = part_i
                m["chunks_total"] = len(chunks)
                metas.append(m)

        if not docs:
            continue

        # Upsert in ChromaDB-safe sub-batches of 166
        batch_start = time.time()
        for i in range(0, len(docs), CHROMA_MAX_BATCH):
            col.upsert(
                documents=docs[i : i + CHROMA_MAX_BATCH],
                ids=ids[i : i + CHROMA_MAX_BATCH],
                metadatas=metas[i : i + CHROMA_MAX_BATCH],
            )

        speed = len(docs) / (time.time() - batch_start + 1e-6)
        bnum  = (batch_idx // PAPERS_PER_BATCH) + 1
        print(f"Batch {bnum}: {len(docs)} chunks at {speed:.2f} chunks/sec")

    print(f"Ingestion complete into {COLLECTION_NAME}")
    print(f"Total time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()