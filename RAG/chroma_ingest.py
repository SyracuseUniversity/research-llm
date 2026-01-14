# chroma_ingest.py
import os
import sqlite3
import time
import re
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

import config_full as config


DB_PATH = config.SQLITE_DB_FULL
CHROMA_DIR = config.CHROMA_DIR_FULL
COLLECTION_NAME = getattr(config, "CHROMA_COLLECTION_FULL", "papers_all")
EMBED_MODEL = getattr(config, "EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

PAPERS_PER_BATCH = int(getattr(config, "PAPERS_PER_BATCH", 200))
CHROMA_MAX_BATCH = int(getattr(config, "CHROMA_MAX_BATCH", 5400))
CHUNK_MAX_CHARS = int(getattr(config, "CHUNK_MAX_CHARS", 12000))

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


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


def _split_text(text: str, max_chars: int = 12000) -> List[str]:
    if not text:
        return []
    t = text.strip()
    if len(t) <= max_chars:
        return [t]
    parts: List[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + max_chars, n)
        parts.append(t[i:j])
        i = j
    return parts


def _join_nonempty(*parts: Any, sep: str = "\n") -> str:
    out = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip()
        if s:
            out.append(s)
    return sep.join(out)


def detect_works_fulltext_column(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(works);")
    cols = [r[1] for r in cur.fetchall()]
    if "full_text" in cols:
        return "full_text"
    if "fultext" in cols:
        return "fultext"
    if "fulltext" in cols:
        return "fulltext"
    raise RuntimeError("Could not find works fulltext column. Expected one of: full_text, fultext, fulltext")


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


def build_paper_document(row: Tuple[Any, ...], works_fulltext_colname: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
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
    summ = (summary or "").strip()
    ft = (fulltext or "").strip()

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
        "paper_id": safe_meta(paper_id_s),
        "research_info_id": safe_meta(r_id),
        "researcher": safe_meta(researcher_name, "Unknown"),
        "title": safe_meta(title, "Untitled"),
        "authors": safe_meta(authors),
        "doi": safe_meta(doi),
        "year": _pick_year(publication_date),
        "publication_date": safe_meta(publication_date),
        "primary_topic": safe_meta(primary_topic),
    }

    return paper_id_s, doc, meta


def main() -> None:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedder)

    conn = sqlite3.connect(DB_PATH)
    works_fulltext_col = detect_works_fulltext_column(conn)
    rows = fetch_rows(conn, works_fulltext_col)
    conn.close()

    by_paper: Dict[str, List[Tuple[Any, ...]]] = {}
    for row in rows:
        pid = row[0]
        if pid is None:
            continue
        pid_s = str(pid).strip()
        if not pid_s:
            continue
        by_paper.setdefault(pid_s, []).append(row)

    paper_ids = list(by_paper.keys())
    total_start = time.time()

    for batch_idx in tqdm(range(0, len(paper_ids), PAPERS_PER_BATCH), desc="Ingesting papers", unit="batch"):
        ids_batch = paper_ids[batch_idx : batch_idx + PAPERS_PER_BATCH]

        docs: List[str] = []
        ids: List[str] = []
        metas: List[Dict[str, Any]] = []

        for pid in ids_batch:
            candidates = by_paper.get(pid) or []
            best = None
            best_score = -1

            for row in candidates:
                built = build_paper_document(row, works_fulltext_col)
                if not built:
                    continue
                _, doc_text, meta = built
                score = len(doc_text)
                if score > best_score:
                    best_score = score
                    best = (doc_text, meta)

            if not best:
                continue

            doc_text, meta = best
            chunks = _split_text(doc_text, max_chars=CHUNK_MAX_CHARS)
            if not chunks:
                continue

            for part_i, chunk in enumerate(chunks, start=1):
                docs.append(chunk)
                ids.append(f"{pid}_part{part_i}")
                m = dict(meta)
                m["chunk"] = part_i
                m["chunks_total"] = len(chunks)
                metas.append(m)

        if not docs:
            continue

        batch_start = time.time()
        for i in range(0, len(docs), CHROMA_MAX_BATCH):
            col.upsert(
                documents=docs[i : i + CHROMA_MAX_BATCH],
                ids=ids[i : i + CHROMA_MAX_BATCH],
                metadatas=metas[i : i + CHROMA_MAX_BATCH],
            )

        speed = len(docs) / (time.time() - batch_start + 1e-6)
        bnum = (batch_idx // PAPERS_PER_BATCH) + 1
        print(f"Batch {bnum}: {len(docs)} chunks at {speed:.2f} chunks/sec")

    print(f"Ingestion complete into {COLLECTION_NAME}")
    print(f"Total time: {time.time() - total_start:.2f} seconds")


if __name__ == "__main__":
    main()
