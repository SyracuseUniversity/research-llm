# topic_tag_openalex.py
import argparse
import json
import logging
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch
from tqdm import tqdm
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed


DB_PATH_DEFAULT = r"C:\codes\t5-db\syr_research_all.db"

MODEL_ID = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
OPENALEX_TOPIC_URL = "https://api.openalex.org/topics/{}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
LOG = logging.getLogger("topic_tag_openalex")

WS_RE = re.compile(r"\s+")
XML_TAG = re.compile(r"<[^>]+>")
LABEL_ID_RE = re.compile(r"^\s*(\d+)\s*:")
TOPIC_LABEL_SPLIT_RE = re.compile(r"^\s*\d+\s*:\s*(.+?)\s*$")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


def norm(s: Any) -> str:
    if s is None:
        return ""
    s = str(s).replace("\x00", " ")
    s = XML_TAG.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def ensure_columns(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute("PRAGMA table_info(research_info);")
    cols = {r[1] for r in cur.fetchall()}

    def add_col(name: str, decl: str) -> None:
        if name in cols:
            return
        cur.execute(f"ALTER TABLE research_info ADD COLUMN {name} {decl};")

    add_col("primary_topic", "TEXT")
    add_col("topics_json", "TEXT")
    add_col("topics_status", "TEXT DEFAULT 'untagged'")
    add_col("subject", "TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_research_info_paper_id ON research_info(paper_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_works_paper_id ON works(paper_id);")
    conn.commit()


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
        conn.commit()

    return inserted


def snapshot(cur: sqlite3.Cursor) -> Dict[str, int]:
    out: Dict[str, int] = {}
    cur.execute("SELECT COUNT(*) FROM research_info;")
    out["research_info_total"] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM research_info WHERE topics_status='tagged';")
    out["research_info_tagged"] = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT COUNT(*)
        FROM research_info
        WHERE topics_status IS NULL OR TRIM(topics_status)='' OR topics_status='untagged'
        """
    )
    out["research_info_untagged"] = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(DISTINCT paper_id) FROM works;")
    out["works_distinct_paper_id"] = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT COUNT(DISTINCT w.paper_id)
        FROM works w
        JOIN research_info ri ON ri.paper_id = w.paper_id
        """
    )
    out["works_paper_ids_with_research_info"] = int(cur.fetchone()[0])
    return out


def print_snapshot(s: Dict[str, int], label: str) -> None:
    print()
    print(label)
    for k in [
        "research_info_total",
        "research_info_tagged",
        "research_info_untagged",
        "works_distinct_paper_id",
        "works_paper_ids_with_research_info",
    ]:
        print(f"{k}:", s.get(k, 0))


def build_input_text(title: str, abstract: str) -> str:
    title = norm(title)
    abstract = norm(abstract)
    if not title and not abstract:
        return ""
    if abstract and not title:
        return f"<TITLE> NONE\n<ABSTRACT> {abstract}"
    if title and not abstract:
        return f"<TITLE> {title}"
    return f"<TITLE> {title}\n<ABSTRACT> {abstract}"


def pick_title(p_title: str, ri_title: str) -> str:
    return norm(ri_title) or norm(p_title)


def pick_abstract(summary: str, full_text: str, max_chars: int) -> str:
    s = norm(summary)
    if s:
        return s
    ft = norm(full_text)
    return ft[:max_chars] if ft else ""


def fetch_candidates(db_path: str, limit: Optional[int], force: bool, max_abstract_chars: int) -> List[Tuple[int, str, str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    where = "1=1" if force else "(ri.topics_status IS NULL OR TRIM(ri.topics_status)='' OR ri.topics_status='untagged')"

    cur.execute(
        f"""
        SELECT
            w.paper_id,
            COALESCE(p.title, '') AS p_title,
            COALESCE(ri.work_title, '') AS ri_title,
            COALESCE(w.summary, '') AS summary,
            COALESCE(w.full_text, '') AS full_text
        FROM works w
        LEFT JOIN papers p ON p.paper_id = w.paper_id
        LEFT JOIN research_info ri ON ri.paper_id = w.paper_id
        WHERE {where}
        GROUP BY w.paper_id
        ORDER BY w.paper_id
        """
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Tuple[int, str, str]] = []
    for paper_id, p_title, ri_title, summary, full_text in rows:
        title = pick_title(p_title, ri_title)
        abstract = pick_abstract(summary, full_text, max_chars=max_abstract_chars)
        if build_input_text(title, abstract):
            out.append((int(paper_id), title, abstract))

    if limit is not None:
        out = out[:limit]
    return out


def label_to_topic_num(label: str) -> Optional[int]:
    m = LABEL_ID_RE.match(label or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def label_to_name_only(label: str) -> str:
    label = norm(label)
    if not label:
        return ""
    m = TOPIC_LABEL_SPLIT_RE.match(label)
    if m:
        return norm(m.group(1))
    return label


def openalex_topic_lookup(topic_num: int, timeout: int = 20) -> Optional[Dict[str, Any]]:
    tid = f"T{topic_num}"
    url = OPENALEX_TOPIC_URL.format(tid)
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "topic-tag-openalex/1.0"})
        if r.status_code != 200:
            return None
        obj = r.json()

        def pick_obj(o: Any) -> Dict[str, str]:
            if not isinstance(o, dict):
                return {"id": "", "display_name": ""}
            return {"id": norm(o.get("id", "")), "display_name": norm(o.get("display_name", ""))}

        return {
            "topic_id": tid,
            "topic_display_name": norm(obj.get("display_name", "")),
            "domain": pick_obj(obj.get("domain")),
            "field": pick_obj(obj.get("field")),
            "subfield": pick_obj(obj.get("subfield")),
        }
    except Exception:
        return None


def build_openalex_cache(topic_nums: List[int], workers: int, timeout: int) -> Dict[int, Dict[str, Any]]:
    uniq = sorted(set([t for t in topic_nums if isinstance(t, int)]))
    if not uniq:
        return {}

    cache: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(openalex_topic_lookup, t, timeout): t for t in uniq}
        for f in tqdm(as_completed(futs), total=len(futs), desc="Fetching topic hierarchy", unit="topic"):
            t = futs[f]
            try:
                data = f.result()
            except Exception:
                data = None
            if data:
                cache[t] = data
    return cache


def run_inference(
    clf,
    candidates: List[Tuple[int, str, str]],
    top_k: int,
    batch_size: int,
    openalex_workers: int,
    timeout: int,
) -> List[Tuple[int, str, str, str, str]]:
    texts: List[str] = []
    paper_ids: List[int] = []
    for paper_id, title, abstract in candidates:
        texts.append(build_input_text(title, abstract))
        paper_ids.append(paper_id)

    raw_preds: List[Tuple[int, List[Dict[str, Any]]]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Classifying", unit="paper"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = paper_ids[i : i + batch_size]

        preds = clf(batch_texts, top_k=top_k, truncation=True, max_length=512)
        if isinstance(preds, dict):
            preds = [preds]

        for pid, pred in zip(batch_ids, preds):
            if not pred:
                continue
            pred_list = [pred] if isinstance(pred, dict) else list(pred)
            pred_list = sorted(pred_list, key=lambda x: float(x.get("score", 0.0)), reverse=True)
            raw_preds.append((pid, pred_list))

    all_topic_nums: List[int] = []
    for _, pred_list in raw_preds:
        for item in pred_list:
            tnum = label_to_topic_num(norm(item.get("label", "")))
            if tnum is not None:
                all_topic_nums.append(tnum)

    cache = build_openalex_cache(all_topic_nums, workers=openalex_workers, timeout=timeout)

    updates: List[Tuple[int, str, str, str, str]] = []
    for pid, pred_list in raw_preds:
        if not pred_list:
            continue

        primary_label = norm(pred_list[0].get("label", ""))
        primary_name_only = label_to_name_only(primary_label)

        field_name = ""
        primary_num = label_to_topic_num(primary_label)
        if primary_num is not None and primary_num in cache:
            meta0 = cache[primary_num]
            field_name = norm((meta0.get("field") or {}).get("display_name", ""))

        subject = "Unknown"
        if field_name:
            parts = ALNUM_RE.findall(field_name)
            if parts:
                subject = parts[0][:1].upper() + parts[0][1:]

        packed: List[Dict[str, Any]] = []
        for item in pred_list:
            label = norm(item.get("label", ""))
            score = float(item.get("score", 0.0))
            tnum = label_to_topic_num(label)

            entry: Dict[str, Any] = {"label": label, "score": score}
            if tnum is not None and tnum in cache:
                meta = cache[tnum]
                entry["topic_num"] = tnum
                entry["topic_id"] = meta.get("topic_id", "")
                entry["topic_display_name"] = meta.get("topic_display_name", "")
                entry["domain"] = meta.get("domain", {})
                entry["field"] = meta.get("field", {})
                entry["subfield"] = meta.get("subfield", {})
            packed.append(entry)

        topics_json = json.dumps(
            {
                "subject_one_word": subject,
                "primary_topic_label": primary_label,
                "ranked_topics": packed,
            },
            ensure_ascii=False,
        )

        updates.append((pid, primary_name_only, topics_json, subject, primary_label))
        print(f"paper_id={pid} subject={subject} primary_topic={primary_name_only}")

    return updates


def apply_updates(db_path: str, updates: List[Tuple[int, str, str, str, str]], db_batch: int) -> None:
    if not updates:
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    for i in tqdm(range(0, len(updates), db_batch), desc="Writing to DB", unit="paper"):
        chunk = updates[i : i + db_batch]
        for paper_id, primary_name_only, topics_json, subject, _primary_label in chunk:
            cur.execute(
                """
                UPDATE research_info
                SET primary_topic = ?,
                    topics_json = ?,
                    topics_status = 'tagged',
                    subject = COALESCE(NULLIF(TRIM(subject), ''), ?)
                WHERE paper_id = ?
                """,
                (
                    primary_name_only or None,
                    topics_json or None,
                    subject or None,
                    paper_id,
                ),
            )
        conn.commit()

    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH_DEFAULT)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--db-batch", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-abstract-chars", type=int, default=3000)
    ap.add_argument("--openalex-workers", type=int, default=24)
    ap.add_argument("--ensure-ri-batch", type=int, default=5000)
    ap.add_argument("--openalex-timeout", type=int, default=20)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    ensure_columns(conn)

    inserted = ensure_research_info_rows_for_works(conn, batch=args.ensure_ri_batch)
    if inserted:
        LOG.info("Inserted %d missing research_info rows for works paper_ids", inserted)

    cur = conn.cursor()
    print_snapshot(snapshot(cur), "Snapshot before")
    conn.close()

    candidates = fetch_candidates(
        db_path=args.db,
        limit=args.limit,
        force=args.force,
        max_abstract_chars=args.max_abstract_chars,
    )
    if not candidates:
        print("No works paper_ids to tag")
        return

    device = 0 if torch.cuda.is_available() else -1
    LOG.info("Device set to %s", "cuda:0" if device == 0 else "cpu")

    clf = pipeline(task="text-classification", model=args.model, device=device)

    t0 = time.time()
    updates = run_inference(
        clf=clf,
        candidates=candidates,
        top_k=args.top_k,
        batch_size=args.batch_size,
        openalex_workers=args.openalex_workers,
        timeout=args.openalex_timeout,
    )

    apply_updates(db_path=args.db, updates=updates, db_batch=args.db_batch)

    dt = time.time() - t0
    LOG.info("Done. Tagged %d paper_ids in %.1f seconds.", len(updates), dt)

    conn2 = sqlite3.connect(args.db)
    cur2 = conn2.cursor()
    print_snapshot(snapshot(cur2), "Snapshot after")
    conn2.close()


if __name__ == "__main__":
    main()
