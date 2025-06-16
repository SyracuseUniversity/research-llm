"""
clean_db.py  –  Normalize `research_info` in place, using each work’s full_text.
It looks for dummy titles/names and overwrites them where needed.
"""

import os
import sqlite3
import re
import logging

DB_PATH = r"C:\codes\t5-db\researchers_all.db"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

doi_pat = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
year_pat = re.compile(r"\b(19|20)\d{2}\b")
DUMMY_PREFIX = re.compile(r"^(arxiv|cms\s+paper|european\s+organization|cern|preprint)", re.I)


def clean(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def looks_dummy_title(t: str) -> bool:
    t = t.strip()
    return (not t) or len(t) > 150 or bool(DUMMY_PREFIX.match(t))


def looks_dummy_name(n: str) -> bool:
    n = n.strip()
    return (not n) or len(n) > 80 or bool(DUMMY_PREFIX.match(n))


def parse_from_fulltext(txt: str):
    """
    Return (researcher_name, title, authors, doi, pub_date, info)
    extracted from the text.
    """
    if not txt:
        return ("", "", "", "", "", "")

    lines = [l.strip() for l in txt.splitlines() if l.strip()]

    filtered = []
    for l in lines:
        low = l.lower()
        if low.startswith("arxiv") or "physics." in low or "hep" in low:
            continue
        if re.match(r"https?://", l):
            continue
        filtered.append(l)

    if not filtered:
        filtered = lines

    # Title: first filtered line with >5 words and <300 chars
    title = ""
    for l in filtered:
        if len(l.split()) > 5 and len(l) < 300:
            title = l
            break
    if not title and filtered:
        title = filtered[0]

    # Authors: the next line after title that contains a comma or ' and '
    authors = ""
    if title in filtered:
        idx = filtered.index(title)
        for l in filtered[idx+1: idx+10]:
            if "," in l or " and " in l:
                authors = l
                break

    doi_m = doi_pat.search(txt[:6000])
    doi = doi_m.group(0) if doi_m else ""

    year_m = year_pat.search(txt[:600])
    pub_date = year_m.group(0) if year_m else ""

    researcher = authors.split(",")[0].strip() if authors else ""
    info_parts = []
    if doi:
        info_parts.append(f"DOI: {doi}")
    if pub_date:
        info_parts.append(f"Date: {pub_date}")
    info = " | ".join(info_parts)

    return tuple(map(clean, (researcher, title, authors, doi, pub_date, info)))


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.text_factory = str
    cur = conn.cursor()

    # Cache id → full_text from works
    cur.execute("SELECT id, full_text FROM works;")
    fulltext = {wid: txt for wid, txt in cur.fetchall()}

    # Get column names of research_info
    cur.execute("PRAGMA table_info(research_info);")
    cols = [row[1] for row in cur.fetchall()]
    if "id" not in cols:
        raise RuntimeError("research_info must have an 'id' PK")

    editable = [c for c in cols if c != "id"]
    cur.execute(f"SELECT {', '.join(cols)} FROM research_info;")
    rows = cur.fetchall()

    updates = []
    filled = 0
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        r = dict(zip(cols, row))
        wid = r["id"]

        # Clean whitespace/non‐ascii
        for c in editable:
            r[c] = clean(r[c])

        need_title = looks_dummy_title(r.get("work_title", ""))
        need_name = looks_dummy_name(r.get("researcher_name", ""))
        need_info = len(r.get("info", "")) > 500 or not r.get("info")

        raw_full = fulltext.get(wid, "")
        if raw_full and (need_title or need_name or need_info):
            try:
                rn, wt, au, doi, pd, info = parse_from_fulltext(raw_full)
            except Exception:
                rn, wt, au, doi, pd, info = ("", "", "", "", "", "")

            if need_name and rn:
                r["researcher_name"] = rn
            if need_title and wt:
                r["work_title"] = wt
            if "authors" in cols and not r.get("authors") and au:
                r["authors"] = au
            if "doi" in cols and not r.get("doi") and doi:
                r["doi"] = doi
            if "publication_date" in cols and not r.get("publication_date") and pd:
                r["publication_date"] = pd
            if "info" in cols and (need_info and info):
                r["info"] = info

            filled += 1

        updates.append(r)

        if i % 1000 == 0 or i == total:
            logging.info("Processed %d/%d rows", i, total)

    # Bulk update
    set_clause = ", ".join(f"{c}=?" for c in editable)
    params = [tuple(r[c] for c in editable) + (r["id"],) for r in updates]
    cur.executemany(f"UPDATE research_info SET {set_clause} WHERE id=?", params)

    conn.commit()
    conn.close()

    logging.info("✅ Normalized %d/%d rows; replaced placeholders in %d rows.", total, total, filled)


if __name__ == "__main__":
    main()
