"""
fetch_and_download.py — Module 1: Fetch from OpenAlex and download fulltexts.

Standalone — run directly:
    python fetch_and_download.py
    python fetch_and_download.py --incremental
    python fetch_and_download.py --skip-download

What it does:
    1. Fetches all Syracuse University works + authors from OpenAlex API
       → data/raw/works.jsonl
       → data/raw/authors.jsonl

    2. For each work, downloads fulltext in priority order:
         a. TEI XML via OpenAlex /works/{id}/fulltext endpoint
         b. Open-access PDF from oa_url / primary_location.pdf_url
         c. PDF via Unpaywall (DOI lookup)
       → data/fulltext/{work_id}.tei.xml  OR  data/fulltext/{work_id}.pdf

    3. Annotates each work record with:
         fulltext_status: 'tei_xml' | 'pdf' | 'none'
         fulltext_path:   local file path or ''
       → data/raw/works_with_fulltext.jsonl

Works with no downloadable fulltext are flagged 'none' and will be
ingested using title + abstract only in later steps.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

# ── Allow running as standalone script from any working directory ─────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

API_BASE     = "https://api.openalex.org"
FULLTEXT_DIR = Path(config.FULLTEXT_DIR)

# fulltext_status constants
FT_TEI  = "tei_xml"
FT_PDF  = "pdf"
FT_NONE = "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Sync state (watermark for incremental fetches)
# ═══════════════════════════════════════════════════════════════════════════════

class SyncState:
    def __init__(self):
        self.path = Path(config.SYNC_STATE_FILE)

    def load(self) -> Dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def save(self, state: Dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def get_watermark(self) -> Optional[str]:
        return self.load().get("last_updated_date")

    def update(self, counts: Dict):
        state = self.load()
        state["last_updated_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        state["last_sync_at"]      = datetime.now(timezone.utc).isoformat()
        for k, v in counts.items():
            state[k] = state.get(k, 0) + v
        self.save(state)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAlex helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _api_params(**extra) -> Dict:
    params: Dict[str, Any] = {}
    if config.OPENALEX_API_KEY:
        params["api_key"] = config.OPENALEX_API_KEY
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    params.update(extra)
    return params


def _reconstruct_abstract(inverted_index: Optional[Dict]) -> str:
    if not inverted_index:
        return ""
    positions: List[Tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for pos in idxs:
            positions.append((pos, word))
    positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in positions)


def _cursor_paginate(endpoint: str, params: Dict, desc: str = "records") -> Iterator[Dict]:
    params = dict(params)
    params["cursor"] = "*"
    total    = 0
    page_num = 0

    while True:
        page_num += 1
        try:
            url  = f"{API_BASE}/{endpoint}"
            resp = requests.get(url, params=params, timeout=30)
            if page_num == 1:
                logger.info("API → %s", resp.url)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error("API error on %s: %s", endpoint, e)
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", 5))
                logger.warning("Rate limited — waiting %ds", wait)
                time.sleep(wait)
                continue
            raise

        data    = resp.json()
        meta    = data.get("meta", {})
        results = data.get("results", [])

        if page_num == 1:
            logger.info("Total available: %s", meta.get("count", "?"))

        for item in results:
            total += 1
            yield item

        next_cursor = meta.get("next_cursor")
        if not next_cursor or not results:
            break

        params["cursor"] = next_cursor
        time.sleep(config.OPENALEX_RATE_DELAY)

        if total % 1000 == 0:
            logger.info("  ... %d %s fetched", total, desc)

    logger.info("Fetched %d %s total", total, desc)


def _verify_institution(inst_id: str) -> bool:
    try:
        resp = requests.get(f"{API_BASE}/institutions/{inst_id}", params=_api_params(), timeout=15)
        if resp.status_code == 200:
            d = resp.json()
            logger.info("Institution: %s (%s) — %d works", inst_id, d.get("display_name"), d.get("works_count", 0))
            return True
        logger.error(
            "Institution %s not found (HTTP %d). "
            "Find ID at: https://api.openalex.org/autocomplete/institutions?q=syracuse+university",
            inst_id, resp.status_code,
        )
        return False
    except Exception:
        logger.error("Could not verify institution %s", inst_id, exc_info=True)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Fetch
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_works(from_updated_date: Optional[str] = None) -> Iterator[Dict]:
    inst_id = config.OPENALEX_INSTITUTION_ID
    if not _verify_institution(inst_id):
        raise RuntimeError(f"Invalid institution ID: {inst_id}")

    filters = [f"authorships.institutions.lineage:{inst_id}"]
    if from_updated_date:
        filters.append(f"from_updated_date:{from_updated_date}")
        logger.info("Incremental fetch — works updated since %s", from_updated_date)
    else:
        logger.info("Full fetch — all works for %s", inst_id)

    params = _api_params(filter=",".join(filters), per_page=config.OPENALEX_PER_PAGE)

    for work in _cursor_paginate("works", params, desc="works"):
        work["abstract_text"] = _reconstruct_abstract(work.pop("abstract_inverted_index", None))
        yield work


def fetch_authors() -> Iterator[Dict]:
    inst_id = config.OPENALEX_INSTITUTION_ID
    logger.info("Fetching authors for %s", inst_id)
    params = _api_params(
        filter=f"affiliations.institution.lineage:{inst_id}",
        per_page=config.OPENALEX_PER_PAGE,
    )
    yield from _cursor_paginate("authors", params, desc="authors")


# ═══════════════════════════════════════════════════════════════════════════════
# Download helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, params: Optional[Dict] = None, stream: bool = False) -> Optional[requests.Response]:
    for attempt in range(config.DOWNLOAD_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params or {}, timeout=config.DOWNLOAD_TIMEOUT, stream=stream)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = int(resp.headers.get("retry-after", 10))
                time.sleep(wait)
                continue
            if resp.status_code in (403, 404, 410):
                return None
        except requests.exceptions.RequestException:
            if attempt < config.DOWNLOAD_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def _save_text(path: Path, content: str) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        logger.warning("Save failed %s: %s", path, e)
        return False


def _save_bytes(path: Path, content: bytes) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return True
    except Exception as e:
        logger.warning("Save failed %s: %s", path, e)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Download strategies
# ═══════════════════════════════════════════════════════════════════════════════

def _try_openalex_tei(work_id: str, tei_path: Path) -> bool:
    url    = f"{API_BASE}/works/{work_id}/fulltext"
    params = _api_params()
    resp   = _http_get(url, params=params, stream=False)
    if resp is None:
        return False
    try:
        data        = resp.json()
        tei_content = data.get("tei_xml") or data.get("fulltext") or ""
        if tei_content and len(tei_content) > 200:
            return _save_text(tei_path, tei_content)
    except Exception:
        pass
    return False


def _try_pdf(url: str, pdf_path: Path) -> bool:
    if not url:
        return False
    resp = _http_get(url, stream=True)
    if resp is None:
        return False
    content = resp.content
    if content and content[:4] == b"%PDF":
        return _save_bytes(pdf_path, content)
    return False


def _try_unpaywall(doi: str, pdf_path: Path) -> bool:
    if not doi or not config.UNPAYWALL_EMAIL:
        return False
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi_clean:
        return False
    resp = _http_get(f"https://api.unpaywall.org/v2/{doi_clean}", params={"email": config.UNPAYWALL_EMAIL})
    if resp is None:
        return False
    try:
        data    = resp.json()
        best    = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or ""
        if not pdf_url:
            for loc in (data.get("oa_locations") or []):
                pdf_url = loc.get("url_for_pdf") or ""
                if pdf_url:
                    break
        if pdf_url:
            return _try_pdf(pdf_url, pdf_path)
    except Exception:
        pass
    return False


def _download_work(work: Dict) -> Tuple[str, str]:
    """
    Download fulltext for a single work.
    Returns (fulltext_status, local_path).
    """
    raw_id  = work.get("id") or ""
    work_id = raw_id.rsplit("/", 1)[-1]
    if not work_id:
        return FT_NONE, ""

    FULLTEXT_DIR.mkdir(parents=True, exist_ok=True)
    tei_path = FULLTEXT_DIR / f"{work_id}.tei.xml"
    pdf_path = FULLTEXT_DIR / f"{work_id}.pdf"

    # Already downloaded?
    if tei_path.exists() and tei_path.stat().st_size > 200:
        return FT_TEI, str(tei_path)
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return FT_PDF, str(pdf_path)

    # Strategy 1: OpenAlex TEI XML
    if _try_openalex_tei(work_id, tei_path):
        return FT_TEI, str(tei_path)

    # Strategy 2: OA PDF from oa_url
    oa     = work.get("open_access") or {}
    oa_url = oa.get("oa_url") or ""
    if oa_url and _try_pdf(oa_url, pdf_path):
        return FT_PDF, str(pdf_path)

    # Strategy 3: PDF from primary_location
    primary = work.get("primary_location") or {}
    pdf_url = primary.get("pdf_url") or ""
    if not pdf_url:
        landing = primary.get("landing_page_url") or ""
        if landing.endswith(".pdf"):
            pdf_url = landing
    if pdf_url and pdf_url != oa_url and _try_pdf(pdf_url, pdf_path):
        return FT_PDF, str(pdf_path)

    # Strategy 4: Unpaywall
    doi = work.get("doi") or ""
    if doi and _try_unpaywall(doi, pdf_path):
        return FT_PDF, str(pdf_path)

    return FT_NONE, ""


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run(incremental: bool = False, skip_download: bool = False) -> Dict[str, Any]:
    """
    Full module run:
      1. Fetch works + authors from OpenAlex
      2. Download fulltexts in parallel
      3. Write annotated JSONL

    Returns summary counts dict.
    """
    os.makedirs(config.RAW_DIR, exist_ok=True)
    sync    = SyncState()
    from_dt = sync.get_watermark() if incremental else None
    mode    = "a" if incremental else "w"

    # ── Step 1: Fetch ─────────────────────────────────────────────────────────
    logger.info("━" * 50)
    logger.info("FETCH: Pulling works from OpenAlex")
    logger.info("━" * 50)

    works_raw_path   = Path(config.RAW_DIR) / "works.jsonl"
    authors_raw_path = Path(config.RAW_DIR) / "authors.jsonl"

    works_list: List[Dict] = []
    with open(works_raw_path, mode, encoding="utf-8") as f:
        for work in fetch_works(from_updated_date=from_dt):
            f.write(json.dumps(work, ensure_ascii=False) + "\n")
            works_list.append(work)
            if len(works_list) % 500 == 0:
                logger.info("  ... %d works written", len(works_list))

    logger.info("Works fetched: %d → %s", len(works_list), works_raw_path)

    authors_count = 0
    with open(authors_raw_path, mode, encoding="utf-8") as f:
        for author in fetch_authors():
            f.write(json.dumps(author, ensure_ascii=False) + "\n")
            authors_count += 1

    logger.info("Authors fetched: %d → %s", authors_count, authors_raw_path)

    if skip_download:
        # Annotate all works with status=none and write output
        out_path = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for work in works_list:
                work["fulltext_status"] = FT_NONE
                work["fulltext_path"]   = ""
                f.write(json.dumps(work, ensure_ascii=False) + "\n")
        logger.info("Download skipped — all works marked as title+abstract only")
        sync.update({"works_fetched": len(works_list), "authors_fetched": authors_count})
        return {"works": len(works_list), "authors": authors_count, FT_TEI: 0, FT_PDF: 0, FT_NONE: len(works_list)}

    # ── Step 2: Download ──────────────────────────────────────────────────────
    logger.info("━" * 50)
    logger.info("DOWNLOAD: Fetching fulltexts (%d workers)", config.DOWNLOAD_WORKERS)
    logger.info("━" * 50)

    # In incremental mode, skip works already in output
    already_done: Dict[str, Dict] = {}
    out_path = Path(config.RAW_DIR) / "works_with_fulltext.jsonl"
    if incremental and out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    w   = json.loads(line)
                    wid = (w.get("id") or "").rsplit("/", 1)[-1]
                    if wid and w.get("fulltext_status"):
                        already_done[wid] = w
                except json.JSONDecodeError:
                    continue

    to_download = [
        w for w in works_list
        if (w.get("id") or "").rsplit("/", 1)[-1] not in already_done
    ]

    logger.info("To download: %d (already done: %d)", len(to_download), len(already_done))

    counts  = {FT_TEI: 0, FT_PDF: 0, FT_NONE: 0}
    results: Dict[str, Tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_WORKERS) as executor:
        future_map = {executor.submit(_download_work, w): w for w in to_download}
        done = 0
        for future in as_completed(future_map):
            work = future_map[future]
            wid  = (work.get("id") or "").rsplit("/", 1)[-1]
            try:
                status, path = future.result()
            except Exception as e:
                logger.warning("Download error %s: %s", wid, e)
                status, path = FT_NONE, ""

            results[wid]        = (status, path)
            counts[status]      = counts.get(status, 0) + 1
            done               += 1

            if done % 100 == 0 or done == len(to_download):
                logger.info(
                    "Progress: %d/%d | TEI: %d | PDF: %d | None: %d",
                    done, len(to_download), counts[FT_TEI], counts[FT_PDF], counts[FT_NONE],
                )

    # ── Step 3: Write output JSONL ─────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for w in already_done.values():
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

        for work in to_download:
            wid               = (work.get("id") or "").rsplit("/", 1)[-1]
            status, path      = results.get(wid, (FT_NONE, ""))
            work["fulltext_status"] = status
            work["fulltext_path"]   = path
            f.write(json.dumps(work, ensure_ascii=False) + "\n")

    summary = {
        "works":   len(works_list),
        "authors": authors_count,
        FT_TEI:    counts[FT_TEI],
        FT_PDF:    counts[FT_PDF],
        FT_NONE:   counts[FT_NONE],
    }
    logger.info("━" * 50)
    logger.info("DONE: %s → %s", summary, out_path)
    logger.info("━" * 50)

    sync.update({"works_fetched": len(works_list), "authors_fetched": authors_count})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Fetch OpenAlex data and download fulltexts")
    parser.add_argument("--incremental",   action="store_true", help="Only fetch records updated since last run")
    parser.add_argument("--skip-download", action="store_true", help="Skip fulltext download (title+abstract only)")
    args = parser.parse_args()

    result = run(incremental=args.incremental, skip_download=args.skip_download)
    print(f"\nResult: {result}")