"""
main.py — Full ingestion pipeline orchestrator.

Modules:
  1. fetch      — fetch_and_download.py   (OpenAlex API + fulltext download)
  2. docling    — docling_process.py      (Docling PDF/TEI processing)
  3. ingest     — normalize + chroma + neo4j

Usage:
  python main.py                          # full pipeline
  python main.py --module fetch           # fetch + download only
  python main.py --module docling         # docling only
  python main.py --module ingest          # normalize + chroma + neo4j
  python main.py --skip-download          # fetch only, no PDFs
  python main.py --skip-docling           # skip docling step
  python main.py --skip-neo4j             # skip neo4j
  python main.py --incremental            # only new records
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _banner(title: str):
    logger.info("")
    logger.info("┌" + "─" * 58 + "┐")
    logger.info("│  %-56s│" % title)
    logger.info("└" + "─" * 58 + "┘")


def main():
    parser = argparse.ArgumentParser(
        description="Syracuse RAG — Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modules:
  fetch    — Pull works/authors from OpenAlex + download PDFs/TEI
  docling  — Process downloaded files through Docling
  ingest   — Normalize → ChromaDB → Neo4j
  all      — Run everything (default)

Examples:
  python main.py                        Full pipeline
  python main.py --module fetch         Fetch + download only
  python main.py --module docling       Docling only
  python main.py --module ingest        Normalize + Chroma + Neo4j
  python main.py --skip-download        No PDF download
  python main.py --skip-docling         Skip Docling
  python main.py --skip-neo4j           No Neo4j
  python main.py --incremental          New records only
        """,
    )
    parser.add_argument(
        "--module",
        choices=["fetch", "docling", "ingest", "all"],
        default="all",
    )
    parser.add_argument("--incremental",   action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-docling",  action="store_true")
    parser.add_argument("--skip-neo4j",    action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    results = {}

    # ── Module 1: Fetch + Download ─────────────────────────────────────────────
    if args.module in ("fetch", "all"):
        _banner("MODULE 1: Fetch from OpenAlex + Download Fulltexts")
        t0 = time.time()
        import fetch_and_download
        result = fetch_and_download.run(
            incremental=args.incremental,
            skip_download=args.skip_download,
        )
        logger.info("Module 1 done in %.1fs: %s", time.time() - t0, result)
        results["fetch"] = result

    # ── Module 2: Docling ──────────────────────────────────────────────────────
    run_docling = (
        args.module in ("docling", "all")
        and not args.skip_docling
        and not args.skip_download
    )
    if run_docling:
        _banner("MODULE 2: Docling Fulltext Processing")
        t0 = time.time()
        import docling_process
        result = docling_process.run(incremental=args.incremental)
        logger.info("Module 2 done in %.1fs: %s", time.time() - t0, result)
        results["docling"] = result
    elif args.module in ("docling", "all"):
        logger.info("Module 2 (Docling): SKIPPED")

    # ── Module 3: Ingest ───────────────────────────────────────────────────────
    if args.module in ("ingest", "all"):

        # 3a. Normalize
        _banner("MODULE 3a: Normalize")
        t0 = time.time()
        import normalize
        result = normalize.run()
        logger.info("Normalize done in %.1fs: %s", time.time() - t0, result)
        results["normalize"] = result

        # 3b. ChromaDB
        _banner("MODULE 3b: Ingest into ChromaDB")
        t0 = time.time()
        import ingest_chroma
        result = ingest_chroma.run(rebuild=not args.incremental)
        logger.info("Chroma done in %.1fs: %s", time.time() - t0, result)
        results["chroma"] = result

        # 3c. Neo4j
        if not args.skip_neo4j:
            _banner("MODULE 3c: Build Neo4j Knowledge Graph")
            t0 = time.time()
            try:
                import ingest_neo4j
                result = ingest_neo4j.run(rebuild=not args.incremental)
                logger.info("Neo4j done in %.1fs: %s", time.time() - t0, result)
                results["neo4j"] = result
            except Exception as e:
                logger.error("Neo4j failed: %s", e)
                logger.info("Use --skip-neo4j to skip")
        else:
            logger.info("Module 3c (Neo4j): SKIPPED")

    # ── Summary ────────────────────────────────────────────────────────────────
    _banner(f"Pipeline complete in {time.time() - t_start:.1f}s")
    for k, v in results.items():
        logger.info("  %-12s %s", k + ":", v)


if __name__ == "__main__":
    main()