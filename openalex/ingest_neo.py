"""
ingest_neo4j.py — Build knowledge graph in Neo4j.

Nodes:  Author, Work, Topic
Edges:  AUTHORED, HAS_TOPIC, CITES, COLLABORATES_WITH

Reads:   data/raw/normalized_works.jsonl
         data/raw/normalized_authors.jsonl

Standalone:
    python ingest_neo4j.py
    python ingest_neo4j.py --incremental
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

BATCH = 500


class Neo4jGraph:
    def __init__(self):
        if not HAS_NEO4J:
            raise RuntimeError("neo4j not installed — run: pip install neo4j")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        self.db = config.NEO4J_DATABASE
        self.driver.verify_connectivity()
        logger.info("Connected to Neo4j at %s (db=%s)", config.NEO4J_URI, self.db)

    def close(self):
        self.driver.close()

    def _run(self, q: str, **p):
        with self.driver.session(database=self.db) as s:
            s.run(q, **p)

    def _batch(self, q: str, key: str, data: list):
        if data:
            with self.driver.session(database=self.db) as s:
                s.run(q, **{key: data})

    def create_constraints(self):
        for q in [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.oid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (w:Work)   REQUIRE w.oid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic)  REQUIRE t.oid IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (w:Work)   ON (w.title)",
            "CREATE INDEX IF NOT EXISTS FOR (w:Work)   ON (w.year)",
        ]:
            try:
                self._run(q)
            except Exception:
                pass

    def clear(self):
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared all Neo4j nodes/edges")

    def upsert_works(self, b):
        self._batch("""
            UNWIND $b AS w
            MERGE (x:Work {oid: w.oid})
            SET x.title = w.title, x.doi = w.doi, x.year = w.year,
                x.cited = w.cited, x.type = w.type,
                x.researcher = w.researcher,
                x.docling_status = w.docling_status
        """, "b", b)

    def upsert_authors(self, b):
        self._batch("""
            UNWIND $b AS a
            MERGE (x:Author {oid: a.oid})
            SET x.name = a.name, x.orcid = a.orcid,
                x.works_count = a.works_count,
                x.cited = a.cited, x.h_index = a.h_index
        """, "b", b)

    def upsert_topics(self, b):
        self._batch("""
            UNWIND $b AS t
            MERGE (x:Topic {oid: t.oid})
            SET x.name = t.name, x.subfield = t.subfield,
                x.field = t.field, x.domain = t.domain
        """, "b", b)

    def create_authored(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (a:Author {oid: e.aid})
            MATCH (w:Work   {oid: e.wid})
            MERGE (a)-[r:AUTHORED]->(w)
            SET r.position = e.pos, r.is_corresponding = e.corr
        """, "b", b)

    def create_has_topic(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (w:Work  {oid: e.wid})
            MATCH (t:Topic {oid: e.tid})
            MERGE (w)-[r:HAS_TOPIC]->(t)
            SET r.score = e.score
        """, "b", b)

    def create_cites(self, b):
        self._batch("""
            UNWIND $b AS e
            MATCH (a:Work {oid: e.src})
            MATCH (b:Work {oid: e.dst})
            MERGE (a)-[:CITES]->(b)
        """, "b", b)

    def derive_collaborations(self):
        # Run in batches to avoid transaction memory limit
        # First get total author count
        with self.driver.session(database=self.db) as s:
            total = s.run("MATCH (a:Author) RETURN count(a) AS c").single()["c"]

        batch_size = 2000
        offset     = 0
        total_edges = 0
        while offset < total:
            with self.driver.session(database=self.db) as s:
                result = s.run("""
                    MATCH (a1:Author)
                    WITH a1 SKIP $skip LIMIT $limit
                    MATCH (a1)-[:AUTHORED]->(w:Work)<-[:AUTHORED]-(a2:Author)
                    WHERE a1.oid < a2.oid
                    WITH a1, a2, COUNT(w) AS shared
                    MERGE (a1)-[r:COLLABORATES_WITH]-(a2)
                    SET r.shared_works = shared
                    RETURN count(r) AS edges
                """, skip=offset, limit=batch_size)
                edges = result.single()["edges"]
                total_edges += edges
            offset += batch_size
            logger.info("  Collaborations batch offset=%d edges_so_far=%d", offset, total_edges)

        logger.info("COLLABORATES_WITH edges derived: %d total", total_edges)


def run(rebuild: bool = True) -> Dict[str, int]:
    if not HAS_NEO4J:
        logger.error("neo4j driver not installed — skipping")
        return {"status": "skipped"}

    raw_dir      = Path(config.RAW_DIR)
    works_file   = raw_dir / "normalized_works.jsonl"
    authors_file = raw_dir / "normalized_authors.jsonl"

    graph = Neo4jGraph()
    valid_cites: List[Dict] = []

    try:
        if rebuild:
            graph.clear()
        graph.create_constraints()

        # ── Author nodes ──────────────────────────────────────────────────────
        author_count = 0
        if authors_file.exists():
            batch = []
            with open(authors_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a = json.loads(line)
                    batch.append({
                        "oid": a["openalex_id"], "name": a["name"],
                        "orcid": a.get("orcid", ""),
                        "works_count": a.get("works_count", 0),
                        "cited": a.get("cited_by_count", 0),
                        "h_index": a.get("h_index", 0),
                    })
                    if len(batch) >= BATCH:
                        graph.upsert_authors(batch)
                        author_count += len(batch)
                        batch.clear()
            if batch:
                graph.upsert_authors(batch)
                author_count += len(batch)
            logger.info("Author nodes: %d", author_count)

        # ── Work nodes + edges ────────────────────────────────────────────────
        work_count       = 0
        seen_topics:     Set[str]   = set()
        known_works:     Set[str]   = set()
        cite_edges:      List[Dict] = []   # collected during pass, filtered after

        work_batch:       List[Dict] = []
        auth_node_batch:  List[Dict] = []
        topic_batch:      List[Dict] = []
        authored_batch:   List[Dict] = []
        topic_edge_batch: List[Dict] = []

        if works_file.exists():
            with open(works_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    w   = json.loads(line)
                    wid = w["openalex_id"]
                    known_works.add(wid)

                    work_batch.append({
                        "oid": wid, "title": w.get("title", ""),
                        "doi": w.get("doi", ""),
                        "year": w.get("publication_year"),
                        "cited": w.get("cited_by_count", 0),
                        "type": w.get("work_type", ""),
                        "researcher": w.get("primary_researcher", ""),
                        "docling_status": w.get("docling_status", "none"),
                    })

                    for a in (w.get("all_authors") or []):
                        aid = a.get("id", "")
                        if not aid:
                            continue
                        auth_node_batch.append({
                            "oid": aid, "name": a.get("name", ""),
                            "orcid": a.get("orcid", ""),
                            "works_count": 0, "cited": 0, "h_index": 0,
                        })
                        authored_batch.append({
                            "aid": aid, "wid": wid,
                            "pos": a.get("position", ""),
                            "corr": a.get("is_corresponding", False),
                        })

                    for t in (w.get("topics") or []):
                        tid = t.get("id", "")
                        if not tid:
                            continue
                        if tid not in seen_topics:
                            seen_topics.add(tid)
                            topic_batch.append({
                                "oid": tid, "name": t.get("name", ""),
                                "subfield": t.get("subfield", ""),
                                "field": t.get("field", ""),
                                "domain": t.get("domain", ""),
                            })
                        topic_edge_batch.append({
                            "wid": wid, "tid": tid,
                            "score": float(t.get("score", 0)),
                        })

                    for ref in (w.get("referenced_works") or []):
                        if ref:
                            # Normalize to short ID (strip URL prefix if present)
                            dst = ref.rsplit("/", 1)[-1] if "/" in ref else ref
                            cite_edges.append({"src": wid, "dst": dst})

                    if len(work_batch) >= BATCH:
                        graph.upsert_works(work_batch);           work_count += len(work_batch); work_batch.clear()
                        graph.upsert_authors(auth_node_batch);    auth_node_batch.clear()
                        graph.upsert_topics(topic_batch);         topic_batch.clear()
                        graph.create_authored(authored_batch);    authored_batch.clear()
                        graph.create_has_topic(topic_edge_batch); topic_edge_batch.clear()
                        logger.info("  ... %d works processed", work_count)

            # Flush remainder
            if work_batch:
                graph.upsert_works(work_batch);           work_count += len(work_batch)
            if auth_node_batch:
                graph.upsert_authors(auth_node_batch)
            if topic_batch:
                graph.upsert_topics(topic_batch)
            if authored_batch:
                graph.create_authored(authored_batch)
            if topic_edge_batch:
                graph.create_has_topic(topic_edge_batch)

            # Citation edges — only between works we actually have
            valid_cites = [e for e in cite_edges if e["dst"] in known_works]
            for i in range(0, len(valid_cites), BATCH):
                graph.create_cites(valid_cites[i:i+BATCH])

            logger.info(
                "Work nodes: %d | Topics: %d | Citations: %d",
                work_count, len(seen_topics), len(valid_cites),
            )

        graph.derive_collaborations()

        result = {
            "authors":   author_count,
            "works":     work_count,
            "topics":    len(seen_topics),
            "citations": len(valid_cites),
        }
        logger.info("Neo4j complete: %s", result)
        return result

    finally:
        graph.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()
    print(run(rebuild=not args.incremental))