# research-llm OpenAlex

OpenAlex ingestion branch for building a research paper corpus directly from OpenAlex and preparing it for retrieval and graph based downstream workflows. This branch fetches institution scoped works and authors, downloads available full text, processes documents with Docling, normalizes records into a clean JSONL format, and ingests the output into both ChromaDB and Neo4j.

## Overview

The codebase is organized as a set of standalone Python scripts that can be run individually or through a single orchestrator. It is designed as a local batch pipeline rather than a packaged service.

At a high level, the pipeline does five things:

1. Fetch works and author metadata from OpenAlex
2. Download available full text as TEI XML or PDF
3. Process downloaded files through Docling
4. Normalize raw records into structured JSONL outputs
5. Ingest the results into ChromaDB and Neo4j

## Repository Goals

This branch appears intended to support a research assistant or discovery workflow by producing a local, reusable academic corpus that includes:

* institution scoped OpenAlex work metadata
* author metadata
* local full text files where available
* Docling structured document sections
* normalized records for chunking and embedding
* vector search storage in ChromaDB
* graph relationships in Neo4j

## Architecture

The OpenAlex pipeline revolves around a staged file based flow rather than a relational database.

### Raw acquisition layer

Stores direct outputs from OpenAlex and download resolution.

* `data/raw/works.jsonl`
* `data/raw/authors.jsonl`
* `data/raw/works_with_fulltext.jsonl`

### Document processing layer

Stores extracted full text artifacts and Docling outputs.

* `data/fulltext/{work_id}.tei.xml`
* `data/fulltext/{work_id}.pdf`
* `data/docling/{work_id}.json`
* `data/raw/works_with_docling.jsonl`

### Normalized ingestion layer

Stores cleaned records for downstream vector and graph indexing.

* `data/raw/normalized_works.jsonl`
* `data/raw/normalized_authors.jsonl`
* `data/chroma_db/`

## Processing Layers

### OpenAlex fetch and download

* `fetch_and_download.py`

Fetches Syracuse University works and authors from OpenAlex, writes raw JSONL outputs, and attempts to download full text in priority order.

Priority order:

1. TEI XML from the OpenAlex fulltext endpoint
2. open access PDF from OpenAlex metadata
3. PDF via Unpaywall using DOI lookup

Works without downloadable full text are retained and marked so they can still be ingested later using title and abstract only.

### Full text processing

* `docling_process.py`

Reads downloaded TEI XML or PDF files and processes them through Docling. When Docling succeeds, it writes rich structured document sections. If Docling fails on a PDF, the script falls back to raw text extraction through `pdfminer`.

`docling_status` values:

* `docling_ok`
* `fallback_pdf`
* `none`

### Record normalization

* `normalize.py`

Transforms raw OpenAlex outputs into a cleaner structured format for downstream indexing. It automatically chooses the best available input source in this order:

1. `works_with_docling.jsonl`
2. `works_with_fulltext.jsonl`
3. `works.jsonl`

It writes:

* `data/raw/normalized_works.jsonl`
* `data/raw/normalized_authors.jsonl`

### Chunking and vector preparation

* `chunker.py`
* `ingest_chroma.py`

`chunker.py` converts normalized works into embeddable units. Chunk types include:

* `title_abstract`
* `keywords`
* `section`
* `fallback_text`
* `table`
* `figure_caption`

`ingest_chroma.py` embeds those chunks and stores them in a persistent ChromaDB collection.

### Graph ingestion

* `ingest_neo.py`

Builds a Neo4j knowledge graph from normalized works and authors.

Node types:

* `Author`
* `Work`
* `Topic`

Edge types:

* `AUTHORED`
* `HAS_TOPIC`
* `CITES`
* `COLLABORATES_WITH`

## End to End Pipeline

The main orchestrator is `main.py`.

The execution order is:

1. `fetch_and_download`
2. `docling_process`
3. `normalize`
4. `ingest_chroma`
5. `ingest_neo`

### Why this order matters

* OpenAlex fetch creates the base work and author records
* download adds local full text availability and status metadata
* Docling produces structured sections that improve later chunking quality
* normalization consolidates all upstream outputs into a stable schema
* ChromaDB ingestion depends on normalized chunkable work records
* Neo4j ingestion depends on normalized work, author, topic, and citation data

## Repository Structure

    openalex/
    ├── chunker.py              # Splits normalized works into embeddable chunks
    ├── config.py               # Central configuration via environment variables
    ├── docling_process.py      # Processes PDFs and TEI with Docling
    ├── fetch_and_download.py   # Fetches works and authors, downloads full text
    ├── ingest_chroma.py        # Embeds and indexes chunks in ChromaDB
    ├── ingest_neo.py           # Builds Neo4j knowledge graph
    ├── main.py                 # Full pipeline orchestrator
    └── normalize.py            # Cleans and structures raw OpenAlex outputs

## Expected Environment

This branch assumes a local environment with:

* Python 3.x
* internet access for OpenAlex and optional Unpaywall lookups
* local write access for `data/`
* optional Docling compatible environment
* optional Neo4j instance running locally or remotely
* optional Torch compatible environment for Docling and embeddings

## Configuration

Configuration is centralized in `config.py` and can be overridden with environment variables.

### OpenAlex

* `OPENALEX_INSTITUTION_ID`
* `OPENALEX_API_KEY`
* `OPENALEX_EMAIL`
* `OPENALEX_PER_PAGE`
* `OPENALEX_RATE_DELAY`

### Paths

* `DATA_DIR`
* `RAW_DIR`
* `FULLTEXT_DIR`
* `DOCLING_DIR`
* `SYNC_STATE_FILE`

### ChromaDB

* `CHROMA_DIR`
* `CHROMA_COLLECTION`

### Neo4j

* `NEO4J_URI`
* `NEO4J_USER`
* `NEO4J_PASSWORD`
* `NEO4J_DATABASE`

### Embeddings

* `EMBED_MODEL`
* `EMBED_DEVICE`

### Chunking

* `CHUNK_MAX_TOKENS`
* `CHUNK_OVERLAP_TOKENS`

### Download behavior

* `DOWNLOAD_WORKERS`
* `DOWNLOAD_TIMEOUT`
* `DOWNLOAD_MAX_RETRIES`
* `DOWNLOAD_RATE_DELAY`
* `UNPAYWALL_EMAIL`

## Installation

Create a virtual environment and install the dependencies needed by the pipeline.

    python -m venv .venv
    source .venv/bin/activate
    pip install requests chromadb sentence-transformers neo4j

For Docling based processing you will also need:

    pip install docling pdfminer.six

If you want Torch with CUDA support, install the appropriate build for your platform.

## Running the Pipeline

Run the full pipeline:

    python main.py

Run specific modules:

    python main.py --module fetch
    python main.py --module docling
    python main.py --module ingest

Useful options:

    python main.py --incremental
    python main.py --skip-download
    python main.py --skip-docling
    python main.py --skip-neo4j

### Standalone script usage

Fetch and download only:

    python fetch_and_download.py
    python fetch_and_download.py --incremental
    python fetch_and_download.py --skip-download

Docling only:

    python docling_process.py
    python docling_process.py --incremental

Normalize only:

    python normalize.py

Chroma only:

    python ingest_chroma.py
    python ingest_chroma.py --incremental

Neo4j only:

    python ingest_neo.py
    python ingest_neo.py --incremental

## Script Reference

### `main.py`

Top level orchestrator for the whole ingestion flow.

Responsibilities:

* routes execution by module
* supports full and partial runs
* supports incremental mode
* supports skipping download, Docling, or Neo4j

### `config.py`

Central configuration module.

Responsibilities:

* loads environment variables
* defines default directories and runtime values
* centralizes embedding, chunking, download, and database settings

### `fetch_and_download.py`

Module 1 of the pipeline.

Responsibilities:

* fetches institution filtered works from OpenAlex
* fetches related author records
* reconstructs abstracts when needed
* downloads TEI XML or PDF when possible
* tracks sync state for incremental updates
* annotates works with `fulltext_status` and `fulltext_path`

### `docling_process.py`

Module 2 of the pipeline.

Responsibilities:

* reads downloaded full text files
* processes TEI XML and PDFs with Docling
* falls back to `pdfminer` text extraction for failed PDFs
* writes structured per work JSON outputs
* annotates works with `docling_status` and `docling_path`

### `normalize.py`

Normalization layer between acquisition and indexing.

Responsibilities:

* chooses the best available upstream input file
* reshapes raw OpenAlex records into a stable schema
* preserves useful metadata such as authors, topics, affiliations, DOI, citation counts, and processing status
* writes normalized works and authors JSONL outputs

### `chunker.py`

Chunk generation utility used by Chroma ingestion.

Responsibilities:

* creates chunk variants from titles, abstracts, keywords, sections, figures, and tables
* applies overlap aware sentence based splitting
* keeps chunk typing explicit for retrieval control

### `ingest_chroma.py`

Vector indexing stage.

Responsibilities:

* reads normalized works
* generates chunks through `chunk_work`
* embeds chunks using a sentence transformer model
* writes to a persistent ChromaDB collection
* supports rebuild or incremental style operation

### `ingest_neo.py`

Graph indexing stage.

Responsibilities:

* creates and clears graph structures when rebuilding
* upserts author, work, and topic nodes
* builds authorship, topic, and citation edges
* derives collaboration edges from coauthorship patterns

## Output Artifacts

After a successful end to end run, the main artifacts are expected to be:

* `data/raw/works.jsonl`
* `data/raw/authors.jsonl`
* `data/raw/works_with_fulltext.jsonl`
* `data/raw/works_with_docling.jsonl`
* `data/raw/normalized_works.jsonl`
* `data/raw/normalized_authors.jsonl`
* `data/fulltext/`
* `data/docling/`
* `data/chroma_db/`
* Neo4j nodes and relationships in the configured database

## Incremental Behavior

The pipeline includes incremental support, mainly through the fetch and downstream ingestion flow.

Expected behavior:

* fetch uses a sync watermark file
* previously downloaded records can be skipped
* Chroma and Neo4j stages can avoid full rebuilds when incremental mode is used
* output quality depends on the consistency of prior artifacts already present in `data/`
