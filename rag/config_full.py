# # config_full.py

# SQLITE_DB_FULL = r"C:\codes\t5-db\syr_research_all.db"

# CHROMA_DIR_FULL = r"C:\codes\new_pipeline\Syr_research_all\chroma_store_full"
# CHROMA_COLLECTION = "papers_all"

# LLAMA_1B = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
# LLAMA_3B = r"C:\codes\llama32\Llama-3.2-3B-Instruct"

# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASS = "OSPOlol@1234"
# NEO4J_DB = "syr-rag-one"


import os

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

SQLITE_DB_FULL = _env("SQLITE_DB_FULL", "path")

CHROMA_DIR_FULL = _env("CHROMA_DIR_FULL", "path")
CHROMA_COLLECTION_FULL = _env("CHROMA_COLLECTION_FULL", "papers_all")

CHROMA_COLLECTION = CHROMA_COLLECTION_FULL

LLAMA_1B = _env("LLAMA_1B", "path")
LLAMA_3B = _env("LLAMA_3B", "path")
LLAMA_8B = _env("LLAMA_8B", "path")
GEMMA_12B = _env("GEMMA_12B", "path")
QWEN_14B = _env("QWEN_14B", "path")
GPT_OSS_20B = _env("GPT_OSS_20B", "path")

EMBED_MODEL = _env("EMBED_MODEL", "path")
SENTENCE_TFORMER = EMBED_MODEL

CHUNK_MAX_CHARS = int(_env("CHUNK_MAX_CHARS", "12000"))
PAPERS_PER_BATCH = int(_env("PAPERS_PER_BATCH", "200"))
CHROMA_MAX_BATCH = int(_env("CHROMA_MAX_BATCH", "5400"))

NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASS = _env("NEO4J_PASS", "OSPOlol@1234")
NEO4J_DB = _env("NEO4J_DB", "syr-rag-one")

# ── OpenAlex dataset ──────────────────────────────────────────────────────────
CHROMA_DIR_OPENALEX = _env("CHROMA_DIR_OPENALEX", "path")
CHROMA_COLLECTION_OPENALEX = _env("CHROMA_COLLECTION_OPENALEX", "syracuse_papers")
NEO4J_DB_OPENALEX = _env("NEO4J_DB_OPENALEX", "syr-rag-openalex")

# ── OpenAlex abstracts-only dataset ──────────────────────────────────────────
CHROMA_DIR_ABSTRACTS = _env("CHROMA_DIR_ABSTRACTS", "path")
CHROMA_COLLECTION_ABSTRACTS = _env("CHROMA_COLLECTION_ABSTRACTS", "syracuse_abstracts")
NEO4J_DB_ABSTRACTS = _env("NEO4J_DB_ABSTRACTS", "syr-rag-abstracts")