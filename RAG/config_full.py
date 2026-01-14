# config_full.py
import os

SQLITE_DB_FULL = r"C:\codes\t5-db\syr_research_all.db"

CHROMA_DIR_FULL = r"C:\codes\new_pipeline\Syr_research_all\chroma_store_full"
CHROMA_DIR_ABSTRACTS = r"C:\codes\new_pipeline\Syr_research_all\chroma_store_abstracts"

CHROMA_COLLECTION_FULL = "papers_all"
CHROMA_COLLECTION_ABSTRACTS = "abstracts_all"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLAMA_1B = r"C:\codes\llama32\Llama-3.2-1B-Instruct"
LLAMA_3B = r"C:\codes\llama32\Llama-3.2-3B-Instruct"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "password")
NEO4J_DB = os.getenv("NEO4J_DB", "neo4j")

PAPERS_PER_BATCH = 200
CHROMA_MAX_BATCH = 5400
CHUNK_MAX_CHARS = 12000

NEO_BATCH_SIZE = 500
NEO_WORKERS = 6
