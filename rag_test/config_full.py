import os

#SQLITE_DB = os.getenv("SQLITE_DB", r"D:\OSPO\KG-RAG1\researchers_fixed.db")
SQLITE_DB = os.getenv("SQLITE_DB", r"D:\OSPO\KG-RAG1\abstracts_only.db")

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "OSPOlol@1234")
NEO4J_DB   = os.getenv("NEO4J_DB",   "syr-rag")

# CHROMA_DIR = os.getenv("CHROMA_DIR", r"D:\OSPO\KG-RAG1\chroma_store_full")
CHROMA_DIR = os.getenv("CHROMA_DIR", r"D:\OSPO\KG-RAG1\chroma_store_abstracts")

LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", r"D:\OSPO\KG-RAG1\Llama-3.2-1B-Instruct")

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "2000"))
CHROMA_BATCH  = int(os.getenv("CHROMA_BATCH", "3000"))
PARALLEL_JOBS = int(os.getenv("PARALLEL_JOBS", "4"))

SENTENCE_TFORMER = os.getenv("SENTENCE_TFORMER", "sentence-transformers/all-MiniLM-L6-v2")
