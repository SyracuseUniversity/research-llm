# import os, sqlite3, hashlib, json

# CACHE_PATH = r"C:\codes\t5-db\rag_cache.db"

# def _init():
#     os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
#     conn = sqlite3.connect(CACHE_PATH)
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS cache (
#             qhash TEXT PRIMARY KEY,
#             question TEXT,
#             answer TEXT,
#             sources TEXT,
#             ts DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     """)
#     conn.commit(); conn.close()

# def clear_cache():
#     conn = sqlite3.connect(CACHE_PATH)
#     conn.execute("DELETE FROM cache")
#     conn.commit(); conn.close()

# def _hash(text):
#     return hashlib.sha256(text.encode()).hexdigest()

# def fetch(question):
#     conn = sqlite3.connect(CACHE_PATH)
#     row = conn.execute("SELECT answer, sources FROM cache WHERE qhash=?",
#                        (_hash(question),)).fetchone()
#     conn.close()
#     if not row: return None
#     ans, sources = row
#     return ans, json.loads(sources)

# def store(question, answer, sources):
#     conn = sqlite3.connect(CACHE_PATH)
#     conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,datetime('now'))",
#                  (_hash(question), question, answer, json.dumps(sources)))
#     conn.commit(); conn.close()

# _init()


# cache_manager.py
import os
import json
import hashlib

CACHE_DIR = os.path.expanduser("~/.rag_memory")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_FILE = os.path.join(CACHE_DIR, "answer_cache.json")

_cache = {}


def _load():
    global _cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}


def _save():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(_cache, f, ensure_ascii=False, indent=2)


_load()


def _key(q):
    return hashlib.sha256(q.strip().encode("utf-8")).hexdigest()


def clear_cache():
    global _cache
    _cache = {}
    _save()


def fetch(question):
    k = _key(question)
    entry = _cache.get(k)
    if not entry:
        return None
    return entry["answer"], entry.get("sources", [])


def store(question, answer, sources):
    k = _key(question)

    _cache[k] = {
        "question": question,
        "answer": answer,
        "sources": sources,
    }

    _save()
