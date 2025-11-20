import os, sqlite3, hashlib, json

CACHE_PATH = r"C:\codes\t5-db\rag_cache.db"

def _init():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            qhash TEXT PRIMARY KEY,
            question TEXT,
            answer TEXT,
            sources TEXT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit(); conn.close()

def clear_cache():
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("DELETE FROM cache")
    conn.commit(); conn.close()

def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def fetch(question):
    conn = sqlite3.connect(CACHE_PATH)
    row = conn.execute("SELECT answer, sources FROM cache WHERE qhash=?",
                       (_hash(question),)).fetchone()
    conn.close()
    if not row: return None
    ans, sources = row
    return ans, json.loads(sources)

def store(question, answer, sources):
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,datetime('now'))",
                 (_hash(question), question, answer, json.dumps(sources)))
    conn.commit(); conn.close()

_init()
