# session_store.py
import sqlite3
import json
from typing import List, Dict, Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_state (
  session_id TEXT PRIMARY KEY,
  rolling_summary TEXT NOT NULL,
  turns_json TEXT NOT NULL
);
"""

class SessionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(SCHEMA_SQL)
        conn.commit()
        conn.close()

    def load(self, session_id: str) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT rolling_summary, turns_json FROM chat_state WHERE session_id = ?",
            (session_id,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return {"rolling_summary": "", "turns": []}

        rolling_summary, turns_json = row
        try:
            turns = json.loads(turns_json) if turns_json else []
        except Exception:
            turns = []

        if not isinstance(turns, list):
            turns = []

        return {"rolling_summary": rolling_summary or "", "turns": turns}

    def save(self, session_id: str, rolling_summary: str, turns: List[Dict[str, str]]) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO chat_state(session_id, rolling_summary, turns_json)
            VALUES(?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
              rolling_summary=excluded.rolling_summary,
              turns_json=excluded.turns_json
            """,
            (session_id, rolling_summary or "", json.dumps(turns or [])),
        )
        conn.commit()
        conn.close()

    def reset(self, session_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM chat_state WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
