# session_store.py
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from runtime_settings import settings

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_state (
  session_id TEXT PRIMARY KEY,
  rolling_summary TEXT NOT NULL,
  turns_json TEXT NOT NULL,
  extra_state_json TEXT NOT NULL DEFAULT '{}'
);
"""

_DEFAULT_ROLLING_SUMMARY = (
    "Current focus:\n- (none)\n"
    "Core entities:\n- (none)\n"
    "Key themes:\n- (none)\n"
    "Constraints:\n- Use only retrieved Syracuse corpus context.\n"
    "Open questions:\n- (none)"
)


def _safe_json_loads(raw: Optional[str], default: Any) -> Any:
    try:
        data = json.loads(raw) if raw else default
    except Exception:
        return default
    return data if isinstance(data, type(default)) else default


def _trim_turns(turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Trim conversation turns based on total character size, with count as a backstop.

    Keeps the most recent turns that fit within the character budget.
    When the total exceeds ``session_turns_max_chars``, older turns are
    dropped until the total falls below ``session_turn_trim_target_chars``.
    The hard count limit ``session_turns_keep`` is still enforced as a
    ceiling to prevent unbounded growth even if turns are very short.
    """
    keep_n = max(2, int(getattr(settings, "session_turns_keep", 24)))
    max_chars = max(2000, int(getattr(settings, "session_turns_max_chars", 32000)))
    target_chars = max(1000, int(getattr(settings, "session_turn_trim_target_chars", 24000)))

    # First pass: normalise and apply hard count limit
    trimmed = list(turns or [])[-keep_n:]
    out: List[Dict[str, str]] = []
    for obj in trimmed:
        role = str((obj or {}).get("role", "") or "").strip().lower()
        text = str((obj or {}).get("text", "") or "").strip()
        if role not in {"user", "assistant"} or not text:
            continue
        out.append({"role": role, "text": text})

    # Second pass: trim by total character size (drop oldest first)
    total_chars = sum(len(t.get("text", "")) for t in out)
    if total_chars > max_chars:
        while len(out) > 2 and total_chars > target_chars:
            removed = out.pop(0)
            total_chars -= len(removed.get("text", ""))

    return out


def _sanitize_anchor(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    anchor_value = str(value.get("value", "") or "").strip()
    if not anchor_value:
        return {}
    try:
        confidence = float(value.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "type": str(value.get("type", "metadata") or "metadata").strip().lower(),
        "value": anchor_value,
        "source": str(value.get("source", "retrieval") or "retrieval").strip(),
        "confidence": confidence,
    }


def _sanitize_extra_state(extra_state: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(extra_state, dict):
        return {}
    out: Dict[str, Any] = {}
    if "last_focus" in extra_state:
        out["last_focus"] = str(extra_state.get("last_focus", "") or "")
    if "last_topic" in extra_state:
        out["last_topic"] = str(extra_state.get("last_topic", "") or "")
    if "anchor_last_action" in extra_state:
        out["anchor_last_action"] = str(extra_state.get("anchor_last_action", "") or "")
    if "summary_updated" in extra_state:
        out["summary_updated"] = bool(extra_state.get("summary_updated"))
    if "retrieval_confidence" in extra_state:
        out["retrieval_confidence"] = str(extra_state.get("retrieval_confidence", "") or "")
    if "anchor_support_ratio" in extra_state:
        try:
            out["anchor_support_ratio"] = float(extra_state.get("anchor_support_ratio", 0.0) or 0.0)
        except Exception:
            out["anchor_support_ratio"] = 0.0
    if "rewrite_anchor_valid" in extra_state:
        out["rewrite_anchor_valid"] = bool(extra_state.get("rewrite_anchor_valid"))
    if "rewrite_blocked" in extra_state:
        out["rewrite_blocked"] = bool(extra_state.get("rewrite_blocked"))

    # Always write the anchor key — even when empty — so an explicit clear
    # (anchor={}) actually overwrites the previous value in the DB merge.
    anchor = _sanitize_anchor(extra_state.get("anchor"))
    out["anchor"] = anchor  # {} when cleared, populated dict when set
    return out


class SessionStore:
    """
    SQLite-backed session store.

    Uses threading.local() so each thread gets its own connection.
    Schema migration runs exactly once across all threads.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._schema_done = False
        self._ensure_connection()

    def _ensure_connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=True)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA temp_store=MEMORY;")
            self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        return self._ensure_connection()

    def _init_db(self) -> None:
        if self._schema_done:
            return
        with self._init_lock:
            if self._schema_done:
                return
            conn = self._conn
            conn.execute(SCHEMA_SQL)
            conn.commit()
            try:
                cols = [
                    row[1]
                    for row in conn.execute("PRAGMA table_info(chat_state);").fetchall()
                ]
                if "extra_state_json" not in cols:
                    conn.execute(
                        "ALTER TABLE chat_state ADD COLUMN extra_state_json TEXT NOT NULL DEFAULT '{}';"
                    )
                    conn.commit()
            except Exception:
                pass
            self._schema_done = True

    def load(self, session_id: str) -> Dict[str, Any]:
        self._init_db()
        conn = self._conn
        try:
            row = conn.execute(
                "SELECT rolling_summary, turns_json, extra_state_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                "SELECT rolling_summary, turns_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()

        if not row:
            return {"rolling_summary": _DEFAULT_ROLLING_SUMMARY, "turns": []}

        if len(row) == 2:
            rolling_summary, turns_json = row
            extra_state_json = "{}"
        else:
            rolling_summary, turns_json, extra_state_json = row

        turns = _trim_turns(_safe_json_loads(turns_json, []))
        extra_state = _sanitize_extra_state(_safe_json_loads(extra_state_json, {}))

        summary_value = str(rolling_summary or "")
        if not summary_value.strip():
            summary_value = _DEFAULT_ROLLING_SUMMARY

        state: Dict[str, Any] = {"rolling_summary": summary_value, "turns": turns}
        if extra_state:
            state["extra_state"] = extra_state
            for key in (
                "last_focus", "last_topic", "anchor_last_action",
                "summary_updated", "retrieval_confidence",
            ):
                if key in extra_state:
                    state[key] = extra_state[key]
            # Only promote a non-empty anchor into top-level state.
            if "anchor" in extra_state:
                anchor_val = extra_state.get("anchor")
                state["anchor"] = anchor_val if isinstance(anchor_val, dict) and anchor_val else {}
        return state

    def save(
        self,
        session_id: str,
        rolling_summary: str,
        turns: List[Dict[str, str]],
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._init_db()
        turns_payload = _trim_turns(turns or [])
        payload = json.dumps(turns_payload, ensure_ascii=False)
        conn = self._conn

        # Use BEGIN IMMEDIATE to acquire a write lock before reading,
        # preventing a race between the SELECT and the INSERT/UPDATE
        # when the utility worker and main thread save concurrently.
        conn.execute("BEGIN IMMEDIATE")
        try:
            existing = conn.execute(
                "SELECT rolling_summary, extra_state_json FROM chat_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            base_summary = str(existing[0] or "") if existing else ""
            base_extra = _safe_json_loads(existing[1], {}) if existing and len(existing) > 1 else {}
            base_extra = _sanitize_extra_state(base_extra)

            if isinstance(extra_state, dict):
                sanitized_new = _sanitize_extra_state(extra_state)
                # Explicit per-key assignment so that an empty anchor ({}) correctly
                # overwrites a previously stored non-empty anchor value.
                for k, v in sanitized_new.items():
                    base_extra[k] = v

            extra_payload = json.dumps(base_extra or {}, ensure_ascii=False)

            summary_payload = str(rolling_summary or "")
            if not summary_payload.strip():
                summary_payload = base_summary if base_summary.strip() else _DEFAULT_ROLLING_SUMMARY

            conn.execute(
                """
                INSERT INTO chat_state(session_id, rolling_summary, turns_json, extra_state_json)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  rolling_summary=excluded.rolling_summary,
                  turns_json=excluded.turns_json,
                  extra_state_json=excluded.extra_state_json
                """,
                (session_id, summary_payload, payload, extra_payload),
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def reset(self, session_id: str) -> None:
        self._init_db()
        conn = self._conn
        conn.execute("DELETE FROM chat_state WHERE session_id = ?", (session_id,))
        conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass