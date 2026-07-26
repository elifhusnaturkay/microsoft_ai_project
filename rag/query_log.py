"""SQLite-backed durable log of questions asked through /api/ask.

Deliberately a separate DB file (see rag/config.py's QUERY_LOG_DB_PATH) from knowledge.db --
that file is owned by scripts/ingest.py, which deletes and repopulates it on every deploy;
this one must persist across those runs.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCHEMA = """
CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    language TEXT NOT NULL,
    created_at TEXT NOT NULL,
    num_sources INTEGER NOT NULL,
    was_answered INTEGER NOT NULL
);
"""


def init_db(db_path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB and ensure the query_log table exists.

    check_same_thread=False for the same reason as rag/store.py's init_db: server.py
    caches this connection per-thread, but FastAPI runs ask() in a thread-pool worker,
    so the connection may be created on one thread and used from another.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def insert_query(
    conn: sqlite3.Connection,
    question: str,
    language: str,
    num_sources: int,
    was_answered: bool,
) -> None:
    conn.execute(
        "INSERT INTO query_log (question, language, created_at, num_sources, was_answered) "
        "VALUES (?, ?, ?, ?, ?)",
        (question, language, datetime.now(timezone.utc).isoformat(), num_sources, int(was_answered)),
    )
    conn.commit()


def get_recent_queries(conn: sqlite3.Connection, limit: int = 100) -> List[Dict]:
    """Return the `limit` most recently logged queries, newest first (see server.py's
    GET /api/admin/queries)."""
    rows = conn.execute(
        "SELECT id, question, language, created_at, num_sources, was_answered "
        "FROM query_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": row[0],
            "question": row[1],
            "language": row[2],
            "created_at": row[3],
            "num_sources": row[4],
            "was_answered": bool(row[5]),
        }
        for row in rows
    ]
