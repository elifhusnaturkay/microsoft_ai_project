"""SQLite-backed chunk store + cosine-similarity retrieval math.

No vector extension assumed (PROJECT_PLAN.md locked decision) -- embeddings are stored as
JSON-serialized float lists in a BLOB column, and similarity is computed with plain
numpy at query time.
"""
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT,
    sources TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


def init_db(db_path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB and ensure the chunks table exists.

    check_same_thread=False: server.py caches a single connection for the process
    lifetime (see its _get_db_connection), but FastAPI runs each sync route handler
    (ask() is `def`, not `async def`) in a thread-pool worker -- a different request
    can land on a different thread than the one that created the connection. Without
    this flag, that raised `sqlite3.ProgrammingError: SQLite objects created in a
    thread can only be used in that same thread` on every request that happened to
    land on a different worker thread than the first one -- intermittent by nature
    (whichever thread the pool picked), which is exactly the "sometimes works,
    sometimes doesn't" behavior reported live on 2026-07-26 (see .claude/HANDOFF.md).
    Safe here: default Python sqlite3 builds use SQLite's serialized threading mode,
    and this DB is read-only at request time (writes only happen offline via
    ingest.py, never concurrently with serving).
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def clear_chunks(conn: sqlite3.Connection) -> None:
    """Delete all existing chunks (used by ingest.py --reset for a clean re-ingest)."""
    conn.execute("DELETE FROM chunks")
    conn.commit()


def _serialize_embedding(embedding: np.ndarray) -> bytes:
    return json.dumps(np.asarray(embedding, dtype=float).tolist()).encode("utf-8")


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.array(json.loads(blob.decode("utf-8")), dtype=np.float32)


def _serialize_sources(sources: List[Dict]) -> str:
    return json.dumps(sources or [])


def _deserialize_sources(raw: str) -> List[Dict]:
    return json.loads(raw) if raw else []


def insert_chunk(
    conn: sqlite3.Connection,
    source_file: str,
    sources: List[Dict],
    content: str,
    embedding: np.ndarray,
) -> None:
    conn.execute(
        "INSERT INTO chunks (source_file, sources, content, embedding) VALUES (?, ?, ?, ?)",
        (source_file, _serialize_sources(sources), content, _serialize_embedding(embedding)),
    )


def insert_chunks(conn: sqlite3.Connection, chunks: List[Dict]) -> None:
    """Bulk-insert chunks as produced by chunker.py, each augmented with an 'embedding' key
    (see scripts/ingest.py, which is what wires embedder.py's output onto each chunk dict).
    Each chunk carries a `sources` list (chunker.py may merge paragraphs from more than one
    source into a single chunk -- see rag/chunker.py's module docstring)."""
    for chunk in chunks:
        insert_chunk(
            conn,
            chunk.get("source_file"),
            chunk.get("sources", []),
            chunk["text"],
            chunk["embedding"],
        )
    conn.commit()


def get_all_chunks(conn: sqlite3.Connection) -> List[Dict]:
    """Fetch every stored chunk with its embedding deserialized back to a numpy array."""
    cursor = conn.execute("SELECT id, source_file, sources, content, embedding FROM chunks")
    rows = []
    for row_id, source_file, sources_raw, content, blob in cursor.fetchall():
        rows.append(
            {
                "id": row_id,
                "source_file": source_file,
                "sources": _deserialize_sources(sources_raw),
                "content": content,
                "embedding": _deserialize_embedding(blob),
            }
        )
    return rows


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Plain numpy cosine similarity -- no vector extension assumed."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
