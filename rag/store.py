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
    source_name TEXT,
    source_url TEXT,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL
);
"""


def init_db(db_path) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite DB and ensure the chunks table exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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


def insert_chunk(
    conn: sqlite3.Connection,
    source_file: str,
    source_name: str,
    source_url: str,
    content: str,
    embedding: np.ndarray,
) -> None:
    conn.execute(
        "INSERT INTO chunks (source_file, source_name, source_url, content, embedding) "
        "VALUES (?, ?, ?, ?, ?)",
        (source_file, source_name, source_url, content, _serialize_embedding(embedding)),
    )


def insert_chunks(conn: sqlite3.Connection, chunks: List[Dict]) -> None:
    """Bulk-insert chunks as produced by chunker.py, each augmented with an 'embedding' key
    (see scripts/ingest.py, which is what wires embedder.py's output onto each chunk dict)."""
    for chunk in chunks:
        insert_chunk(
            conn,
            chunk.get("source_file"),
            chunk.get("source_name"),
            chunk.get("source_url"),
            chunk["text"],
            chunk["embedding"],
        )
    conn.commit()


def get_all_chunks(conn: sqlite3.Connection) -> List[Dict]:
    """Fetch every stored chunk with its embedding deserialized back to a numpy array."""
    cursor = conn.execute(
        "SELECT id, source_file, source_name, source_url, content, embedding FROM chunks"
    )
    rows = []
    for row_id, source_file, source_name, source_url, content, blob in cursor.fetchall():
        rows.append(
            {
                "id": row_id,
                "source_file": source_file,
                "source_name": source_name,
                "source_url": source_url,
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
