"""Top-K retrieval (fixed K=3 per PROJECT_PLAN.md) using store.py's cosine similarity."""
import sqlite3
from typing import Dict, List

from . import config, store
from .embedder import Embedder


def get_top_chunks(
    query: str,
    embedder: Embedder,
    conn: sqlite3.Connection,
    k: int = config.TOP_K,
) -> List[Dict]:
    """Embed the query, score it against every stored chunk via cosine similarity, and
    return the top-k chunks (each augmented with a 'similarity' score), highest first.

    Returns an empty list if the store has no chunks yet (e.g. before ingest.py has run).
    """
    rows = store.get_all_chunks(conn)
    if not rows:
        return []

    query_vector = embedder.embed_one(query)

    scored = [
        {**row, "similarity": store.cosine_similarity(query_vector, row["embedding"])}
        for row in rows
    ]
    scored.sort(key=lambda r: r["similarity"], reverse=True)
    return scored[:k]
