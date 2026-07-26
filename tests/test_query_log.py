"""Unit tests for rag/query_log.py: the durable, knowledge.db-independent question log."""
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import query_log


def test_init_db_creates_the_query_log_table():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "nested" / "queries.db"  # parent dir doesn't exist yet
        conn = query_log.init_db(str(db_path))

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'")
        assert cursor.fetchone() is not None


def test_insert_query_persists_all_fields():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queries.db"
        conn = query_log.init_db(str(db_path))

        query_log.insert_query(conn, "how much is tuition?", "en", num_sources=2, was_answered=True)

        row = conn.execute(
            "SELECT question, language, num_sources, was_answered, created_at FROM query_log"
        ).fetchone()
        question, language, num_sources, was_answered, created_at = row
        assert question == "how much is tuition?"
        assert language == "en"
        assert num_sources == 2
        assert was_answered == 1
        assert created_at  # non-empty ISO 8601 timestamp string


def test_insert_query_stores_was_answered_false_as_zero():
    with tempfile.TemporaryDirectory() as tmp:
        conn = query_log.init_db(str(Path(tmp) / "queries.db"))

        query_log.insert_query(conn, "meow?", "tr", num_sources=0, was_answered=False)

        row = conn.execute("SELECT was_answered FROM query_log").fetchone()
        assert row[0] == 0


def test_get_recent_queries_returns_newest_first():
    with tempfile.TemporaryDirectory() as tmp:
        conn = query_log.init_db(str(Path(tmp) / "queries.db"))
        query_log.insert_query(conn, "first?", "en", num_sources=1, was_answered=True)
        query_log.insert_query(conn, "second?", "en", num_sources=0, was_answered=False)

        rows = query_log.get_recent_queries(conn)

        assert [row["question"] for row in rows] == ["second?", "first?"]


def test_get_recent_queries_respects_limit():
    with tempfile.TemporaryDirectory() as tmp:
        conn = query_log.init_db(str(Path(tmp) / "queries.db"))
        for i in range(5):
            query_log.insert_query(conn, f"q{i}?", "en", num_sources=0, was_answered=False)

        rows = query_log.get_recent_queries(conn, limit=2)

        assert [row["question"] for row in rows] == ["q4?", "q3?"]


def test_get_recent_queries_shape_and_types():
    with tempfile.TemporaryDirectory() as tmp:
        conn = query_log.init_db(str(Path(tmp) / "queries.db"))
        query_log.insert_query(conn, "how much is tuition?", "en", num_sources=2, was_answered=True)

        row = query_log.get_recent_queries(conn)[0]

        assert set(row.keys()) == {"id", "question", "language", "created_at", "num_sources", "was_answered"}
        assert row["question"] == "how much is tuition?"
        assert row["language"] == "en"
        assert row["num_sources"] == 2
        assert row["was_answered"] is True  # int-in-DB coerced to a real bool, not 1/0


def test_connection_from_init_db_is_usable_from_a_different_thread():
    # Same reasoning as rag/store.py's init_db -- server.py caches this connection per
    # thread, but FastAPI runs ask() in a thread-pool worker (see server.py's
    # _get_query_log_conn), so a connection can be created on one thread and used from
    # another.
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "queries.db"
        conn = query_log.init_db(str(db_path))  # created on the main/test thread

        errors = []

        def insert_from_another_thread():
            try:
                query_log.insert_query(conn, "q", "en", num_sources=0, was_answered=False)
            except sqlite3.Error as e:
                errors.append(e)

        thread = threading.Thread(target=insert_from_another_thread)
        thread.start()
        thread.join()

        assert errors == []
