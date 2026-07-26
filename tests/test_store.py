"""Unit tests for rag/store.py: the cross-thread SQLite connection regression (2026-07-26)."""
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import store


def test_connection_from_init_db_is_usable_from_a_different_thread():
    # server.py caches one connection for the process lifetime, but FastAPI runs each
    # sync route handler in a thread-pool worker -- a request can land on a different
    # thread than the one that created the connection. Without check_same_thread=False
    # in init_db, this raised sqlite3.ProgrammingError on whichever request happened to
    # land on a different worker thread -- intermittent "sometimes works, sometimes
    # doesn't" 503s in prod (see .claude/HANDOFF.md's 2026-07-26 incident).
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        conn = store.init_db(str(db_path))  # created on the main/test thread

        errors = []

        def query_from_another_thread():
            try:
                store.get_all_chunks(conn)
            except Exception as e:
                errors.append(e)

        thread = threading.Thread(target=query_from_another_thread)
        thread.start()
        thread.join()

        assert errors == []
