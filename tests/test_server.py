"""Unit tests for server.py's /api/ask handler: the MIN_SIMILARITY relevance filter and
the backend-unavailable -> 503 error path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import HTTPException

import server
from rag import config


class FakeBackend:
    def generate(self, system_prompt, user_prompt, max_tokens=None):
        return "unused"


def _patch_pipeline(monkeypatch, chunks):
    monkeypatch.setattr(server, "get_chat_backend", lambda: FakeBackend())
    monkeypatch.setattr(server, "_get_embedder", lambda: object())
    monkeypatch.setattr(server, "_get_db_connection", lambda: object())
    monkeypatch.setattr(server, "get_top_chunks", lambda *a, **k: chunks)


def test_ask_returns_empty_when_best_chunk_is_below_min_similarity(monkeypatch):
    # Off-topic question ("what's your favorite color?") -- retrieval still returns
    # its top-K nearest chunks, but none of them are a real match.
    chunks = [
        {"text": "Irrelevant.", "source_file": "a.md", "similarity": config.MIN_SIMILARITY - 0.01,
         "sources": [{"name": "A", "url": "https://x/a"}]},
    ]
    _patch_pipeline(monkeypatch, chunks)

    result = server.ask(server.AskRequest(question="what's your favorite color?", language="en"))

    assert result == {"segments": [], "sources": []}


def test_ask_keeps_chunks_at_or_above_min_similarity(monkeypatch):
    chunks = [
        {"text": "Relevant.", "source_file": "a.md", "similarity": config.MIN_SIMILARITY,
         "sources": [{"name": "A", "url": "https://x/a"}]},
    ]
    _patch_pipeline(monkeypatch, chunks)
    monkeypatch.setattr(
        server, "answer_query",
        lambda question, kept_chunks, language, backend: {"segments": [{"txt": "ok"}], "sources": [], "_kept": kept_chunks},
    )

    result = server.ask(server.AskRequest(question="how much is tuition?", language="en"))

    assert result["_kept"] == chunks


def test_ask_drops_only_the_chunks_below_threshold(monkeypatch):
    good = {"text": "Relevant.", "source_file": "a.md", "similarity": 0.6, "sources": []}
    bad = {"text": "Noise.", "source_file": "b.md", "similarity": 0.1, "sources": []}
    _patch_pipeline(monkeypatch, [good, bad])
    monkeypatch.setattr(
        server, "answer_query",
        lambda question, kept_chunks, language, backend: {"segments": [], "sources": [], "_kept": kept_chunks},
    )

    result = server.ask(server.AskRequest(question="how much is tuition?", language="en"))

    assert result["_kept"] == [good]


def test_ask_returns_empty_when_all_chunks_below_threshold(monkeypatch):
    chunks = [
        {"text": "a", "source_file": "a.md", "similarity": 0.1, "sources": []},
        {"text": "b", "source_file": "b.md", "similarity": 0.2, "sources": []},
    ]
    _patch_pipeline(monkeypatch, chunks)

    result = server.ask(server.AskRequest(question="meow?", language="tr"))

    assert result == {"segments": [], "sources": []}


def test_ask_returns_503_when_the_chat_backend_is_unreachable(monkeypatch):
    def raise_connection_error():
        raise ConnectionError("Ollama not running")

    monkeypatch.setattr(server, "get_chat_backend", raise_connection_error)

    with pytest.raises(HTTPException) as exc_info:
        server.ask(server.AskRequest(question="how much is tuition?", language="en"))

    assert exc_info.value.status_code == 503


def test_ask_returns_503_when_retrieval_fails(monkeypatch):
    def raise_connection_error(*a, **k):
        raise ConnectionError("embedding backend not running")

    monkeypatch.setattr(server, "get_chat_backend", lambda: FakeBackend())
    monkeypatch.setattr(server, "_get_embedder", lambda: object())
    monkeypatch.setattr(server, "_get_db_connection", lambda: object())
    monkeypatch.setattr(server, "get_top_chunks", raise_connection_error)

    with pytest.raises(HTTPException) as exc_info:
        server.ask(server.AskRequest(question="how much is tuition?", language="en"))

    assert exc_info.value.status_code == 503
