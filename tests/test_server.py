"""Unit tests for server.py's /api/ask handler: the MIN_SIMILARITY relevance filter and
the backend-unavailable -> 503 error path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import HTTPException

import server
from rag import config
from rag.generator import ContentBlocked


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


def test_ask_returns_503_when_the_chat_backend_is_unreachable(monkeypatch, caplog):
    def raise_connection_error():
        raise ConnectionError("Gemini API key missing or invalid")

    monkeypatch.setattr(server, "get_chat_backend", raise_connection_error)

    with caplog.at_level("ERROR"):
        with pytest.raises(HTTPException) as exc_info:
            server.ask(server.AskRequest(question="how much is tuition?", language="en"))

    assert exc_info.value.status_code == 503
    # The detail must not name a specific backend -- the public deployment runs Gemini,
    # not the offline Foundry Local/Ollama backend the old hardcoded string assumed,
    # and that mismatch previously misdirected diagnosis of prod failures.
    assert exc_info.value.detail == "Chat backend unavailable"
    # The log line itself (not just the attached traceback) must carry the real
    # exception type/message so failures are diagnosable without a full stack trace.
    log_message = caplog.records[-1].getMessage()
    assert "ConnectionError" in log_message
    assert "Gemini API key missing or invalid" in log_message
    assert "foundry" not in log_message.lower()
    assert "ollama" not in log_message.lower()


def test_ask_returns_refusal_text_when_backend_blocks_content(monkeypatch):
    # A backend's own safety layer refusing a request is a deliberate response, not an
    # outage -- this should surface as 200 + REFUSAL_TEXT, not the generic 503 path.
    chunks = [{"text": "Relevant.", "source_file": "a.md", "similarity": config.MIN_SIMILARITY,
               "sources": [{"name": "A", "url": "https://x/a"}]}]
    _patch_pipeline(monkeypatch, chunks)

    def raise_blocked(question, kept_chunks, language, backend):
        raise ContentBlocked("Gemini declined to respond (finish_reason=SAFETY)")

    monkeypatch.setattr(server, "answer_query", raise_blocked)

    result_tr = server.ask(server.AskRequest(question="kotu bir sey soyle", language="tr"))
    assert result_tr == {"segments": [{"txt": server.REFUSAL_TEXT["tr"]}], "sources": []}

    result_en = server.ask(server.AskRequest(question="say something bad", language="en"))
    assert result_en == {"segments": [{"txt": server.REFUSAL_TEXT["en"]}], "sources": []}


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
