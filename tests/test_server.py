"""Unit tests for server.py's /api/ask handler: the MIN_SIMILARITY relevance filter and
the backend-unavailable -> 503 error path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

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


# --- R-02: a greeting/small-talk opener has no semantic similarity to the transfer-process
# docs, so retrieval always scores it below MIN_SIMILARITY -- it must still reach the chat
# model (which has its own greeting-handling instructions, see rag/generator.py's
# SYSTEM_PROMPT_TR/EN) instead of hitting the empty short-circuit before the model ever runs.

@pytest.mark.parametrize("question,language", [
    ("ne naber", "tr"),
    ("naber", "tr"),
    ("merhaba", "tr"),
    ("hey", "en"),
    ("hi", "en"),
    ("what's up", "en"),
])
def test_ask_routes_greetings_to_the_model_instead_of_the_empty_short_circuit(monkeypatch, question, language):
    # No chunk clears MIN_SIMILARITY -- same retrieval outcome as a genuinely off-topic
    # question, but this one must NOT hit the empty short-circuit.
    chunks = [
        {"text": "Irrelevant.", "source_file": "a.md", "similarity": config.MIN_SIMILARITY - 0.01, "sources": []},
    ]
    _patch_pipeline(monkeypatch, chunks)
    monkeypatch.setattr(
        server, "answer_query",
        lambda q, kept_chunks, language, backend: {"segments": [{"txt": "warm greeting"}], "sources": []},
    )

    result = server.ask(server.AskRequest(question=question, language=language))

    assert result == {"segments": [{"txt": "warm greeting"}], "sources": []}


def test_ask_still_returns_empty_for_genuinely_off_topic_question(monkeypatch):
    # Guards against the greeting heuristic over-firing: a real (if silly) question that
    # is not a greeting match must keep the original "I don't have information" behavior.
    chunks = [
        {"text": "Irrelevant.", "source_file": "a.md", "similarity": config.MIN_SIMILARITY - 0.01, "sources": []},
    ]
    _patch_pipeline(monkeypatch, chunks)

    result = server.ask(server.AskRequest(question="what's a good cookie recipe?", language="en"))

    assert result == {"segments": [], "sources": []}


def test_looks_like_greeting_does_not_match_a_real_question_with_a_greeting_prefix():
    # "hi" followed by an actual question must still go through retrieval normally --
    # this is a whole-message match, not a substring/keyword scan.
    assert server._looks_like_greeting("hi, how much is tuition?") is False
    assert server._looks_like_greeting("hi") is True
    assert server._looks_like_greeting("Hi!") is True
    assert server._looks_like_greeting("  Merhaba  ") is True


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


# --- S-02: question length cap + per-IP rate limiting ------------------------------------
# These go through the real ASGI/TestClient path (not a direct server.ask(...) call) because
# both defenses live at the HTTP boundary: the length cap is a pydantic body-validation error
# FastAPI only raises during request parsing, and the rate limiter only activates when a real
# Request is injected (see ask()'s http_request param).

@pytest.fixture
def client(monkeypatch):
    _patch_pipeline(monkeypatch, chunks=[])
    server._request_log.clear()
    yield TestClient(server.app)
    server._request_log.clear()


def test_question_over_max_length_returns_422(client):
    response = client.post("/api/ask", json={"question": "a" * 2001, "language": "en"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "question"]


def test_question_at_max_length_is_accepted(client):
    response = client.post("/api/ask", json={"question": "a" * 2000, "language": "en"})

    assert response.status_code == 200


def test_rate_limit_returns_429_after_threshold_exceeded_within_window(client):
    for _ in range(server.RATE_LIMIT_MAX_REQUESTS):
        response = client.post("/api/ask", json={"question": "hi", "language": "en"})
        assert response.status_code == 200

    over_limit_response = client.post("/api/ask", json={"question": "hi", "language": "en"})

    assert over_limit_response.status_code == 429
    assert "Too many requests" in over_limit_response.json()["detail"]


def test_rate_limit_is_scoped_per_ip(monkeypatch, client):
    for _ in range(server.RATE_LIMIT_MAX_REQUESTS):
        client.post("/api/ask", json={"question": "hi", "language": "en"})
    assert client.post("/api/ask", json={"question": "hi", "language": "en"}).status_code == 429

    # A different client IP should not be affected by the first client's exhausted quota.
    assert server._check_rate_limit("203.0.113.7") is True


# --- S-01: the served page must reflect the *actual* active chat backend, not a hardcoded
# "Offline - Foundry Local" claim -- the public Render deployment runs RAG_CHAT_BACKEND=gemini
# (a cloud API), and telling users their questions stay local/offline when they're actually
# sent to Google's Gemini API is a privacy-transparency bug, not a cosmetic one.

def test_index_page_embeds_the_configured_chat_backend_when_gemini(monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "gemini")
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'window.__RAG_CHAT_BACKEND__ = "gemini";' in response.text
    # The raw template placeholder must never leak to the browser, and the identifier
    # name itself (which shares a substring with the placeholder) must survive intact --
    # a naive global string replace could clobber "window.__RAG_CHAT_BACKEND__" too.
    assert "__RAG_CHAT_BACKEND_VALUE__" not in response.text
    assert "window.__RAG_CHAT_BACKEND__ =" in response.text


def test_index_page_embeds_the_configured_chat_backend_when_local(monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "foundry")
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'window.__RAG_CHAT_BACKEND__ = "foundry";' in response.text


def test_index_html_direct_path_also_gets_the_backend_injected(monkeypatch):
    # StaticFiles(html=True) would otherwise also serve the raw, un-templated file directly
    # at /index.html, bypassing the injection and re-exposing the stale claim.
    monkeypatch.setattr(config, "CHAT_BACKEND", "gemini")
    client = TestClient(server.app)

    response = client.get("/index.html")

    assert response.status_code == 200
    assert 'window.__RAG_CHAT_BACKEND__ = "gemini";' in response.text
