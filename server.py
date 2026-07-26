#!/usr/bin/env python3
"""FastAPI app: serves the chat UI in static/ (adapted from the Claude Design handoff
bundle) and exposes POST /api/ask, backed by the real RAG pipeline (rag/retriever.py +
rag/generator.py). Replaces the Streamlit shell (app.py) -- the delivered UI design is a
custom HTML/CSS/JS single-page app that doesn't map onto Streamlit's widget model, so this
project now ships its own frontend instead (see static/index.html's business-logic script
for the client side of this).

Run with: uvicorn server:app --reload
(Requires knowledge.db to already exist -- run `python scripts/ingest.py` first.)
"""
import logging
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from threading import Lock, local
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import config, query_log, store
from rag.embedder import get_embedder
from rag.generator import ContentBlocked, answer_query, get_chat_backend, translate_query_for_retrieval
from rag.retriever import get_top_chunks

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger(__name__)

# Small talk/greetings ("ne naber", "hey", "merhaba") have no semantic overlap with the
# transfer-process docs, so retrieval always scores them below MIN_SIMILARITY -- without
# this check they'd hit the empty short-circuit below before the chat model, which already
# has greeting-handling instructions in its system prompt (see rag/generator.py's
# SYSTEM_PROMPT_TR/EN), ever runs. Matched as the WHOLE normalized message, not a
# substring/keyword scan, so a real question that happens to start with a greeting word
# (e.g. "hi, how much is tuition?") still goes through retrieval normally and can still
# get the correct "I don't have information on that" refusal if it's genuinely off-topic.
_GREETING_PHRASES = {
    "hi", "hey", "hello", "hiya", "yo", "sup", "howdy",
    "what's up", "whats up", "how are you", "how's it going", "hows it going",
    "good morning", "good afternoon", "good evening",
    "naber", "n'aber", "ne naber", "ne haber", "nasilsin", "nasılsın",
    "nasilsiniz", "nasılsınız", "selam", "selamlar", "merhaba", "merhabalar",
    "gunaydin", "günaydın", "iyi aksamlar", "iyi akşamlar", "iyi geceler",
}


def _looks_like_greeting(question: str) -> bool:
    """True for short small-talk openers matched against the WHOLE (normalized)
    message -- see _GREETING_PHRASES above for why this must not be a substring scan."""
    normalized = re.sub(r"[!.,?¿¡]+$", "", question.strip().lower()).strip()
    return normalized in _GREETING_PHRASES

# Shown when a backend's own safety layer refuses a request (see rag/generator.py's
# ContentBlocked) -- a deliberate response, not the generic "backend unavailable" error.
REFUSAL_TEXT = {
    "tr": "Buna hiç yakıştıramadım.",
    "en": "I really can't see myself doing that.",
}

app = FastAPI(title="SHSU Transfer Assistant")

_embedder = None
_db_conn_local = local()
_query_log_conn_local = local()


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_query_log_conn():
    """One SQLite connection per thread -- same reasoning as _get_db_connection below
    (ask() runs in a FastAPI thread-pool worker, not always the thread that opened it)."""
    if not hasattr(_query_log_conn_local, "conn"):
        _query_log_conn_local.conn = query_log.init_db(config.QUERY_LOG_DB_PATH)
    return _query_log_conn_local.conn


def _get_db_connection():
    """One SQLite connection per thread, not one shared globally.

    FastAPI runs sync route handlers (ask() is `def`, not `async def`) in a thread-pool,
    and a single connection shared across threads broke two different ways: first, using
    it from a thread other than the one that created it raised outright (fixed by
    check_same_thread=False in rag/store.py's init_db) -- but that flag only lifts
    Python's same-thread check, it does not make ONE connection object safe for multiple
    threads to query at the *same instant*. A synthetic burst of 6 concurrent requests
    still failed 5/6 even with that flag, and real concurrent traffic (a genuine question
    landing at the same moment as Render's own health-check GET / on another worker
    thread) reproduced it live -- see .claude/HANDOFF.md's 2026-07-26 incident. Giving each
    thread its own dedicated connection removes the sharing entirely, so there's nothing
    left to race on; SQLite's file-level locking already handles concurrent reads safely
    across separate connections.
    """
    if not hasattr(_db_conn_local, "conn"):
        _db_conn_local.conn = store.init_db(config.DB_PATH)
    return _db_conn_local.conn


class HistoryMessage(BaseModel):
    role: str
    text: str = Field(max_length=2000)


class AskRequest(BaseModel):
    # Bounded so an unauthenticated caller can't send an arbitrarily large body -- each
    # question triggers a billed embedding + completion call against the configured backend.
    # 2000 chars is generous for a chat question while keeping a single request cheap.
    question: str = Field(max_length=2000)
    language: str = "tr"
    # Prior turns of the CURRENT chat (static/index.html sends its own client-side message
    # list) so follow-up questions can refer back to what was just discussed -- see
    # rag/generator.py's build_history_block. Bounded independently of config.
    # MAX_HISTORY_MESSAGES (which trims from the *end*) so a caller can't force this app to
    # hold an arbitrarily large request body in memory regardless of what's actually used.
    history: List[HistoryMessage] = Field(default_factory=list, max_length=50)


# Per-IP sliding-window rate limit for POST /api/ask, kept dependency-free (no slowapi/redis)
# since this runs as a single Render instance -- an in-process dict is sufficient and avoids
# adding a new third-party package to a live deployment. Not shared across multiple instances
# or process restarts; that's an accepted v1 limitation, not a correctness bug for this app's
# current single-instance deployment.
RATE_LIMIT_MAX_REQUESTS = 15
RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_lock = Lock()
_request_log: dict = defaultdict(deque)


def _is_gemini_quota_error(e: Exception) -> bool:
    """True for the Gemini SDK's 429 (RESOURCE_EXHAUSTED/quota) error specifically --
    distinguishing it from a genuine backend outage matters because it's transient and
    self-resolving, not a bug to page someone about. Before this, both collapsed into the
    same generic 503 in the except-block below, which is exactly what made a real quota
    exhaustion (from concurrent testing against the shared prod GEMINI_API_KEY) look
    identical to the actual "gemini-2.5-flash retired" outage it had just been fixed
    from -- see .claude/HANDOFF.md's 2026-07-26 note. google-genai is imported lazily
    here (not at module level) so this stays a no-op for the foundry/ollama backends,
    matching rag/generator.py's own lazy-import pattern for the same package."""
    try:
        from google.genai.errors import APIError
    except ImportError:
        return False
    return isinstance(e, APIError) and e.code == 429


def _check_rate_limit(client_ip: str) -> bool:
    """Record a request from client_ip and return whether it's within the allowed rate.
    Returns False (and does NOT record the request) once client_ip already has
    RATE_LIMIT_MAX_REQUESTS timestamps within the trailing RATE_LIMIT_WINDOW_SECONDS."""
    now = time.monotonic()
    with _rate_limit_lock:
        timestamps = _request_log[client_ip]
        while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:
            timestamps.popleft()
        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            return False
        timestamps.append(now)
        return True


def _log_query(question: str, language: str, num_sources: int, was_answered: bool) -> None:
    """Best-effort durable log of a question asked through /api/ask (see rag/query_log.py).
    Never allowed to break the request it's logging -- a full disk or similar here would
    otherwise turn an observability nice-to-have into a user-facing failure, so any error
    is swallowed and only surfaced via a warning log."""
    try:
        conn = _get_query_log_conn()
        query_log.insert_query(conn, question, language, num_sources, was_answered)
    except Exception:
        logger.warning("Failed to record query log entry", exc_info=True)


@app.post("/api/ask")
def ask(request: AskRequest, http_request: Request = None):
    # http_request defaults to None so existing direct-call unit tests (server.ask(AskRequest(...)))
    # keep working without a real ASGI request; FastAPI itself always injects the real Request
    # for actual HTTP traffic, which is the only path rate-limiting needs to cover.
    if http_request is not None:
        client_ip = http_request.client.host if http_request.client else "unknown"
        if not _check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many requests -- please wait a minute before asking again.",
            )

    question = request.question.strip()
    language = request.language if request.language in ("tr", "en") else "tr"
    if not question:
        return {"segments": [], "sources": []}

    try:
        chat_backend = get_chat_backend()

        # docs/ is English-only; retrieve using an English version of the question so
        # retrieval quality doesn't depend on the embedding backend's cross-lingual ability
        # (see rag/generator.py's translate_query_for_retrieval docstring for measurements).
        # The ORIGINAL question is still what answer_query sees below.
        retrieval_query = question
        if language != "en":
            retrieval_query = translate_query_for_retrieval(question, chat_backend)

        embedder = _get_embedder()
        conn = _get_db_connection()
        chunks = get_top_chunks(retrieval_query, embedder, conn, k=config.TOP_K)
        chunks = [c for c in chunks if c["similarity"] >= config.MIN_SIMILARITY]
        # A greeting still short-circuits an empty *retrieval* result, but not the model
        # call itself -- the model has its own greeting-handling instructions and will
        # ignore the (empty) sources per the system prompt. Genuinely off-topic questions
        # (not a greeting match) keep the original behavior below.
        if not chunks and not _looks_like_greeting(question):
            _log_query(question, language, num_sources=0, was_answered=False)
            return {"segments": [], "sources": []}

        history = [{"role": m.role, "text": m.text} for m in request.history]
        response = answer_query(question, chunks, language=language, backend=chat_backend, history=history)
        _log_query(
            question, language,
            num_sources=len(response.get("sources", [])),
            was_answered=bool(response.get("segments")),
        )
        return response
    except ContentBlocked:
        _log_query(question, language, num_sources=0, was_answered=True)
        return {"segments": [{"txt": REFUSAL_TEXT[language]}], "sources": []}
    except Exception as e:
        # Any failure here means the configured chat/embedding backend (local Foundry
        # Local/Ollama, or the Gemini backend used on the public Render deployment) didn't
        # respond -- not a bug in this app's own logic. Surface a distinct status so the
        # frontend's existing "couldn't reach the chat backend" message (static/index.html's
        # fetchAnswer) is accurate, and log the real cause for whoever's diagnosing it.
        logger.exception("Chat backend failed while answering a question: %s: %s", type(e).__name__, e)
        _log_query(question, language, num_sources=0, was_answered=False)
        if _is_gemini_quota_error(e):
            raise HTTPException(
                status_code=429,
                detail="Chat backend is rate-limited -- please wait a minute before asking again.",
            )
        raise HTTPException(status_code=503, detail="Chat backend unavailable")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index() -> HTMLResponse:
    # S-01: static/index.html's connection-status badge used to hardcode "Offline - Foundry
    # Local" text, which is false on the public Render deployment (RAG_CHAT_BACKEND=gemini --
    # see rag/config.py). Inject the *actual* configured chat backend into the page so the
    # frontend (see static/index.html's business-logic script, near
    # `__RAG_CHAT_BACKEND_VALUE__`) can show an honest label instead of a claim that doesn't
    # match where questions go.
    # (Also registered at /index.html, not just /, so the raw un-templated file underneath
    # the StaticFiles mount below can't be reached directly and re-expose the stale claim.)
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("__RAG_CHAT_BACKEND_VALUE__", config.CHAT_BACKEND)
    return HTMLResponse(html)


# Registered last: explicit routes above (like /api/ask and /) always win over this
# catch-all mount, but anything else (styles.css, vendor/*, uploads/*, ...) falls through
# to the static asset directory.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
