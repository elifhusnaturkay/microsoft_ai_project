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
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config, store
from rag.embedder import get_embedder
from rag.generator import ContentBlocked, answer_query, get_chat_backend, translate_query_for_retrieval
from rag.retriever import get_top_chunks

STATIC_DIR = Path(__file__).resolve().parent / "static"

logger = logging.getLogger(__name__)

# Shown when a backend's own safety layer refuses a request (see rag/generator.py's
# ContentBlocked) -- a deliberate response, not the generic "backend unavailable" error.
REFUSAL_TEXT = {
    "tr": "Buna hiç yakıştıramadım.",
    "en": "I really can't see myself doing that.",
}

app = FastAPI(title="SHSU Transfer Assistant")

_embedder = None
_db_conn = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_db_connection():
    global _db_conn
    if _db_conn is None:
        _db_conn = store.init_db(config.DB_PATH)
    return _db_conn


class AskRequest(BaseModel):
    question: str
    language: str = "tr"


@app.post("/api/ask")
def ask(request: AskRequest):
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
        if not chunks:
            return {"segments": [], "sources": []}

        return answer_query(question, chunks, language=language, backend=chat_backend)
    except ContentBlocked:
        return {"segments": [{"txt": REFUSAL_TEXT[language]}], "sources": []}
    except Exception as e:
        # Any failure here means the configured chat/embedding backend (local Foundry
        # Local/Ollama, or the Gemini backend used on the public Render deployment) didn't
        # respond -- not a bug in this app's own logic. Surface a distinct status so the
        # frontend's existing "couldn't reach the chat backend" message (static/index.html's
        # fetchAnswer) is accurate, and log the real cause for whoever's diagnosing it.
        logger.exception("Chat backend failed while answering a question: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=503, detail="Chat backend unavailable")


# Registered last: an explicit route above (like /api/ask) always wins over this
# catch-all mount, but anything else falls through to the static chat UI.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
