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
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config, store
from rag.embedder import get_embedder
from rag.generator import answer_query
from rag.retriever import get_top_chunks

STATIC_DIR = Path(__file__).resolve().parent / "static"

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

    embedder = _get_embedder()
    conn = _get_db_connection()
    chunks = get_top_chunks(question, embedder, conn, k=config.TOP_K)
    if not chunks:
        return {"segments": [], "sources": []}

    return answer_query(question, chunks, language=language)


# Registered last: an explicit route above (like /api/ask) always wins over this
# catch-all mount, but anything else falls through to the static chat UI.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
