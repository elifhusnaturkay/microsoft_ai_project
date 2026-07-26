# SHSU Transfer Assistant

A bilingual (Turkish/English) question-answering chatbot for Firat University software
engineering students transferring to Sam Houston State University. Offline by default —
embedding and chat inference run entirely on the user's own machine via Microsoft Foundry
Local (with an Ollama fallback) — with an optional cloud deploy mode (Gemini) for sharing
it as a web link.

**Live:** [bearkat-transfer-assistant.onrender.com](https://bearkat-transfer-assistant.onrender.com) (Render, cloud/Gemini mode)

## Why this exists

Transfer students have to work through the same scattered set of admissions,
visa, tuition, and enrollment questions every year, and the answers live across
many different university pages and PDFs. This project turns that material into
a retrieval-augmented (RAG) chatbot: ask a question in Turkish or English, and it
answers using only the ingested source documents, with inline citations back to
where each piece of information came from. Run locally (offline mode), it works
without an internet connection and without sending any data to a third party;
deployed as a web link (cloud mode), students can just open a URL.

## How it works

1. **Ingestion** — source documents are loaded, split into paragraph-sized
   chunks, embedded, and stored in a local SQLite database.
2. **Retrieval** — an incoming question is embedded and compared against the
   stored chunks with cosine similarity; the closest matches above a minimum
   relevance threshold are kept, everything else is discarded as noise.
3. **Translation bridge** — since the source documents are English-only,
   non-English questions are translated to English before retrieval so
   retrieval quality doesn't depend on the embedding model's cross-lingual
   ability. The original question (and language) is still what the model
   answers in.
4. **Conversation memory** — the last few turns of the current chat are folded
   into the prompt so follow-up questions ("what about housing?") resolve
   against what was just discussed. Retrieval itself still runs fresh against
   only the current question; history feeds the answering call, not what gets
   retrieved.
5. **Generation** — the question, retrieved chunks, and recent conversation
   history are sent to the configured chat model, which answers in the
   question's original language and cites its sources inline.
6. **UI** — a custom, mobile-responsive HTML/CSS/JS chat interface (not a
   generic template) is served by the backend, with a language toggle, chat
   history, and clickable inline citations.

## Stack

- **Backend:** Python, [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/), serving both the static frontend and a `POST /api/ask` endpoint
- **Frontend:** a custom-designed chat UI (HTML/CSS/vanilla JS, React vendored locally — no CDN dependencies, so offline mode also works without one)
- **Storage:** SQLite — chunks/embeddings/source metadata in one DB, plus a separate durable query log (`rag/query_log.py`) that survives re-ingestion
- **Inference backends** (`RAG_CHAT_BACKEND` / `RAG_EMBED_BACKEND` in `rag/config.py`):
  - `foundry` (default) — [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local), fully on-device
  - `ollama` — local fallback via [Ollama](https://ollama.com/) when Foundry Local isn't available
  - `gemini` — optional cloud backend ([google-genai](https://github.com/googleapis/python-genai) SDK) used by the public Render deployment; includes retry-with-backoff on transient errors and a distinct 429 response when rate-limited
- **Testing:** pytest

## Running it

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the knowledge base from the documents in `docs/`:

```bash
python scripts/ingest.py
```

Start the server:

```bash
uvicorn server:app --reload
```

Then open the served address in a browser. All configuration (which inference
backend to use, model names, similarity threshold, chunk sizes, conversation
history length, etc.) lives in `rag/config.py` and can be overridden with
environment variables without touching code — see that file for the full list.

To run in cloud mode instead of the offline default, set `GEMINI_API_KEY` plus:

```bash
RAG_CHAT_BACKEND=gemini RAG_EMBED_BACKEND=gemini uvicorn server:app --reload
```

## Tests

```bash
pytest
```
