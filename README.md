# SHSU Transfer Assistant

A fully offline, bilingual (Turkish/English) question-answering chatbot for Firat
University software engineering students transferring to Sam Houston State University.
Everything runs locally on the user's own machine — no cloud API calls, no internet
connection required after setup.

## Why this exists

Transfer students have to work through the same scattered set of admissions,
visa, tuition, and enrollment questions every year, and the answers live across
many different university pages and PDFs. This project turns that material into
a retrieval-augmented (RAG) chatbot: ask a question in Turkish or English, and it
answers using only the ingested source documents, with inline citations back to
where each piece of information came from. Running entirely on-device means it
works without an internet connection and without sending any data to a third
party.

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
4. **Generation** — the question plus the retrieved chunks are sent to a local
   chat model, which answers in the question's original language and cites its
   sources inline.
5. **UI** — a custom HTML/CSS/JS chat interface (not a generic template) is
   served by the backend, with a language toggle, chat history, and clickable
   inline citations.

## Stack

- **Backend:** Python, [FastAPI](https://fastapi.tiangolo.com/) + [uvicorn](https://www.uvicorn.org/), serving both the static frontend and a `POST /api/ask` endpoint
- **Frontend:** a custom-designed chat UI (HTML/CSS/vanilla JS, React vendored locally — no CDN dependencies, so it also works offline)
- **Storage:** SQLite (chunks, embeddings, and source metadata)
- **Local inference:** [Microsoft Foundry Local](https://github.com/microsoft/Foundry-Local) as the primary backend (embedding + chat models running entirely on-device), with [Ollama](https://ollama.com/) as a fallback backend when Foundry Local isn't available
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
backend to use, model names, similarity threshold, chunk sizes, etc.) lives in
`rag/config.py` and can be overridden with environment variables without
touching code — see that file for the full list.

## Tests

```bash
pytest
```
