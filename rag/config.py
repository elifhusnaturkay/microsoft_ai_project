"""Shared constants for the RAG pipeline.

Everything here can be overridden with an environment variable so the same
code runs unchanged on macOS and Windows, and so tests/CI can point at a
throwaway DB path without touching source.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = Path(os.environ.get("RAG_DOCS_DIR", str(PROJECT_ROOT / "docs")))
DB_PATH = os.environ.get("RAG_DB_PATH", str(PROJECT_ROOT / "knowledge.db"))

# Retrieval
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))

# Chunking (paragraph-based, ~200-500 words per PROJECT_PLAN.md)
MIN_CHUNK_WORDS = int(os.environ.get("RAG_MIN_CHUNK_WORDS", "200"))
MAX_CHUNK_WORDS = int(os.environ.get("RAG_MAX_CHUNK_WORDS", "500"))

# Backend selection: "foundry" (primary) or "ollama" (fallback, see PROJECT_PLAN.md "Yedek Plan")
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "foundry")
CHAT_BACKEND = os.environ.get("RAG_CHAT_BACKEND", "foundry")

# Foundry Local model names (locked decisions, PROJECT_PLAN.md)
LOCAL_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "qwen3-embedding-0.6b")
LOCAL_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "phi-3.5-mini")

# Ollama fallback model names -- these are guesses at reasonable equivalents, not locked
# decisions. Adjust to whatever is actually pulled locally (`ollama pull ...`).
OLLAMA_EMBED_MODEL = os.environ.get("RAG_OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.environ.get("RAG_OLLAMA_CHAT_MODEL", "phi3.5")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
