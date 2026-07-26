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

# Below this cosine similarity, a chunk is treated as noise rather than a real match --
# guards against off-topic/nonsense questions (e.g. "what's your favorite color?") getting
# an answer synthesized from irrelevant context. Measured via gemma3:12b-translated queries
# on this project's docs: genuine questions scored ~0.44-0.69, off-topic ones ~0.35-0.43.
MIN_SIMILARITY = float(os.environ.get("RAG_MIN_SIMILARITY", "0.42"))

# Generation length caps (speed lever: local-model latency scales with output token count).
# Measured on an M2/16GB via Ollama: an uncapped, verbose answer took ~50s; the query
# translation step only ever needs a short phrase, not a paragraph, so it gets a much
# tighter cap. Both overridable per-deployment/hardware.
MAX_ANSWER_TOKENS = int(os.environ.get("RAG_MAX_ANSWER_TOKENS", "300"))
MAX_TRANSLATE_TOKENS = int(os.environ.get("RAG_MAX_TRANSLATE_TOKENS", "60"))

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
# gemma3:12b over phi3.5: measured on an M2 with both models pulled, phi3.5 is faster
# (~25-50s vs ~50-90s) but its Turkish translation/generation is unreliable enough to
# misread real questions (mistranslated a genuine registration question outright).
# Override via RAG_OLLAMA_CHAT_MODEL if speed matters more than Turkish quality.
OLLAMA_CHAT_MODEL = os.environ.get("RAG_OLLAMA_CHAT_MODEL", "gemma3:12b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Gemini model names -- optional cloud-deploy path (RAG_CHAT_BACKEND/RAG_EMBED_BACKEND=gemini),
# not the offline default. See rag/generator.py's GeminiChat / rag/embedder.py's GeminiEmbedder.
GEMINI_CHAT_MODEL = os.environ.get("RAG_GEMINI_CHAT_MODEL", "gemini-flash-latest")
GEMINI_EMBED_MODEL = os.environ.get("RAG_GEMINI_EMBED_MODEL", "gemini-embedding-001")
