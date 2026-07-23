"""Embedding backend abstraction.

Two backends implement the `Embedder` interface so the rest of the pipeline (ingest.py,
retriever.py) never hardcodes a specific SDK call:

- FoundryLocalEmbedder: qwen3-embedding-0.6b via Microsoft Foundry Local (primary, locked
  decision per PROJECT_PLAN.md).
- OllamaEmbedder: fallback if Foundry Local has issues (also a locked decision, see
  PROJECT_PLAN.md "Yedek Plan").

TODO -- verify before real ingestion runs:
    PROJECT_PLAN.md's own environment-check snippet only demonstrates loading a *chat*
    model via `FoundryLocalManager().get_model("phi-3.5-mini")`; it does not document how
    to request *embeddings* for qwen3-embedding-0.6b. The implementation below assumes
    Foundry Local exposes an OpenAI-compatible local endpoint (a documented pattern for
    Foundry Local's chat models) and that the same endpoint serves `.embeddings.create(...)`
    for the embedding model. That assumption is NOT independently confirmed in this pass --
    once `foundry-local-sdk` is actually installed and running, check its real API
    (`FoundryLocalManager`, `manager.endpoint`, `manager.api_key`, `manager.get_model_info`)
    and adjust `FoundryLocalEmbedder._ensure_client` / `.embed` accordingly. If Foundry Local
    doesn't serve qwen3-embedding-0.6b this way, switch RAG_EMBED_BACKEND=ollama in the
    meantime (see rag/config.py) -- the rest of the pipeline doesn't care which backend
    is behind this interface.
"""
from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np

from . import config


class Embedder(ABC):
    """Backend-agnostic embedding interface. Implementations must return unit-shaped
    (or at least consistently-shaped) float vectors; cosine similarity in store.py
    normalizes at query time so exact scale doesn't matter."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of texts. Returns an array of shape (len(texts), embedding_dim)."""
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper for embedding a single string (e.g. a user query)."""
        return self.embed([text])[0]


class FoundryLocalEmbedder(Embedder):
    """qwen3-embedding-0.6b via Microsoft Foundry Local. See module TODO above."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.LOCAL_EMBED_MODEL
        self._client = None
        self._model_id = None

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            from foundry_local import FoundryLocalManager
        except ImportError as e:
            raise RuntimeError(
                "foundry-local-sdk is not installed. Run `pip install foundry-local-sdk` "
                "(see requirements.txt), or set RAG_EMBED_BACKEND=ollama to use the fallback."
            ) from e
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The `openai` package is required to talk to Foundry Local's local "
                "OpenAI-compatible endpoint. Run `pip install openai`."
            ) from e

        # TODO: confirm this against the real foundry-local-sdk API -- see module docstring.
        manager = FoundryLocalManager(self.model_name)
        self._model_id = manager.get_model_info(self.model_name).id
        self._client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_client()
        # TODO: confirm Foundry Local actually serves qwen3-embedding-0.6b through
        # `.embeddings.create` on the OpenAI-compatible endpoint -- see module docstring.
        response = self._client.embeddings.create(model=self._model_id, input=list(texts))
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


class OllamaEmbedder(Embedder):
    """Fallback embedding backend using Ollama's local HTTP API (PROJECT_PLAN.md "Yedek Plan").

    Uses `requests` directly against `POST /api/embeddings` rather than the `ollama` pip
    package, so the fallback path only needs a dependency that's already in requirements.txt.
    """

    def __init__(self, model_name: str = None, host: str = None):
        self.model_name = model_name or config.OLLAMA_EMBED_MODEL
        self.host = host or config.OLLAMA_HOST

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import requests

        vectors: List[List[float]] = []
        for text in texts:
            resp = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=60,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return np.array(vectors, dtype=np.float32)


def get_embedder(backend: str = None) -> Embedder:
    """Factory: returns the configured Embedder implementation.

    `backend` defaults to config.EMBED_BACKEND ("foundry" or "ollama"), overridable via the
    RAG_EMBED_BACKEND env var or by passing it explicitly (e.g. from a CLI flag).
    """
    backend = (backend or config.EMBED_BACKEND).lower()
    if backend == "foundry":
        return FoundryLocalEmbedder()
    if backend == "ollama":
        return OllamaEmbedder()
    raise ValueError(f"Unknown embed backend: {backend!r} (expected 'foundry' or 'ollama')")
