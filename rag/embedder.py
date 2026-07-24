"""Embedding backend abstraction.

Two backends implement the `Embedder` interface so the rest of the pipeline (ingest.py,
retriever.py) never hardcodes a specific SDK call:

- FoundryLocalEmbedder: qwen3-embedding-0.6b via Microsoft Foundry Local (primary, locked
  decision per PROJECT_PLAN.md).
- OllamaEmbedder: fallback if Foundry Local has issues (also a locked decision, see
  PROJECT_PLAN.md "Yedek Plan").

Verified against the real foundry-local-sdk==0.5.1 source (downloaded and inspected the
wheel directly -- see requirements.txt for why that exact version is pinned): its
`FoundryLocalManager(alias, bootstrap=True)` constructor asserts Foundry Local is
installed, starts the service, downloads and loads the given model, and exposes
`.endpoint` (an OpenAI-compatible `{service_uri}/v1` base URL good for chat AND embedding
models alike) plus `.api_key` and `.get_model_info(alias).id` (the concrete loaded model
ID required by the OpenAI-compatible request payload). `FoundryLocalEmbedder._ensure_client`
below matches that API exactly. Note: foundry-local-sdk had a breaking API redesign at
1.0.0 (package renamed foundry_local_sdk, singleton Configuration/native-interop design) --
this module is written against the pre-1.0 API on purpose, see requirements.txt.
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

        manager = FoundryLocalManager(self.model_name)
        self._model_id = manager.get_model_info(self.model_name, raise_on_not_found=True).id
        self._client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_client()
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


class GeminiEmbedder(Embedder):
    """gemini-embedding-001 via the Gemini API -- optional cloud-deploy path
    (RAG_EMBED_BACKEND=gemini), not the offline default. `genai.Client()` reads
    GEMINI_API_KEY from the environment itself.

    Embeds one text per call, like OllamaEmbedder above -- some gemini-embedding model
    versions aggregate a batch `contents` list into a single vector instead of returning
    one per input, which would silently corrupt the vector store.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.GEMINI_EMBED_MODEL
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return

        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Run `pip install google-genai` "
                "(see requirements.txt), or set RAG_EMBED_BACKEND=foundry/ollama instead."
            ) from e

        self._client = genai.Client()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_client()

        vectors: List[List[float]] = []
        for text in texts:
            result = self._client.models.embed_content(model=self.model_name, contents=text)
            vectors.append(result.embeddings[0].values)
        return np.array(vectors, dtype=np.float32)


def get_embedder(backend: str = None) -> Embedder:
    """Factory: returns the configured Embedder implementation.

    `backend` defaults to config.EMBED_BACKEND ("foundry", "ollama", or "gemini"), overridable
    via the RAG_EMBED_BACKEND env var or by passing it explicitly (e.g. from a CLI flag).
    """
    backend = (backend or config.EMBED_BACKEND).lower()
    if backend == "foundry":
        return FoundryLocalEmbedder()
    if backend == "ollama":
        return OllamaEmbedder()
    if backend == "gemini":
        return GeminiEmbedder()
    raise ValueError(f"Unknown embed backend: {backend!r} (expected 'foundry', 'ollama', or 'gemini')")
