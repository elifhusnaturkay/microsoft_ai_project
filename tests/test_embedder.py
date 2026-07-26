"""Unit tests for rag/embedder.py: backend factory selection (no real network calls)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.embedder import FoundryLocalEmbedder, GeminiEmbedder, OllamaEmbedder, get_embedder


def test_get_embedder_foundry():
    assert isinstance(get_embedder("foundry"), FoundryLocalEmbedder)


def test_get_embedder_ollama():
    assert isinstance(get_embedder("ollama"), OllamaEmbedder)


def test_get_embedder_gemini():
    assert isinstance(get_embedder("gemini"), GeminiEmbedder)


def test_get_embedder_unknown_raises_with_all_three_names():
    with pytest.raises(ValueError, match="foundry.*ollama.*gemini"):
        get_embedder("nonexistent")


class _FakeEmbedding:
    def __init__(self, values):
        self.values = values


class _FakeEmbedContentResult:
    def __init__(self, values):
        self.embeddings = [_FakeEmbedding(values)]


class _FakeEmbedModels:
    def __init__(self, side_effects):
        self._side_effects = list(side_effects)
        self.call_count = 0

    def embed_content(self, model, contents):
        self.call_count += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeEmbedClient:
    def __init__(self, side_effects):
        self.models = _FakeEmbedModels(side_effects)


def _gemini_embedder_with_side_effects(side_effects):
    embedder = GeminiEmbedder(model_name="gemini-embedding-001")
    client = _FakeEmbedClient(side_effects)
    embedder._client = client  # bypasses _ensure_client's genai.Client() construction
    return embedder, client


def test_gemini_embedder_retries_on_429_then_succeeds(monkeypatch):
    # Every /api/ask call embeds its query BEFORE the chat step runs -- an unprotected
    # embed() call is a single point of failure even if the chat step tolerates blips
    # (see .claude/HANDOFF.md's 2026-07-26 incident).
    from google.genai.errors import ClientError

    monkeypatch.setattr("rag.embedder.time.sleep", lambda seconds: None)
    quota_error = ClientError(429, {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"})
    embedder, client = _gemini_embedder_with_side_effects(
        [quota_error, _FakeEmbedContentResult([0.1, 0.2, 0.3])]
    )

    result = embedder.embed(["some query"])

    assert result.shape == (1, 3)
    assert client.models.call_count == 2


def test_gemini_embedder_does_not_retry_on_400_bad_argument(monkeypatch):
    from google.genai.errors import ClientError

    monkeypatch.setattr("rag.embedder.time.sleep", lambda seconds: (_ for _ in ()).throw(
        AssertionError("should not sleep/retry on a non-retryable 400")
    ))
    bad_arg_error = ClientError(400, {"message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"})
    embedder, client = _gemini_embedder_with_side_effects([bad_arg_error])

    with pytest.raises(ClientError):
        embedder.embed(["some query"])
    assert client.models.call_count == 1
