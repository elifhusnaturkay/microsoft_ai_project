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
