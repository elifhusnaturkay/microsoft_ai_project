"""Unit tests for rag/generator.py: source numbering + inline [n] citation parsing."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generator import (
    ContentBlocked,
    FoundryLocalChat,
    GeminiChat,
    OllamaChat,
    answer_query,
    build_context_and_sources,
    get_chat_backend,
    parse_segments,
    translate_query_for_retrieval,
)


def test_build_context_and_sources_numbers_by_distinct_source_not_chunk():
    chunks = [
        {"text": "Tuition info.", "source_file": "a.md", "sources": [{"name": "Cost of Attendance", "url": "https://x/a"}]},
        {"text": "Housing info.", "source_file": "b.md", "sources": [{"name": "Housing Options", "url": "https://x/b"}]},
    ]
    context, sources = build_context_and_sources(chunks)

    assert sources == [
        {"name": "Cost of Attendance", "url": "https://x/a"},
        {"name": "Housing Options", "url": "https://x/b"},
    ]
    assert "[1]\nTuition info." in context
    assert "[2]\nHousing info." in context


def test_build_context_and_sources_dedupes_repeated_source_across_chunks():
    chunks = [
        {"text": "First.", "source_file": "a.md", "sources": [{"name": "Same Source", "url": "https://x/a"}]},
        {"text": "Second.", "source_file": "a.md", "sources": [{"name": "Same Source", "url": "https://x/a"}]},
    ]
    context, sources = build_context_and_sources(chunks)

    assert len(sources) == 1
    assert "[1]\nFirst." in context
    assert "[1]\nSecond." in context


def test_build_context_and_sources_chunk_with_multiple_sources_gets_multiple_numbers():
    # Mirrors chunker.py merging several small sections into one chunk.
    chunks = [
        {
            "text": "Merged paragraph.",
            "source_file": "faq.md",
            "sources": [{"name": "Source A", "url": "https://x/a"}, {"name": "Source B", "url": "https://x/b"}],
        }
    ]
    context, sources = build_context_and_sources(chunks)

    assert len(sources) == 2
    assert "[1][2]\nMerged paragraph." in context


def test_parse_segments_splits_text_and_citation():
    segments = parse_segments("Cost is $41,860 [1]. More info [2].", num_sources=2)
    assert segments == [
        {"txt": "Cost is $41,860 "},
        {"c": 1},
        {"txt": ". More info "},
        {"c": 2},
        {"txt": "."},
    ]


def test_parse_segments_adjacent_citations_in_one_sentence():
    segments = parse_segments("Required [1][2].", num_sources=2)
    assert segments == [{"txt": "Required "}, {"c": 1}, {"c": 2}, {"txt": "."}]


def test_parse_segments_no_citations_returns_single_text_segment():
    segments = parse_segments("I don't have information on that.", num_sources=0)
    assert segments == [{"txt": "I don't have information on that."}]


def test_parse_segments_out_of_range_number_is_left_as_literal_text():
    # Model hallucinated source [5] when only 2 sources exist -- must not crash or
    # produce a chip pointing at a nonexistent source.
    segments = parse_segments("Some fact [5].", num_sources=2)
    assert segments == [{"txt": "Some fact [5]."}]


def test_answer_query_returns_segments_and_sources_shape():
    chunks = [
        {"text": "Tuition is $41,860.", "source_file": "a.md", "sources": [{"name": "Cost of Attendance", "url": "https://x/a"}]},
    ]

    class FakeBackend:
        def generate(self, system_prompt, user_prompt, max_tokens=None):
            assert "[1]" in user_prompt
            return "Tuition is $41,860 [1]."

    result = answer_query("How much is tuition?", chunks, language="en", backend=FakeBackend())

    assert result["sources"] == [{"name": "Cost of Attendance", "url": "https://x/a"}]
    assert result["segments"] == [{"txt": "Tuition is $41,860 "}, {"c": 1}, {"txt": "."}]


def test_answer_query_drops_uncited_sources_and_renumbers_the_rest():
    # Retrieval can pass back chunks the model never actually uses -- e.g. it answers
    # "I don't have information on that," or the message is a greeting with no citations
    # at all. `sources` must reflect what was actually cited, not everything retrieved
    # (showing a pile of unrelated sources under a non-answer reads as a bug).
    chunks = [
        {"text": "Irrelevant chunk.", "source_file": "a.md", "sources": [{"name": "A", "url": "https://x/a"}]},
        {"text": "Tuition is $41,860.", "source_file": "b.md", "sources": [{"name": "B", "url": "https://x/b"}]},
        {"text": "Also irrelevant.", "source_file": "c.md", "sources": [{"name": "C", "url": "https://x/c"}]},
    ]

    class FakeBackend:
        def generate(self, system_prompt, user_prompt, max_tokens=None):
            return "Tuition is $41,860 [2]."  # only cites the 2nd of 3 retrieved sources

    result = answer_query("How much is tuition?", chunks, language="en", backend=FakeBackend())

    assert result["sources"] == [{"name": "B", "url": "https://x/b"}]
    assert result["segments"] == [{"txt": "Tuition is $41,860 "}, {"c": 1}, {"txt": "."}]


def test_answer_query_returns_no_sources_when_answer_has_no_citations():
    # e.g. a greeting, or "I don't have information on that" -- both are uncited by design.
    chunks = [
        {"text": "Tuition is $41,860.", "source_file": "a.md", "sources": [{"name": "A", "url": "https://x/a"}]},
    ]

    class FakeBackend:
        def generate(self, system_prompt, user_prompt, max_tokens=None):
            return "Hey! How can I help with your transfer to SHSU?"

    result = answer_query("hey", chunks, language="en", backend=FakeBackend())

    assert result["sources"] == []
    assert result["segments"] == [{"txt": "Hey! How can I help with your transfer to SHSU?"}]


def test_translate_query_for_retrieval_returns_backend_output():
    class FakeBackend:
        def generate(self, system_prompt, user_prompt, max_tokens=None):
            assert user_prompt == "yillik maliyet ne kadar?"
            assert "English" in system_prompt
            return "What is the annual cost?"

    result = translate_query_for_retrieval("yillik maliyet ne kadar?", FakeBackend())
    assert result == "What is the annual cost?"


def test_translate_query_for_retrieval_falls_back_to_original_on_empty_response():
    class EmptyBackend:
        def generate(self, system_prompt, user_prompt, max_tokens=None):
            return "   "

    result = translate_query_for_retrieval("some question", EmptyBackend())
    assert result == "some question"


def test_get_chat_backend_foundry():
    assert isinstance(get_chat_backend("foundry"), FoundryLocalChat)


def test_get_chat_backend_ollama():
    assert isinstance(get_chat_backend("ollama"), OllamaChat)


def test_get_chat_backend_gemini():
    assert isinstance(get_chat_backend("gemini"), GeminiChat)


def test_get_chat_backend_unknown_raises_with_all_three_names():
    with pytest.raises(ValueError, match="foundry.*ollama.*gemini"):
        get_chat_backend("nonexistent")


class _FakeCandidate:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


class _FakeGeminiResponse:
    def __init__(self, text, finish_reason=None):
        self.text = text
        self.candidates = [_FakeCandidate(finish_reason)] if finish_reason is not None else []


class _FakeModels:
    def __init__(self, side_effects):
        # Each entry is either a response object to return, or an Exception instance
        # to raise -- consumed in order, one per generate_content() call.
        self._side_effects = list(side_effects)
        self.last_config = None
        self.call_count = 0

    def generate_content(self, model, contents, config):
        self.last_config = config
        self.call_count += 1
        effect = self._side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


class _FakeGeminiClient:
    def __init__(self, side_effects):
        self.models = _FakeModels(side_effects)


def _gemini_chat_with_fake_response(response) -> tuple[GeminiChat, _FakeGeminiClient]:
    return _gemini_chat_with_side_effects([response])


def _gemini_chat_with_side_effects(side_effects) -> tuple[GeminiChat, _FakeGeminiClient]:
    chat = GeminiChat(model_name="gemini-flash-latest")
    client = _FakeGeminiClient(side_effects)
    chat._client = client  # bypasses _ensure_client's genai.Client() construction
    return chat, client


def test_gemini_chat_uses_thinking_level_minimal_not_thinking_budget():
    # thinking_budget=0 (the old hard-disable knob) started raising 400 INVALID_ARGUMENT
    # once gemini-flash-latest moved to a newer model family that doesn't support it --
    # this pins the fix (thinking_level=MINIMAL) so a future SDK/model swap can't silently
    # regress back to the field that broke prod (see .claude/HANDOFF.md's 2026-07-26 note).
    from google.genai import types

    chat, client = _gemini_chat_with_fake_response(_FakeGeminiResponse("An answer."))
    result = chat.generate("system", "user", max_tokens=100)

    assert result == "An answer."
    thinking_config = client.models.last_config.thinking_config
    assert thinking_config.thinking_level == types.ThinkingLevel.MINIMAL
    assert thinking_config.thinking_budget is None


def test_gemini_chat_raises_runtime_error_when_thinking_exhausts_the_token_budget():
    from google.genai import types

    chat, _ = _gemini_chat_with_fake_response(
        _FakeGeminiResponse(None, finish_reason=types.FinishReason.MAX_TOKENS)
    )

    # MAX_TOKENS must NOT raise ContentBlocked -- that would show the user the bot's
    # in-character refusal text for what is actually a token-budget failure.
    with pytest.raises(RuntimeError) as exc_info:
        chat.generate("system", "user", max_tokens=60)
    assert not isinstance(exc_info.value, ContentBlocked)


def test_gemini_chat_still_raises_content_blocked_on_safety_finish_reason():
    from google.genai import types

    chat, _ = _gemini_chat_with_fake_response(
        _FakeGeminiResponse(None, finish_reason=types.FinishReason.SAFETY)
    )

    with pytest.raises(ContentBlocked):
        chat.generate("system", "user", max_tokens=100)


def test_gemini_chat_retries_on_429_then_succeeds(monkeypatch):
    # Concurrent real users can trip Gemini's per-minute quota transiently (see
    # .claude/HANDOFF.md's 2026-07-26 incident) -- a request that would otherwise fail
    # outright should recover if the very next attempt succeeds.
    from google.genai.errors import ClientError

    monkeypatch.setattr("rag.generator.time.sleep", lambda seconds: None)
    quota_error = ClientError(429, {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"})
    chat, client = _gemini_chat_with_side_effects([quota_error, _FakeGeminiResponse("Recovered.")])

    result = chat.generate("system", "user", max_tokens=100)

    assert result == "Recovered."
    assert client.models.call_count == 2


def test_gemini_chat_gives_up_after_max_attempts_of_persistent_429(monkeypatch):
    from google.genai.errors import ClientError

    monkeypatch.setattr("rag.generator.time.sleep", lambda seconds: None)
    quota_error = ClientError(429, {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"})
    chat, client = _gemini_chat_with_side_effects([quota_error, quota_error, quota_error, quota_error])

    with pytest.raises(ClientError):
        chat.generate("system", "user", max_tokens=100)
    assert client.models.call_count == 3  # default max_attempts, not endless


def test_gemini_chat_does_not_retry_on_400_bad_argument(monkeypatch):
    # A bad argument (e.g. an incompatible thinking_config) will fail identically on
    # every attempt -- retrying just adds latency for a guaranteed-to-fail request.
    from google.genai.errors import ClientError

    monkeypatch.setattr("rag.generator.time.sleep", lambda seconds: (_ for _ in ()).throw(
        AssertionError("should not sleep/retry on a non-retryable 400")
    ))
    bad_arg_error = ClientError(400, {"message": "Request contains an invalid argument.", "status": "INVALID_ARGUMENT"})
    chat, client = _gemini_chat_with_side_effects([bad_arg_error])

    with pytest.raises(ClientError):
        chat.generate("system", "user", max_tokens=100)
    assert client.models.call_count == 1
