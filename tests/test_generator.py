"""Unit tests for rag/generator.py: source numbering + inline [n] citation parsing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.generator import answer_query, build_context_and_sources, parse_segments


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
        def generate(self, system_prompt, user_prompt):
            assert "[1]" in user_prompt
            return "Tuition is $41,860 [1]."

    result = answer_query("How much is tuition?", chunks, language="en", backend=FakeBackend())

    assert result["sources"] == [{"name": "Cost of Attendance", "url": "https://x/a"}]
    assert result["segments"] == [{"txt": "Tuition is $41,860 "}, {"c": 1}, {"txt": "."}]
