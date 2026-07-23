"""Basic unit tests for rag/chunker.py: paragraph splitting + word-count bounds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunker import chunk_document, chunk_documents, chunk_text, split_paragraphs


def _words(word: str, count: int) -> str:
    return " ".join([word] * count)


def test_split_paragraphs_splits_on_blank_lines():
    text = "First paragraph here.\n\nSecond paragraph here.\n\n\nThird one, extra blank line."
    paragraphs = split_paragraphs(text)
    assert paragraphs == [
        "First paragraph here.",
        "Second paragraph here.",
        "Third one, extra blank line.",
    ]


def test_split_paragraphs_empty_text_returns_empty_list():
    assert split_paragraphs("") == []
    assert split_paragraphs("   \n\n  ") == []


def test_chunk_text_short_document_is_a_single_chunk_within_bounds():
    text = _words("lorem", 250)
    chunks = chunk_text(text, min_words=200, max_words=500)
    assert len(chunks) == 1
    assert 1 <= len(chunks[0].split()) <= 500


def test_chunk_text_long_document_splits_into_multiple_chunks_within_bounds():
    # 6 paragraphs x 150 words = 900 words total -> should split into >= 2 chunks.
    paragraphs = [_words("word", 150) for _ in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, min_words=200, max_words=500)

    assert len(chunks) >= 2
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 500, f"chunk exceeded max_words: {word_count}"
    # every chunk except possibly the last should meet the minimum
    for chunk in chunks[:-1]:
        assert len(chunk.split()) >= 200


def test_chunk_text_oversized_single_paragraph_is_hard_split_by_word_count():
    text = _words("word", 1200)
    chunks = chunk_text(text, min_words=200, max_words=500)
    assert len(chunks) == 3  # 1200 / 500 -> 500, 500, 200
    for chunk in chunks:
        assert len(chunk.split()) <= 500


def test_chunk_document_preserves_source_metadata_and_word_count():
    document = {
        "text": _words("word", 250),
        "source_file": "tuition_fees.md",
        "source_name": "SHSU Cost of Attendance",
        "source_url": "https://www.shsu.edu/cost-aid/cost-attendance",
    }
    chunks = chunk_document(document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk["source_file"] == "tuition_fees.md"
    assert chunk["source_name"] == "SHSU Cost of Attendance"
    assert chunk["source_url"] == "https://www.shsu.edu/cost-aid/cost-attendance"
    assert chunk["word_count"] == 250


def test_chunk_documents_aggregates_across_multiple_documents():
    documents = [
        {"text": _words("a", 220), "source_file": "d1.md", "source_name": "D1", "source_url": None},
        {"text": _words("b", 220), "source_file": "d2.md", "source_name": "D2", "source_url": None},
    ]
    chunks = chunk_documents(documents)

    assert len(chunks) == 2
    assert {c["source_file"] for c in chunks} == {"d1.md", "d2.md"}
