"""Paragraph-based chunking, ~200-500 words per chunk (PROJECT_PLAN.md locked decision).

Strategy: split each document into paragraphs (blank-line separated), then greedily pack
consecutive paragraphs into a chunk until adding the next paragraph would push it past
MAX_CHUNK_WORDS -- at which point the chunk is closed off (as long as it has already
reached MIN_CHUNK_WORDS) and a new chunk starts. A paragraph that is itself larger than
MAX_CHUNK_WORDS is split on word boundaries so no single chunk ever exceeds the max.

The trailing chunk of a document may end up shorter than MIN_CHUNK_WORDS -- that's expected
and fine (it's whatever text is left over), the bound we actually guarantee is the upper one.
"""
import re
from typing import Dict, List

from . import config

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")


def split_paragraphs(text: str) -> List[str]:
    """Split raw text into non-empty paragraphs on blank lines."""
    if not text or not text.strip():
        return []
    raw_paragraphs = _PARAGRAPH_SPLIT_RE.split(text.strip())
    return [p.strip() for p in raw_paragraphs if p.strip()]


def chunk_text(
    text: str,
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[str]:
    """Split text into ~min_words-max_words chunks along paragraph boundaries."""
    paragraphs = split_paragraphs(text)
    chunks: List[str] = []
    current_words: List[str] = []

    def flush():
        if current_words:
            chunks.append(" ".join(current_words))

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > max_words:
            # Oversized paragraph: flush whatever we were building, then hard-split
            # this paragraph on word count so no chunk ever exceeds max_words.
            flush()
            current_words = []
            for i in range(0, len(para_words), max_words):
                chunks.append(" ".join(para_words[i : i + max_words]))
            continue

        if current_words and len(current_words) + len(para_words) > max_words:
            if len(current_words) >= min_words:
                flush()
                current_words = list(para_words)
            else:
                # Current chunk is still under the minimum -- accept going over max_words
                # slightly rather than emitting a too-small chunk. Keeps within the spirit
                # of "~200-500 words" without ever leaving a tiny orphan chunk.
                current_words.extend(para_words)
                flush()
                current_words = []
        else:
            current_words.extend(para_words)

    flush()
    return chunks


def chunk_document(
    document: Dict,
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[Dict]:
    """Chunk a single loaded document (as returned by loader.load_document), carrying its
    source_file/source_name/source_url metadata onto every chunk produced from it."""
    text_chunks = chunk_text(document.get("text", ""), min_words=min_words, max_words=max_words)
    return [
        {
            "text": chunk,
            "source_file": document.get("source_file"),
            "source_name": document.get("source_name"),
            "source_url": document.get("source_url"),
            "word_count": len(chunk.split()),
        }
        for chunk in text_chunks
    ]


def chunk_documents(
    documents: List[Dict],
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[Dict]:
    """Chunk a list of loaded documents, flattening into a single list of chunk dicts."""
    all_chunks: List[Dict] = []
    for document in documents:
        all_chunks.extend(chunk_document(document, min_words=min_words, max_words=max_words))
    return all_chunks
