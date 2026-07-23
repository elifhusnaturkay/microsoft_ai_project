"""Paragraph-based chunking, ~200-500 words per chunk (PROJECT_PLAN.md locked decision).

Strategy: within a single physical file, flatten every section's paragraphs into one
ordered stream (each paragraph tagged with the source it came from -- a file may have
several `**Source:**` sections, see loader.py), then greedily pack consecutive paragraphs
into a chunk until adding the next paragraph would push it past MAX_CHUNK_WORDS -- at which
point the chunk is closed off (as long as it has already reached MIN_CHUNK_WORDS) and a new
chunk starts. A paragraph that is itself larger than MAX_CHUNK_WORDS is split on word
boundaries so no single chunk ever exceeds the max.

Packing across section boundaries (rather than forcing a chunk break at every source
change) is deliberate: many of the collected docs/*.md sections are much shorter than
MIN_CHUNK_WORDS on their own (e.g. faq_practical.md's per-question sections), and forcing
a chunk-per-section would produce a lot of tiny, under-sized chunks. Instead, a chunk
carries a `sources` list (one entry per distinct source that contributed a paragraph to
it, in first-seen order) rather than a single source_name/source_url -- so citation
accuracy is preserved even when a chunk blends paragraphs pulled from more than one
source. Packing never crosses a *file* boundary (chunk_documents groups by source_file
first), and a hard-split of an oversized paragraph always attributes to that paragraph's
single source only.

The trailing chunk of a file may end up shorter than MIN_CHUNK_WORDS -- that's expected
and fine (it's whatever text is left over), the bound we actually guarantee is the upper one.
"""
import re
from typing import Dict, List, Optional, Tuple

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


def _source_key(document: Dict) -> Dict:
    return {"name": document.get("source_name"), "url": document.get("source_url")}


def _dedupe_sources(sources: List[Dict]) -> List[Dict]:
    """Dedupe {'name', 'url'} dicts, preserving first-seen order."""
    seen = set()
    deduped = []
    for src in sources:
        key = (src.get("name"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(src)
    return deduped


def _tag_paragraphs(documents: List[Dict]) -> List[Tuple[List[str], Dict]]:
    """Flatten a group of same-file documents (sections) into an ordered list of
    (paragraph_words, source) tuples, one per paragraph."""
    items = []
    for document in documents:
        source = _source_key(document)
        for para in split_paragraphs(document.get("text", "")):
            items.append((para.split(), source))
    return items


def _pack_tagged_paragraphs(
    items: List[Tuple[List[str], Dict]],
    min_words: int,
    max_words: int,
) -> List[Dict]:
    """Same greedy packing algorithm as chunk_text, but each paragraph carries a source
    tag and each emitted chunk carries the deduped list of sources it drew from."""
    chunks: List[Dict] = []
    current: List[Tuple[List[str], Dict]] = []
    current_word_count = 0

    def flush():
        if not current:
            return
        words = [w for para_words, _ in current for w in para_words]
        sources = _dedupe_sources([source for _, source in current])
        chunks.append({"text": " ".join(words), "sources": sources, "word_count": len(words)})

    for para_words, source in items:
        if len(para_words) > max_words:
            flush()
            current = []
            current_word_count = 0
            # Oversized paragraph: hard-split on word count. It came from exactly one
            # source, so every resulting sub-chunk attributes to that source only.
            for i in range(0, len(para_words), max_words):
                sub_words = para_words[i : i + max_words]
                chunks.append({"text": " ".join(sub_words), "sources": [source], "word_count": len(sub_words)})
            continue

        if current and current_word_count + len(para_words) > max_words:
            if current_word_count >= min_words:
                flush()
                current = [(para_words, source)]
                current_word_count = len(para_words)
            else:
                current.append((para_words, source))
                current_word_count += len(para_words)
                flush()
                current = []
                current_word_count = 0
        else:
            current.append((para_words, source))
            current_word_count += len(para_words)

    flush()
    return chunks


def chunk_document_group(
    documents: List[Dict],
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[Dict]:
    """Chunk every section belonging to the SAME source_file together, letting small
    adjacent sections merge into properly-sized chunks. `documents` must all share the
    same source_file (see chunk_documents, which groups before calling this)."""
    if not documents:
        return []

    source_file = documents[0].get("source_file")
    items = _tag_paragraphs(documents)
    packed = _pack_tagged_paragraphs(items, min_words=min_words, max_words=max_words)
    return [
        {
            "text": c["text"],
            "source_file": source_file,
            "sources": c["sources"],
            "word_count": c["word_count"],
        }
        for c in packed
    ]


def chunk_document(
    document: Dict,
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[Dict]:
    """Chunk a single loaded document/section in isolation (no merging with siblings).
    Each chunk carries a one-entry `sources` list for that document's source_name/url."""
    return chunk_document_group([document], min_words=min_words, max_words=max_words)


def chunk_documents(
    documents: List[Dict],
    min_words: int = config.MIN_CHUNK_WORDS,
    max_words: int = config.MAX_CHUNK_WORDS,
) -> List[Dict]:
    """Chunk a list of loaded documents/sections, grouping by source_file first (preserving
    first-seen file order) so small adjacent sections from the same file can merge into
    properly-sized chunks -- packing never crosses a file boundary. Each resulting chunk
    carries a `sources` list (one entry per distinct source it drew from)."""
    groups: "Dict[Optional[str], List[Dict]]" = {}
    file_order: List[Optional[str]] = []
    for document in documents:
        key = document.get("source_file")
        if key not in groups:
            groups[key] = []
            file_order.append(key)
        groups[key].append(document)

    all_chunks: List[Dict] = []
    for key in file_order:
        all_chunks.extend(chunk_document_group(groups[key], min_words=min_words, max_words=max_words))
    return all_chunks
