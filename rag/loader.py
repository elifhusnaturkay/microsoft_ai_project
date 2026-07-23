"""Loads raw text + source metadata from everything under docs/.

Handles two kinds of files generically:
- .pdf  -> extracted via pypdf. PDFs carry no embedded URL, so source_url is None.
- .md/.txt -> read as plain text, one or more `**Source:**`/`**URL:**` header blocks,
  each followed by its own body text and separated by a `---` horizontal rule:

      **Source:** SHSU Cost of Attendance
      **URL:** https://www.shsu.edu/cost-aid/cost-attendance

      International students pay approximately $41,860 per year...

      ---

      **Source:** SHSU Catalog - Tuition & Fees
      **URL:** https://catalog.shsu.edu/...

      Per semester credit hour...

  This is the actual convention the collected docs/*.md files use (verified against the
  real files, several of which -- e.g. faq_practical.md, tuition_costs.md -- cite a
  different source per section within the same file). Each `---`-delimited section is
  therefore treated as its own logical document with its own source_name/source_url, so
  chunking never blends two sections' text under one citation. A file with no `---`
  separators and no header at all is treated as a single section, falling back to the
  filename as source_name (e.g. hand-written notes with no citation convention).
"""
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only when pypdf truly isn't installed
    PdfReader = None

SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
SUPPORTED_PDF_SUFFIXES = {".pdf"}

_SOURCE_RE = re.compile(r"^\*\*Source:\*\*\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_URL_RE = re.compile(r"^\*\*URL:\*\*\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SECTION_SPLIT_RE = re.compile(r"^\s*---\s*$", re.MULTILINE)


def _parse_source_header(text: str) -> (Optional[str], Optional[str]):
    """Pull source_name/source_url out of `**Source:**` / `**URL:**` header lines, if present."""
    source_name = None
    source_url = None

    m = _SOURCE_RE.search(text)
    if m:
        source_name = m.group(1).strip() or None

    m = _URL_RE.search(text)
    if m:
        source_url = m.group(1).strip() or None

    return source_name, source_url


def _strip_header_lines(text: str) -> str:
    """Remove **Source:**/**URL:** header lines from the body so they aren't duplicated in chunks."""
    kept_lines = [
        line
        for line in text.splitlines()
        if not _SOURCE_RE.match(line) and not _URL_RE.match(line)
    ]
    return "\n".join(kept_lines).strip()


def load_markdown(path: Path) -> List[Dict]:
    """Load a .md/.txt file as one or more `---`-separated sections, each with its own
    Source/URL header if present. Returns one document dict per non-empty section."""
    raw = Path(path).read_text(encoding="utf-8")
    fallback_name = Path(path).stem

    documents = []
    for section in _SECTION_SPLIT_RE.split(raw):
        source_name, source_url = _parse_source_header(section)
        body = _strip_header_lines(section)
        if not body.strip():
            continue
        documents.append(
            {
                "text": body,
                "source_file": Path(path).name,
                "source_name": source_name or fallback_name,
                "source_url": source_url,
            }
        )
    return documents


def load_pdf(path: Path) -> Dict:
    """Load a .pdf file via pypdf. PDFs have no embedded URL, so source_url is None."""
    if PdfReader is None:
        raise ImportError(
            "pypdf is required to load PDF files. Install it with `pip install pypdf` "
            "(it's listed in requirements.txt)."
        )
    reader = PdfReader(str(path))
    page_texts = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(t for t in page_texts if t.strip())
    return {
        "text": text,
        "source_file": Path(path).name,
        "source_name": Path(path).stem,
        "source_url": None,
    }


def load_document(path: Path) -> List[Dict]:
    """Dispatch on file extension. Returns [] for unsupported file types (skipped, not an
    error) or empty files. Always returns a list -- one .md/.txt file may yield multiple
    section-documents; a .pdf always yields at most one."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_PDF_SUFFIXES:
        doc = load_pdf(path)
        return [doc] if doc["text"].strip() else []
    if suffix in SUPPORTED_TEXT_SUFFIXES:
        return load_markdown(path)
    return []


def load_documents(docs_dir) -> List[Dict]:
    """Load every supported file under docs_dir (recursively), skipping unsupported types.

    Returns a flat list of dicts: {text, source_file, source_name, source_url} -- one per
    section for multi-source .md/.txt files, one per file for .pdf. Empty
    documents/sections (no extractable text) are skipped.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []

    documents = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        documents.extend(load_document(path))
    return documents
