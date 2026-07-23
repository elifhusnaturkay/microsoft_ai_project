"""Basic unit tests for rag/loader.py: Source/URL header parsing + generic file dispatch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.loader import load_documents, load_markdown


def test_load_markdown_parses_source_and_url_header(tmp_path):
    content = (
        "**Source:** SHSU Cost of Attendance\n"
        "**URL:** https://www.shsu.edu/cost-aid/cost-attendance\n"
        "\n"
        "International students pay approximately $41,860 per year, covering tuition, "
        "housing, meals, and personal expenses.\n"
    )
    path = tmp_path / "tuition_fees.md"
    path.write_text(content, encoding="utf-8")

    docs = load_markdown(path)

    assert len(docs) == 1
    doc = docs[0]
    assert doc["source_name"] == "SHSU Cost of Attendance"
    assert doc["source_url"] == "https://www.shsu.edu/cost-aid/cost-attendance"
    assert "**Source:**" not in doc["text"]
    assert "**URL:**" not in doc["text"]
    assert "$41,860" in doc["text"]
    assert doc["source_file"] == "tuition_fees.md"


def test_load_markdown_without_header_falls_back_to_filename(tmp_path):
    path = tmp_path / "plain_notes.md"
    path.write_text("Just some body text with no header at all.\n", encoding="utf-8")

    docs = load_markdown(path)

    assert len(docs) == 1
    assert docs[0]["source_name"] == "plain_notes"
    assert docs[0]["source_url"] is None
    assert "Just some body text" in docs[0]["text"]


def test_load_markdown_splits_multiple_sections_with_distinct_sources(tmp_path):
    # Mirrors the real docs/*.md convention (e.g. tuition_costs.md, faq_practical.md):
    # multiple **Source:**/**URL:** blocks in one file, separated by a `---` rule.
    content = (
        "# Tuition, Fees, and Cost of Attendance\n"
        "\n"
        "**Source:** SHSU Cost of Attendance\n"
        "**URL:** https://www.shsu.edu/cost-aid/cost-attendance\n"
        "\n"
        "Total budget for a non-resident undergraduate is $41,860 per year.\n"
        "\n"
        "---\n"
        "\n"
        "**Source:** SHSU Catalog - Tuition & Fees\n"
        "**URL:** https://catalog.shsu.edu/undergraduate/financial-information/tuition-fees/\n"
        "\n"
        "Nonresident students pay $649 per semester credit hour.\n"
    )
    path = tmp_path / "tuition_costs.md"
    path.write_text(content, encoding="utf-8")

    docs = load_markdown(path)

    assert len(docs) == 2
    assert docs[0]["source_name"] == "SHSU Cost of Attendance"
    assert docs[0]["source_url"] == "https://www.shsu.edu/cost-aid/cost-attendance"
    assert "$41,860" in docs[0]["text"]
    assert "$649" not in docs[0]["text"]

    assert docs[1]["source_name"] == "SHSU Catalog - Tuition & Fees"
    assert docs[1]["source_url"] == "https://catalog.shsu.edu/undergraduate/financial-information/tuition-fees/"
    assert "$649" in docs[1]["text"]
    assert "$41,860" not in docs[1]["text"]

    # Both sections still trace back to the same physical file.
    assert {d["source_file"] for d in docs} == {"tuition_costs.md"}


def test_load_documents_reads_md_and_txt_and_skips_unsupported_extensions(tmp_path):
    (tmp_path / "doc1.md").write_text(
        "**Source:** Doc One\n**URL:** https://example.com/one\n\nBody one.\n", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text("Body two, plain txt file.\n", encoding="utf-8")
    (tmp_path / "ignored.png").write_bytes(b"\x89PNG\r\n\x1a\n")  # not a supported type
    (tmp_path / "empty.md").write_text("   \n\n  ", encoding="utf-8")  # no extractable text

    documents = load_documents(tmp_path)

    source_files = {d["source_file"] for d in documents}
    assert source_files == {"doc1.md", "doc2.txt"}

    doc1 = next(d for d in documents if d["source_file"] == "doc1.md")
    assert doc1["source_name"] == "Doc One"
    assert doc1["source_url"] == "https://example.com/one"


def test_load_documents_missing_dir_returns_empty_list(tmp_path):
    assert load_documents(tmp_path / "does_not_exist") == []
