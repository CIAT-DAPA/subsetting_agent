"""Unit tests for the document processing module (PDF -> Markdown -> store)."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pymupdf
import pytest

from document_processing import (
    ConvertedDocument,
    DocumentStore,
    PdfConversionError,
    build_cache_file_name,
    compute_document_id,
    convert_pdf_to_markdown,
    find_cached_markdown,
)

CACHE_NAME_PATTERN = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{12}\.md$")


def write_pdf(path: Path, pages: list[list[tuple[str, int]]], title: str | None = None) -> Path:
    """Create a small text PDF for tests.

    Args:
        path: Where to write the PDF.
        pages: One list per page of ``(text, font_size)`` lines.
        title: Optional metadata title.
    """
    document = pymupdf.open()

    # Each page gets its blocks laid out top to bottom inside wrapped text
    # boxes, so long paragraphs stay within the page and remain extractable;
    # large font sizes are detected as headings by pymupdf4llm.
    for lines in pages:
        page = document.new_page()
        y = 72.0

        for text, size in lines:
            chars_per_line = max(int(468 / (size * 0.5)), 1)
            # Text is written in slices that fit the remaining page height; a
            # new page is opened whenever the current one is full.
            chars_per_page = chars_per_line * int(640 / (size * 1.4))
            remaining = text

            while remaining:
                if y > 640:
                    page = document.new_page()
                    y = 72.0

                slice_, remaining = remaining[:chars_per_page], remaining[chars_per_page:]
                line_count = len(slice_) // chars_per_line + 1
                height = line_count * size * 1.4 + size
                rect = pymupdf.Rect(72, y, 540, min(y + height, 770))
                page.insert_textbox(rect, slice_, fontsize=size)
                y += height + size

    if title is not None:
        document.set_metadata({"title": title})

    document.save(path)
    document.close()

    return path


@pytest.fixture
def paper_pdf(tmp_path: Path) -> Path:
    """A two-page PDF resembling a short paper with headings and body text."""
    body = "Accessions collected in dry regions showed higher drought tolerance. " * 6
    methods = "We evaluated 120 bean accessions from Genesys under water stress. " * 6
    results = "Landraces from Mexico had the best yield under heat stress conditions. " * 6

    return write_pdf(
        tmp_path / "paper.pdf",
        [
            [("Drought tolerance in bean landraces", 20), (body, 10)],
            [("Methods", 16), (methods, 10), ("Results", 16), (results, 10)],
        ],
    )


class TestCacheNaming:
    """Cache files follow ``YYYYMMDD_HHmmss_<hash>.md``."""

    def test_build_cache_file_name_format(self) -> None:
        """The timestamp is compact (no colons) and the hash is appended."""
        name = build_cache_file_name("abcdef123456", datetime(2026, 9, 15, 14, 32, 5))

        assert name == "20260915_143205_abcdef123456.md"
        assert CACHE_NAME_PATTERN.match(name)

    def test_document_id_is_content_hash(self, tmp_path: Path) -> None:
        """Identical bytes give the same id; different bytes a different one."""
        first = tmp_path / "a.pdf"
        second = tmp_path / "b.pdf"
        third = tmp_path / "c.pdf"
        first.write_bytes(b"%PDF-1.4 same")
        second.write_bytes(b"%PDF-1.4 same")
        third.write_bytes(b"%PDF-1.4 other")

        assert compute_document_id(first) == compute_document_id(second)
        assert compute_document_id(first) != compute_document_id(third)
        assert len(compute_document_id(first)) == 12

    def test_find_cached_markdown_returns_newest(self, tmp_path: Path) -> None:
        """When several timestamps exist for a hash, the newest wins."""
        (tmp_path / "20260101_000000_abcdef123456.md").write_text("old")
        (tmp_path / "20260915_120000_abcdef123456.md").write_text("new")
        (tmp_path / "20260915_120000_ffffffffffff.md").write_text("other")

        found = find_cached_markdown(tmp_path, "abcdef123456")

        assert found is not None
        assert found.name == "20260915_120000_abcdef123456.md"

    def test_find_cached_markdown_missing(self, tmp_path: Path) -> None:
        """No file for the hash (or no directory) yields ``None``."""
        assert find_cached_markdown(tmp_path, "abcdef123456") is None
        assert find_cached_markdown(tmp_path / "nope", "abcdef123456") is None


class TestConvertPdfToMarkdown:
    """Conversion validates input, writes the cache and reuses it."""

    def test_converts_and_caches(self, paper_pdf: Path, tmp_path: Path) -> None:
        """The Markdown is written with the cache naming and metadata is filled."""
        cache = tmp_path / "cache"

        document = convert_pdf_to_markdown(paper_pdf, cache)

        assert isinstance(document, ConvertedDocument)
        assert document.from_cache is False
        assert document.page_count == 2
        assert document.char_count > 0
        assert document.markdown_path.parent == cache
        assert CACHE_NAME_PATTERN.match(document.markdown_path.name)
        assert document.markdown_path.name.endswith(f"_{document.document_id}.md")
        assert "drought tolerance" in document.markdown_path.read_text().lower()

    def test_second_conversion_uses_cache(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Converting the same PDF again reuses the file instead of rewriting it."""
        cache = tmp_path / "cache"
        first = convert_pdf_to_markdown(paper_pdf, cache)

        second = convert_pdf_to_markdown(paper_pdf, cache)

        assert second.from_cache is True
        assert second.markdown_path == first.markdown_path
        assert len(list(cache.glob("*.md"))) == 1

    def test_force_reconverts(self, paper_pdf: Path, tmp_path: Path) -> None:
        """``force=True`` ignores the cache and produces a fresh conversion."""
        cache = tmp_path / "cache"
        convert_pdf_to_markdown(paper_pdf, cache)

        again = convert_pdf_to_markdown(paper_pdf, cache, force=True)

        assert again.from_cache is False

    def test_title_from_metadata(self, tmp_path: Path) -> None:
        """PDF metadata title takes precedence over headings."""
        pdf = write_pdf(
            tmp_path / "meta.pdf", [[("Some heading", 20), ("body " * 20, 10)]], title="Meta Title"
        )

        document = convert_pdf_to_markdown(pdf, tmp_path / "cache")

        assert document.title == "Meta Title"

    def test_title_falls_back_to_heading_or_stem(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Without metadata the title comes from the first heading or the file name."""
        document = convert_pdf_to_markdown(paper_pdf, tmp_path / "cache")

        assert document.title in ("Drought tolerance in bean landraces", "Methods", "paper")

    def test_missing_file(self, tmp_path: Path) -> None:
        """A path that does not exist is rejected with a clear error."""
        with pytest.raises(PdfConversionError, match="not found"):
            convert_pdf_to_markdown(tmp_path / "missing.pdf", tmp_path)

    def test_wrong_extension(self, tmp_path: Path) -> None:
        """Only ``.pdf`` files are accepted."""
        text = tmp_path / "notes.txt"
        text.write_text("hello")

        with pytest.raises(PdfConversionError, match="Not a PDF"):
            convert_pdf_to_markdown(text, tmp_path)

    def test_empty_file(self, tmp_path: Path) -> None:
        """A zero-byte PDF is rejected before conversion."""
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")

        with pytest.raises(PdfConversionError, match="empty"):
            convert_pdf_to_markdown(empty, tmp_path)

    def test_corrupt_file(self, tmp_path: Path) -> None:
        """A file with the right extension but invalid content is rejected."""
        corrupt = tmp_path / "corrupt.pdf"
        corrupt.write_bytes(b"this is not a pdf at all")

        with pytest.raises(PdfConversionError, match="Cannot open"):
            convert_pdf_to_markdown(corrupt, tmp_path)


class TestDocumentStore:
    """The store indexes documents into sections and searches them."""

    def test_add_pdf_and_list(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Added documents are listed with their sections available."""
        store = DocumentStore(tmp_path / "cache")

        document = store.add_pdf(paper_pdf)

        assert not store.is_empty
        assert [d.document_id for d in store.list_documents()] == [document.document_id]
        assert len(store.get_outline(document.document_id)) >= 1

    def test_adding_same_pdf_twice_is_idempotent(self, paper_pdf: Path, tmp_path: Path) -> None:
        """The same content is indexed once."""
        store = DocumentStore(tmp_path / "cache")
        first = store.add_pdf(paper_pdf)

        second = store.add_pdf(paper_pdf)

        assert first is second
        assert len(store.list_documents()) == 1

    def test_add_pdfs_collects_errors(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Bad files are reported without preventing good ones from loading."""
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"garbage")
        store = DocumentStore(tmp_path / "cache")

        added, errors = store.add_pdfs([paper_pdf, bad, tmp_path / "missing.pdf"])

        assert len(added) == 1
        assert len(errors) == 2

    def test_sections_are_bounded(self, tmp_path: Path) -> None:
        """No section exceeds ``max_section_chars``."""
        long_body = " ".join(f"Sentence number {i} about accessions." for i in range(400))
        pdf = write_pdf(tmp_path / "long.pdf", [[("Long paper", 20), (long_body, 8)]])
        store = DocumentStore(tmp_path / "cache", max_section_chars=600, min_section_chars=100)

        document = store.add_pdf(pdf)
        outline = store.get_outline(document.document_id)

        assert len(outline) > 1
        assert all(chars <= 600 for _, _, chars in outline)
        assert [index for index, _, _ in outline] == list(range(len(outline)))

    def test_read_section_and_bounds(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Sections are readable by index; bad indexes raise ``IndexError``."""
        store = DocumentStore(tmp_path / "cache")
        document = store.add_pdf(paper_pdf)

        section = store.read_section(document.document_id, 0)

        assert section.document_id == document.document_id
        assert section.text

        with pytest.raises(IndexError, match="does not exist"):
            store.read_section(document.document_id, 999)

    def test_unknown_document_id(self, tmp_path: Path) -> None:
        """Unknown ids raise ``KeyError`` listing the known ids."""
        store = DocumentStore(tmp_path / "cache")

        with pytest.raises(KeyError, match="Unknown document id"):
            store.get_outline("nope")

    def test_search_finds_relevant_section(self, paper_pdf: Path, tmp_path: Path) -> None:
        """Query terms rank the section that contains them first."""
        store = DocumentStore(tmp_path / "cache")
        document = store.add_pdf(paper_pdf)

        hits = store.search("heat stress yield Mexico")

        assert hits
        assert hits[0].section.document_id == document.document_id
        assert "mexico" in hits[0].matched_terms
        assert "Mexico" in hits[0].section.text
        assert hits[0].snippet

    def test_search_ignores_stop_words_and_unknown_terms(
        self, paper_pdf: Path, tmp_path: Path
    ) -> None:
        """Stop-word-only queries and absent terms give no hits."""
        store = DocumentStore(tmp_path / "cache")
        store.add_pdf(paper_pdf)

        assert store.search("the and of") == []
        assert store.search("photosynthesis chlorophyll") == []

    def test_search_restricted_to_document(self, paper_pdf: Path, tmp_path: Path) -> None:
        """``document_id`` limits the hits to one document."""
        other = write_pdf(
            tmp_path / "other.pdf",
            [[("Cassava report", 20), ("Cassava accessions tolerate drought well. " * 8, 10)]],
        )
        store = DocumentStore(tmp_path / "cache")
        paper = store.add_pdf(paper_pdf)
        cassava = store.add_pdf(other)

        all_hits = store.search("drought", top_k=10)
        only_cassava = store.search("drought", top_k=10, document_id=cassava.document_id)

        assert {h.section.document_id for h in all_hits} == {paper.document_id, cassava.document_id}
        assert {h.section.document_id for h in only_cassava} == {cassava.document_id}

    def test_top_k_limits_results(self, tmp_path: Path) -> None:
        """At most ``top_k`` hits are returned."""
        long_body = " ".join(f"Drought paragraph {i}." for i in range(200))
        pdf = write_pdf(tmp_path / "many.pdf", [[("Many", 20), (long_body, 8)]])
        store = DocumentStore(tmp_path / "cache", max_section_chars=300, min_section_chars=50)
        store.add_pdf(pdf)

        assert len(store.search("drought", top_k=2)) == 2

    def test_invalid_bounds(self, tmp_path: Path) -> None:
        """Inconsistent section bounds are rejected."""
        with pytest.raises(ValueError):
            DocumentStore(tmp_path, max_section_chars=100, min_section_chars=200)

    def test_clear_keeps_cache_and_cleanup_removes_it(
        self, paper_pdf: Path, tmp_path: Path
    ) -> None:
        """``clear`` forgets documents only; ``cleanup`` also deletes the cache."""
        cache = tmp_path / "cache"
        store = DocumentStore(cache)
        store.add_pdf(paper_pdf)

        store.clear()

        assert store.is_empty
        assert cache.is_dir()

        store.cleanup()

        assert not cache.exists()
