"""Data structures produced by the document processing module."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ConvertedDocument:
    """A PDF converted to Markdown and cached on disk.

    Attributes:
        document_id: Short content hash of the PDF. Identical PDFs share the id,
            which is what makes the cache reusable across chat turns.
        source_path: Path of the original PDF as uploaded by the user.
        markdown_path: Path of the cached ``.md`` file
            (``<cache_dir>/YYYYMMDD_HHmmss_<document_id>.md``).
        title: Best-effort title: PDF metadata, else the first Markdown heading,
            else the file name.
        page_count: Number of pages of the PDF.
        char_count: Length of the Markdown text.
        from_cache: Whether an existing conversion was reused.
    """

    document_id: str
    source_path: Path
    markdown_path: Path
    title: str
    page_count: int
    char_count: int
    from_cache: bool = False

    @property
    def file_name(self) -> str:
        """Return the original file name of the PDF."""
        return self.source_path.name


@dataclass(frozen=True)
class DocumentSection:
    """A chunk of a converted document, bounded in size for LLM consumption.

    Attributes:
        document_id: Id of the document the section belongs to.
        index: Position of the section inside the document (0-based).
        heading: Heading the section falls under, or the document title when the
            Markdown has no headings before it.
        text: Section text, without the heading line.
    """

    document_id: str
    index: int
    heading: str
    text: str

    @property
    def char_count(self) -> int:
        """Return the number of characters of the section text."""
        return len(self.text)


@dataclass(frozen=True)
class SearchHit:
    """A section matched by a lexical search.

    Attributes:
        section: Matched section.
        score: Relevance score; higher is better. Comparable only within one
            search call.
        matched_terms: Query terms that occur in the section.
        snippet: Short excerpt around the first match, for compact display.
    """

    section: DocumentSection
    score: float
    matched_terms: list[str] = field(default_factory=list)
    snippet: str = ""
