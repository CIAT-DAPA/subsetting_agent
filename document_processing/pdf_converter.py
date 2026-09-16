"""Conversion of PDF files to Markdown with an on-disk cache.

Business rule: PDFs uploaded by the user are converted to Markdown in a
temporary file first; the agent only ever reads the Markdown. Conversions are
cached under ``<cache_dir>/YYYYMMDD_HHmmss_<hash>.md`` so that the same PDF is
not converted again on every chat turn.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pymupdf
import pymupdf4llm

from document_processing.models import ConvertedDocument

logger = logging.getLogger(__name__)

# Number of hexadecimal characters of the SHA-256 digest kept as document id.
HASH_LENGTH = 12

# Timestamp format of the cache file name (Windows forbids ':' in file names).
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Name of the environment variable that overrides the cache directory.
CACHE_DIR_ENV = "DOCUMENT_CACHE_DIR"

# Markdown heading at the start of a line, used to detect the title.
_HEADING_PATTERN = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


class PdfConversionError(Exception):
    """Raised when a PDF cannot be read or converted to Markdown."""


def default_cache_dir() -> Path:
    """Return the cache directory, from ``DOCUMENT_CACHE_DIR`` or the system temp dir."""
    configured = os.getenv(CACHE_DIR_ENV)

    # An explicit directory wins; otherwise a dedicated folder under the system
    # temporary directory keeps conversions apart from other temp files.
    if configured:
        return Path(configured)

    return Path(tempfile.gettempdir()) / "subsetting_agent_documents"


def compute_document_id(pdf_path: Path) -> str:
    """Return the short SHA-256 digest of a file's content.

    Args:
        pdf_path: File to hash.
    """
    digest = hashlib.sha256()

    # Read in blocks so large PDFs do not have to be loaded fully in memory.
    with pdf_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()[:HASH_LENGTH]


def build_cache_file_name(document_id: str, moment: datetime | None = None) -> str:
    """Build the ``YYYYMMDD_HHmmss_<hash>.md`` cache file name.

    Args:
        document_id: Content hash of the PDF.
        moment: Conversion time; defaults to now.
    """
    stamp = (moment or datetime.now()).strftime(TIMESTAMP_FORMAT)

    return f"{stamp}_{document_id}.md"


def find_cached_markdown(cache_dir: Path, document_id: str) -> Path | None:
    """Return the most recent cached Markdown for a document id, if any.

    Args:
        cache_dir: Directory where conversions are stored.
        document_id: Content hash of the PDF.
    """
    if not cache_dir.is_dir():
        return None

    # Several timestamps may exist for the same hash (e.g. a manual re-run);
    # the lexicographically greatest name is the newest thanks to the format.
    candidates = sorted(cache_dir.glob(f"*_{document_id}.md"))

    if not candidates:
        return None

    return candidates[-1]


def extract_title(markdown: str, metadata_title: str | None, fallback: str) -> str:
    """Pick the best available title for a converted document.

    Args:
        markdown: Converted Markdown text.
        metadata_title: Title from the PDF metadata, possibly empty.
        fallback: Value used when neither metadata nor headings give a title.
    """
    # PDF metadata is the most reliable source when authors filled it in.
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()

    match = _HEADING_PATTERN.search(markdown)

    # The first Markdown heading is the next best guess.
    if match:
        return match.group(1).strip()

    return fallback


def convert_pdf_to_markdown(
    pdf_path: str | Path,
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> ConvertedDocument:
    """Convert a PDF to Markdown, reusing a cached conversion when available.

    Args:
        pdf_path: PDF uploaded by the user.
        cache_dir: Where the ``.md`` files live; defaults to :func:`default_cache_dir`.
        force: Convert again even if a cached file exists.

    Returns:
        Metadata of the converted document, including the Markdown path.

    Raises:
        PdfConversionError: If the file is missing, empty, not a PDF or unreadable.
    """
    source = Path(pdf_path)
    target_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()

    # Validate the input before touching the converter, so errors are explicit.
    if not source.is_file():
        raise PdfConversionError(f"PDF not found: {source}")

    if source.suffix.lower() != ".pdf":
        raise PdfConversionError(f"Not a PDF file: {source.name}")

    if source.stat().st_size == 0:
        raise PdfConversionError(f"PDF is empty: {source.name}")

    target_dir.mkdir(parents=True, exist_ok=True)
    document_id = compute_document_id(source)

    # Reading page count and metadata is cheap and needed for both paths.
    try:
        with pymupdf.open(source) as pdf:
            page_count = pdf.page_count
            metadata_title = (pdf.metadata or {}).get("title")

    except Exception as exc:
        raise PdfConversionError(f"Cannot open PDF {source.name}: {exc}") from exc

    cached = None if force else find_cached_markdown(target_dir, document_id)

    # A cached conversion is reused as-is; only the metadata is recomputed.
    if cached is not None:
        markdown = cached.read_text(encoding="utf-8")
        logger.info("Reusing cached Markdown for %s: %s", source.name, cached.name)

        return ConvertedDocument(
            document_id=document_id,
            source_path=source,
            markdown_path=cached,
            title=extract_title(markdown, metadata_title, source.stem),
            page_count=page_count,
            char_count=len(markdown),
            from_cache=True,
        )

    try:
        markdown = pymupdf4llm.to_markdown(str(source))

    except Exception as exc:
        raise PdfConversionError(f"Cannot convert PDF {source.name}: {exc}") from exc

    # A PDF made only of images yields no text; the agent must know that
    # instead of silently working with an empty document.
    if not markdown.strip():
        raise PdfConversionError(
            f"No text could be extracted from {source.name}; it may be a scanned document."
        )

    markdown_path = target_dir / build_cache_file_name(document_id)
    markdown_path.write_text(markdown, encoding="utf-8")
    logger.info("Converted %s to %s", source.name, markdown_path.name)

    return ConvertedDocument(
        document_id=document_id,
        source_path=source,
        markdown_path=markdown_path,
        title=extract_title(markdown, metadata_title, source.stem),
        page_count=page_count,
        char_count=len(markdown),
        from_cache=False,
    )
