"""Per-session store of user documents converted to Markdown.

The store receives the PDFs uploaded through the chat, converts them (with
caching), splits the Markdown into bounded sections and offers listing, outline,
reading and lexical search operations. Those operations are what the agent
exposes to the LLM as tools, so a whole paper never has to be pushed into the
model context at once.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from pathlib import Path

from document_processing.models import ConvertedDocument, DocumentSection, SearchHit
from document_processing.pdf_converter import PdfConversionError, convert_pdf_to_markdown
from subsetting_sdk.catalog import normalize_text

logger = logging.getLogger(__name__)

# Default upper bound of a section, in characters. Chosen so that a few hits fit
# comfortably in the context window of a small local model.
DEFAULT_MAX_SECTION_CHARS = 1500

# Minimum size of a section before it is merged with the following one, so the
# store does not produce dozens of one-line sections.
DEFAULT_MIN_SECTION_CHARS = 200

# Very common words ignored during search, in English and Spanish.
STOP_WORDS = frozenset(
    """
    a an and are as at be by for from has in is it its of on or that the to was were
    with which this these those their there
    de la el los las y o en un una que por con para del al se su sus es son como
    """.split()
)

_HEADING_LINE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")


class DocumentStore:
    """Holds the converted documents of one chat session.

    Attributes:
        cache_dir: Directory where Markdown conversions are stored.
        max_section_chars: Upper bound of a section in characters.
        min_section_chars: Sections shorter than this are merged forward.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
        min_section_chars: int = DEFAULT_MIN_SECTION_CHARS,
    ) -> None:
        """Create an empty store.

        Args:
            cache_dir: Where conversions live (configured by the application).
            max_section_chars: Upper bound of a section in characters.
            min_section_chars: Sections shorter than this are merged forward.

        Raises:
            ValueError: If the size bounds are inconsistent.
        """
        # The bounds must leave room for splitting; otherwise merging and
        # splitting would fight each other.
        if min_section_chars <= 0 or max_section_chars <= min_section_chars:
            raise ValueError("Expected 0 < min_section_chars < max_section_chars.")

        self.cache_dir = Path(cache_dir)
        self.max_section_chars = max_section_chars
        self.min_section_chars = min_section_chars

        self._documents: dict[str, ConvertedDocument] = {}
        self._sections: dict[str, list[DocumentSection]] = {}

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def add_pdf(self, pdf_path: str | Path) -> ConvertedDocument:
        """Convert one PDF and index its sections.

        Args:
            pdf_path: PDF uploaded by the user.

        Returns:
            The converted document. Adding the same PDF twice returns the
            existing entry without reprocessing.

        Raises:
            PdfConversionError: If the file cannot be converted.
        """
        document = convert_pdf_to_markdown(pdf_path, self.cache_dir)

        # Identical content is already indexed: nothing else to do.
        if document.document_id in self._documents:
            return self._documents[document.document_id]

        markdown = document.markdown_path.read_text(encoding="utf-8")
        self._documents[document.document_id] = document
        self._sections[document.document_id] = self._split_into_sections(document, markdown)

        return document

    def add_pdfs(self, pdf_paths: list[str | Path]) -> tuple[list[ConvertedDocument], list[str]]:
        """Convert several PDFs, collecting failures instead of aborting.

        Args:
            pdf_paths: PDFs uploaded by the user.

        Returns:
            The documents added successfully and one message per failed file.
        """
        added: list[ConvertedDocument] = []
        errors: list[str] = []

        # Each file is processed independently so one bad upload does not
        # prevent the others from being available to the agent.
        for pdf_path in pdf_paths:
            try:
                added.append(self.add_pdf(pdf_path))

            except PdfConversionError as exc:
                logger.warning("Skipping document %s: %s", pdf_path, exc)
                errors.append(str(exc))

        return added, errors

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    @property
    def is_empty(self) -> bool:
        """Whether no document has been added."""
        return not self._documents

    def list_documents(self) -> list[ConvertedDocument]:
        """Return the documents in insertion order."""
        return list(self._documents.values())

    def get_document(self, document_id: str) -> ConvertedDocument:
        """Return a document by id.

        Args:
            document_id: Id returned by :meth:`add_pdf`.

        Raises:
            KeyError: If the id is unknown.
        """
        # Fail with a message listing the known ids to help the LLM recover.
        if document_id not in self._documents:
            known = ", ".join(self._documents) or "none"
            raise KeyError(f"Unknown document id '{document_id}'. Known ids: {known}.")

        return self._documents[document_id]

    def get_outline(self, document_id: str) -> list[tuple[int, str, int]]:
        """Return ``(index, heading, char_count)`` for every section of a document.

        Args:
            document_id: Id of the document.
        """
        self.get_document(document_id)

        return [(s.index, s.heading, s.char_count) for s in self._sections[document_id]]

    def read_section(self, document_id: str, index: int) -> DocumentSection:
        """Return one section of a document.

        Args:
            document_id: Id of the document.
            index: 0-based section index, as listed by :meth:`get_outline`.

        Raises:
            IndexError: If the index is out of range.
        """
        self.get_document(document_id)
        sections = self._sections[document_id]

        # Guard the index explicitly to return a readable message.
        if index < 0 or index >= len(sections):
            raise IndexError(
                f"Section {index} does not exist; document '{document_id}' has "
                f"{len(sections)} sections (0-{len(sections) - 1})."
            )

        return sections[index]

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[SearchHit]:
        """Find the sections most relevant to a query using term matching.

        The score is the sum, over query terms present in the section, of the
        term frequency weighted by term length, normalized by section length,
        with a bonus for matches inside the heading.

        Args:
            query: Free-text query in any language supported by the stop list.
            top_k: Maximum number of hits.
            document_id: Restrict the search to one document.
        """
        terms = self._query_terms(query)

        # Without meaningful terms there is nothing to match.
        if not terms:
            return []

        hits: list[SearchHit] = []

        # Score each candidate section; documents are filtered when requested.
        for doc_id, sections in self._sections.items():
            if document_id is not None and doc_id != document_id:
                continue

            for section in sections:
                hit = self._score_section(section, terms)

                if hit is not None:
                    hits.append(hit)

        hits.sort(key=lambda hit: hit.score, reverse=True)

        return hits[:top_k]

    # ------------------------------------------------------------------ #
    # Maintenance
    # ------------------------------------------------------------------ #

    def clear(self) -> None:
        """Forget every document without touching the cache on disk."""
        self._documents.clear()
        self._sections.clear()

    def cleanup(self) -> None:
        """Forget every document and delete the cache directory."""
        self.clear()

        # The directory may not exist if nothing was ever converted.
        if self.cache_dir.is_dir():
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Sectioning
    # ------------------------------------------------------------------ #

    def _split_into_sections(
        self, document: ConvertedDocument, markdown: str
    ) -> list[DocumentSection]:
        """Split Markdown into heading-based sections bounded in size.

        Args:
            document: Document the Markdown belongs to.
            markdown: Converted text.
        """
        blocks = self._split_by_headings(markdown, document.title)
        blocks = self._merge_small_blocks(blocks)

        sections: list[DocumentSection] = []

        # Oversized blocks are cut at paragraph boundaries; every resulting
        # piece keeps the heading of the block it came from.
        for heading, text in blocks:
            for piece in self._chunk_text(text):
                sections.append(
                    DocumentSection(
                        document_id=document.document_id,
                        index=len(sections),
                        heading=heading,
                        text=piece,
                    )
                )

        return sections

    @staticmethod
    def _split_by_headings(markdown: str, default_heading: str) -> list[tuple[str, str]]:
        """Group lines under their Markdown heading.

        Args:
            markdown: Converted text.
            default_heading: Heading used for text before the first heading.

        Returns:
            ``(heading, text)`` pairs with empty texts removed.
        """
        blocks: list[tuple[str, list[str]]] = [(default_heading, [])]

        # Each heading line opens a new block; other lines join the current one.
        for line in markdown.splitlines():
            match = _HEADING_LINE.match(line)

            if match:
                blocks.append((match.group(2).strip(), []))

            else:
                blocks[-1][1].append(line)

        result: list[tuple[str, str]] = []

        # Drop blocks that carry no text (e.g. consecutive headings).
        for heading, lines in blocks:
            text = "\n".join(lines).strip()

            if text:
                result.append((heading, text))

        return result

    def _merge_small_blocks(self, blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Merge blocks shorter than ``min_section_chars`` into the next block.

        Args:
            blocks: ``(heading, text)`` pairs in document order.
        """
        merged: list[tuple[str, str]] = []
        pending: tuple[str, str] | None = None

        # A small block is held and prepended (with its heading as a label) to
        # the following block, which keeps its own heading, so short
        # subsections do not become noise.
        for heading, text in blocks:
            if pending is not None:
                text = f"{pending[0]}\n{pending[1]}\n\n{text}"
                pending = None

            if len(text) < self.min_section_chars:
                pending = (heading, text)

            else:
                merged.append((heading, text))

        # A trailing small block has nothing to merge into and is kept alone.
        if pending is not None:
            merged.append(pending)

        return merged

    def _chunk_text(self, text: str) -> list[str]:
        """Cut a text into pieces no longer than ``max_section_chars``.

        Cuts happen at paragraph boundaries when possible, then at line
        boundaries, and as a last resort at a fixed length.

        Args:
            text: Text to cut.
        """
        if len(text) <= self.max_section_chars:
            return [text]

        pieces: list[str] = []
        current = ""

        # Paragraphs are appended while they fit; a paragraph that alone
        # exceeds the bound is hard-split.
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) > self.max_section_chars:
                if current:
                    pieces.append(current)
                    current = ""

                pieces.extend(self._hard_split(paragraph))
                continue

            candidate = f"{current}\n\n{paragraph}" if current else paragraph

            if len(candidate) > self.max_section_chars:
                pieces.append(current)
                current = paragraph

            else:
                current = candidate

        if current:
            pieces.append(current)

        return pieces

    def _hard_split(self, paragraph: str) -> list[str]:
        """Split a single oversized paragraph at line or fixed boundaries.

        Args:
            paragraph: Paragraph longer than ``max_section_chars``.
        """
        pieces: list[str] = []
        current = ""

        # Lines are accumulated up to the bound; a single line longer than the
        # bound is cut at fixed positions.
        for line in paragraph.splitlines():
            while len(line) > self.max_section_chars:
                if current:
                    pieces.append(current)
                    current = ""

                pieces.append(line[: self.max_section_chars])
                line = line[self.max_section_chars :]

            candidate = f"{current}\n{line}" if current else line

            if len(candidate) > self.max_section_chars:
                pieces.append(current)
                current = line

            else:
                current = candidate

        if current:
            pieces.append(current)

        return pieces

    # ------------------------------------------------------------------ #
    # Search helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _query_terms(query: str) -> list[str]:
        """Normalize a query into distinct, meaningful terms.

        Args:
            query: Free-text query.
        """
        terms: list[str] = []

        # Stop words and single characters carry no signal and are dropped.
        for term in normalize_text(query).split():
            if term in STOP_WORDS or len(term) < 2:
                continue

            if term not in terms:
                terms.append(term)

        return terms

    def _score_section(self, section: DocumentSection, terms: list[str]) -> SearchHit | None:
        """Score one section against the query terms.

        Args:
            section: Section to score.
            terms: Normalized query terms.

        Returns:
            A hit, or ``None`` when no term occurs in the section.
        """
        body_tokens = normalize_text(section.text).split()
        heading_tokens = normalize_text(section.heading).split()
        body_counts = Counter(body_tokens)
        heading_counts = Counter(heading_tokens)

        score = 0.0
        matched: list[str] = []

        # Each term contributes its weighted frequency; heading matches count
        # extra because headings summarize the section.
        for term in terms:
            occurrences = body_counts.get(term, 0)
            in_heading = heading_counts.get(term, 0)

            if occurrences == 0 and in_heading == 0:
                continue

            matched.append(term)
            score += (occurrences + 3 * in_heading) * len(term)

        if not matched:
            return None

        # Normalize by length so long sections do not dominate by volume, and
        # reward sections that match more distinct terms.
        score = score / max(len(body_tokens), 1) * (1 + len(matched) / len(terms))

        return SearchHit(
            section=section,
            score=round(score, 6),
            matched_terms=matched,
            snippet=self._snippet(section.text, matched[0]),
        )

    @staticmethod
    def _snippet(text: str, term: str, width: int = 160) -> str:
        """Return an excerpt of ``text`` centred on the first occurrence of ``term``.

        Args:
            text: Section text.
            term: Normalized term to locate.
            width: Approximate length of the excerpt.
        """
        position = normalize_text(text).find(term)
        flat = " ".join(text.split())

        # Normalization may shift offsets slightly; the excerpt is approximate
        # and falls back to the beginning of the text when the term is absent.
        if position < 0:
            return flat[:width] + ("..." if len(flat) > width else "")

        start = max(0, position - width // 2)
        end = min(len(flat), start + width)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(flat) else ""

        return f"{prefix}{flat[start:end]}{suffix}"
