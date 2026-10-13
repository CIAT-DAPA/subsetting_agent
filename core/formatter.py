"""Builds the final chat response and the CSV exports.

Every answer returned to the user contains:

1. A one-line status of the session (Original -> Candidate counts, mode).
2. The text produced by the agent.
3. The analysis report of the turn, when a cluster/group skill produced one.
4. A summary of the activities performed so far in the session.
5. A preview of the first N rows of the Candidate list, ONLY in the turns
   where a list was loaded or the Candidate list was filtered/reset.

The fixed blocks are rendered in the user's language (``core.language``).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.language import strings_for
from core.logger import get_logger
from core.session import SessionPaths
from core.state import CANDIDATE_EXTRA_COLUMNS, SessionState, SourceMode

logger = get_logger(__name__)

# Maximum number of columns shown in the markdown preview to keep it readable.
_MAX_PREVIEW_COLUMNS = 12
# Identification columns shown first in the preview when present (MCPD names and common local spellings).
_PRIORITY_COLUMNS = (
    "ACCENUMB", "accession_number", "accession", "INSTCODE", "GENUS", "SPECIES", "TAXONNAME",
    "ACCENAME", "ORIGCTY", "country", "SAMPSTAT", "DECLATITUDE", "DECLONGITUDE", "cellid",
)


# Headings of the blocks the system adds; if the model writes them itself they are cut out.
_SYSTEM_BLOCK_MARKERS = (
    "**Session:**", "**Sesión:**",
    "**Activities performed so far", "**Actividades realizadas",
    "**Candidate list preview", "**Vista previa de la lista candidata",
    "### Environmental clustering report", "### Informe de la agrupación por ambientes",
    "### Trait grouping report", "### Informe de la agrupación por rasgos",
)


def strip_system_blocks(text: str) -> str:
    """Remove any copy of the system blocks that the model wrote in its own answer.

    Small models tend to imitate the previous answers; everything from the first
    system heading onwards is dropped because the formatter adds the real blocks.

    Args:
        text: Raw text produced by the model.

    Returns:
        The text without imitated blocks (trailing separators removed).
    """
    cut = len(text)

    # Keep only what precedes the earliest imitated heading.
    for marker in _SYSTEM_BLOCK_MARKERS:
        position = text.find(marker)
        if position != -1:
            cut = min(cut, position)

    cleaned = text[:cut].rstrip()
    while cleaned.endswith("---"):
        cleaned = cleaned[:-3].rstrip()

    return cleaned


class ResponseFormatter:
    """Compose chat responses and export session lists to CSV."""

    def __init__(self, preview_rows: int = 15) -> None:
        """Create the formatter.

        Args:
            preview_rows: Number of Candidate list rows included in each preview.
        """
        self.preview_rows = preview_rows

    # ---------------------------------------------------------------- previews
    def build_status_line(self, state: SessionState) -> str:
        """Render the one-line session status shown at the top of every answer.

        Args:
            state: Session to describe.

        Returns:
            Markdown line with Original/Candidate counts and the active mode.
        """
        texts = strings_for(state.language)

        if not state.has_data:
            return texts["status_no_data"]

        mode = texts["mode_genesys"] if state.mode == SourceMode.GENESYS else texts["mode_local"]
        return texts["status"].format(original=f"{state.original_count:,}", candidate=f"{state.candidate_count:,}", mode=mode)

    def build_activity_summary(self, state: SessionState) -> str:
        """Render the activity log as a markdown list.

        Args:
            state: Session whose log is rendered.

        Returns:
            Markdown text, or an empty string when nothing has been done yet.
        """
        # Nothing to summarise before the first action.
        if not state.activity_log:
            return ""

        texts = strings_for(state.language)
        lines = [texts["activities"], ""]

        # One bullet per action, with the resulting row count when known.
        for index, entry in enumerate(state.activity_log, start=1):
            rows = f" ({entry.rows_after:,} {texts['accessions']})" if entry.rows_after is not None else ""
            lines.append(f"{index}. {entry.description}{rows}")

        return "\n".join(lines)

    @staticmethod
    def select_preview_columns(candidate: pd.DataFrame, limit: int = _MAX_PREVIEW_COLUMNS) -> list[str]:
        """Choose the most useful columns for the preview.

        Order: identification columns, then the ``criteria_*`` / ``cluster_*``
        columns that hold values, then the remaining non-empty columns in file
        order. Completely empty columns are never shown.

        Args:
            candidate: Candidate list.
            limit: Maximum number of columns returned.

        Returns:
            Column names in display order.
        """
        columns = [str(column) for column in candidate.columns]
        lowered = {column.lower(): column for column in columns}
        chosen: list[str] = []

        def add(column: str) -> None:
            """Append a column once, skipping completely empty ones."""
            if column not in chosen and candidate[column].notna().any() and (candidate[column].astype("string").str.strip() != "").any():
                chosen.append(column)

        # Identification columns first.
        for name in _PRIORITY_COLUMNS:
            if name.lower() in lowered:
                add(lowered[name.lower()])

        # Annotation columns with content next.
        for name in CANDIDATE_EXTRA_COLUMNS:
            if name in columns:
                add(name)

        # Everything else in original order.
        for column in columns:
            add(column)

        return chosen[:limit]

    def build_candidate_preview(self, state: SessionState) -> str:
        """Render the first rows of the Candidate list as a markdown table.

        Args:
            state: Session whose Candidate list is previewed.

        Returns:
            Markdown text, or an empty string when no data is loaded.
        """
        # Without a Candidate list there is nothing to preview.
        if state.candidate_list is None:
            return ""

        texts = strings_for(state.language)
        candidate = state.candidate_list

        # An empty Candidate list is a valid outcome (filters removed everything).
        if candidate.empty:
            return texts["preview_empty"]

        columns = self.select_preview_columns(candidate)
        preview = candidate.head(self.preview_rows)[columns]

        # Too many columns make the table unreadable in the chat: trim and say so.
        note = ""
        hidden = len(candidate.columns) - len(columns)
        if hidden > 0:
            note = "\n\n" + texts["hidden_columns"].format(hidden=hidden)

        table = preview.to_markdown(index=False)
        header = texts["preview"].format(shown=len(preview), total=f"{state.candidate_count:,}")

        return f"{header}\n\n{table}{note}"

    def build_response(self, agent_text: str, state: SessionState) -> str:
        """Assemble the full chat answer.

        Args:
            agent_text: Natural-language answer produced by the LLM.
            state: Current session state.

        Returns:
            Markdown text ready to be shown in the chat.
        """
        sections = [self.build_status_line(state), strip_system_blocks(agent_text).strip()]

        # Analysis reports are generated by the skills from real results and shown verbatim.
        if state.turn_report:
            sections.append(state.turn_report.strip())

        summary = self.build_activity_summary(state)
        # The preview is only useful right after the list changed (load / filter / reset).
        preview = self.build_candidate_preview(state) if state.show_preview else ""

        # Only append the optional blocks when they carry information.
        if summary:
            sections.append(summary)
        if preview:
            sections.append(preview)

        return "\n\n---\n\n".join(section for section in sections if section)

    # ----------------------------------------------------------------- exports
    @staticmethod
    def _export(dataframe: pd.DataFrame, paths: SessionPaths, prefix: str) -> Path:
        """Write a DataFrame to a timestamped CSV in the session ``outputs`` folder.

        Args:
            dataframe: Data to export.
            paths: Folders of the session.
            prefix: File name prefix (``candidate_list`` or ``original_list``).

        Returns:
            Path of the written CSV file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = paths.outputs / f"{prefix}_{timestamp}.csv"
        dataframe.to_csv(destination, index=False, encoding="utf-8")
        logger.info("Exported %s rows to %s", len(dataframe), destination)
        return destination

    def export_candidate_list(self, state: SessionState, paths: SessionPaths) -> Path:
        """Export the full Candidate list to CSV.

        Args:
            state: Session state holding the Candidate list.
            paths: Folders of the session.

        Returns:
            Path of the CSV file.

        Raises:
            ValueError: If no Candidate list is loaded.
        """
        # The export is meaningless without data; let the caller explain it.
        if state.candidate_list is None:
            raise ValueError("There is no Candidate list to export in this session.")

        return self._export(state.candidate_list, paths, "candidate_list")

    def export_original_list(self, state: SessionState, paths: SessionPaths) -> Path:
        """Export the full Original list to CSV.

        Args:
            state: Session state holding the Original list.
            paths: Folders of the session.

        Returns:
            Path of the CSV file.

        Raises:
            ValueError: If no Original list is loaded.
        """
        # Same guard as the Candidate export.
        if state.original_list is None:
            raise ValueError("There is no Original list to export in this session.")

        return self._export(state.original_list, paths, "original_list")
