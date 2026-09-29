"""Builds the final chat response and the CSV exports.

Every answer returned to the user contains:

1. The text produced by the agent.
2. A summary of the activities performed so far in the session.
3. A preview of the first N rows of the Candidate list (when data is loaded).
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState

logger = get_logger(__name__)

# Maximum number of columns shown in the markdown preview to keep it readable.
_MAX_PREVIEW_COLUMNS = 12


class ResponseFormatter:
    """Compose chat responses and export session lists to CSV."""

    def __init__(self, preview_rows: int = 15) -> None:
        """Create the formatter.

        Args:
            preview_rows: Number of Candidate list rows included in each preview.
        """
        self.preview_rows = preview_rows

    # ---------------------------------------------------------------- previews
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

        lines = ["**Activities performed so far:**", ""]

        # One bullet per action, with the resulting row count when known.
        for index, entry in enumerate(state.activity_log, start=1):
            rows = f" ({entry.rows_after} accessions)" if entry.rows_after is not None else ""
            lines.append(f"{index}. {entry.description}{rows}")

        return "\n".join(lines)

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

        candidate = state.candidate_list

        # An empty Candidate list is a valid outcome (filters removed everything).
        if candidate.empty:
            return "**Candidate list preview:** the Candidate list is currently empty."

        preview = candidate.head(self.preview_rows)

        # Too many columns make the table unreadable in the chat: trim and say so.
        note = ""
        if len(preview.columns) > _MAX_PREVIEW_COLUMNS:
            hidden = len(preview.columns) - _MAX_PREVIEW_COLUMNS
            preview = preview.iloc[:, :_MAX_PREVIEW_COLUMNS]
            note = f"\n\n_{hidden} additional columns not shown in the preview._"

        table = preview.to_markdown(index=False)
        header = (
            f"**Candidate list preview** (showing {len(preview)} of "
            f"{state.candidate_count} accessions):"
        )

        return f"{header}\n\n{table}{note}"

    def build_response(self, agent_text: str, state: SessionState) -> str:
        """Assemble the full chat answer.

        Args:
            agent_text: Natural-language answer produced by the LLM.
            state: Current session state.

        Returns:
            Markdown text ready to be shown in the chat.
        """
        sections = [agent_text.strip()]

        summary = self.build_activity_summary(state)
        preview = self.build_candidate_preview(state)

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
