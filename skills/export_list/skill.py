"""Skill that exports the Candidate list or the Original list as a CSV file.

The file is written to the session ``outputs`` folder and its path is returned
under the ``output_files`` key so the agent can attach it to the chat answer.
"""

from typing import Any

from core.formatter import ResponseFormatter
from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState
from skills.base import Skill

logger = get_logger(__name__)

# Key of the tool result that the agent scans to attach files to the answer.
OUTPUT_FILES_KEY = "output_files"


class ExportListSkill(Skill):
    """Write the full Candidate or Original list to a CSV file."""

    name = "export_list"
    description = (
        "Export a list of the session as a CSV file that the user can download. "
        "Use which='candidate' when the user asks for the final/current/filtered list "
        "and which='original' when the user asks for the original/initial list."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "which": {
                "type": "string",
                "enum": ["candidate", "original"],
                "description": "List to export: 'candidate' (working list) or 'original'.",
            }
        },
        "required": ["which"],
    }

    def __init__(self) -> None:
        """Create the skill with its own formatter (only the export methods are used)."""
        self._formatter = ResponseFormatter()

    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        which: str = "candidate",
        **_: Any,
    ) -> dict[str, Any]:
        """Export the requested list.

        Args:
            state: Session state holding the lists.
            paths: Session folders; the CSV goes to ``paths.outputs``.
            which: ``"candidate"`` or ``"original"``.
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result with the CSV path in ``output_files``.
        """
        target = (which or "").strip().lower()

        # Nothing can be exported before a list is loaded.
        if not state.has_data:
            return self.error(
                "No accession list is loaded in this session yet. Ask the user to upload a "
                "file or to request accessions from Genesys first."
            )

        try:
            # Choose the list and let the formatter write the timestamped CSV.
            if target == "candidate":
                csv_path = self._formatter.export_candidate_list(state, paths)
                rows = state.candidate_count
            elif target == "original":
                csv_path = self._formatter.export_original_list(state, paths)
                rows = state.original_count
            else:
                return self.error(f"Unknown list '{which}'. Use 'candidate' or 'original'.")
        except ValueError as exc:
            return self.error(str(exc))

        description = f"Exported the {target} list ({rows} accessions) to '{csv_path.name}'"
        state.log_activity(self.name, description)

        return self.ok(
            f"{description}. The file is attached to the answer for download.",
            list=target,
            rows=rows,
            file_name=csv_path.name,
            **{OUTPUT_FILES_KEY: [str(csv_path)]},
        )
