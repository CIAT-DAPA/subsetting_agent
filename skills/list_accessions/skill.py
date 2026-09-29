"""Skill that loads the first list of accessions of a session.

Two source modes exist:

* ``local``   - the user uploaded an Excel/CSV file; every column is passport data.
* ``genesys`` - accessions are searched in Genesys PGR through its API.

Loading a list creates the Original list and the Candidate list in the session
state (see :meth:`core.state.SessionState.set_original_list`).
"""

from pathlib import Path
from typing import Any

from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState, SourceMode
from skills.base import Skill
from skills.list_accessions.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileError,
    detect_coordinate_columns,
    read_accession_file,
)

logger = get_logger(__name__)

# Maximum number of column names echoed back to the model.
_MAX_COLUMNS_IN_RESULT = 60


class ListAccessionsSkill(Skill):
    """Load accessions from a local file or from Genesys and start the session lists."""

    name = "list_accessions"
    description = (
        "Load the first list of accessions of the session and create the Original list "
        "and the Candidate list. Use source='local' with the path of a file the user "
        "attached (Excel or CSV; every column becomes passport data). Use "
        "source='genesys' with a search query when the user gives no file and wants "
        "accessions from Genesys PGR. Calling it again replaces the current lists."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["local", "genesys"],
                "description": "Where the accessions come from: 'local' (attached file) or 'genesys'.",
            },
            "file_path": {
                "type": "string",
                "description": "Path of the attached Excel/CSV file (required when source='local').",
            },
            "sheet_name": {
                "type": "string",
                "description": "Excel sheet to read. Defaults to the first sheet.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Search text for Genesys PGR: accession number, crop, taxon, country... "
                    "(required when source='genesys')."
                ),
            },
        },
        "required": ["source"],
    }

    # ---------------------------------------------------------------- run
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        source: str = "",
        file_path: str | None = None,
        sheet_name: str | None = None,
        query: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Dispatch to the local or Genesys loader.

        Args:
            state: Session state to populate.
            paths: Session folders (uploads must live in ``paths.inputs``).
            source: ``"local"`` or ``"genesys"``.
            file_path: Attached file path (local mode).
            sheet_name: Excel sheet (local mode, optional).
            query: Genesys search text (genesys mode).
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result dictionary.
        """
        mode = (source or "").strip().lower()

        # Route by mode; anything else is a model mistake we report explicitly.
        if mode == SourceMode.LOCAL.value:
            return self._run_local(state, paths, file_path, sheet_name)

        if mode == SourceMode.GENESYS.value:
            return self._run_genesys(state, query)

        return self.error(
            f"Unknown source '{source}'. Use 'local' for an attached file or 'genesys' "
            "to search accessions in Genesys PGR."
        )

    # ------------------------------------------------------------- local
    def _run_local(
        self,
        state: SessionState,
        paths: SessionPaths,
        file_path: str | None,
        sheet_name: str | None,
    ) -> dict[str, Any]:
        """Load accessions from an uploaded Excel/CSV file.

        Args:
            state: Session state to populate.
            paths: Session folders.
            file_path: Path of the uploaded file.
            sheet_name: Excel sheet to read (optional).

        Returns:
            Tool result with the shape of the loaded table.
        """
        # Without a path the model must ask the user for the file.
        if not file_path:
            return self.error(
                "Local mode needs the path of an attached Excel or CSV file. "
                "Ask the user to upload the file."
            )

        resolved = self._resolve_input_file(paths, Path(file_path))

        # Only files inside the session inputs folder may be read.
        if resolved is None:
            uploaded = [path.name for path in state.uploaded_files]
            return self.error(
                f"File '{file_path}' is not available in this session. "
                f"Files uploaded so far: {uploaded or 'none'}."
            )

        try:
            dataframe = read_accession_file(resolved, sheet_name)
        except UnsupportedFileError as exc:
            return self.error(str(exc), supported_extensions=list(SUPPORTED_EXTENSIONS))
        except (FileNotFoundError, ValueError) as exc:
            return self.error(str(exc))

        replaced = state.has_data
        state.set_original_list(dataframe, SourceMode.LOCAL)
        state.extras["source_file"] = str(resolved)

        description = f"Loaded {len(dataframe)} accessions from file '{resolved.name}' (local mode)"

        # Tell the user the previous lists were discarded when re-loading.
        if replaced:
            description += ", replacing the previous Original and Candidate lists"

        state.log_activity(self.name, description)

        columns = [str(column) for column in dataframe.columns]
        coordinates = detect_coordinate_columns(columns)

        return self.ok(
            f"{description}. All {len(columns)} columns are treated as passport data.",
            mode=SourceMode.LOCAL.value,
            accessions=len(dataframe),
            columns=columns[:_MAX_COLUMNS_IN_RESULT],
            total_columns=len(columns),
            coordinate_columns=coordinates,
            replaced_previous_lists=replaced,
        )

    @staticmethod
    def _resolve_input_file(paths: SessionPaths, candidate: Path) -> Path | None:
        """Locate an uploaded file inside the session ``inputs`` folder.

        The model may pass the full path or just the file name; both are accepted
        as long as the final file lives under ``paths.inputs``.

        Args:
            paths: Session folders.
            candidate: Path or name given by the model.

        Returns:
            The resolved path, or ``None`` when the file is outside ``inputs`` or missing.
        """
        inputs_root = paths.inputs.resolve()
        options = [candidate, paths.inputs / candidate.name]

        # Try the path as given first, then the same file name inside inputs.
        for option in options:
            resolved = option.resolve()

            try:
                resolved.relative_to(inputs_root)
            except ValueError:
                # Outside the session folder: never read it, even if it exists.
                continue

            if resolved.is_file():
                return resolved

        return None

    # ----------------------------------------------------------- genesys
    def _run_genesys(self, state: SessionState, query: str | None) -> dict[str, Any]:
        """Load accessions from Genesys PGR.

        The Genesys SDK is not available yet; the branch is kept so the tool
        interface does not change when it is implemented.

        Args:
            state: Session state to populate.
            query: Search text given by the user.

        Returns:
            Tool result (currently always an error explaining the limitation).
        """
        # The model must collect a query before we can search anything.
        if not query:
            return self.error(
                "Genesys mode needs a search query (accession number, crop, taxon, country...)."
            )

        logger.info("Genesys mode requested for session %s with query '%s'", state.session_id, query)

        return self.error(
            "Genesys mode is not available yet in this version. Please ask the user to "
            "upload an Excel or CSV file with the accessions to use local mode.",
            mode=SourceMode.GENESYS.value,
            query=query,
        )
