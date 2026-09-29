"""In-memory state of a user session.

The business rules define two accession lists:

* **Original list** - the first list of accessions loaded in the session. It is
  created once and never modified afterwards.
* **Candidate list** - the list returned to the user. Every skill filters or
  annotates it; it is *not* cumulative, each decision narrows the previous one.

The Candidate list carries seven extra columns that skills fill in:
``criteria_passport``, ``criteria_traits``, ``criteria_research``,
``criteria_climate``, ``cluster_traits``, ``cluster_research`` and
``cluster_climate``.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

# Extra columns appended to the Candidate list when it is created.
CANDIDATE_EXTRA_COLUMNS: tuple[str, ...] = (
    "criteria_passport",
    "criteria_traits",
    "criteria_research",
    "criteria_climate",
    "cluster_traits",
    "cluster_research",
    "cluster_climate",
)


class SourceMode(str, Enum):
    """Origin of the accession data loaded in the session."""

    LOCAL = "local"  # An Excel/CSV file uploaded by the user.
    GENESYS = "genesys"  # Accessions retrieved through the Genesys PGR API.


@dataclass
class ActivityEntry:
    """A single action performed by the system during the session.

    Attributes:
        timestamp: When the action happened.
        skill: Name of the skill (or component) that performed it.
        description: Human readable description of what was done.
        rows_after: Number of Candidate list rows after the action, if applicable.
    """

    timestamp: datetime
    skill: str
    description: str
    rows_after: int | None = None


@dataclass
class SessionState:
    """Everything the agent must remember about one session between turns.

    Attributes:
        session_id: Identifier of the session.
        mode: ``SourceMode`` of the loaded accessions, ``None`` until data is loaded.
        original_list: Immutable first list of accessions.
        candidate_list: Working list returned to the user.
        climate_indicators: Climate indicators selected for the climate skill.
        uploaded_files: Files uploaded by the user during the session.
        activity_log: Ordered list of actions performed so far.
        extras: Free key/value store for skills that need extra memory.
    """

    session_id: str
    mode: SourceMode | None = None
    original_list: pd.DataFrame | None = None
    candidate_list: pd.DataFrame | None = None
    climate_indicators: list[str] = field(default_factory=list)
    uploaded_files: list[Path] = field(default_factory=list)
    activity_log: list[ActivityEntry] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ queries
    @property
    def has_data(self) -> bool:
        """Whether an Original list has already been loaded in this session."""
        return self.original_list is not None

    @property
    def candidate_count(self) -> int:
        """Number of rows currently in the Candidate list (0 when empty)."""
        return 0 if self.candidate_list is None else len(self.candidate_list)

    @property
    def original_count(self) -> int:
        """Number of rows in the Original list (0 when not loaded)."""
        return 0 if self.original_list is None else len(self.original_list)

    # ---------------------------------------------------------------- mutators
    def set_original_list(self, dataframe: pd.DataFrame, mode: SourceMode) -> None:
        """Store the first list of accessions and derive the Candidate list.

        Two independent copies are kept: the Original list, which is never
        modified again, and the Candidate list, which receives the empty
        ``criteria_*`` / ``cluster_*`` columns and is narrowed by later skills.

        Args:
            dataframe: Accessions loaded from a file or from Genesys.
            mode: Where the accessions came from.
        """
        self.mode = mode
        self.original_list = dataframe.copy(deep=True)

        candidate = dataframe.copy(deep=True)

        # Add each extra column only when the source does not already provide it,
        # so a re-uploaded export keeps whatever the user had.
        for column in CANDIDATE_EXTRA_COLUMNS:
            if column not in candidate.columns:
                candidate[column] = pd.NA

        self.candidate_list = candidate.reset_index(drop=True)

        logger.info(
            "Session %s: original list set (%s rows, mode=%s)",
            self.session_id,
            len(self.original_list),
            mode.value,
        )

    def update_candidate_list(self, dataframe: pd.DataFrame) -> None:
        """Replace the Candidate list with a new (filtered or annotated) version.

        Args:
            dataframe: The new Candidate list. Must keep the extra columns.

        Raises:
            ValueError: If any of the mandatory extra columns is missing.
        """
        missing = [column for column in CANDIDATE_EXTRA_COLUMNS if column not in dataframe.columns]

        # Skills must never drop the annotation columns; fail loudly if they do.
        if missing:
            raise ValueError(f"Candidate list is missing mandatory columns: {missing}")

        self.candidate_list = dataframe.reset_index(drop=True)

    def log_activity(self, skill: str, description: str) -> None:
        """Append an entry to the activity log.

        Args:
            skill: Name of the skill or component performing the action.
            description: What was done, in plain language.
        """
        entry = ActivityEntry(
            timestamp=datetime.now(),
            skill=skill,
            description=description,
            rows_after=self.candidate_count if self.candidate_list is not None else None,
        )
        self.activity_log.append(entry)
        logger.info("Session %s | %s | %s", self.session_id, skill, description)

    def register_uploaded_file(self, path: Path) -> None:
        """Remember a file uploaded by the user.

        Args:
            path: Path of the stored copy inside the session ``inputs`` folder.
        """
        # Keep the list free of duplicates while preserving upload order.
        if path not in self.uploaded_files:
            self.uploaded_files.append(path)

    def summary_context(self) -> str:
        """Build a short textual description of the state for the system prompt.

        Returns:
            Plain text the LLM can read to know what is loaded in the session.
        """
        # Without data the model must guide the user towards loading accessions.
        if not self.has_data:
            files = ", ".join(path.name for path in self.uploaded_files) or "none"
            return (
                "No accession list has been loaded yet. "
                f"Uploaded files available in this session: {files}."
            )

        columns = ", ".join(str(column) for column in self.candidate_list.columns)
        indicators = ", ".join(self.climate_indicators) or "none selected"
        files = ", ".join(path.name for path in self.uploaded_files) or "none"

        return (
            f"Source mode: {self.mode.value}. "
            f"Original list: {self.original_count} accessions. "
            f"Candidate list: {self.candidate_count} accessions. "
            f"Candidate list columns: {columns}. "
            f"Climate indicators in memory: {indicators}. "
            f"Uploaded files: {files}."
        )


class SessionStore:
    """Registry of ``SessionState`` objects indexed by session id (process memory)."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._states: dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        """Return the state of a session, creating it on first access.

        Args:
            session_id: Identifier of the session.

        Returns:
            The ``SessionState`` bound to ``session_id``.
        """
        # Lazily create the state so that the first message of a session works.
        if session_id not in self._states:
            self._states[session_id] = SessionState(session_id=session_id)
            logger.info("Created state for session %s", session_id)

        return self._states[session_id]

    def reset(self, session_id: str) -> None:
        """Forget everything about a session.

        Args:
            session_id: Identifier of the session to drop.
        """
        self._states.pop(session_id, None)

    def __contains__(self, session_id: str) -> bool:
        """Whether a state exists for ``session_id``."""
        return session_id in self._states
