"""Unit tests for the core package (sessions, state and formatter)."""

from pathlib import Path

import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from core.formatter import ResponseFormatter
from core.session import SessionManager
from core.state import CANDIDATE_EXTRA_COLUMNS, SessionState, SessionStore, SourceMode


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Small accession table used across the tests."""
    return pd.DataFrame(
        {
            "accession_number": ["G1", "G2", "G3"],
            "country": ["COL", "PER", "MEX"],
            "latitude": [3.4, -12.0, 19.4],
            "longitude": [-76.5, -77.0, -99.1],
        }
    )


def test_session_manager_creates_inputs_and_outputs(tmp_path: Path) -> None:
    """Every session gets ``inputs`` and ``outputs`` sub folders under tmp."""
    manager = SessionManager(tmp_path / "tmp")
    paths = manager.get_paths("abc/../123")

    assert paths.inputs.is_dir()
    assert paths.outputs.is_dir()
    # Unsafe characters of the id are replaced so no path traversal is possible.
    assert paths.root.parent == tmp_path / "tmp"


def test_store_input_file_copies_and_avoids_overwrite(tmp_path: Path) -> None:
    """Uploading two files with the same name keeps both copies."""
    manager = SessionManager(tmp_path / "tmp")
    source = tmp_path / "data.xlsx"
    source.write_bytes(b"fake")

    first = manager.store_input_file("s1", source)
    second = manager.store_input_file("s1", source)

    assert first.exists() and second.exists()
    assert first != second
    assert len(manager.list_input_files("s1")) == 2


def test_set_original_list_creates_candidate_with_extra_columns(
    sample_dataframe: pd.DataFrame,
) -> None:
    """The Candidate list is a copy plus the seven annotation columns."""
    state = SessionState(session_id="s1")
    state.set_original_list(sample_dataframe, SourceMode.LOCAL)

    assert state.mode is SourceMode.LOCAL
    assert state.original_count == 3
    assert state.candidate_count == 3

    # Original list must not carry the extra columns.
    for column in CANDIDATE_EXTRA_COLUMNS:
        assert column in state.candidate_list.columns
        assert column not in state.original_list.columns


def test_update_candidate_list_rejects_missing_columns(sample_dataframe: pd.DataFrame) -> None:
    """Skills must not drop the mandatory annotation columns."""
    state = SessionState(session_id="s1")
    state.set_original_list(sample_dataframe, SourceMode.GENESYS)

    with pytest.raises(ValueError):
        state.update_candidate_list(sample_dataframe)


def test_session_store_reuses_state() -> None:
    """The same session id always returns the same state object."""
    store = SessionStore()
    first = store.get_or_create("x")
    second = store.get_or_create("x")

    assert first is second
    assert "x" in store


def test_formatter_builds_preview_and_exports(
    tmp_path: Path, sample_dataframe: pd.DataFrame
) -> None:
    """The response includes the activity summary and preview; CSV exports work."""
    manager = SessionManager(tmp_path / "tmp")
    paths = manager.get_paths("s1")
    state = SessionState(session_id="s1")
    state.set_original_list(sample_dataframe, SourceMode.LOCAL)
    state.log_activity("list_accessions", "Loaded 3 accessions from data.xlsx")

    formatter = ResponseFormatter(preview_rows=15)
    response = formatter.build_response("Done.", state)

    assert "Activities performed so far" in response
    assert "Candidate list preview" in response
    assert "G1" in response

    candidate_csv = formatter.export_candidate_list(state, paths)
    original_csv = formatter.export_original_list(state, paths)

    assert candidate_csv.parent == paths.outputs
    assert len(pd.read_csv(candidate_csv)) == 3
    assert len(pd.read_csv(original_csv).columns) == 4


def test_registry_discovers_without_skills() -> None:
    """With no concrete skills the registry is empty but usable."""
    registry = SkillRegistry().discover()

    assert len(registry) == 0
    assert registry.tools() == []
    assert "no tools" in registry.describe()
