"""Tests for the ``list_accessions`` and ``export_list`` skills."""

import shutil
from pathlib import Path

import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from agent.subsetting_agent import OUTPUT_FILES_KEY, SubsettingAgent
from core.session import SessionManager, SessionPaths
from core.state import CANDIDATE_EXTRA_COLUMNS, SessionState, SourceMode
from skills.export_list.skill import ExportListSkill
from skills.list_accessions.loaders import detect_coordinate_columns, normalize_columns
from skills.list_accessions.skill import ListAccessionsSkill

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def session(tmp_path: Path) -> tuple[SessionState, SessionPaths, SessionManager]:
    """Fresh state, folders and manager for one test session."""
    manager = SessionManager(tmp_path / "tmp")
    paths = manager.get_paths("s1")
    return SessionState(session_id="s1"), paths, manager


def _upload(manager: SessionManager, name: str) -> Path:
    """Copy a fixture into the session inputs folder, like Gradio uploads do."""
    return manager.store_input_file("s1", FIXTURES / name)


# ----------------------------------------------------------------- loaders
def test_normalize_columns_handles_unnamed_and_duplicates() -> None:
    """Empty headers get stable names and duplicates receive suffixes."""
    frame = pd.DataFrame([[1, 2, 3, 4]], columns=[" A ", "Unnamed: 1", "A", "B"])
    result = normalize_columns(frame)

    assert list(result.columns) == ["A", "column_2", "A_1", "B"]


def test_detect_coordinate_columns() -> None:
    """Common passport coordinate names are detected case-insensitively."""
    detected = detect_coordinate_columns(["ACCENUMB", "DECLATITUDE", "DECLONGITUDE"])

    assert detected == {"latitude": "DECLATITUDE", "longitude": "DECLONGITUDE"}
    assert detect_coordinate_columns(["ACCENUMB"]) == {"latitude": None, "longitude": None}


# ------------------------------------------------------- list_accessions
def test_local_mode_loads_excel(session) -> None:
    """An uploaded xlsx creates Original and Candidate lists in local mode."""
    state, paths, manager = session
    uploaded = _upload(manager, "accessions_sample.xlsx")

    result = ListAccessionsSkill().run(state, paths, source="local", file_path=str(uploaded))

    assert result["status"] == "ok"
    assert result["mode"] == "local"
    assert result["accessions"] == 5
    assert result["coordinate_columns"]["latitude"] == "DECLATITUDE"
    assert state.mode is SourceMode.LOCAL
    assert state.original_count == 5

    # The Candidate list carries the annotation columns; the Original does not.
    for column in CANDIDATE_EXTRA_COLUMNS:
        assert column in state.candidate_list.columns
        assert column not in state.original_list.columns

    assert len(state.activity_log) == 1


def test_local_mode_accepts_file_name_only_and_csv(session) -> None:
    """The model may pass just the file name; CSV with ';' is auto-detected."""
    state, paths, manager = session
    _upload(manager, "accessions_sample.csv")

    result = ListAccessionsSkill().run(
        state, paths, source="local", file_path="accessions_sample.csv"
    )

    assert result["status"] == "ok"
    assert result["total_columns"] == 7


def test_local_mode_reads_named_sheet_and_rejects_unknown_sheet(session) -> None:
    """Sheet selection works and unknown sheets produce a helpful error."""
    state, paths, manager = session
    uploaded = _upload(manager, "accessions_sample.xlsx")
    skill = ListAccessionsSkill()

    ok = skill.run(state, paths, source="local", file_path=str(uploaded), sheet_name="Notes")
    bad = skill.run(state, paths, source="local", file_path=str(uploaded), sheet_name="Nope")

    assert ok["status"] == "ok" and ok["accessions"] == 1
    assert bad["status"] == "error" and "Available sheets" in bad["message"]


def test_local_mode_rejects_files_outside_inputs(session, tmp_path: Path) -> None:
    """Files that are not in the session inputs folder are never read."""
    state, paths, _ = session
    outside = tmp_path / "outside.xlsx"
    shutil.copy(FIXTURES / "accessions_sample.xlsx", outside)

    result = ListAccessionsSkill().run(state, paths, source="local", file_path=str(outside))

    assert result["status"] == "error"
    assert not state.has_data


def test_local_mode_requires_file_path_and_reports_replacement(session) -> None:
    """Missing path is an error; re-loading replaces the lists and says so."""
    state, paths, manager = session
    skill = ListAccessionsSkill()

    missing = skill.run(state, paths, source="local")
    assert missing["status"] == "error"

    uploaded = _upload(manager, "accessions_sample.xlsx")
    first = skill.run(state, paths, source="local", file_path=str(uploaded))
    second = skill.run(state, paths, source="local", file_path=str(uploaded))

    assert first["replaced_previous_lists"] is False
    assert second["replaced_previous_lists"] is True
    assert "replacing" in state.activity_log[-1].description


def test_genesys_mode_not_available_yet(session) -> None:
    """Genesys mode returns a formal error and needs a query."""
    state, paths, _ = session
    skill = ListAccessionsSkill()

    no_query = skill.run(state, paths, source="genesys")
    with_query = skill.run(state, paths, source="genesys", query="Phaseolus vulgaris")

    assert no_query["status"] == "error"
    assert with_query["status"] == "error"
    assert with_query["mode"] == "genesys"
    assert not state.has_data


def test_unknown_source_is_rejected(session) -> None:
    """Any source other than local/genesys is reported as an error."""
    state, paths, _ = session
    result = ListAccessionsSkill().run(state, paths, source="cloud")

    assert result["status"] == "error"


# ------------------------------------------------------------ export_list
def test_export_list_writes_csv_and_reports_output_files(session) -> None:
    """Both lists export to outputs/ and the path is exposed for attachment."""
    state, paths, manager = session
    uploaded = _upload(manager, "accessions_sample.xlsx")
    ListAccessionsSkill().run(state, paths, source="local", file_path=str(uploaded))

    candidate = ExportListSkill().run(state, paths, which="candidate")
    original = ExportListSkill().run(state, paths, which="original")

    assert candidate["status"] == "ok"
    assert Path(candidate[OUTPUT_FILES_KEY][0]).parent == paths.outputs
    assert "cluster_climate" in pd.read_csv(candidate[OUTPUT_FILES_KEY][0]).columns
    assert "cluster_climate" not in pd.read_csv(original[OUTPUT_FILES_KEY][0]).columns


def test_export_list_without_data_and_unknown_list(session) -> None:
    """Exporting before loading, or an unknown list, returns an error."""
    state, paths, manager = session
    skill = ExportListSkill()

    assert skill.run(state, paths, which="candidate")["status"] == "error"

    uploaded = _upload(manager, "accessions_sample.xlsx")
    ListAccessionsSkill().run(state, paths, source="local", file_path=str(uploaded))

    assert skill.run(state, paths, which="everything")["status"] == "error"


# ------------------------------------------------------------- integration
def test_registry_discovers_both_skills() -> None:
    """Auto-discovery registers list_accessions and export_list."""
    registry = SkillRegistry().discover()

    assert set(registry.names()) == {"list_accessions", "export_list"}
    assert all(tool["type"] == "function" for tool in registry.tools())


def test_agent_collects_output_files(tmp_path: Path) -> None:
    """The agent attaches only existing files listed under output_files."""
    existing = tmp_path / "out.csv"
    existing.write_text("a,b\n1,2\n", encoding="utf-8")
    collected: list[Path] = []

    SubsettingAgent._collect_output_files(
        {OUTPUT_FILES_KEY: [str(existing), str(tmp_path / "missing.csv")]}, collected
    )
    SubsettingAgent._collect_output_files({OUTPUT_FILES_KEY: [str(existing)]}, collected)
    SubsettingAgent._collect_output_files({"message": "no files"}, collected)

    assert collected == [existing]
