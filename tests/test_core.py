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
    assert "Candidate list preview" in response  # a list was just loaded in this turn
    assert "G1" in response
    assert response.startswith("**Session:** 3 accessions")

    candidate_csv = formatter.export_candidate_list(state, paths)
    original_csv = formatter.export_original_list(state, paths)

    assert candidate_csv.parent == paths.outputs
    assert len(pd.read_csv(candidate_csv)) == 3
    assert len(pd.read_csv(original_csv).columns) == 4


def test_registry_is_usable_before_discovery() -> None:
    """A registry without skills is empty but usable; discovery fills it."""
    registry = SkillRegistry()

    assert len(registry) == 0
    assert registry.tools() == []
    assert "no tools" in registry.describe()

    registry.discover()

    assert len(registry) > 0
    assert "no tools" not in registry.describe()


# ----------------------------------------------------------- response layout
def test_detect_language() -> None:
    """Spanish and English messages are recognised; empty text keeps the default."""
    from core.language import detect_language

    assert detect_language("Busca en Genesys los frijoles de Colombia") == "es"
    assert detect_language("Find the bean accessions from Colombia") == "en"
    assert detect_language("", default="es") == "es"
    assert detect_language("G1234") is None


def test_preview_only_after_load_or_filter(sample_dataframe: pd.DataFrame) -> None:
    """The preview is shown when a list is loaded or filtered, not after other actions."""
    state = SessionState(session_id="s1")
    formatter = ResponseFormatter(preview_rows=15)

    # Nothing loaded: status line says so, no preview.
    assert "no accession list loaded" in formatter.build_response("Hi", state)

    state.begin_turn()
    state.set_original_list(sample_dataframe, SourceMode.LOCAL)
    assert state.show_preview and "Candidate list preview" in formatter.build_response("Loaded.", state)

    # Next turn with an annotation (cluster-like) action: no preview.
    state.begin_turn()
    state.log_activity("climate_analysis", "Clustered")
    response = formatter.build_response("Clustered.", state)
    assert "Candidate list preview" not in response and "Activities performed so far" in response
    assert "**3** in the Candidate list" in response

    # Filtering shows it again; reset as well.
    state.begin_turn()
    state.mark_event("filtered")
    assert "Candidate list preview" in formatter.build_response("Filtered.", state)
    state.begin_turn()
    state.reset_candidate_list()
    assert state.show_preview


def test_fixed_blocks_follow_the_user_language(sample_dataframe: pd.DataFrame) -> None:
    """Status, activity and preview headers are rendered in Spanish when the user writes Spanish."""
    state = SessionState(session_id="s1")
    state.language = "es"
    state.set_original_list(sample_dataframe, SourceMode.GENESYS)
    state.log_activity("list_accessions", "Cargadas 3 accesiones")
    response = ResponseFormatter().build_response("Listo.", state)

    assert "Sesión:" in response and "modo Genesys" in response
    assert "Actividades realizadas hasta ahora" in response
    assert "Vista previa de la lista candidata" in response


def test_preview_columns_prioritise_ids_and_filled_annotations() -> None:
    """Identification columns first, filled criteria/cluster columns next, empty columns hidden."""
    frame = pd.DataFrame({
        "extra": ["x", "y"], "ORIGCTY": ["COL", "PER"], "ACCENUMB": ["G1", "G2"], "empty": [None, None],
    })
    state = SessionState(session_id="s1")
    state.set_original_list(frame, SourceMode.LOCAL)
    state.candidate_list["cluster_climate"] = [1, 2]

    columns = ResponseFormatter.select_preview_columns(state.candidate_list)
    assert columns[:2] == ["ACCENUMB", "ORIGCTY"]
    assert columns[2] == "cluster_climate" and "extra" in columns
    assert "empty" not in columns and "criteria_passport" not in columns


def test_export_defaults_to_candidate_and_prompt_language(tmp_path: Path, sample_dataframe: pd.DataFrame) -> None:
    """export_list without 'which' exports the Candidate list; the prompt names the user's language."""
    from agent.prompts import build_system_prompt
    from skills.export_list.skill import ExportListSkill

    state = SessionState(session_id="s1")
    state.set_original_list(sample_dataframe, SourceMode.LOCAL)
    paths = SessionManager(tmp_path / "tmp").get_paths("s1")
    result = ExportListSkill().run(state, paths)

    assert result["status"] == "ok" and result["list"] == "candidate"

    prompt = build_system_prompt("tools", "context", language="es")["content"]
    assert "answer in SPANISH" in prompt
    assert "SAME LANGUAGE" in build_system_prompt("tools", "context")["content"]


def test_strip_system_blocks_and_raw_conversation(sample_dataframe: pd.DataFrame) -> None:
    """Imitated system blocks are cut from the model text; the raw memory keeps only clean texts."""
    from core.formatter import strip_system_blocks

    text = "Listo, filtré la lista.\n\n---\n\n**Actividades realizadas hasta ahora:**\n\n1. algo\n\n**Vista previa de la lista candidata**"
    assert strip_system_blocks(text) == "Listo, filtré la lista."
    assert strip_system_blocks("Plain answer") == "Plain answer"

    state = SessionState(session_id="s1")
    state.remember_exchange("hola", "Hola, ¿qué necesitas?")
    assert state.conversation == [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "Hola, ¿qué necesitas?"}]
