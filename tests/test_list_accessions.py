"""Tests for the ``list_accessions`` and ``export_list`` skills."""

import json
import shutil
from pathlib import Path

import httpx

import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from agent.subsetting_agent import OUTPUT_FILES_KEY, SubsettingAgent
from core.session import SessionManager, SessionPaths
from core.state import CANDIDATE_EXTRA_COLUMNS, SessionState, SourceMode
from sdks.genesys import ApiTokenAuth, GenesysClient
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


def test_genesys_mode_requires_a_criterion(session) -> None:
    """Genesys mode without any criterion is rejected before calling the API."""
    state, paths, _ = session
    skill = ListAccessionsSkill(client_factory=lambda: _fail("factory must not be called"))

    result = skill.run(state, paths, source="genesys", crop=[], text="   ")

    assert result["status"] == "error"
    assert "at least one" in result["message"]
    assert not state.has_data


def _fail(message: str):
    """Helper raising inside a lambda (used to assert a factory is not used)."""
    raise AssertionError(message)


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


# ---------------------------------------------------------- genesys mode
def _genesys_dto(number: str, lat: float = 3.4) -> dict:
    """Minimal AccessionDTO for the mocked Genesys API."""
    return {
        "instituteCode": "COL003",
        "accessionNumber": number,
        "taxonomy": {"genus": "Phaseolus", "species": "vulgaris"},
        "origCty": "COL",
        "sampStat": 300,
        "geo": {"latitude": lat, "longitude": -76.5},
        "storage": [13],
    }


MOCK_CROPS = [
    {"shortName": "beans", "name": "Beans", "otherNames": ["bean", "frijol", "common bean"]},
    {"shortName": "maize", "name": "Maize", "otherNames": ["corn", "maíz"]},
]


def _mock_genesys_client(total: int, page_size_hint: int = 500) -> tuple[GenesysClient, list[httpx.Request]]:
    """Genesys client backed by a mock transport that serves ``total`` accessions."""
    numbers = [f"G{i:05d}" for i in range(total)]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        # Crop catalogue used by resolve_crop_codes (real Genesys shape).
        if request.url.path == "/api/v2/crop":
            return httpx.Response(200, json=MOCK_CROPS)

        page = int(request.url.params["p"])
        size = int(request.url.params["l"])
        chunk = numbers[page * size : (page + 1) * size]
        total_pages = max((total + size - 1) // size, 1)
        return httpx.Response(
            200,
            json={
                "content": [_genesys_dto(n) for n in chunk],
                "number": page,
                "size": size,
                "totalElements": total,
                "totalPages": total_pages,
                "last": page >= total_pages - 1,
            },
        )

    client = GenesysClient(
        "https://api.test.genesys",
        ApiTokenAuth("token"),
        transport=httpx.MockTransport(handler),
        page_size=page_size_hint,
        max_records=1000,
    )
    return client, requests


def test_build_genesys_filter_normalises_values() -> None:
    """Scalars become lists, codes are upper-cased, empty values are dropped."""
    accession_filter = ListAccessionsSkill.build_genesys_filter(
        crop="Bean",
        genus="phaseolus",
        species=["VULGARIS"],
        country_of_origin=["col", " per "],
        institute_code="col003",
        samp_stat=["300", 100],
        accession_numbers=None,
        text="",
        historic=None,
    )

    assert accession_filter.to_body() == {
        "crop": ["bean"],
        "taxonomy": {"genus": ["Phaseolus"], "species": ["vulgaris"]},
        "countryOfOrigin": {"code3": ["COL", "PER"]},
        "institute": {"code": ["COL003"]},
        "sampStat": [300, 100],
    }


def test_genesys_mode_loads_lists(session) -> None:
    """A Genesys search fills Original/Candidate lists with MCPD columns."""
    state, paths, _ = session
    client, requests = _mock_genesys_client(total=7)
    skill = ListAccessionsSkill(client_factory=lambda: client)

    result = skill.run(
        state, paths, source="genesys", genus=["Phaseolus"], country_of_origin=["COL"], samp_stat=[300]
    )

    assert result["status"] == "ok"
    assert result["mode"] == "genesys"
    assert result["accessions"] == 7
    assert result["total_matching"] == 7
    assert result["truncated"] is False
    assert result["filter"] == {
        "taxonomy": {"genus": ["Phaseolus"]},
        "countryOfOrigin": {"code3": ["COL"]},
        "sampStat": [300],
    }
    assert result["coordinate_columns"] == {"latitude": "DECLATITUDE", "longitude": "DECLONGITUDE"}

    assert state.mode is SourceMode.GENESYS
    assert state.original_count == 7
    assert "ACCENUMB" in state.original_list.columns
    for column in CANDIDATE_EXTRA_COLUMNS:
        assert column in state.candidate_list.columns
    assert state.extras["genesys_filter"] == result["filter"]

    # The body sent to Genesys is the pruned filter and uses the API-Token header.
    body = json.loads(requests[-1].content)
    assert body == result["filter"]
    assert requests[-1].headers["Authorization"] == "API-Token token"


def test_genesys_mode_reports_truncation(session) -> None:
    """When more accessions match than the cap, the result says so."""
    state, paths, _ = session
    client, requests = _mock_genesys_client(total=50)
    skill = ListAccessionsSkill(client_factory=lambda: client)

    result = skill.run(state, paths, source="genesys", crop=["bean"], max_records=20)

    assert result["status"] == "ok"
    assert result["accessions"] == 20
    assert result["total_matching"] == 50
    assert result["truncated"] is True
    assert "only the first 20" in result["message"]
    assert state.candidate_count == 20

    # The count call asks for 1 record, the download never exceeds the cap per page.
    sizes = [int(r.url.params["l"]) for r in requests if "l" in r.url.params]
    assert sizes[0] == 1
    assert max(sizes[1:]) <= 20


def test_genesys_mode_no_results_keeps_state(session) -> None:
    """Zero matches return an error and leave the current lists untouched."""
    state, paths, manager = session
    uploaded = _upload(manager, "accessions_sample.xlsx")
    ListAccessionsSkill().run(state, paths, source="local", file_path=str(uploaded))

    client, _ = _mock_genesys_client(total=0)
    result = ListAccessionsSkill(client_factory=lambda: client).run(
        state, paths, source="genesys", text="nothing-matches"
    )

    assert result["status"] == "error"
    assert result["total_matching"] == 0
    assert state.mode is SourceMode.LOCAL
    assert state.original_count == 5


def test_genesys_mode_auth_error_is_formal(session) -> None:
    """A 401 from Genesys becomes a formal error message, not an exception."""
    state, paths, _ = session
    client = GenesysClient(
        "https://api.test.genesys",
        ApiTokenAuth("bad"),
        transport=httpx.MockTransport(lambda r: httpx.Response(401, text="unauthorized")),
    )
    result = ListAccessionsSkill(client_factory=lambda: client).run(
        state, paths, source="genesys", genus=["Zea"]
    )

    assert result["status"] == "error"
    assert "credentials" in result["message"]
    assert not state.has_data


def test_genesys_mode_query_alias_and_invalid_samp_stat(session) -> None:
    """`query` still works as alias of `text`; bad samp_stat is reported."""
    state, paths, _ = session
    client, requests = _mock_genesys_client(total=1)
    skill = ListAccessionsSkill(client_factory=lambda: client)

    ok = skill.run(state, paths, source="genesys", query="drought")
    assert ok["status"] == "ok"
    assert json.loads(requests[-1].content) == {"_text": "drought"}

    bad = skill.run(state, paths, source="genesys", samp_stat=["landrace"])
    assert bad["status"] == "error"
    assert "Invalid Genesys criteria" in bad["message"]


def test_genesys_mode_tolerates_text_encoded_arguments(session) -> None:
    """Arguments serialised as text by small models are decoded correctly."""
    state, paths, _ = session
    client, requests = _mock_genesys_client(total=2)
    skill = ListAccessionsSkill(client_factory=lambda: client)

    # Exactly what llama3.1 sent in the real session.
    result = skill.run(
        state,
        paths,
        source="genesys",
        country_of_origin='["COL"]',
        crop='["bean"]',
        samp_stat="[300]",
        text="",
        historic="false",
        max_records="20",
    )

    assert result["status"] == "ok"
    assert json.loads(requests[-1].content) == {
        "crop": ["beans"],
        "countryOfOrigin": {"code3": ["COL"]},
        "sampStat": [300],
        "historic": False,
    }


def test_list_normalisation_edge_cases() -> None:
    """Nested lists, separated strings and floats are all normalised."""
    body = ListAccessionsSkill.build_genesys_filter(
        samp_stat=[[300], "100, 500", 999.0],
        country_of_origin="COL; per",
        genus=['["Phaseolus"]'],
        text=["drought", "tolerance"],
    ).to_body()

    assert body["sampStat"] == [300, 100, 500, 999]
    assert body["countryOfOrigin"] == {"code3": ["COL", "PER"]}
    assert body["taxonomy"] == {"genus": ["Phaseolus"]}
    assert body["_text"] == "drought tolerance"

    with pytest.raises(ValueError) as info:
        ListAccessionsSkill.build_genesys_filter(samp_stat=["landrace", 300.5])

    assert "landrace" in str(info.value)


def test_genesys_mode_resolves_crop_names_and_reports_unknown(session) -> None:
    """User crop names map to Genesys codes; unknown ones are dropped and reported."""
    state, paths, _ = session
    client, requests = _mock_genesys_client(total=3)
    skill = ListAccessionsSkill(client_factory=lambda: client)

    result = skill.run(state, paths, source="genesys", crop=["frijol", "Corn", "papaya"])

    assert result["status"] == "ok"
    assert result["filter"] == {"crop": ["beans", "maize"]}
    assert result["unresolved_crops"] == ["papaya"]
    assert "papaya" in result["message"]

    # The catalogue is fetched once and cached for later calls.
    crop_calls = [r for r in requests if r.url.path == "/api/v2/crop"]
    assert len(crop_calls) == 1
    skill.run(state, paths, source="genesys", crop=["bean"])
    assert len([r for r in requests if r.url.path == "/api/v2/crop"]) == 1


def test_genesys_mode_retries_without_crop_when_taxonomy_given(session) -> None:
    """Crop + taxonomy returning 0 triggers a taxonomy-only retry."""
    state, paths, _ = session
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v2/crop":
            return httpx.Response(200, json=MOCK_CROPS)

        body = json.loads(request.content)
        total = 0 if "crop" in body else 4
        size = int(request.url.params["l"])
        content = [_genesys_dto(f"G{i}") for i in range(min(total, size))]
        return httpx.Response(
            200,
            json={"content": content, "number": 0, "size": size, "totalElements": total,
                  "totalPages": 1, "last": True},
        )

    client = GenesysClient(
        "https://api.test.genesys", ApiTokenAuth("t"), transport=httpx.MockTransport(handler)
    )
    result = ListAccessionsSkill(client_factory=lambda: client).run(
        state, paths, source="genesys", crop=["bean"], genus=["Phaseolus"], species=["vulgaris"],
        country_of_origin=["COL"], samp_stat=[300],
    )

    assert result["status"] == "ok"
    assert result["accessions"] == 4
    assert "crop" not in result["filter"]
    assert result["filter"]["taxonomy"] == {"genus": ["Phaseolus"], "species": ["vulgaris"]}
    assert any("repeated" in note for note in result["notes"])
    assert "taxonomy" in state.activity_log[-1].description or state.candidate_count == 4
