"""Tests for the ``traits_analysis`` skill and its engine (mocked Genesys API)."""

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from core.session import SessionManager
from core.state import SessionState, SourceMode
from sdks.genesys import ApiTokenAuth, GenesysClient
from sdks.genesys.traits import Descriptor
from skills.traits_analysis.engine import (
    TraitsError,
    aggregate_observations,
    aggregate_values,
    build_groups,
    detect_trait_columns,
    resolve_trait_column,
    select_descriptors,
    tercile_labels,
)
from skills.traits_analysis.skill import TraitsAnalysisSkill

BASE_URL = "https://api.test"
DS_FE = "ds-fe"
DS_CHAR = "ds-char"
DESCRIPTORS_FE = [
    {"uuid": "d-fe", "title": "Seed iron concentration", "columnName": "Fe.Mean", "dataType": "NUMERIC", "uom": "mg/kg", "category": "EVALUATION"},
    {"uuid": "d-zn", "title": "Seed zinc concentration", "columnName": "Zn.Mean", "dataType": "NUMERIC", "uom": "mg/kg", "category": "EVALUATION"},
]
DESCRIPTORS_CHAR = [
    {"uuid": "d-color", "title": "Seed coat colour", "columnName": "SEEDCOL", "dataType": "CODED", "category": "CHARACTERIZATION",
     "terms": [{"code": "1", "title": "White"}, {"code": "2", "title": "Red"}]},
    {"uuid": "d-100sw", "title": "100 seed weight", "columnName": "SW100", "dataType": "NUMERIC", "uom": "g", "category": "CHARACTERIZATION"},
]
# Data rows in the real production shape (accession uuid string, list values).
DATA_FE = [
    {"accession": "u1", "accessionNumber": "G1", "d-fe": [55.9], "d-zn": [30.0]},
    {"accession": "u2", "accessionNumber": "G2", "d-fe": [70.0, 74.0], "d-zn": [35.0]},
    {"accession": "u3", "accessionNumber": "G3", "d-fe": [90.0]},
]
DATA_CHAR = [
    {"accession": "u1", "accessionNumber": "G1", "d-color": ["Red", "Red", "White"], "d-100sw": [20.0]},
    {"accession": "u2", "accessionNumber": "G2", "d-color": ["White"], "d-100sw": [25.0]},
    {"accession": "u4", "accessionNumber": "G4", "d-color": ["Red"], "d-100sw": [40.0]},
]


def genesys_handler(request: httpx.Request) -> httpx.Response:
    """Scripted Genesys responses for the trait workflow."""
    path = request.url.path

    if path == "/api/v2/dataset/accessions-datasets":
        return httpx.Response(200, json=[DS_FE, DS_CHAR])
    if path == f"/api/v2/dataset/{DS_FE}":
        return httpx.Response(200, json={"uuid": DS_FE, "title": "Seed Iron and Zinc", "accessionCount": 3, "descriptorCount": 2})
    if path == f"/api/v2/dataset/{DS_CHAR}":
        return httpx.Response(200, json={"uuid": DS_CHAR, "title": "Beans characterization", "accessionCount": 3, "descriptorCount": 2})
    if path == f"/api/v2/dataset/{DS_FE}/descriptors":
        return httpx.Response(200, json=DESCRIPTORS_FE)
    if path == f"/api/v2/dataset/{DS_CHAR}/descriptors":
        return httpx.Response(200, json=DESCRIPTORS_CHAR)
    if path == "/api/v2/dataset/data":
        datasets = request.url.params.get_list("datasetUuids")
        fields = set(request.url.params.get_list("fields"))
        body = json.loads(request.content)
        wanted = set(body["filters"]["accession"]["uuid"])
        source = DATA_FE if DS_FE in datasets else DATA_CHAR
        rows = [
            {key: value for key, value in row.items() if key in ("accession", "accessionNumber") or key in fields}
            for row in source if row["accession"] in wanted
        ]
        return httpx.Response(200, json={"content": rows, "number": 0, "size": 500, "totalElements": len(rows), "totalPages": 1, "last": True})

    return httpx.Response(404, text=f"unexpected {path}")


def make_state(frame: pd.DataFrame, mode: SourceMode = SourceMode.GENESYS) -> SessionState:
    """Session state with the given list loaded."""
    state = SessionState(session_id="s1")
    state.set_original_list(frame, mode)
    return state


def make_skill(handler=genesys_handler) -> TraitsAnalysisSkill:
    """Skill wired to a mocked Genesys client."""
    client = GenesysClient(BASE_URL, ApiTokenAuth("tok"), transport=httpx.MockTransport(handler))
    return TraitsAnalysisSkill(client_factory=lambda: client)


def genesys_frame() -> pd.DataFrame:
    """Candidate list in Genesys mode (UUID column), one accession without any trait data."""
    return pd.DataFrame({
        "INSTCODE": ["COL003"] * 5,
        "ACCENUMB": ["G1", "G2", "G3", "G4", "G5"],
        "UUID": ["u1", "u2", "u3", "u4", "u5"],
        "GENUS": ["Phaseolus"] * 5,
        "ORIGCTY": ["COL", "COL", "PER", "COL", "MEX"],
    })


@pytest.fixture
def paths(tmp_path: Path):
    """Session folders in a temporary directory."""
    return SessionManager(tmp_path).get_paths("s1")


# ------------------------------------------------------------------ engine
def test_detect_trait_columns_ignores_passport_and_annotations() -> None:
    """Only non-passport, non-empty columns are reported as traits."""
    frame = pd.DataFrame({
        "ACCENUMB": ["G1", "G2", "G3"], "ORIGCTY": ["COL"] * 3, "Latitude": [1.0, 2.0, 3.0],
        "yield_kg": [1000, 1200, 800], "seed_color": ["red", "white", "red"], "empty": [None] * 3,
        "criteria_traits": [None] * 3, "cluster_traits": [None] * 3,
    })
    found = {item.column: item for item in detect_trait_columns(frame)}

    assert set(found) == {"yield_kg", "seed_color"}
    assert found["yield_kg"].numeric and found["yield_kg"].summary["max"] == 1200
    assert not found["seed_color"].numeric and found["seed_color"].summary["categories"] == ["red", "white"]


def test_resolve_trait_column_variants() -> None:
    """Exact, case-insensitive, prefix-less and substring names resolve."""
    frame = pd.DataFrame({"trait_Fe.Mean": [1], "trait_Zn.Mean": [2], "seed_color": ["red"]})

    assert resolve_trait_column("trait_Fe.Mean", frame) == "trait_Fe.Mean"
    assert resolve_trait_column("fe.mean", frame) == "trait_Fe.Mean"
    assert resolve_trait_column("Zn", frame) == "trait_Zn.Mean"
    assert resolve_trait_column("Seed Color", frame) == "seed_color"

    with pytest.raises(TraitsError):
        resolve_trait_column("protein", frame)
    with pytest.raises(TraitsError):
        resolve_trait_column("mean", frame)  # ambiguous


def test_select_descriptors_by_query_and_names() -> None:
    """Descriptors are chosen by free text words or by column names."""
    descriptors = [Descriptor.model_validate(item) for item in DESCRIPTORS_FE + DESCRIPTORS_CHAR]

    assert [d.label for d in select_descriptors(descriptors, query="iron zinc")] == ["Fe.Mean", "Zn.Mean"]
    assert [d.label for d in select_descriptors(descriptors, names=["trait_SEEDCOL", "Fe"])] == ["SEEDCOL", "Fe.Mean"]
    assert len(select_descriptors(descriptors)) == 4
    assert select_descriptors(descriptors, query="protein") == []


def test_aggregate_values_and_observations() -> None:
    """Numeric traits are averaged, categorical traits take the mode; several rows per accession pool."""
    assert aggregate_values([70.0, 74.0], numeric=True) == 72.0
    assert aggregate_values(["Red", "Red", "White"], numeric=False) == "Red"
    assert aggregate_values([], numeric=True) is None
    assert aggregate_values("5", numeric=True) == 5.0

    descriptors = [Descriptor.model_validate(item) for item in DESCRIPTORS_FE]
    frame = pd.DataFrame([
        {"uuid": "u1", "accessionNumber": "G1", "Fe.Mean": 55.9, "Zn.Mean": 30.0},
        {"uuid": "u1", "accessionNumber": "G1", "Fe.Mean": [60.1], "Zn.Mean": None},
        {"uuid": "u2", "accessionNumber": "G2", "Fe.Mean": [70.0, 74.0], "Zn.Mean": 35.0},
    ])
    aggregated = aggregate_observations(frame, descriptors).set_index("uuid")

    assert aggregated.loc["u1", "trait_Fe.Mean"] == 58.0 and aggregated.loc["u1", "trait_Zn.Mean"] == 30.0
    assert aggregated.loc["u2", "trait_Fe.Mean"] == 72.0


def test_tercile_labels_and_groups() -> None:
    """Numeric columns split into terciles, categorical by value, combined with ' & '."""
    frame = pd.DataFrame({"fe": [10, 20, 30, 40, 50, 60, None], "color": ["red", "red", "white", "white", "red", "white", "red"]})

    assert list(tercile_labels(frame["fe"]).dropna()) == ["low", "low", "medium", "medium", "high", "high"]

    labels, codes = build_groups(frame, ["fe", "color"])
    assert labels[0] == "fe=low & color=red" and labels[5] == "fe=high & color=white"
    assert pd.isna(labels[6]) and pd.isna(codes[6])
    assert codes[0] == 1 and codes.dropna().max() == len(set(labels.dropna()))

    with pytest.raises(TraitsError):
        build_groups(frame, [])


# ------------------------------------------------------------------- skill
def test_registry_discovers_traits_skill() -> None:
    """The skill is auto-discovered with its tool schema."""
    registry = SkillRegistry()
    registry.discover()

    assert "traits_analysis" in registry.names()
    tool = registry.get("traits_analysis").to_openai_tool()
    assert tool["function"]["parameters"]["properties"]["action"]["enum"] == ["detect", "fetch", "group"]


def test_requires_loaded_list(paths) -> None:
    """Every action needs a loaded list."""
    result = make_skill().run(SessionState(session_id="s1"), paths, action="detect")
    assert result["status"] == "error" and "Load accessions" in result["message"]


def test_detect_local_columns(paths) -> None:
    """Local lists report their trait columns and do not create subsets."""
    frame = pd.DataFrame({"ACCENUMB": ["G1", "G2"], "ORIGCTY": ["COL", "PER"], "yield_kg": [1000, 800]})
    state = make_state(frame, SourceMode.LOCAL)
    result = make_skill().run(state, paths, action="detect")

    assert result["status"] == "ok" and result["subsets_created"] is False
    assert [item["column"] for item in result["trait_columns"]] == ["yield_kg"]


def test_fetch_adds_trait_columns_with_coverage(paths) -> None:
    """fetch downloads the selected descriptors, aggregates and merges them by UUID."""
    state = make_state(genesys_frame())
    result = make_skill().run(state, paths, action="fetch", query="iron zinc")

    assert result["status"] == "ok" and result["subsets_created"] is False
    assert [t["column"] for t in result["traits"]] == ["trait_Fe.Mean", "trait_Zn.Mean"]
    assert result["accessions_with_data"] == 3 and result["accessions_without_data"] == 2
    assert result["coverage"] == {"trait_Fe.Mean": 3, "trait_Zn.Mean": 2}

    candidate = state.candidate_list.set_index("UUID")
    assert candidate.loc["u1", "trait_Fe.Mean"] == 55.9
    assert candidate.loc["u2", "trait_Fe.Mean"] == 72.0  # mean of two observations
    assert pd.isna(candidate.loc["u4", "trait_Fe.Mean"]) and pd.isna(candidate.loc["u5", "trait_Fe.Mean"])
    assert len(state.candidate_list) == 5  # nothing removed
    assert state.original_list is not None and "trait_Fe.Mean" not in state.original_list.columns
    assert "trait_Fe.Mean" in state.extras["trait_descriptors"]


def test_fetch_without_selection_takes_all_descriptors_up_to_cap(paths) -> None:
    """No query and no names -> every descriptor (capped) across datasets, categorical uses the mode."""
    state = make_state(genesys_frame())
    result = make_skill().run(state, paths, action="fetch")

    assert result["status"] == "ok" and len(result["traits"]) == 4
    candidate = state.candidate_list.set_index("UUID")
    assert candidate.loc["u1", "trait_SEEDCOL"] == "Red" and candidate.loc["u4", "trait_SW100"] == 40.0

    capped = make_state(genesys_frame())
    assert len(make_skill().run(capped, paths, action="fetch", max_descriptors=1)["traits"]) == 1


def test_fetch_unknown_query_lists_available(paths) -> None:
    """An unmatched query returns the available descriptors instead of downloading."""
    result = make_skill().run(make_state(genesys_frame()), paths, action="fetch", query="protein")
    assert result["status"] == "error" and "Fe.Mean" in result["available"]


def test_fetch_requires_identifiers(paths) -> None:
    """A local list without UUID / accession numbers cannot be matched in Genesys."""
    state = make_state(pd.DataFrame({"name": ["a", "b"], "yield": [1, 2]}), SourceMode.LOCAL)
    result = make_skill().run(state, paths, action="fetch")
    assert result["status"] == "error" and "UUID" in result["message"]


def test_fetch_no_datasets(paths) -> None:
    """Genesys without datasets for the accessions -> informative ok result."""
    skill = make_skill(lambda request: httpx.Response(200, json=[]))
    result = skill.run(make_state(genesys_frame()), paths, action="fetch")
    assert result["status"] == "ok" and result["datasets"] == 0 and result["subsets_created"] is False


def test_group_single_condition_is_binary(paths) -> None:
    """One trait + one condition -> cluster_traits 1/0 and criteria with values."""
    state = make_state(genesys_frame())
    skill = make_skill()
    skill.run(state, paths, action="fetch", query="iron")
    result = skill.run(state, paths, action="group", conditions=[{"trait": "Fe.Mean", "operator": "gte", "value": 60}])

    assert result["status"] == "ok" and result["subsets_created"] is True
    candidate = state.candidate_list.set_index("UUID")
    assert [(-1 if pd.isna(v) else int(v)) for v in candidate["cluster_traits"]] == [0, 1, 1, -1, -1]
    assert candidate.loc["u2", "criteria_traits"].startswith("'trait_Fe.Mean >= 60' -> cluster_traits 1 (meets) / 0 (does not meet): meets [trait_Fe.Mean=72.00]")
    assert "no trait data" in candidate.loc["u5", "criteria_traits"]
    assert result["assigned"] == 3 and result["without_data"] == 2
    assert {g["cluster_traits"]: g["accessions"] for g in result["groups"]} == {0: 1, 1: 2}
    assert len(state.candidate_list) == 5


def test_group_several_traits_builds_combinations(paths) -> None:
    """Several traits -> terciles/categories combined; local columns work too."""
    frame = pd.DataFrame({
        "ACCENUMB": [f"G{i}" for i in range(6)],
        "yield_kg": [500, 700, 900, 1100, 1300, 1500],
        "seed_color": ["red", "white", "red", "white", "red", "white"],
    })
    state = make_state(frame, SourceMode.LOCAL)
    result = make_skill().run(state, paths, action="group", traits=["yield", "seed color"])

    assert result["status"] == "ok" and result["subsets_created"] is True
    assert result["assigned"] == 6 and len(result["groups"]) == 6
    first = state.candidate_list.loc[0]
    assert first["cluster_traits"] == 1 and "yield_kg=low & seed_color=red" in first["criteria_traits"]


def test_group_conditions_and_traits_chain_criteria(paths) -> None:
    """Conditions on several traits become met/not met parts; criteria chain across calls."""
    state = make_state(genesys_frame())
    skill = make_skill()
    skill.run(state, paths, action="fetch")
    skill.run(state, paths, action="group", conditions=[{"trait": "Fe.Mean", "operator": "gte", "value": 60}])
    result = skill.run(
        state, paths, action="group",
        conditions=[{"trait": "Fe.Mean", "operator": "gte", "value": 60}, {"trait": "SEEDCOL", "operator": "equals", "value": "Red"}],
    )

    assert result["status"] == "ok"
    candidate = state.candidate_list.set_index("UUID")
    assert "trait_Fe.Mean >= 60: not met & trait_SEEDCOL = Red: met" in candidate.loc["u1", "criteria_traits"]
    assert " | " in candidate.loc["u1", "criteria_traits"]  # two rounds chained
    assert candidate.loc["u1", "cluster_traits"] in (1, 2) and pd.isna(candidate.loc["u3", "cluster_traits"])  # u3 has no colour
    assert len(result["groups"]) == 2  # u1 (not met, met) and u2 (met, not met); u3/u4 lack one trait


def test_group_errors(paths) -> None:
    """Missing traits or unknown columns produce formal errors."""
    state = make_state(genesys_frame())
    skill = make_skill()

    assert skill.run(state, paths, action="group")["status"] == "error"
    result = skill.run(state, paths, action="group", traits=["protein"])
    assert result["status"] == "error" and "not found" in result["message"]
