"""Tests for the ``passport_filter`` skill and its filtering engine."""

import json
from pathlib import Path

import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from core.session import SessionManager
from core.state import CANDIDATE_EXTRA_COLUMNS, SessionState, SourceMode
from skills.arguments import as_dict_list
from skills.passport_filter.engine import (
    FilterError,
    build_condition,
    describe_columns,
    normalise_operator,
    resolve_column,
)
from skills.passport_filter.skill import PassportFilterSkill


@pytest.fixture
def frame() -> pd.DataFrame:
    """Passport table as loaded from Excel (all text) with a few gaps."""
    return pd.DataFrame(
        {
            "ACCENUMB": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "GENUS": ["Phaseolus"] * 6,
            "SPECIES": ["vulgaris", "vulgaris", "lunatus", "vulgaris", "coccineus", "vulgaris"],
            "ORIGCTY": ["COL", "PER", "MEX", "col ", "COL", None],
            "SAMPSTAT": ["300", "300", "100", "300", "999", "300"],
            "DECLATITUDE": ["3.4", "-12.0", "19.4", None, "4.7", "6.1"],
            "ELEVATION": ["1000", "3200", "2100", "800", None, "1500"],
            "COLLSITE": ["Palmira, Valle del Cauca", "Cusco", "Oaxaca", "Popayán", "Bogotá", "Cauca"],
        }
    )


@pytest.fixture
def session(tmp_path: Path, frame: pd.DataFrame):
    """Session with the fixture loaded as Original/Candidate lists."""
    manager = SessionManager(tmp_path / "tmp")
    paths = manager.get_paths("s1")
    state = SessionState(session_id="s1")
    state.set_original_list(frame, SourceMode.LOCAL)
    return state, paths


# ------------------------------------------------------------------ engine
def test_resolve_column_variants(frame: pd.DataFrame) -> None:
    """Exact, case-insensitive, alias and substring resolution all work."""
    columns = list(frame.columns)

    assert resolve_column("ORIGCTY", columns) == "ORIGCTY"
    assert resolve_column("origcty", columns) == "ORIGCTY"
    assert resolve_column("country", columns) == "ORIGCTY"
    assert resolve_column("País", columns) == "ORIGCTY"
    assert resolve_column("latitude", columns) == "DECLATITUDE"
    assert resolve_column("status", columns) == "SAMPSTAT"
    assert resolve_column("coll site", columns) == "COLLSITE"

    with pytest.raises(FilterError) as info:
        resolve_column("longitude", columns)

    assert "Available columns" in str(info.value)


def test_normalise_operator_aliases() -> None:
    """Symbols and synonyms map to the canonical operators."""
    assert normalise_operator("=") == "equals"
    assert normalise_operator("EQ") == "equals"
    assert normalise_operator(">=") == "gte"
    assert normalise_operator("not in") == "not_in"
    assert normalise_operator(None) == "equals"

    with pytest.raises(FilterError):
        normalise_operator("approximately")


def test_build_condition_shapes(frame: pd.DataFrame) -> None:
    """List values, delimited strings and unary operators are normalised."""
    columns = list(frame.columns)

    listed = build_condition({"column": "country", "operator": "in", "value": "COL, PER"}, columns)
    assert listed.value == ["COL", "PER"]

    promoted = build_condition({"column": "ORIGCTY", "operator": "equals", "value": ["COL", "PER"]}, columns)
    assert promoted.operator == "in"

    unary = build_condition({"column": "latitude", "operator": "not_null", "value": "ignored"}, columns)
    assert unary.value is None
    assert unary.describe() == "DECLATITUDE is not empty"

    with pytest.raises(FilterError):
        build_condition({"column": "ELEVATION", "operator": "between", "value": [1]}, columns)

    with pytest.raises(FilterError):
        build_condition({"operator": "equals", "value": 1}, columns)


def test_describe_columns_types(frame: pd.DataFrame) -> None:
    """Numeric columns report ranges; text columns report frequent values."""
    summaries = {item["column"]: item for item in describe_columns(frame, ["ELEVATION", "ORIGCTY"])}

    assert summaries["ELEVATION"]["type"] == "numeric"
    assert summaries["ELEVATION"]["min"] == 800.0
    assert summaries["ORIGCTY"]["type"] == "text"
    assert summaries["ORIGCTY"]["top_values"]["COL"] == 2
    assert summaries["ORIGCTY"]["nulls"] == 1


# ------------------------------------------------------------------- skill
def test_filter_text_in_is_case_and_space_insensitive(session) -> None:
    """'in' matches 'col ' and 'COL'; criteria are recorded on surviving rows."""
    state, paths = session
    result = PassportFilterSkill().run(
        state, paths, conditions=[{"column": "country", "operator": "in", "value": ["col"]}]
    )

    assert result["status"] == "ok"
    assert result["accessions_before"] == 6
    assert result["accessions_after"] == 3
    assert result["criteria"] == "ORIGCTY in [col]"
    assert list(state.candidate_list["ACCENUMB"]) == ["G1", "G4", "G5"]
    assert set(state.candidate_list["criteria_passport"]) == {"ORIGCTY in [col]"}

    # The Original list is never touched.
    assert state.original_count == 6
    assert "criteria_passport" not in state.original_list.columns


def test_filters_chain_and_accumulate_criteria(session) -> None:
    """A second call narrows the first result and appends its criteria."""
    state, paths = session
    skill = PassportFilterSkill()

    skill.run(state, paths, conditions=[{"column": "SAMPSTAT", "operator": "equals", "value": 300}])
    assert state.candidate_count == 4

    result = skill.run(
        state, paths, conditions=[{"column": "ELEVATION", "operator": "gte", "value": "1000"}]
    )

    assert result["accessions_after"] == 3
    assert list(state.candidate_list["ACCENUMB"]) == ["G1", "G2", "G6"]
    assert state.candidate_list["criteria_passport"].iloc[0] == "SAMPSTAT = 300 | ELEVATION >= 1000"


def test_numeric_between_contains_and_or_logic(session) -> None:
    """between, contains and OR logic evaluate correctly."""
    state, paths = session
    result = PassportFilterSkill().run(
        state,
        paths,
        conditions=[
            {"column": "latitude", "operator": "between", "value": [3, 7]},
            {"column": "COLLSITE", "operator": "contains", "value": "cusco"},
        ],
        logic="or",
    )

    assert result["status"] == "ok"
    assert list(state.candidate_list["ACCENUMB"]) == ["G1", "G2", "G5", "G6"]
    assert result["criteria"] == "DECLATITUDE between [3, 7] OR COLLSITE contains cusco"


def test_conditions_as_json_text_and_null_checks(session) -> None:
    """Conditions serialised as text by the model are decoded; not_null works."""
    state, paths = session
    conditions = json.dumps([{"column": "DECLATITUDE", "operator": "not_null"}])
    result = PassportFilterSkill().run(state, paths, conditions=conditions)

    assert result["status"] == "ok"
    assert result["accessions_after"] == 5
    assert "G4" not in list(state.candidate_list["ACCENUMB"])


def test_zero_matches_keeps_list_and_gives_hints(session) -> None:
    """A filter matching nothing does not change the list and returns hints."""
    state, paths = session
    result = PassportFilterSkill().run(
        state, paths, conditions=[{"column": "ORIGCTY", "operator": "equals", "value": "BRA"}]
    )

    assert result["status"] == "error"
    assert state.candidate_count == 6
    assert result["column_hints"][0]["column"] == "ORIGCTY"
    assert "COL" in result["column_hints"][0]["top_values"]
    assert "unchanged" in state.activity_log[-1].description


def test_errors_are_reported(session) -> None:
    """Missing data, bad column, bad operator, missing conditions and bad JSON."""
    state, paths = session
    skill = PassportFilterSkill()

    assert skill.run(state, paths, conditions=[])["status"] == "error"
    assert skill.run(state, paths, conditions="not json")["status"] == "error"

    bad_column = skill.run(state, paths, conditions=[{"column": "nope", "operator": "equals", "value": 1}])
    assert bad_column["status"] == "error" and "Available columns" in bad_column["message"]

    bad_operator = skill.run(state, paths, conditions=[{"column": "ORIGCTY", "operator": "~", "value": 1}])
    assert bad_operator["status"] == "error" and "operators" in bad_operator

    text_numeric = skill.run(state, paths, conditions=[{"column": "COLLSITE", "operator": "gt", "value": 1}])
    assert text_numeric["status"] == "error" and "numeric" in text_numeric["message"]

    empty_state = SessionState(session_id="empty")
    assert skill.run(empty_state, paths, action="describe")["status"] == "error"
    assert skill.run(state, paths, action="explode")["status"] == "error"


def test_describe_and_reset(session) -> None:
    """describe summarises requested columns; reset restores the Original list."""
    state, paths = session
    skill = PassportFilterSkill()

    described = skill.run(state, paths, action="describe", columns=["country", "elevation"], top="3")
    assert described["status"] == "ok"
    assert [item["column"] for item in described["columns"]] == ["ORIGCTY", "ELEVATION"]
    assert len(described["columns"][0]["top_values"]) <= 3

    everything = skill.run(state, paths, action="describe")
    assert all(not item["column"].startswith("criteria_") for item in everything["columns"])

    skill.run(state, paths, conditions=[{"column": "ORIGCTY", "operator": "equals", "value": "PER"}])
    assert state.candidate_count == 1

    reset = skill.run(state, paths, action="reset")
    assert reset["status"] == "ok"
    assert reset["accessions_after"] == 6
    assert state.candidate_count == 6
    assert state.candidate_list["criteria_passport"].isna().all()
    for column in CANDIDATE_EXTRA_COLUMNS:
        assert column in state.candidate_list.columns


def test_registry_discovers_passport_filter() -> None:
    """The skill is auto-registered with the others."""
    registry = SkillRegistry().discover()

    assert "passport_filter" in registry.names()


def test_as_dict_list_shapes() -> None:
    """The shared helper accepts lists, dicts, JSON text and rejects garbage."""
    assert as_dict_list(None) == []
    assert as_dict_list({"a": 1}) == [{"a": 1}]
    assert as_dict_list('[{"a": 1}]') == [{"a": 1}]
    assert as_dict_list(['{"a": 1}']) == [{"a": 1}]

    with pytest.raises(ValueError):
        as_dict_list([1, 2])
