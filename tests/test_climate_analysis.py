"""Tests for the ``climate_analysis`` skill and its engine (mocked Subsetting API)."""

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest

from agent.skill_registry import SkillRegistry
from core.session import SessionManager
from core.state import SessionState, SourceMode
from sdks.subsetting import FallbackAuth, SubsettingClient
from sdks.subsetting.models import Indicator
from skills.climate_analysis.engine import (
    ClimateError,
    aggregate_indicator,
    infer_crop,
    month_columns,
    normalise_months,
    normalise_statistic,
    summarise_catalogue,
)
from skills.climate_analysis.skill import ClimateAnalysisSkill

FIXTURES = Path(__file__).parent / "fixtures"

INDICATORS = [
    {"category": "Drought stress", "indicators": [
        {"id": "i_cdd", "name": "Consecutive dry days", "pref": "CDD", "indicator_type": "generic", "crop": "NC", "unit": "days"},
        {"id": "i_train", "name": "Total precipitation", "pref": "t_rain", "indicator_type": "generic", "crop": "NC", "unit": "mm"},
    ]},
    {"category": "Heat stress", "indicators": [
        {"id": "i_tx", "name": "Average maximum temperature", "pref": "TX", "indicator_type": "generic", "crop": "NC", "unit": "°C"},
    ]},
    {"category": "Crop-specific indicators", "indicators": [
        {"id": "i_heat_beans", "name": "Number of days high daytime temperatures", "pref": "days_heat", "indicator_type": "specific", "crop": "Beans", "unit": "days"},
        {"id": "i_heat_maize", "name": "Number of days high daytime temperatures", "pref": "days_heat", "indicator_type": "specific", "crop": "Maize", "unit": "days"},
    ]},
    {"category": "Soil indicators", "indicators": [
        {"id": "i_ph", "name": "pH", "pref": "PHIHOX", "indicator_type": "extracted", "crop": "NC", "unit": None},
        {"id": "i_tex", "name": "Soil texture type", "pref": "TEXMHT", "indicator_type": "categorical", "crop": "NC", "unit": ""},
    ]},
]
PERIODS = [{"id": f"p_{item['id']}_{label}", "indicator": item["id"], "period": label, "ssp": "hist"}
           for group in INDICATORS for item in group["indicators"] for label in ("mean", "1983-2016")]

# Per-cell synthetic values: cells 1..4 have data, cell 5 has none.
CELL_VALUES = {
    1: {"CDD": 5.0, "t_rain": 100.0, "TX": 25.0, "days_heat": 2.0, "PHIHOX": 6.5, "TEXMHT": 3},
    2: {"CDD": 15.0, "t_rain": 50.0, "TX": 30.0, "days_heat": 20.0, "PHIHOX": 5.5, "TEXMHT": 4},
    3: {"CDD": 25.0, "t_rain": 20.0, "TX": 33.0, "days_heat": 30.0, "PHIHOX": 7.2, "TEXMHT": 3},
    4: {"CDD": 10.0, "t_rain": 80.0, "TX": 28.0, "days_heat": 10.0, "PHIHOX": 6.0, "TEXMHT": 5},
}


def _data_row(pref: str, value: float) -> dict:
    """Build one indicators-data row (monthly for generic/specific, value/category otherwise)."""
    if pref == "PHIHOX":
        return {"pref_indicator": pref, "value": value}
    if pref == "TEXMHT":
        return {"pref_indicator": pref, "category": int(value)}
    row = {"pref_indicator": pref, **{f"month{m}": value for m in range(1, 13)}}
    row["month1"] = None  # a missing month, as seen in the real API
    return row


def make_handler(cluster_of=None):
    """Mock Subsetting API following the real response shapes."""
    pref_of = {item["id"]: item["pref"] for group in INDICATORS for item in group["indicators"]}
    indicator_of_period = {p["id"]: p["indicator"] for p in PERIODS}
    cluster_of = cluster_of or {1: 0, 2: 1, 3: 1, 4: 0}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path.endswith("/indicators"):
            return httpx.Response(200, json=INDICATORS)
        if path.endswith("/indicator-period"):
            return httpx.Response(200, json=PERIODS)
        body = json.loads(request.content)
        if path.endswith("/indicators-data"):
            prefs = {pref_of[indicator_of_period[p]] for p in body["indicators"]}
            response = []
            for cell in body["cellid"]:
                if cell in CELL_VALUES:
                    response.append({"cellid": cell, "data": [_data_row(p, CELL_VALUES[cell][p]) for p in prefs]})
            if not response:
                return httpx.Response(500, text="KeyError: 'cellid'")
            return httpx.Response(200, json={"response": response})
        if path.endswith("/cluster"):
            cells = body["cellid_list"][0]["cellids"]
            prefs = [pref_of[indicator_of_period[d["indicator"][0]]] for d in body["data"]]
            rows = []
            for cell in cells:
                if cell in CELL_VALUES and cell in cluster_of:
                    row = {"cellid": cell, "cluster_hac": cluster_of[cell], "crop_name": body["cellid_list"][0]["crop"]}
                    for pref in prefs:
                        row[f"{pref}_month1"] = CELL_VALUES[cell][pref]
                    rows.append(row)
            return httpx.Response(200, json={"data": rows, "summary": [{"indicator": prefs[0], "cluster_hac": 0, "mean": 1.0}], "calculate": []})
        return httpx.Response(404)

    handler.requests = requests  # type: ignore[attr-defined]
    return handler


def make_skill(handler=None) -> tuple[ClimateAnalysisSkill, SubsettingClient]:
    """Skill wired to a mocked Subsetting client."""
    handler = handler or make_handler()
    client = SubsettingClient("https://sb.test/api/subsetting/v1", FallbackAuth(access_token="jwt"),
                              transport=httpx.MockTransport(handler))
    return ClimateAnalysisSkill(client_factory=lambda: client), client


@pytest.fixture
def session(tmp_path: Path):
    """Session with 6 accessions: cells 1..4 with data, cell 5 without, one row without cellid."""
    manager = SessionManager(tmp_path / "tmp")
    paths = manager.get_paths("s1")
    frame = pd.DataFrame({
        "ACCENUMB": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "CROPNAME": ["beans"] * 6,
        "DECLATITUDE": [1, 2, 3, 4, 5, None],
        "DECLONGITUDE": [1, 2, 3, 4, 5, None],
    })
    state = SessionState(session_id="s1")
    state.set_original_list(frame, SourceMode.LOCAL)
    candidate = state.candidate_list.copy()
    candidate["cellid"] = pd.array([1, 2, 3, 4, 5, None], dtype="Int64")
    state.update_candidate_list(candidate)
    return state, paths


# ------------------------------------------------------------------ engine
def test_statistics_months_and_windows() -> None:
    """Statistic defaults/aliases, month ranges and window labels."""
    assert normalise_statistic(None, "t_rain") == "sum"
    assert normalise_statistic(None, "TX") == "mean"
    assert normalise_statistic("promedio", "t_rain") == "mean"
    with pytest.raises(ClimateError):
        normalise_statistic("median", "TX")

    assert normalise_months(None) == (1, 12)
    assert normalise_months("3-8") == (3, 8)
    assert normalise_months([10, 2]) == (10, 2)
    assert normalise_months(6) == (6, 6)
    assert month_columns((11, 2)) == ["month11", "month12", "month1", "month2"]
    with pytest.raises(ClimateError):
        normalise_months([0, 5])


def test_aggregate_indicator_kinds() -> None:
    """Monthly, extracted and categorical indicators aggregate correctly (NaN ignored)."""
    data = pd.DataFrame([
        {"cellid": 1, **_data_row("t_rain", 10.0)},
        {"cellid": 2, **_data_row("t_rain", 20.0)},
        {"cellid": 1, **_data_row("PHIHOX", 6.5)},
        {"cellid": 1, **_data_row("TEXMHT", 3)},
    ])
    train = Indicator(id="x", name="rain", pref="t_rain", indicator_type="generic")
    ph = Indicator(id="y", name="ph", pref="PHIHOX", indicator_type="extracted")
    tex = Indicator(id="z", name="tex", pref="TEXMHT", indicator_type="categorical")

    total = aggregate_indicator(data, train, "sum", (1, 12))
    assert total[1] == 110.0 and total[2] == 220.0  # month1 is missing -> 11 months
    assert aggregate_indicator(data, train, "mean", (1, 12))[1] == 10.0
    assert aggregate_indicator(data, train, "max", (2, 3))[2] == 20.0
    assert aggregate_indicator(data, ph, "mean", (1, 12))[1] == 6.5
    assert aggregate_indicator(data, tex, "mean", (1, 12))[1] == 3
    assert aggregate_indicator(data, Indicator(id="n", name="n", pref="CDD", indicator_type="generic"), "mean", (1, 12)).empty


def test_infer_crop_and_catalogue_summary() -> None:
    """Crop inference from CROPNAME and prefix-level catalogue summary."""
    assert infer_crop(pd.DataFrame({"CROPNAME": ["beans", "Common bean"]}), ["Beans", "Maize"]) == "Beans"
    assert infer_crop(pd.DataFrame({"CROPNAME": ["beans", "maize"]}), ["Beans", "Maize"]) is None
    assert infer_crop(pd.DataFrame({"ACCENUMB": ["x"]}), ["Beans"]) is None

    catalogue = [Indicator.model_validate({**item, "category": g["category"]}) for g in INDICATORS for item in g["indicators"]]
    summary = summarise_catalogue(catalogue, "drought")
    assert [item["pref"] for item in summary] == ["CDD", "t_rain"]
    heat = next(item for item in summarise_catalogue(catalogue) if item["pref"] == "days_heat")
    assert heat["crops"] == ["Beans", "Maize"]


# ------------------------------------------------------------------- skill
def test_list_indicators_remembers_selection(session) -> None:
    """list_indicators returns the catalogue and stores the chosen indicators."""
    state, paths = session
    skill, _ = make_skill()

    result = skill.run(state, paths, action="list_indicators", query="drought", indicators=["CDD", "t_rain"])

    assert result["status"] == "ok"
    assert [i["pref"] for i in result["indicators"]] == ["CDD", "t_rain"]
    assert result["remembered"] == ["CDD", "t_rain"]
    assert state.climate_indicators == ["i_cdd", "i_train"]
    assert "mean" in result["periods"]


def test_filter_by_climate_records_criteria_and_values(session) -> None:
    """Filter keeps matching cells, writes criteria+values and cluster_climate=0."""
    state, paths = session
    skill, _ = make_skill()

    result = skill.run(
        state, paths, action="filter",
        conditions=[{"indicator": "CDD", "operator": "gte", "value": 10, "statistic": "mean"},
                    {"indicator": "t_rain", "operator": "lt", "value": 1000}],
    )

    assert result["status"] == "ok"
    # CDD mean >= 10 -> cells 2,3,4 ; t_rain sum(11 months) < 1000 -> cells 2 (550), 3 (220), 4 (880)
    assert result["accessions_after"] == 3
    assert result["without_cellid"] == 1 and result["without_climate_data"] == 1
    assert list(state.candidate_list["ACCENUMB"]) == ["A2", "A3", "A4"]
    assert (state.candidate_list["cluster_climate"] == 0).all()
    text = state.candidate_list["criteria_climate"].iloc[0]
    assert text.startswith("CDD mean(m1-12) >= 10 AND t_rain sum(m1-12) < 1000 [CDD=15.00, t_rain=550.00]")
    assert state.climate_indicators == ["i_cdd", "i_train"]


def test_filter_json_text_conditions_and_or_logic(session) -> None:
    """Conditions as JSON text and OR logic work; criteria chain across calls."""
    state, paths = session
    skill, _ = make_skill()

    skill.run(state, paths, action="filter", conditions='[{"indicator":"TX","operator":"gt","value":32}]')
    assert list(state.candidate_list["ACCENUMB"]) == ["A3"]

    state.reset_candidate_list()
    candidate = state.candidate_list.copy()
    candidate["cellid"] = pd.array([1, 2, 3, 4, 5, None], dtype="Int64")
    state.update_candidate_list(candidate)

    first = skill.run(state, paths, action="filter", conditions=[{"indicator": "CDD", "operator": "lte", "value": 5, "statistic": "mean"}])
    second = skill.run(state, paths, action="filter",
                       conditions=[{"indicator": "TX", "operator": "gt", "value": 40}, {"indicator": "PHIHOX", "operator": "gte", "value": 6}],
                       logic="or")
    assert first["accessions_after"] == 1 and second["accessions_after"] == 1
    assert " | " in state.candidate_list["criteria_climate"].iloc[0]


def test_filter_zero_matches_keeps_list(session) -> None:
    """A filter nobody passes leaves the list unchanged and reports ranges."""
    state, paths = session
    skill, _ = make_skill()

    result = skill.run(state, paths, action="filter", conditions=[{"indicator": "CDD", "operator": "gt", "value": 999}])

    assert result["status"] == "error"
    assert state.candidate_count == 6
    # CDD is summed by default (11 available months): 25 * 11 = 275
    assert result["value_ranges"]["CDD sum(m1-12)"]["max"] == 275.0


def test_cluster_assigns_clusters_and_keeps_rows(session) -> None:
    """Cluster writes cluster_climate/criteria_climate without filtering."""
    state, paths = session
    skill, client = make_skill()
    skill.run(state, paths, action="list_indicators", indicators=["CDD", "TX"])

    result = skill.run(state, paths, action="cluster", months=[3, 8])

    assert result["status"] == "ok"
    assert state.candidate_count == 6
    assert result["clusters"] == 2 and result["cluster_sizes"] == {0: 2, 1: 2}
    assert result["assigned"] == 4 and result["unassigned"] == 2
    clusters = state.candidate_list.set_index("ACCENUMB")["cluster_climate"]
    assert clusters["A1"] == 0 and clusters["A2"] == 1 and pd.isna(clusters["A5"]) and pd.isna(clusters["A6"])
    assert "cluster by CDD, TX (period mean, months 3-8) [CDD=5.00, TX=25.00]" in state.candidate_list["criteria_climate"].iloc[0]
    assert result["summary"][0]["indicator"] == "CDD"

    body = json.loads([r for r in client._http._transport.handler.requests if r.url.path.endswith("/cluster")][-1].content)  # type: ignore[attr-defined]
    assert body["analysis"]["hyperparameter"] == {"n_clusters": 10, "min_cluster": 2}
    assert body["months"] == [3, 8]
    assert body["cellid_list"][0]["crop"] == "Beans"


def test_crop_specific_indicator_resolution(session) -> None:
    """Crop-specific prefixes resolve via inferred crop; explicit wrong crop errors."""
    state, paths = session
    skill, client = make_skill()

    result = skill.run(state, paths, action="filter", conditions=[{"indicator": "days_heat", "operator": "gte", "value": 10}])
    assert result["status"] == "ok"
    assert result["crop"] == "Beans"
    assert result["accessions_after"] == 4  # sum over 11 months >= 10 -> cells 1,2,3,4

    state.extras.pop("climate_crop")
    state.candidate_list["CROPNAME"] = "unknown crop"
    ambiguous = skill.run(state, paths, action="cluster", indicators=["days_heat"])
    assert ambiguous["status"] == "error" and "crop-specific" in ambiguous["message"]


def test_errors_are_formal(session, tmp_path: Path) -> None:
    """Missing data, missing cellid, unknown indicator, no conditions, auth failures."""
    state, paths = session
    skill, _ = make_skill()

    assert skill.run(SessionState(session_id="e"), paths, action="filter", conditions=[])["status"] == "error"
    assert skill.run(state, paths, action="filter", conditions=[])["status"] == "error"
    assert skill.run(state, paths, action="cluster")["status"] == "error"

    unknown = skill.run(state, paths, action="filter", conditions=[{"indicator": "banana", "operator": "gt", "value": 1}])
    assert unknown["status"] == "error" and "not found" in unknown["message"]

    no_cell = SessionState(session_id="n")
    no_cell.set_original_list(pd.DataFrame({"ACCENUMB": ["x"]}), SourceMode.LOCAL)
    assert "cellid" in skill.run(no_cell, paths, action="filter", conditions=[{"indicator": "CDD", "operator": "gt", "value": 1}])["message"]

    denied, _ = make_skill(lambda r: httpx.Response(403, text="Forbidden"))
    assert "credentials" in denied.run(state, paths, action="list_indicators")["message"]


def test_registry_discovers_climate_skill() -> None:
    """The skill is auto-registered."""
    assert "climate_analysis" in SkillRegistry().discover().names()
