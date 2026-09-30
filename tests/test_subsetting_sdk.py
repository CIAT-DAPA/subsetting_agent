"""Tests for the Subsetting SDK (auth fallback, catalogue, data, clusters, cellid)."""

import json

import httpx
import pandas as pd
import pytest

from core.config import GridSettings, Settings
from sdks.subsetting import (
    FallbackAuth,
    SubsettingAuthError,
    SubsettingClient,
    SubsettingError,
    SubsettingRequestError,
    add_cellid_column,
    compute_cellid,
)
from sdks.subsetting.models import ClusterResult

BASE_URL = "https://sandbox.test/api/subsetting"

INDICATORS_PAYLOAD = [
    {
        "category": "Drought",
        "checked": False,
        "indicators": [
            {"id": "ind_cdd", "name": "Consecutive dry days", "pref": "cdd", "indicator_type": "generic",
             "crop": "generic", "category": "Drought", "checked": False, "unit": "days"},
            {"id": "ind_train", "name": "Total rainfall", "pref": "t_rain", "indicator_type": "generic",
             "crop": "generic", "category": "Drought", "unit": "mm"},
        ],
    },
    {
        "category": "Heat",
        "indicators": [
            {"id": "ind_heat_bean", "name": "Heat stress days (bean)", "pref": "hs_bean",
             "indicator_type": "specific", "crop": "bean", "category": "Heat", "unit": "days"},
        ],
    },
]

PERIODS_PAYLOAD = [
    {"id": "p1", "indicator": "ind_cdd", "period": "mean", "ssp": "historic"},
    {"id": "p2", "indicator": "ind_cdd", "period": "1983", "ssp": "historic"},
    {"id": "p3", "indicator": "ind_train", "period": "mean", "ssp": "historic"},
    {"id": "p4", "indicator": "ind_heat_bean", "period": "mean", "ssp": "historic"},
]


def _data_payload(cells: list[int]) -> dict:
    """indicators-data response for the given cells (two indicators each)."""
    return {
        "response": [
            {
                "cellid": cell,
                "data": [
                    {"pref_indicator": "cdd", **{f"month{m}": float(m) for m in range(1, 13)}},
                    {"pref_indicator": "t_rain", **{f"month{m}": 10.0 * m for m in range(1, 13)}},
                ],
            }
            for cell in cells
        ]
    }


def _cluster_payload(cells: list[int]) -> dict:
    """cluster response assigning cells alternately to clusters 0/1."""
    return {
        "data": [
            {"cellid": cell, "cdd_month1": 1.0, "t_rain_month1": 10.0, "cluster_hac": index % 2, "crop_name": "generic"}
            for index, cell in enumerate(cells)
        ],
        "calculate": [],
        "quantile": [],
        "summary": [{"indicator": "cdd", "cluster_hac": 0, "mean": 1.0}],
        "proportion": [],
    }


class Recorder:
    """Mock transport recording requests and serving a handler."""

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def default_handler(request: httpx.Request) -> httpx.Response:
    """Serve every endpoint with fixture data."""
    path = request.url.path

    if path.endswith("/indicators"):
        return httpx.Response(200, json=INDICATORS_PAYLOAD)
    if path.endswith("/indicator-period"):
        return httpx.Response(200, json=PERIODS_PAYLOAD)
    if path.endswith("/indicators-data"):
        body = json.loads(request.content)
        return httpx.Response(200, json=_data_payload(body["cellid"]))
    if path.endswith("/cluster"):
        body = json.loads(request.content)
        return httpx.Response(200, json=_cluster_payload(body["cellid_list"][0]["cellids"]))

    return httpx.Response(404, text="not found")


def _client(handler=default_handler, **kwargs) -> tuple[SubsettingClient, Recorder]:
    recorder = Recorder(handler)
    auth = kwargs.pop("auth", FallbackAuth(api_token="api-tok", access_token="acc-tok"))
    client = SubsettingClient(BASE_URL, auth, transport=recorder.transport, **kwargs)
    return client, recorder


# -------------------------------------------------------------------- grid
def test_compute_cellid_matches_raster_convention() -> None:
    """Cell ids follow raster::cellFromXY on the global grid: 1-based, row 1 at 90N."""
    grid = GridSettings()  # defaults = global 7200 x 3600, 0.05 degrees

    # North-west corner cell and its eastern neighbour.
    assert compute_cellid(89.99, -179.99, grid) == 1
    assert compute_cellid(89.99, -179.94, grid) == 2
    # First cell of the second row.
    assert compute_cellid(89.94, -179.99, grid) == 7201
    # Last cell of the grid (south-east corner).
    assert compute_cellid(-89.99, 179.99, grid) == 7200 * 3600

    # Real accessions verified against the Subsetting API (probe + indicators-data).
    assert compute_cellid(11.39, -72.22, grid) == 11320556
    assert compute_cellid(7.39, -72.65, grid) == 11896547
    assert compute_cellid(5.75, -72.84, grid) == 12134144

    # Outside the extent, missing and invalid coordinates.
    assert compute_cellid(91.0, 0.0, grid) is None
    assert compute_cellid(-90.5, 0.0, grid) is None
    assert compute_cellid(0.0, 180.0, grid) is None
    assert compute_cellid(None, 0.0, grid) is None
    assert compute_cellid("abc", 0.0, grid) is None


def test_compute_cellid_with_custom_grid() -> None:
    """The formula honours whatever grid is configured (legacy 7198x2000 example)."""
    grid = GridSettings(
        subsetting_grid_ncols=7198, subsetting_grid_nrows=2000,
        subsetting_grid_xmin=-180, subsetting_grid_ymin=-50, subsetting_grid_cellsize=0.05,
    )
    assert compute_cellid(49.99, -179.99, grid) == 1
    assert compute_cellid(3.5, -76.35, grid) == 930 * 7198 + 2073 + 1
    assert compute_cellid(60.0, 0.0, grid) is None


def test_add_cellid_column_from_text_coordinates() -> None:
    """Coordinates stored as text are converted; missing ones give <NA>."""
    grid = GridSettings()
    frame = pd.DataFrame({"DECLATITUDE": ["3.5", None, "x"], "DECLONGITUDE": ["-76.35", "-76.0", "1"]})

    result = add_cellid_column(frame, "DECLATITUDE", "DECLONGITUDE", grid)

    # (3.5, -76.35): row floor((90-3.5)/0.05)=1730, col floor((180-76.35)/0.05)=2073
    assert result["cellid"].dtype.name == "Int64"
    assert result.loc[0, "cellid"] == 1730 * 7200 + 2073 + 1
    assert pd.isna(result.loc[1, "cellid"]) and pd.isna(result.loc[2, "cellid"])
    assert "cellid" not in frame.columns  # original untouched


# -------------------------------------------------------------------- auth
def _credential(request: httpx.Request) -> str:
    """Describe how a request authenticated (header value or cookie)."""
    if "Authorization" in request.headers:
        return request.headers["Authorization"]
    if "Cookie" in request.headers:
        return "cookie:" + request.headers["Cookie"]
    return "none"


def test_fallback_auth_switches_once_on_401() -> None:
    """API-Token is tried first; after a 401 the Bearer token is used for good."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = _credential(request)
        seen.append(auth)
        if auth.startswith("API-Token"):
            return httpx.Response(401, text="bad token")
        return default_handler(request)

    client, _ = _client(handler)
    assert len(client.list_indicators()) == 3
    assert len(client.list_indicator_periods()) == 4

    assert seen[0] == "API-Token api-tok"
    assert seen[1] == "Bearer acc-tok"
    assert seen[2] == "Bearer acc-tok"  # no second attempt with the API token
    assert client.auth.scheme == "Bearer"


def test_fallback_auth_ends_with_access_token_cookie() -> None:
    """When both headers are rejected the JWT is sent as the access_token cookie."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth = _credential(request)
        seen.append(auth)
        if "Authorization" in request.headers:
            return httpx.Response(403, text="Forbidden")
        return default_handler(request)

    client, _ = _client(handler)
    assert len(client.list_indicators()) == 3
    client.list_indicator_periods()

    assert seen[:3] == ["API-Token api-tok", "Bearer acc-tok", "cookie:access_token=acc-tok"]
    assert seen[3] == "cookie:access_token=acc-tok"
    assert client.auth.scheme == "Cookie"
    assert client.auth.headers() == {}


def test_all_credentials_rejected_and_missing() -> None:
    """A 403 on every scheme raises; no credentials at all raises at construction."""
    client, recorder = _client(lambda r: httpx.Response(403, text="nope"))

    with pytest.raises(SubsettingAuthError) as info:
        client.list_indicators()

    assert len(recorder.requests) == 3  # API-Token, Bearer, Cookie
    assert "SUBSETTING_API_URL" in str(info.value)

    with pytest.raises(SubsettingAuthError):
        FallbackAuth(api_token="", access_token=None)

    only_api = FallbackAuth(api_token="x")
    assert only_api.headers() == {"Authorization": "API-Token x"}
    assert only_api.cookies() == {}
    assert not only_api.has_fallback()

    only_access = FallbackAuth(access_token="jwt")
    assert only_access.scheme == "Bearer" and only_access.has_fallback()


def test_from_settings_requires_url_and_uses_full_url() -> None:
    """The URL is used as given; endpoints are appended to it."""
    settings = Settings(subsetting_api_url=BASE_URL + "/", subsetting_api_token="t")
    client = SubsettingClient.from_settings(settings)
    assert client.base_url == BASE_URL

    with pytest.raises(SubsettingError):
        SubsettingClient.from_settings(Settings(subsetting_api_url=None, subsetting_api_token="t", _env_file=None))


# --------------------------------------------------------------- catalogue
def test_catalogue_flattening_search_and_periods() -> None:
    """Indicators are flattened with their category; search and period lookup work."""
    client, recorder = _client()
    indicators = client.list_indicators()

    assert [i.pref for i in indicators] == ["cdd", "t_rain", "hs_bean"]
    assert indicators[0].category == "Drought"
    assert [i.id for i in client.find_indicators("drought")] == ["ind_cdd", "ind_train"]
    assert client.find_indicators("rain")[0].id == "ind_train"
    assert client.get_indicator("T_RAIN").id == "ind_train"
    assert client.get_indicator("nothing") is None

    assert client.resolve_periods(["ind_cdd", "ind_train"]) == {"ind_cdd": ["p1"], "ind_train": ["p3"]}
    assert client.resolve_periods(["ind_cdd"], period="1983") == {"ind_cdd": ["p2"]}

    with pytest.raises(SubsettingError) as info:
        client.resolve_periods(["ind_train"], period="1983")
    assert "Available periods" in str(info.value)

    # Catalogue and periods are fetched once each.
    paths = [r.url.path for r in recorder.requests]
    assert paths.count(f"/api/subsetting/indicators") == 1
    assert paths.count(f"/api/subsetting/indicator-period") == 1


# -------------------------------------------------------------------- data
def test_get_indicators_data_builds_body_and_long_frame() -> None:
    """The body carries cell ids and period ids; the answer becomes a long table."""
    client, recorder = _client()
    frame = client.get_indicators_data([200, 100, 100], ["ind_cdd", "ind_train"])

    body = json.loads(recorder.requests[-1].content)
    assert recorder.requests[-1].url.path.endswith("/indicators-data")
    assert body == {"cellid": [100, 200], "indicators": ["p1", "p3"]}

    assert len(frame) == 4
    assert set(frame["pref_indicator"]) == {"cdd", "t_rain"}
    assert frame.loc[(frame.cellid == 100) & (frame.pref_indicator == "t_rain"), "month12"].iloc[0] == 120.0
    assert "value" in frame.columns and "category" in frame.columns

    empty = client.get_indicators_data([], ["ind_cdd"])
    assert empty.empty and "pref_indicator" in empty.columns


# ---------------------------------------------------------------- clusters
def test_generate_clusters_body_and_result() -> None:
    """The cluster body follows the API contract and the result maps cell -> cluster."""
    client, recorder = _client()
    result = client.generate_clusters([1, 2, 3, 4], ["ind_cdd", "ind_heat_bean"], crop="bean", max_clusters=6)

    body = json.loads(recorder.requests[-1].content)
    assert body["cellid_list"] == [{"crop": "bean", "cellids": [1, 2, 3, 4]}]
    assert body["analysis"] == {
        "algorithm": ["agglomerative"],
        "hyperparameter": {"n_clusters": 6, "min_cluster": 2},
        "summary": False,
    }
    assert body["months"] == [1, 12]
    assert body["data"] == [
        {"name": "Consecutive dry days", "indicator": ["p1"], "type": "generic", "crop": "generic"},
        {"name": "Heat stress days (bean)", "indicator": ["p4"], "type": "specific", "crop": "bean"},
    ]

    assert result.cluster_column == "cluster_hac"
    assert result.assignments == {1: 0, 2: 1, 3: 0, 4: 1}
    assert result.cluster_count == 2
    assert result.indicator_values[1] == {"cdd_month1": 1.0, "t_rain_month1": 10.0}
    assert result.summary[0]["indicator"] == "cdd"


def test_generate_clusters_validations() -> None:
    """Too few cells, bad ranges and empty analyses are reported."""
    client, _ = _client()

    with pytest.raises(SubsettingError):
        client.generate_clusters([1, 2], ["ind_cdd"])
    with pytest.raises(SubsettingError):
        client.generate_clusters([1, 2, 3], ["ind_cdd"], min_clusters=1)
    with pytest.raises(SubsettingError):
        client.generate_clusters([1, 2, 3], ["ind_cdd"], min_clusters=5, max_clusters=3)

    def empty_cluster(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/cluster"):
            return httpx.Response(200, json={})
        return default_handler(request)

    client, _ = _client(empty_cluster)
    with pytest.raises(SubsettingError) as info:
        client.generate_clusters([1, 2, 3], ["ind_cdd"])
    assert "no clusters" in str(info.value)


def test_cluster_result_prefers_requested_column() -> None:
    """When several algorithms are present the requested column is used."""
    payload = {"data": [{"cellid": 5, "cluster_dbscan": -1, "cluster_hac": 2}]}
    assert ClusterResult.from_response(payload, "cluster_hac").assignments == {5: 2}
    assert ClusterResult.from_response(payload).cluster_column == "cluster_dbscan"
    assert ClusterResult.from_response({}).assignments == {}


# ------------------------------------------------------------------ errors
def test_request_errors_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """400 is typed; 503 is retried and gives up after max_retries."""
    monkeypatch.setattr("sdks.subsetting.client.time.sleep", lambda _: None)

    client, _ = _client(lambda r: httpx.Response(400, text="bad"))
    with pytest.raises(SubsettingRequestError) as info:
        client.list_indicators()
    assert info.value.status_code == 400

    client, recorder = _client(lambda r: httpx.Response(503, text="busy"), max_retries=2)
    with pytest.raises(SubsettingRequestError):
        client.list_indicators()
    assert len(recorder.requests) == 2


def test_indicators_data_500_without_data_is_not_retried() -> None:
    """The empty-DataFrame crash of /indicators-data becomes SubsettingNoDataError at once."""
    from sdks.subsetting import SubsettingNoDataError

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/indicators-data"):
            return httpx.Response(500, text="<h1>Internal Server Error</h1> KeyError: 'cellid'")
        return default_handler(request)

    client, recorder = _client(handler)
    with pytest.raises(SubsettingNoDataError):
        client.get_indicators_data([1, 2], ["ind_cdd"])

    data_calls = [r for r in recorder.requests if r.url.path.endswith("/indicators-data")]
    assert len(data_calls) == 1


def test_parse_coordinates_accepts_negatives() -> None:
    """The smoke helper parses negative pairs separated by spaces or semicolons."""
    from scripts.subsetting_smoke import parse_coordinates

    assert parse_coordinates("3.5,-76.35 -12.0,-77.0;19.4,-99.1") == [(3.5, -76.35), (-12.0, -77.0), (19.4, -99.1)]
    assert parse_coordinates("") == []
    with pytest.raises(SubsettingError):
        parse_coordinates("3.5")


def test_get_indicator_is_crop_aware() -> None:
    """Shared prefixes resolve by crop, fall back to generic, and list variants."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/indicators"):
            return httpx.Response(200, json=[
                {"category": "Crop-specific indicators", "indicators": [
                    {"id": "h_beans", "name": "Heat days", "pref": "days_heat", "indicator_type": "specific", "crop": "Beans", "unit": "days"},
                    {"id": "h_maize", "name": "Heat days", "pref": "days_heat", "indicator_type": "specific", "crop": "Maize", "unit": "days"},
                    {"id": "h_nc", "name": "Heat days", "pref": "days_heat", "indicator_type": "specific", "crop": "NC", "unit": "days"},
                ]},
                {"category": "Drought stress", "indicators": [
                    {"id": "cdd", "name": "Consecutive dry days", "pref": "CDD", "indicator_type": "generic", "crop": "NC", "unit": "days"},
                ]},
            ])
        return default_handler(request)

    client, _ = _client(handler)

    assert client.get_indicator("days_heat", crop="maize").id == "h_maize"
    assert client.get_indicator("days_heat").id == "h_nc"
    assert client.get_indicator("cdd", crop="Beans").id == "cdd"
    assert [i.id for i in client.get_indicator_variants("Heat days")] == ["h_beans", "h_maize", "h_nc"]
    assert client.catalogue_crops() == ["Beans", "Maize"]
