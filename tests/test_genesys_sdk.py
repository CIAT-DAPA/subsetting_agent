"""Tests for the Genesys SDK (filter serialisation, auth, client, flattening)."""

import json
from typing import Any

import httpx
import pandas as pd
import pytest

from core.config import Settings
from sdks.genesys import (
    AccessionFilter,
    AccessionRecord,
    ApiTokenAuth,
    ClientCredentialsAuth,
    CountryFilter,
    GenesysAuthError,
    GenesysClient,
    GenesysRequestError,
    GeoFilter,
    InstituteFilter,
    NumberFilter,
    TaxonomyFilter,
)
from sdks.genesys.client import LIST_ACCESSIONS_PATH

BASE_URL = "https://api.test.genesys"


# ----------------------------------------------------------------- helpers
def _dto(number: str, genus: str = "Phaseolus", lat: float | None = 3.4) -> dict[str, Any]:
    """Build a minimal AccessionDTO-like dictionary."""
    return {
        "instituteCode": "COL003",
        "accessionNumber": number,
        "uuid": f"uuid-{number}",
        "doi": None,
        "taxonomy": {"genus": genus, "species": "vulgaris", "taxonName": f"{genus} vulgaris"},
        "cropName": "Common bean",
        "crop": {"shortName": "bean"},
        "sampStat": 300,
        "origCty": "COL",
        "countryOfOrigin": {"code3": "COL", "name": "Colombia"},
        "geo": None if lat is None else {"latitude": lat, "longitude": -76.5, "elevation": 1000},
        "storage": [11, 13],
        "duplSite": [],
        "coll": {"collSite": "Palmira", "collCode": ["CIAT"]},
        "institute": {"fullName": "CIAT", "countryCode3": "COL"},
        "pdci": {"score": 7.5},
        "available": True,
        "historic": False,
    }


def _page(content: list[dict[str, Any]], number: int, total: int, size: int) -> dict[str, Any]:
    """Build a FilteredPage-like dictionary."""
    total_pages = max((total + size - 1) // size, 1)
    return {
        "content": content,
        "number": number,
        "size": size,
        "totalElements": total,
        "totalPages": total_pages,
        "last": number >= total_pages - 1,
        "filterCode": "abc123",
    }


class Recorder:
    """Mock transport that records requests and serves scripted responses."""

    def __init__(self, handler) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._handler(request)


def _client(handler, **kwargs) -> tuple[GenesysClient, Recorder]:
    """Create a client wired to a recording mock transport."""
    recorder = Recorder(handler)
    client = GenesysClient(
        BASE_URL, ApiTokenAuth("secret-token"), transport=recorder.transport, **kwargs
    )
    return client, recorder


# ------------------------------------------------------ filter serialisation
def test_filter_body_contains_only_values() -> None:
    """None, empty lists, blank strings and empty sub-filters are omitted."""
    accession_filter = AccessionFilter(
        crop=["bean"],
        taxonomy=TaxonomyFilter(genus=["Phaseolus"], species=[]),
        country_of_origin=CountryFilter(code3=["COL", "PER"]),
        institute=InstituteFilter(code=[], country=CountryFilter()),
        geo=GeoFilter(latitude=NumberFilter(ge=-5, le=15), referenced=True),
        samp_stat=[300],
        historic=False,
        text="   ",
    )

    body = accession_filter.to_body()

    assert body == {
        "crop": ["bean"],
        "taxonomy": {"genus": ["Phaseolus"]},
        "countryOfOrigin": {"code3": ["COL", "PER"]},
        "geo": {"latitude": {"ge": -5.0, "le": 15.0}, "referenced": True},
        "sampStat": [300],
        "historic": False,
    }
    # ``False`` is a real criterion and must survive pruning.
    assert body["historic"] is False


def test_filter_aliases_and_operators() -> None:
    """Python names map to the exact JSON names of the API, including _text/NOT."""
    accession_filter = AccessionFilter(
        text="drought",
        accession_numbers=["G50001"],
        mls_status=True,
        not_=AccessionFilter(country_of_origin=CountryFilter(code3=["USA"])),
    )

    body = accession_filter.to_body()

    assert body["_text"] == "drought"
    assert body["accessionNumbers"] == ["G50001"]
    assert body["mlsStatus"] is True
    assert body["NOT"] == {"countryOfOrigin": {"code3": ["USA"]}}


def test_filter_accepts_json_names_and_reports_empty() -> None:
    """Filters can be built from API JSON names; an empty filter is detected."""
    accession_filter = AccessionFilter.model_validate({"sampStat": [100], "_text": "x"})

    assert accession_filter.samp_stat == [100]
    assert accession_filter.text == "x"
    assert AccessionFilter().is_empty()
    assert AccessionFilter(taxonomy=TaxonomyFilter()).is_empty()


# ------------------------------------------------------------------- auth
def test_api_token_header() -> None:
    """The personal token is sent with the API-Token scheme, not Bearer."""
    headers = ApiTokenAuth(" tok ").headers(httpx.Client())

    assert headers == {"Authorization": "API-Token tok"}

    with pytest.raises(GenesysAuthError):
        ApiTokenAuth("  ")


def test_client_credentials_fetches_and_caches_token() -> None:
    """The JWT is requested once with basic auth and reused while valid."""
    token_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal token_calls
        token_calls += 1
        assert request.url.path == "/oauth/token"
        assert request.headers["Authorization"].startswith("Basic ")
        assert b"grant_type=client_credentials" in request.content
        return httpx.Response(200, json={"access_token": "jwt-1", "expires_in": 3600})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    auth = ClientCredentialsAuth("cid", "csecret", f"{BASE_URL}/oauth/token")

    assert auth.headers(http) == {"Authorization": "Bearer jwt-1"}
    assert auth.headers(http) == {"Authorization": "Bearer jwt-1"}
    assert token_calls == 1


def test_client_credentials_rejected() -> None:
    """A 401 from the token endpoint raises GenesysAuthError."""
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad")))
    auth = ClientCredentialsAuth("cid", "csecret", f"{BASE_URL}/oauth/token")

    with pytest.raises(GenesysAuthError):
        auth.headers(http)


def test_from_settings_prefers_token_and_requires_credentials() -> None:
    """Settings wiring: token first, then client credentials, else error."""
    with_token = Settings(genesys_api_token="t", genesys_api_url=BASE_URL)
    assert isinstance(GenesysClient.from_settings(with_token).auth, ApiTokenAuth)

    with_oauth = Settings(genesys_client_id="a", genesys_client_secret="b", genesys_api_url=BASE_URL)
    assert isinstance(GenesysClient.from_settings(with_oauth).auth, ClientCredentialsAuth)

    with pytest.raises(GenesysAuthError):
        GenesysClient.from_settings(Settings(genesys_api_url=BASE_URL, _env_file=None))


# ----------------------------------------------------------------- client
def test_list_accessions_sends_minimal_body_and_params() -> None:
    """The request hits /acn/list with p/l params, auth header and pruned body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_page([_dto("G1")], 0, 1, 50))

    client, recorder = _client(handler)
    page = client.list_accessions(
        AccessionFilter(crop=["bean"], taxonomy=TaxonomyFilter()), page=0, size=50, sort="accessionNumber"
    )

    request = recorder.requests[0]
    assert request.method == "POST"
    assert request.url.path == LIST_ACCESSIONS_PATH
    assert request.url.params["p"] == "0"
    assert request.url.params["l"] == "50"
    assert request.url.params["s"] == "accessionNumber"
    assert request.headers["Authorization"] == "API-Token secret-token"
    assert json.loads(request.content) == {"crop": ["bean"]}

    assert page.total_elements == 1
    assert page.filter_code == "abc123"
    assert page.content[0].accession_number == "G1"


def test_iter_accessions_paginates_and_respects_cap() -> None:
    """Pages are requested until last=true; the cap stops iteration early."""
    total, size = 7, 3
    numbers = [f"G{i}" for i in range(total)]

    def handler(request: httpx.Request) -> httpx.Response:
        page_number = int(request.url.params["p"])
        chunk = numbers[page_number * size : (page_number + 1) * size]
        return httpx.Response(200, json=_page([_dto(n) for n in chunk], page_number, total, size))

    client, recorder = _client(handler, page_size=size, max_records=100)
    records = list(client.iter_accessions(AccessionFilter(crop=["bean"])))

    assert [r.accession_number for r in records] == numbers
    assert len(recorder.requests) == 3

    client, recorder = _client(handler, page_size=size, max_records=4)
    capped = list(client.iter_accessions())

    assert len(capped) == 4
    assert len(recorder.requests) == 2


def test_list_accessions_dataframe_flattens_mcpd() -> None:
    """The DataFrame has MCPD columns, joined lists and nested values resolved."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_page([_dto("G1"), _dto("G2", lat=None)], 0, 2, 500))

    client, _ = _client(handler)
    frame = client.list_accessions_dataframe(AccessionFilter(crop=["bean"]))

    assert list(frame["ACCENUMB"]) == ["G1", "G2"]
    assert frame.loc[0, "GENUS"] == "Phaseolus"
    assert frame.loc[0, "DECLATITUDE"] == 3.4
    assert pd.isna(frame.loc[1, "DECLATITUDE"])
    assert frame.loc[0, "STORAGE"] == "11;13"
    assert frame.loc[0, "COLLCODE"] == "CIAT"
    assert frame.loc[0, "PDCI"] == 7.5
    assert frame.loc[0, "INSTNAME"] == "CIAT"


def test_empty_result_keeps_schema() -> None:
    """An empty answer still yields a DataFrame with the MCPD columns."""
    client, _ = _client(lambda r: httpx.Response(200, json=_page([], 0, 0, 500)))
    frame = client.list_accessions_dataframe(AccessionFilter(crop=["nothing"]))

    assert frame.empty
    assert "ACCENUMB" in frame.columns
    assert client.count_accessions() == 0


def test_auth_error_and_bad_request_are_typed() -> None:
    """401 raises GenesysAuthError; 400 raises GenesysRequestError with the body."""
    client, _ = _client(lambda r: httpx.Response(401, text="nope"))
    with pytest.raises(GenesysAuthError):
        client.list_accessions()

    client, _ = _client(lambda r: httpx.Response(400, text="bad filter"))
    with pytest.raises(GenesysRequestError) as info:
        client.list_accessions()

    assert info.value.status_code == 400
    assert "bad filter" in info.value.body


def test_transient_errors_are_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 503 followed by a 200 succeeds; persistent 503 raises after retries."""
    monkeypatch.setattr("sdks.genesys.client.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_page([_dto("G1")], 0, 1, 500))

    client, _ = _client(flaky, max_retries=3)
    assert client.list_accessions().total_elements == 1
    assert attempts["count"] == 2

    client, recorder = _client(lambda r: httpx.Response(503, text="busy"), max_retries=2)
    with pytest.raises(GenesysRequestError):
        client.list_accessions()
    assert len(recorder.requests) == 2


def test_record_flatten_handles_missing_nested() -> None:
    """Missing nested objects become None instead of raising."""
    record = AccessionRecord(raw={"accessionNumber": "X"})
    flat = record.flatten()

    assert flat["ACCENUMB"] == "X"
    assert flat["GENUS"] is None
    assert flat["DECLATITUDE"] is None


def test_iter_accessions_shrinks_page_to_cap() -> None:
    """A small cap never requests a page larger than the cap."""

    def handler(request: httpx.Request) -> httpx.Response:
        size = int(request.url.params["l"])
        return httpx.Response(200, json=_page([_dto(f"G{i}") for i in range(size)], 0, 5000, size))

    client, recorder = _client(handler, page_size=500)
    records = list(client.iter_accessions(max_records=20))

    assert len(records) == 20
    assert [int(r.url.params["l"]) for r in recorder.requests] == [20]


def test_list_crops_and_resolve_codes() -> None:
    """Crop names resolve through shortName/name/otherNames, cached after one GET."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert request.method == "GET" and request.url.path == "/api/v2/crop"
        assert request.headers["Authorization"] == "API-Token secret-token"
        return httpx.Response(
            200,
            json=[
                {"shortName": "beans", "name": "Beans", "otherNames": ["bean", "frijol"]},
                {"shortName": "maize", "name": "Maize", "otherNames": "corn, maíz"},
            ],
        )

    client, _ = _client(handler)
    crops = client.list_crops()

    assert [c.short_name for c in crops] == ["beans", "maize"]
    assert crops[1].other_names == ["corn", "maíz"]

    resolved, unresolved = client.resolve_crop_codes(["Frijol", "MAIZE", "corn", "papaya", ""])

    assert resolved == ["beans", "maize"]
    assert unresolved == ["papaya", ""]
    assert calls["n"] == 1
