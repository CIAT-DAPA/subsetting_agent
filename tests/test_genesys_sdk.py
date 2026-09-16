"""Unit tests for the Genesys SDK (filters, response models and client)."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from genesys_sdk import (
    Accession,
    AccessionFilter,
    AccessionObservations,
    AccessionPage,
    GenesysApiError,
    GenesysAuthError,
    GenesysBadRequestError,
    GenesysClient,
    GenesysConnectionError,
    GenesysNotFoundError,
    NumberFilter,
    StringFilter,
    TaxonomyFilter,
    read_path,
)

BASE = "https://genesys.example"


def make_client(**kwargs) -> GenesysClient:
    """Create a client against the mocked host with retries disabled by default.

    Args:
        **kwargs: Overrides forwarded to :class:`GenesysClient`.
    """
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("backoff_seconds", 0.0)
    kwargs.setdefault("token", "secret-token")

    return GenesysClient(BASE, **kwargs)


def accession_payload(uuid: str, tile: int | None = 555, tile3: int | None = 777) -> dict:
    """Build a minimal ``AccessionDTO`` payload.

    Args:
        uuid: Accession UUID.
        tile: Value of ``geo.tileIndex`` (``None`` omits the geo block).
        tile3: Value of ``tileIndex3min``.
    """
    payload: dict = {
        "uuid": uuid,
        "id": 1,
        "accessionNumber": f"G{uuid[-3:]}",
        "instituteCode": "COL003",
        "cropName": "Common bean",
        "crop": {"shortName": "bean", "name": "Bean"},
        "genus": "Phaseolus",
        "taxonomy": {
            "genus": "Phaseolus",
            "species": "vulgaris",
            "taxonName": "Phaseolus vulgaris L.",
        },
        "origCty": "COL",
        "countryOfOrigin": {"code3": "COL", "name": "Colombia"},
        "sampStat": 300,
        "available": True,
        "tileIndex3min": tile3,
    }

    if tile is not None:
        payload["geo"] = {
            "latitude": 4.5,
            "longitude": -74.1,
            "tileIndex": tile,
            "referenced": True,
        }

    return payload


def page_payload(uuids: list[str], *, number: int, total: int, last: bool, size: int = 2) -> dict:
    """Build a ``FilteredPageAccessionDTOAccessionFilter`` payload.

    Args:
        uuids: UUIDs of the accessions in the page.
        number: Zero-based page index.
        total: Total matching accessions.
        last: Whether this is the last page.
        size: Page size.
    """
    return {
        "content": [accession_payload(u) for u in uuids],
        "totalElements": total,
        "totalPages": -(-total // size),
        "number": number,
        "size": size,
        "last": last,
        "filterCode": "abc123",
    }


class TestFilterSerialization:
    """Filters use API property names and omit unset fields."""

    def test_empty_filter_is_empty_object(self) -> None:
        """An unconstrained filter serializes to ``{}``."""
        assert AccessionFilter().to_api() == {}

    def test_aliases_and_nesting(self) -> None:
        """Snake-case fields map to the camelCase names Genesys expects."""
        payload = AccessionFilter(
            crop=["bean"],
            samp_stat=[300],
            taxonomy=TaxonomyFilter(
                genus=["Phaseolus"], taxon_name=StringFilter(contains=["vulg"])
            ),
            has_doi=True,
            text="drought",
        ).to_api()

        assert payload == {
            "crop": ["bean"],
            "sampStat": [300],
            "taxonomy": {"genus": ["Phaseolus"], "taxonName": {"contains": ["vulg"]}},
            "hasDoi": True,
            "_text": "drought",
        }

    def test_boolean_combinators(self) -> None:
        """AND/OR/NOT nest full filters under upper-case keys."""
        payload = AccessionFilter(
            crop=["bean"], not_=AccessionFilter(country_of_origin={"code3": ["USA"]})
        ).to_api()

        assert payload == {"crop": ["bean"], "NOT": {"countryOfOrigin": {"code3": ["USA"]}}}

    def test_unknown_field_is_rejected(self) -> None:
        """Typos in filter fields fail fast instead of being silently ignored."""
        with pytest.raises(ValueError):
            AccessionFilter(cropp=["bean"])  # type: ignore[call-arg]

    def test_number_filter_between(self) -> None:
        """``between`` builds inclusive bounds and returns None when unbounded."""
        assert NumberFilter.between(1, 5).to_api() == {"ge": 1.0, "le": 5.0}
        assert NumberFilter.between(None, 5).to_api() == {"le": 5.0}
        assert NumberFilter.between(None, None) is None


class TestPassportBuilder:
    """``AccessionFilter.passport`` maps domain criteria onto the API filter."""

    def test_full_builder(self) -> None:
        """Every criterion lands in the right place."""
        payload = AccessionFilter.passport(
            crop_codes=["bean"],
            genus=["Phaseolus"],
            species=["vulgaris"],
            origin_countries=["COL", "PER"],
            institute_codes=["COL003"],
            sample_status=[300, 100],
            available=True,
            with_coordinates=True,
            latitude=(-5, 12),
            elevation=(None, 2500),
        ).to_api()

        assert payload == {
            "crop": ["bean"],
            "taxonomy": {"genus": ["Phaseolus"], "species": ["vulgaris"]},
            "countryOfOrigin": {"code3": ["COL", "PER"]},
            "institute": {"code": ["COL003"]},
            "sampStat": [300, 100],
            "geo": {
                "referenced": True,
                "latitude": {"ge": -5.0, "le": 12.0},
                "elevation": {"le": 2500.0},
            },
            "available": True,
        }

    def test_builder_omits_empty_criteria(self) -> None:
        """Empty lists and None arguments produce no filter properties."""
        assert AccessionFilter.passport(crop_codes=[], genus=None).to_api() == {}

    def test_builder_crop_name_and_taxon_name_use_contains(self) -> None:
        """Free-text names become substring filters."""
        payload = AccessionFilter.passport(crop_name="frijol", taxon_name="vulgaris").to_api()

        assert payload["cropName"] == {"contains": ["frijol"]}
        assert payload["taxonomy"] == {"taxonName": {"contains": ["vulgaris"]}}


class TestAccessionModel:
    """Accessions expose derived properties and the configurable cellid."""

    def test_derived_properties(self) -> None:
        """Crop code, coordinates, country and taxon are read from nested data."""
        accession = Accession.from_api(accession_payload("u-001"))

        assert accession.crop_code == "bean"
        assert accession.latitude == 4.5
        assert accession.longitude == -74.1
        assert accession.has_coordinates
        assert accession.country_code == "COL"
        assert accession.taxon_name == "Phaseolus vulgaris L."

    def test_cellid_default_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """By default the cellid is ``geo.tileIndex``."""
        monkeypatch.delenv("GENESYS_CELLID_FIELD", raising=False)
        accession = Accession.from_api(accession_payload("u-001", tile=555, tile3=777))

        assert accession.cellid() == 555

    def test_cellid_configured_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``GENESYS_CELLID_FIELD`` switches the source field without code changes."""
        monkeypatch.setenv("GENESYS_CELLID_FIELD", "tileIndex3min")
        accession = Accession.from_api(accession_payload("u-001", tile=555, tile3=777))

        assert accession.cellid() == 777
        assert accession.cellid("geo.tileIndex") == 555

    def test_cellid_missing(self) -> None:
        """Accessions without coordinates have no cellid."""
        accession = Accession.from_api(accession_payload("u-002", tile=None, tile3=None))

        assert accession.cellid() is None
        assert not accession.has_coordinates

    def test_crop_code_falls_back_to_crop_name(self) -> None:
        """Without a curated crop the genebank's crop name is used."""
        payload = accession_payload("u-003")
        del payload["crop"]

        assert Accession.from_api(payload).crop_code == "Common bean"

    def test_read_path(self) -> None:
        """Dotted paths resolve nested values and tolerate missing segments."""
        data = {"a": {"b": {"c": 1}}}

        assert read_path(data, "a.b.c") == 1
        assert read_path(data, "a.x.c") is None
        assert read_path(None, "a") is None


class TestListAccessions:
    """``/acn/list`` sends the filter, paging params and the token header."""

    async def test_request_and_response(self, httpx_mock: HTTPXMock) -> None:
        """Body, query string, auth header and parsed page are all correct."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=0&l=50",
            json=page_payload(["u-001", "u-002"], number=0, total=2, last=True),
        )

        async with make_client() as client:
            page = await client.list_accessions(AccessionFilter.passport(crop_codes=["bean"]))

        request = httpx_mock.get_requests()[0]

        assert request.headers["Authorization"] == "API-Token secret-token"
        assert json.loads(request.content) == {"crop": ["bean"]}
        assert page.total_elements == 2
        assert page.filter_code == "abc123"
        assert [a.uuid for a in page.content] == ["u-001", "u-002"]
        assert page.content[0].cellid() == 555

    async def test_filter_code_reuse_and_page_size_cap(self, httpx_mock: HTTPXMock) -> None:
        """A filter code is sent as ``f`` and the page size is capped at 500."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=3&l=500&f=abc123",
            json=page_payload([], number=3, total=0, last=True),
        )

        async with make_client() as client:
            await client.list_accessions(page=3, page_size=9999, filter_code="abc123")

        assert json.loads(httpx_mock.get_requests()[0].content) == {}

    async def test_requires_filter_or_code(self) -> None:
        """Calling without any filter is a programming error."""
        async with make_client() as client:
            with pytest.raises(ValueError):
                await client.list_accessions()

    async def test_no_token_sends_no_header(self, httpx_mock: HTTPXMock) -> None:
        """Without a configured token the Authorization header is absent."""
        httpx_mock.add_response(json=page_payload([], number=0, total=0, last=True))

        async with make_client(token="") as client:
            await client.list_accessions(AccessionFilter())

        assert "Authorization" not in httpx_mock.get_requests()[0].headers


class TestIterAccessions:
    """Pagination follows ``last`` and honours the record cap."""

    async def test_iterates_all_pages_with_filter_code(self, httpx_mock: HTTPXMock) -> None:
        """The first page sends the body, later pages reuse the filter code."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=0&l=2",
            json=page_payload(["u-001", "u-002"], number=0, total=3, last=False),
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=1&l=2&f=abc123",
            json=page_payload(["u-003"], number=1, total=3, last=True),
        )

        async with make_client() as client:
            accessions, total = await client.collect_accessions(
                AccessionFilter(crop=["bean"]), page_size=2, max_records=100
            )

        requests = httpx_mock.get_requests()

        assert [a.uuid for a in accessions] == ["u-001", "u-002", "u-003"]
        assert total == 3
        assert json.loads(requests[0].content) == {"crop": ["bean"]}
        assert json.loads(requests[1].content) == {}

    async def test_record_cap_truncates_and_stops(self, httpx_mock: HTTPXMock) -> None:
        """Only ``max_records`` accessions are returned and no extra page is fetched."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=0&l=2",
            json=page_payload(["u-001", "u-002"], number=0, total=10, last=False),
        )

        async with make_client() as client:
            accessions, total = await client.collect_accessions(
                AccessionFilter(crop=["bean"]), page_size=2, max_records=1
            )

        assert [a.uuid for a in accessions] == ["u-001"]
        assert total == 10
        assert len(httpx_mock.get_requests()) == 1

    async def test_cap_from_environment(
        self, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``GENESYS_MAX_ACCESSIONS`` is the default cap."""
        monkeypatch.setenv("GENESYS_MAX_ACCESSIONS", "2")
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/list?p=0&l=50",
            json=page_payload(["u-001", "u-002", "u-003"], number=0, total=3, last=False),
        )

        async with make_client() as client:
            accessions, _ = await client.collect_accessions(AccessionFilter(crop=["bean"]))

        assert len(accessions) == 2


class TestOtherEndpoints:
    """Overview, details, observations, crops and descriptors parse correctly."""

    async def test_overview(self, httpx_mock: HTTPXMock) -> None:
        """Counts per field are typed."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/overview?limit=5",
            json={
                "filterCode": "abc123",
                "accessionCount": 120,
                "overview": {
                    "countryOfOrigin.code3": {
                        "terms": [{"term": "COL", "count": 80}, {"term": "PER", "count": 40}],
                        "total": 120,
                        "other": 0,
                        "missing": 0,
                    }
                },
                "suggestions": {},
            },
        )

        async with make_client() as client:
            overview = await client.accession_overview(AccessionFilter(crop=["bean"]), limit=5)

        assert overview.accession_count == 120
        assert overview.overview["countryOfOrigin.code3"].terms[0].term == "COL"

    async def test_details_and_observations(self, httpx_mock: HTTPXMock) -> None:
        """Details wrap an accession; observations merge both providers."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/details/u-001",
            json={"details": accession_payload("u-001")},
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/acn/u-001/observations",
            json={"firstPartyData": [{"trait": "DT", "value": 3}], "thirdPartyData": []},
        )

        async with make_client() as client:
            details = await client.get_accession("u-001")
            observations = await client.get_observations("u-001")

        assert details.details.uuid == "u-001"
        assert details.details.cellid() == 555
        assert isinstance(observations, AccessionObservations)
        assert observations.all_records == [{"trait": "DT", "value": 3}]
        assert not observations.is_empty

    async def test_empty_observations_body(self, httpx_mock: HTTPXMock) -> None:
        """An empty body is treated as no observations."""
        httpx_mock.add_response(url=f"{BASE}/api/v2/acn/u-009/observations", text="")

        async with make_client() as client:
            observations = await client.get_observations("u-009")

        assert observations.is_empty

    async def test_crops_and_descriptors(self, httpx_mock: HTTPXMock) -> None:
        """Crops and descriptor pages parse their aliases."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/crop",
            json=[{"shortName": "bean", "name": "Bean", "accessionCount": 10}],
        )
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/descriptor/list/details?p=0&l=50",
            json={
                "content": [
                    {
                        "uuid": "d-1",
                        "title": "Drought tolerance",
                        "dataType": "CODED",
                        "crop": "bean",
                        "terms": [{"code": "1", "title": "Tolerant"}],
                    }
                ],
                "totalElements": 1,
                "totalPages": 1,
                "number": 0,
                "last": True,
            },
        )

        async with make_client() as client:
            crops = await client.list_crops()
            descriptors = await client.list_descriptors()

        assert crops[0].short_name == "bean"
        assert descriptors.content[0].data_type == "CODED"
        assert descriptors.content[0].terms[0].title == "Tolerant"


class TestErrors:
    """HTTP failures map to the SDK exception hierarchy."""

    @pytest.mark.parametrize(
        ("status", "exception"),
        [
            (401, GenesysAuthError),
            (403, GenesysAuthError),
            (404, GenesysNotFoundError),
            (400, GenesysBadRequestError),
            (500, GenesysApiError),
        ],
    )
    async def test_status_mapping(
        self, httpx_mock: HTTPXMock, status: int, exception: type
    ) -> None:
        """Each status raises its dedicated exception with the API's message."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v2/crop",
            status_code=status,
            json={"error": "Nope", "localizedError": "No"},
        )

        async with make_client() as client:
            with pytest.raises(exception) as info:
                await client.list_crops()

        assert info.value.status_code == status
        assert "No" in str(info.value)

    async def test_auth_error_hints_missing_token(self, httpx_mock: HTTPXMock) -> None:
        """A 401 without a token tells the operator which variable to set."""
        httpx_mock.add_response(url=f"{BASE}/api/v2/crop", status_code=401, text="Unauthorized")

        async with make_client(token="") as client:
            with pytest.raises(GenesysAuthError, match="GENESYS_API_TOKEN"):
                await client.list_crops()

    async def test_connection_error_after_retries(self, httpx_mock: HTTPXMock) -> None:
        """Transport errors are retried and then surfaced."""
        httpx_mock.add_exception(httpx.ConnectError("boom"), url=f"{BASE}/api/v2/crop")
        httpx_mock.add_exception(httpx.ConnectError("boom"), url=f"{BASE}/api/v2/crop")

        async with make_client(max_retries=1) as client:
            with pytest.raises(GenesysConnectionError):
                await client.list_crops()

        assert len(httpx_mock.get_requests()) == 2

    async def test_gateway_error_is_retried(self, httpx_mock: HTTPXMock) -> None:
        """A 503 followed by a 200 succeeds."""
        httpx_mock.add_response(url=f"{BASE}/api/v2/crop", status_code=503)
        httpx_mock.add_response(url=f"{BASE}/api/v2/crop", json=[])

        async with make_client(max_retries=1) as client:
            assert await client.list_crops() == []

    async def test_invalid_json(self, httpx_mock: HTTPXMock) -> None:
        """A 200 with a non-JSON body is an API error."""
        httpx_mock.add_response(url=f"{BASE}/api/v2/crop", text="<html>")

        async with make_client() as client:
            with pytest.raises(GenesysApiError, match="Invalid JSON"):
                await client.list_crops()


class TestPageModel:
    """Page parsing keeps raw payloads for the cellid lookup."""

    def test_from_api_keeps_raw(self) -> None:
        """Each accession retains its raw dictionary."""
        page = AccessionPage.from_api(page_payload(["u-001"], number=0, total=1, last=True))

        assert page.content[0].raw["tileIndex3min"] == 777
        assert not page.has_next
