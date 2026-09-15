"""Unit tests for :class:`SubsettingClient` using a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.exceptions import (
    CoreCollectionError,
    NoMatchingDataError,
    SubsettingApiError,
    SubsettingConnectionError,
)
from subsetting_sdk.models import (
    ClusterRequest,
    CropCellIds,
    IndicatorFilter,
    IndicatorType,
    MonthWindow,
)

BASE = "https://example.org/api/subsetting"


def make_client(**kwargs) -> SubsettingClient:
    """Create a client pointed at the mocked host with retries disabled by default.

    Args:
        **kwargs: Overrides forwarded to :class:`SubsettingClient`.
    """
    kwargs.setdefault("max_retries", 0)
    kwargs.setdefault("backoff_seconds", 0.0)

    return SubsettingClient(BASE, api_prefix="/api/v1", **kwargs)


def generic_filter(value_range: tuple[float, float] | None = (0.0, 1000.0)) -> IndicatorFilter:
    """Build a generic precipitation filter for request tests.

    Args:
        value_range: Range to attach, or ``None`` to omit it.
    """
    return IndicatorFilter(
        type=IndicatorType.GENERIC,
        name="Total precipitation",
        indicator_periods=["64a000000000000000000001"],
        months=MonthWindow(start=1, end=12),
        range=value_range,
    )


class TestUrlBuilding:
    """Base URL, prefix and environment fallbacks compose the endpoint URL."""

    def test_prefix_and_base_are_joined(self) -> None:
        """Trailing and leading slashes are normalized."""
        client = SubsettingClient("https://h/api/", api_prefix="api/v1/", max_retries=0)

        assert client._url("/subset") == "https://h/api/api/v1/subset"

    def test_empty_prefix_is_allowed(self) -> None:
        """An empty prefix means the proxy strips ``/api/v1``."""
        client = SubsettingClient("https://h/api", api_prefix="", max_retries=0)

        assert client._url("/subset") == "https://h/api/subset"

    def test_environment_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without arguments, configuration comes from environment variables."""
        monkeypatch.setenv("SUBSETTING_API_URL", "https://env.example/x/")
        monkeypatch.setenv("SUBSETTING_API_PREFIX", "")
        monkeypatch.setenv("SUBSETTING_API_TIMEOUT", "5")

        client = SubsettingClient(max_retries=0)

        assert client._url("/indicators") == "https://env.example/x/indicators"
        assert client.timeout == 5.0


class TestCatalogEndpoints:
    """GET endpoints return typed lists even when served as ``text/html``."""

    async def test_get_indicators_parses_string_body(
        self, httpx_mock: HTTPXMock, indicators_payload: list[dict]
    ) -> None:
        """``/indicators`` returns ``json.dumps`` text; it is decoded anyway."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/indicators",
            content=json.dumps(indicators_payload).encode(),
            headers={"content-type": "text/html; charset=utf-8"},
        )

        async with make_client() as client:
            categories = await client.get_indicators()

        assert [c.category for c in categories] == [
            "Drought stress",
            "Heat stress",
            "Crop specific",
            "Soil properties",
        ]
        assert categories[2].indicators[0].indicator_type is IndicatorType.SPECIFIC

    async def test_get_indicator_periods(
        self, httpx_mock: HTTPXMock, periods_payload: list[dict]
    ) -> None:
        """``/indicator-period`` rows map to :class:`IndicatorPeriod`."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicator-period", json=periods_payload)

        async with make_client() as client:
            periods = await client.get_indicator_periods()

        assert len(periods) == len(periods_payload)
        assert periods[1].ssp == "ssp245"


class TestSubsetEndpoint:
    """``/subset`` sends the expected body and maps its 400 to a business error."""

    async def test_request_body_and_response(self, httpx_mock: HTTPXMock) -> None:
        """The body uses ``cellid_list`` and ``data``; the response is typed."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/subset",
            json={
                "filtered_cellids": [{"crop": "bean", "cellid": [1]}],
                "quantile": [],
                "proportion": [],
            },
        )

        async with make_client() as client:
            result = await client.filter_subset(
                [CropCellIds(crop="bean", cellids=[1, 2])], [generic_filter()]
            )

        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert sent["cellid_list"] == [{"crop": "bean", "cellids": [1, 2]}]
        assert sent["data"][0]["indicator"] == ["64a000000000000000000001"]
        assert sent["data"][0]["range"] == [0.0, 1000.0]
        assert result.all_cellids() == [1]

    async def test_filter_without_range_is_rejected_locally(self) -> None:
        """A filter without range would crash the server; fail before sending."""
        async with make_client() as client:
            with pytest.raises(ValueError, match="needs a range"):
                await client.filter_subset(
                    [CropCellIds(crop="bean", cellids=[1])], [generic_filter(None)]
                )

    async def test_no_match_400_becomes_no_matching_data_error(self, httpx_mock: HTTPXMock) -> None:
        """The API's plain-text 400 is surfaced as :class:`NoMatchingDataError`."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/subset",
            status_code=400,
            text="Bad request! No data matching the selected filters!",
        )

        async with make_client() as client:
            with pytest.raises(NoMatchingDataError) as info:
                await client.filter_subset(
                    [CropCellIds(crop="bean", cellids=[1])], [generic_filter()]
                )

        assert info.value.status_code == 400
        assert "No data matching" in str(info.value.response_body)


class TestClusterEndpoint:
    """``/cluster`` posts the nested body and tolerates empty responses."""

    async def test_cluster_round_trip(self, httpx_mock: HTTPXMock, cluster_payload: dict) -> None:
        """The response rows are grouped into clusters."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/cluster", json=cluster_payload)
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="bean", cellids=[101, 102, 103])],
            filters=[generic_filter(None)],
        )

        async with make_client() as client:
            result = await client.cluster(request)

        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert sent["analysis"]["algorithm"] == ["agglomerative"]
        assert result.clusters(crop="bean") == {0: [101, 102], 1: [103]}

    async def test_empty_response_yields_no_rows(self, httpx_mock: HTTPXMock) -> None:
        """The API returns ``{}`` when the analysis fails; no exception is raised."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/cluster", json={})
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="bean", cellids=[1])], filters=[generic_filter(None)]
        )

        async with make_client() as client:
            result = await client.cluster(request)

        assert result.rows == []


class TestCoreCollectionEndpoint:
    """``/core-collection`` maps its 422 to :class:`CoreCollectionError`."""

    async def test_success(self, httpx_mock: HTTPXMock) -> None:
        """The response exposes the selected cellids and the body uses ``cellIds``."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/core-collection", json={"cellids": [1, 3]})

        async with make_client() as client:
            result = await client.core_collection([1, 2, 3, 3], [generic_filter(None)], amount=2)

        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert sent["cellIds"] == [1, 2, 3]
        assert sent["amount"] == 2
        assert result.cellids == [1, 3]

    async def test_422_becomes_core_collection_error(self, httpx_mock: HTTPXMock) -> None:
        """Too large an amount is reported as a business error, not a generic one."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/core-collection",
            status_code=422,
            text="Core collection cannot be applied! ...",
        )

        async with make_client() as client:
            with pytest.raises(CoreCollectionError) as info:
                await client.core_collection([1], [generic_filter(None)], amount=50)

        assert info.value.status_code == 422


class TestTransportErrors:
    """Network failures and unexpected statuses become SDK exceptions."""

    async def test_unexpected_status_is_generic_api_error(self, httpx_mock: HTTPXMock) -> None:
        """A 500 on any endpoint raises :class:`SubsettingApiError`."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", status_code=500, text="boom")

        async with make_client() as client:
            with pytest.raises(SubsettingApiError) as info:
                await client.get_indicators()

        assert info.value.status_code == 500
        assert "[HTTP 500]" in str(info.value)

    async def test_invalid_json_is_reported(self, httpx_mock: HTTPXMock) -> None:
        """A 200 with a non-JSON body is an API error, not a crash."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", text="<html>oops</html>")

        async with make_client() as client:
            with pytest.raises(SubsettingApiError, match="Invalid JSON"):
                await client.get_indicators()

    async def test_connection_error_after_retries(self, httpx_mock: HTTPXMock) -> None:
        """Transport errors are retried and finally raised as connection errors."""
        httpx_mock.add_exception(httpx.ConnectError("refused"), url=f"{BASE}/api/v1/indicators")
        httpx_mock.add_exception(httpx.ConnectError("refused"), url=f"{BASE}/api/v1/indicators")

        async with make_client(max_retries=1) as client:
            with pytest.raises(SubsettingConnectionError):
                await client.get_indicators()

        assert len(httpx_mock.get_requests()) == 2

    async def test_gateway_error_is_retried_then_succeeds(
        self, httpx_mock: HTTPXMock, periods_payload: list[dict]
    ) -> None:
        """A 503 followed by a 200 yields the successful result."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicator-period", status_code=503)
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicator-period", json=periods_payload)

        async with make_client(max_retries=1) as client:
            periods = await client.get_indicator_periods()

        assert len(periods) == len(periods_payload)
        assert len(httpx_mock.get_requests()) == 2

    async def test_business_4xx_is_not_retried(self, httpx_mock: HTTPXMock) -> None:
        """A 400 from ``/subset`` is final: only one request is sent."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/subset", status_code=400, text="no data")

        async with make_client(max_retries=3) as client:
            with pytest.raises(NoMatchingDataError):
                await client.filter_subset(
                    [CropCellIds(crop="bean", cellids=[1])], [generic_filter()]
                )

        assert len(httpx_mock.get_requests()) == 1


class TestInjectedHttpClient:
    """An injected ``httpx.AsyncClient`` is not closed by the SDK."""

    async def test_external_client_stays_open(self) -> None:
        """Leaving the context does not close a client owned by the caller."""
        external = httpx.AsyncClient()

        async with SubsettingClient(BASE, http_client=external, max_retries=0):
            pass

        assert not external.is_closed
        await external.aclose()
