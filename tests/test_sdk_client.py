"""Unit tests for :class:`SubsettingClient` using a mocked HTTP transport."""

from __future__ import annotations

import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.exceptions import (
    SubsettingApiError,
    SubsettingAuthError,
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
    kwargs.setdefault("token", "")

    return SubsettingClient(BASE, api_prefix="/api/v1", **kwargs)


def generic_filter() -> IndicatorFilter:
    """Build a generic precipitation filter for request tests."""
    return IndicatorFilter(
        type=IndicatorType.GENERIC,
        name="Total precipitation",
        indicator_periods=["64a000000000000000000001"],
        months=MonthWindow(start=1, end=12),
    )


def mock_catalog(httpx_mock: HTTPXMock, indicators: list[dict], periods: list[dict]) -> None:
    """Register both catalog endpoints on the mock.

    Args:
        httpx_mock: Active HTTP mock.
        indicators: Body of ``/indicators``.
        periods: Body of ``/indicator-period``.
    """
    httpx_mock.add_response(
        url=f"{BASE}/api/v1/indicators",
        content=json.dumps(indicators).encode(),
        headers={"content-type": "text/html; charset=utf-8"},
    )
    httpx_mock.add_response(url=f"{BASE}/api/v1/indicator-period", json=periods)


class TestUrlBuilding:
    """Base URL and prefix compose the endpoint URL."""

    def test_prefix_and_base_are_joined(self) -> None:
        """Trailing and leading slashes are normalized."""
        client = SubsettingClient("https://h/api/", api_prefix="api/v1/", max_retries=0)

        assert client._url("/cluster") == "https://h/api/api/v1/cluster"

    def test_empty_prefix_is_allowed(self) -> None:
        """An empty prefix means the proxy strips ``/api/v1``."""
        client = SubsettingClient("https://h/api", api_prefix="", max_retries=0)

        assert client._url("/cluster") == "https://h/api/cluster"

    def test_explicit_auth_scheme(self) -> None:
        """The scheme given by the caller is used in the header."""
        client = SubsettingClient("https://h", token="t", auth_scheme="Bearer", max_retries=0)

        assert client._headers()["Authorization"] == "Bearer t"


class TestAuthentication:
    """The API token travels in the Authorization header when configured."""

    async def test_token_header_default_scheme(self, httpx_mock: HTTPXMock) -> None:
        """The default scheme is ``API-Token``, like the Genesys API."""
        mock_catalog(httpx_mock, [], [])

        async with make_client(token="secret") as client:
            await client.get_indicators()

        for request in httpx_mock.get_requests():
            assert request.headers["Authorization"] == "API-Token secret"

    async def test_no_token_sends_no_header(self, httpx_mock: HTTPXMock) -> None:
        """Without a token the header is absent (anonymous access)."""
        mock_catalog(httpx_mock, [], [])

        async with make_client(token="") as client:
            await client.get_indicators()

        assert "Authorization" not in httpx_mock.get_requests()[0].headers

    @pytest.mark.parametrize("status", [401, 403])
    async def test_auth_failure_without_token_hints_variable(
        self, httpx_mock: HTTPXMock, status: int
    ) -> None:
        """401/403 without a token name the environment variable to set."""
        httpx_mock.add_response(
            url=f"{BASE}/api/v1/indicators", status_code=status, text="Forbidden"
        )

        async with make_client(token="") as client:
            with pytest.raises(SubsettingAuthError, match="SUBSETTING_API_TOKEN") as info:
                await client.get_indicators()

        assert info.value.status_code == status

    async def test_auth_failure_with_token_hints_scheme(self, httpx_mock: HTTPXMock) -> None:
        """403 with a token points at the token and the scheme."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", status_code=403, text="Forbidden")

        async with make_client(token="bad") as client:
            with pytest.raises(SubsettingAuthError, match="SUBSETTING_API_AUTH_SCHEME"):
                await client.get_indicators()

    async def test_auth_failure_is_not_retried(self, httpx_mock: HTTPXMock) -> None:
        """A 403 is final even when retries are enabled."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", status_code=403, text="Forbidden")

        async with make_client(max_retries=3) as client:
            with pytest.raises(SubsettingAuthError):
                await client.get_indicators()

        assert len(httpx_mock.get_requests()) == 1


class TestGetIndicators:
    """``get_indicators`` merges the catalog with the datasets of each indicator."""

    async def test_periods_are_attached(
        self, httpx_mock: HTTPXMock, indicators_payload: list[dict], periods_payload: list[dict]
    ) -> None:
        """Every indicator receives its periods; the text/html body is decoded."""
        mock_catalog(httpx_mock, indicators_payload, periods_payload)

        async with make_client() as client:
            categories = await client.get_indicators()

        by_id = {i.id: i for c in categories for i in c.indicators}

        assert [c.category for c in categories] == [
            "Drought stress",
            "Heat stress",
            "Crop specific",
            "Soil properties",
        ]
        assert {p.ssp for p in by_id["prec"].periods} == {"historical", "ssp245"}
        assert [p.id for p in by_id["tmax"].periods] == ["64a000000000000000000003"]
        assert by_id["opt_bean"].indicator_type is IndicatorType.SPECIFIC

    async def test_indicator_without_periods(
        self, httpx_mock: HTTPXMock, indicators_payload: list[dict]
    ) -> None:
        """Indicators missing from ``/indicator-period`` keep an empty list."""
        mock_catalog(httpx_mock, indicators_payload, [])

        async with make_client() as client:
            categories = await client.get_indicators()

        assert all(not i.periods for c in categories for i in c.indicators)


class TestCluster:
    """``/cluster`` posts the nested body and tolerates empty responses."""

    async def test_cluster_round_trip(self, httpx_mock: HTTPXMock, cluster_payload: dict) -> None:
        """The response rows are grouped into clusters with statistics."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/cluster", json=cluster_payload)
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="bean", cellids=[101, 102, 103])],
            filters=[generic_filter()],
        )

        async with make_client(token="secret") as client:
            result = await client.cluster(request)

        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert httpx_mock.get_requests()[0].headers["Authorization"] == "API-Token secret"
        assert sent["analysis"]["algorithm"] == ["agglomerative"]
        assert "range" not in sent["data"][0]
        assert result.clusters(crop="bean") == {0: [101, 102], 1: [103]}
        assert result.cluster_statistics()[1]["prec"]["mean"] == 85.0

    async def test_empty_response_yields_no_rows(self, httpx_mock: HTTPXMock) -> None:
        """The API returns ``{}`` when the analysis fails; no exception is raised."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/cluster", json={})
        request = ClusterRequest(
            cellid_list=[CropCellIds(crop="bean", cellids=[1])], filters=[generic_filter()]
        )

        async with make_client() as client:
            result = await client.cluster(request)

        assert result.rows == []


class TestTransportErrors:
    """Network failures and unexpected statuses become SDK exceptions."""

    async def test_unexpected_status_is_generic_api_error(self, httpx_mock: HTTPXMock) -> None:
        """A 500 raises :class:`SubsettingApiError`."""
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

    async def test_gateway_error_is_retried_then_succeeds(self, httpx_mock: HTTPXMock) -> None:
        """A 503 followed by a 200 yields the successful result."""
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", status_code=503)
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicators", json=[])
        httpx_mock.add_response(url=f"{BASE}/api/v1/indicator-period", json=[])

        async with make_client(max_retries=1) as client:
            categories = await client.get_indicators()

        assert categories == []
        assert len(httpx_mock.get_requests()) == 3


class TestInjectedHttpClient:
    """An injected ``httpx.AsyncClient`` is not closed by the SDK."""

    async def test_external_client_stays_open(self) -> None:
        """Leaving the context does not close a client owned by the caller."""
        external = httpx.AsyncClient()

        async with SubsettingClient(BASE, http_client=external, max_retries=0):
            pass

        assert not external.is_closed
        await external.aclose()
