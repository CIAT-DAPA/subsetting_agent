"""Asynchronous HTTP client for the Genesys Subsetting API.

The client wraps the eight Flask endpoints of the API with typed methods. It is
independent from the agent and the LLM so it can be used from scripts, notebooks
or tests.

Example:
    async with SubsettingClient() as client:
        categories = await client.get_indicators()
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import TracebackType
from typing import Any

import httpx

from subsetting_sdk.exceptions import (
    CoreCollectionError,
    NoMatchingDataError,
    SubsettingApiError,
    SubsettingConnectionError,
)
from subsetting_sdk.models import (
    AnaloguesResult,
    ClusterRequest,
    ClusterResult,
    CoreCollectionResult,
    CropCellIds,
    IndicatorCategory,
    IndicatorDataResult,
    IndicatorFilter,
    IndicatorPeriod,
    IndicatorRangesResult,
    MonthWindow,
    SubsetResult,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://sandbox.genesys-pgr.org/api/subsetting"
DEFAULT_API_PREFIX = "/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0

# HTTP statuses that are worth retrying: the request never reached a healthy
# backend, so repeating it cannot duplicate any side effect (the API is read-only).
_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class SubsettingClient:
    """Async client for the Subsetting API.

    Attributes:
        base_url: Root URL of the deployment (without the ``/api/v1`` prefix).
        api_prefix: Path prefix of the Flask routes; empty if the proxy strips it.
        timeout: Per-request timeout in seconds.
        max_retries: Number of additional attempts for network errors and 5xx
            gateway errors. Business errors (4xx) are never retried.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_prefix: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client.

        Every argument falls back to an environment variable and then to a
        default, so the client can be created with no arguments in production.

        Args:
            base_url: Root URL of the API (``SUBSETTING_API_URL``).
            api_prefix: Route prefix (``SUBSETTING_API_PREFIX``).
            timeout: Per-request timeout in seconds (``SUBSETTING_API_TIMEOUT``).
            max_retries: Additional attempts for retryable failures.
            backoff_seconds: Base delay between retries; doubles on each attempt.
            http_client: Pre-configured ``httpx.AsyncClient``. Mainly for tests;
                when provided, the caller owns its lifecycle.
        """
        self.base_url = (base_url or os.getenv("SUBSETTING_API_URL", DEFAULT_BASE_URL)).rstrip("/")

        # ``api_prefix`` may legitimately be an empty string, so ``None`` is the
        # only value that triggers the environment/default fallback.
        if api_prefix is None:
            api_prefix = os.getenv("SUBSETTING_API_PREFIX", DEFAULT_API_PREFIX)

        self.api_prefix = "/" + api_prefix.strip("/") if api_prefix.strip("/") else ""
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("SUBSETTING_API_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        )
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        self._external_client = http_client is not None
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> SubsettingClient:
        """Enter the async context; the client is ready to use immediately."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying HTTP client when leaving the context."""
        await self.aclose()

    async def aclose(self) -> None:
        """Release the HTTP connection pool, unless the client was injected."""
        # An injected client belongs to the caller, who decides when to close it.
        if not self._external_client:
            await self._http.aclose()

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #

    async def get_indicators(self) -> list[IndicatorCategory]:
        """List every indicator grouped by stress category (``GET /indicators``)."""
        payload = await self._request("GET", "/indicators")

        return [IndicatorCategory.model_validate(item) for item in payload]

    async def get_indicator_periods(self) -> list[IndicatorPeriod]:
        """List every indicator period, i.e. dataset per indicator, period and SSP."""
        payload = await self._request("GET", "/indicator-period")

        return [IndicatorPeriod.model_validate(item) for item in payload]

    async def get_indicator_ranges(
        self,
        cellid_list: list[CropCellIds],
        indicator_periods: list[str],
        months: MonthWindow | None = None,
    ) -> IndicatorRangesResult:
        """Get the observed min/max of indicators over the selected cells.

        Useful to propose sensible range bounds before calling ``filter_subset``.

        Args:
            cellid_list: Selected cellids grouped by crop.
            indicator_periods: Ids of the indicator periods to summarize.
            months: Month window for monthly indicators; defaults to the full year.
        """
        body = {
            "months": (months or MonthWindow(start=1, end=12)).to_api(),
            "ind_periods": indicator_periods,
            "cellid_list": [item.to_api() for item in cellid_list],
        }
        payload = await self._request("POST", "/indicators-range", json=body)

        return IndicatorRangesResult.model_validate(payload)

    async def filter_subset(
        self,
        cellid_list: list[CropCellIds],
        filters: list[IndicatorFilter],
    ) -> SubsetResult:
        """Keep the cells whose aggregated indicator values fall inside every range.

        Args:
            cellid_list: Selected cellids grouped by crop.
            filters: One filter per indicator; each must define ``range``.

        Raises:
            ValueError: If a filter has no range, since the API requires it here.
            NoMatchingDataError: If no cell satisfies all the filters.
        """
        # The API indexes ``range[0]`` and ``range[1]`` unconditionally, so a
        # filter without range would crash the server instead of failing cleanly.
        for indicator_filter in filters:
            if indicator_filter.range is None:
                raise ValueError(
                    f"Filter for '{indicator_filter.name}' needs a range to be used with /subset."
                )

        body = {
            "cellid_list": [item.to_api() for item in cellid_list],
            "data": [item.to_api() for item in filters],
        }
        payload = await self._request("POST", "/subset", json=body)

        return SubsetResult.model_validate(payload)

    async def cluster(self, request: ClusterRequest) -> ClusterResult:
        """Run the multivariate clustering over the selected cells (``POST /cluster``).

        Args:
            request: Validated cluster request.

        Returns:
            Parsed result. ``rows`` is empty when the API could not run the
            analysis (it answers ``{}`` in that case instead of an error).
        """
        payload = await self._request("POST", "/cluster", json=request.to_api())

        # The API swallows analysis errors and returns an empty object, which
        # would otherwise look like a successful run with zero clusters.
        if not payload:
            logger.warning("Cluster endpoint returned an empty response; analysis likely failed.")

        return ClusterResult.from_api(payload or {})

    async def core_collection(
        self,
        cellids: list[int],
        filters: list[IndicatorFilter],
        amount: int,
        months: MonthWindow | None = None,
    ) -> CoreCollectionResult:
        """Select a representative core collection of ``amount`` cells.

        Args:
            cellids: Cells to sample from (usually one cluster).
            filters: Indicators that define the environmental space.
            amount: Number of cells to keep.
            months: Month window for monthly indicators; defaults to the full year.

        Raises:
            CoreCollectionError: If ``amount`` exceeds the cells with complete data.
        """
        body = {
            "indicators": [item.to_api() for item in filters],
            "amount": amount,
            "cellIds": list(dict.fromkeys(cellids)),
            "months": (months or MonthWindow(start=1, end=12)).to_api(),
        }
        payload = await self._request("POST", "/core-collection", json=body)

        return CoreCollectionResult.model_validate(payload)

    async def analogues(
        self,
        reference_cellid: int,
        reference_indicator_periods: list[str],
        cellids: list[int],
        indicator_periods: list[str],
    ) -> AnaloguesResult:
        """Rank cells by climatic similarity to a reference cell.

        Args:
            reference_cellid: Cell whose climate is the target.
            reference_indicator_periods: Indicator periods describing the reference.
            cellids: Candidate cells to compare.
            indicator_periods: Indicator periods describing the candidates.
        """
        body = {
            "cellid_ref": reference_cellid,
            "indicator_cellid_ref": reference_indicator_periods,
            "indicator_cellids": indicator_periods,
            "cellids": list(dict.fromkeys(cellids)),
        }
        payload = await self._request("POST", "/analogues-multivariate", json=body)

        return AnaloguesResult.model_validate(payload)

    async def get_indicators_data(
        self,
        cellids: list[int],
        indicator_periods: list[str],
    ) -> IndicatorDataResult:
        """Fetch the raw indicator values of the given cells (``POST /indicators-data``).

        Args:
            cellids: Cells to read.
            indicator_periods: Indicator periods to read.
        """
        body = {"cellid": list(dict.fromkeys(cellids)), "indicators": indicator_periods}
        payload = await self._request("POST", "/indicators-data", json=body)

        return IndicatorDataResult.model_validate(payload)

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        """Build the absolute URL of an endpoint path such as ``/subset``."""
        return f"{self.base_url}{self.api_prefix}{path}"

    async def _request(self, method: str, path: str, *, json: Any = None) -> Any:
        """Send a request with retries and translate failures into SDK errors.

        Args:
            method: HTTP method.
            path: Endpoint path relative to the API prefix.
            json: JSON body for POST requests.

        Returns:
            Decoded JSON body. Some endpoints return ``json.dumps`` strings with
            ``text/html`` content type, so the body is decoded manually.
        """
        url = self._url(path)
        attempt = 0

        # Retry loop: the request is repeated only for transport errors and
        # gateway statuses, up to ``max_retries`` additional times.
        while True:
            try:
                response = await self._http.request(method, url, json=json)

            except httpx.HTTPError as exc:
                # Network-level failure: retry if attempts remain, else surface it.
                if attempt < self.max_retries:
                    attempt += 1
                    await self._sleep_before_retry(attempt, f"{method} {path}: {exc}")
                    continue

                raise SubsettingConnectionError(
                    f"Could not reach the Subsetting API at {url}: {exc}"
                ) from exc

            # Gateway errors mean the backend was unavailable; retry them.
            if response.status_code in _RETRYABLE_STATUSES and attempt < self.max_retries:
                attempt += 1
                await self._sleep_before_retry(
                    attempt, f"{method} {path}: HTTP {response.status_code}"
                )
                continue

            return self._handle_response(response, path)

    async def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        """Wait with exponential backoff before the next attempt.

        Args:
            attempt: Number of the upcoming attempt (1-based).
            reason: Description logged for diagnostics.
        """
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        logger.warning(
            "Retrying (%s/%s) in %.1fs after %s", attempt, self.max_retries, delay, reason
        )
        await asyncio.sleep(delay)

    @staticmethod
    def _handle_response(response: httpx.Response, path: str) -> Any:
        """Map an HTTP response to decoded JSON or to the matching SDK exception.

        Args:
            response: Response received from the API.
            path: Endpoint path, used to pick the business exception.
        """
        body_text = response.text

        # Successful responses are decoded manually because two endpoints
        # return ``json.dumps`` output without a JSON content type.
        if response.is_success:
            try:
                return response.json()

            except ValueError as exc:
                raise SubsettingApiError(
                    f"Invalid JSON received from {path}: {body_text[:200]}",
                    status_code=response.status_code,
                    response_body=body_text,
                ) from exc

        # A 400 from /subset is the API's way of saying "no cell matched".
        if response.status_code == 400 and path == "/subset":
            raise NoMatchingDataError(
                "No accession location matches the selected climate filters.",
                status_code=400,
                response_body=body_text,
            )

        # A 422 from /core-collection means the requested amount is too large.
        if response.status_code == 422 and path == "/core-collection":
            raise CoreCollectionError(
                body_text.strip() or "Core collection cannot be computed for this selection.",
                status_code=422,
                response_body=body_text,
            )

        raise SubsettingApiError(
            f"Subsetting API request to {path} failed: {body_text.strip()[:300]}",
            status_code=response.status_code,
            response_body=body_text,
        )
