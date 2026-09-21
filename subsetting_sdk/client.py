"""Asynchronous HTTP client for the Genesys Subsetting API.

The client exposes the two operations the agent needs: listing the available
climate/soil indicators (with their datasets) and clustering a set of grid
cells by indicator values. It is independent from the agent and the LLM so it
can be used from scripts, notebooks or tests.

Authentication: when ``SUBSETTING_API_TOKEN`` is set, every request carries
``Authorization: <scheme> <token>`` (scheme ``API-Token`` by default, as in the
Genesys API; configurable with ``SUBSETTING_API_AUTH_SCHEME``).

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
    SubsettingApiError,
    SubsettingAuthError,
    SubsettingConnectionError,
)
from subsetting_sdk.models import (
    ClusterRequest,
    ClusterResult,
    IndicatorCategory,
    IndicatorPeriod,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://sandbox.genesys-pgr.org/api/subsetting"
DEFAULT_API_PREFIX = "/api/v1"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_AUTH_SCHEME = "API-Token"

# HTTP statuses that are worth retrying: the request never reached a healthy
# backend, so repeating it cannot duplicate any side effect (the API is read-only).
_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class SubsettingClient:
    """Async client for the Subsetting API.

    Attributes:
        base_url: Root URL of the deployment (without the ``/api/v1`` prefix).
        api_prefix: Path prefix of the Flask routes; empty if the proxy strips it.
        token: API token sent in the ``Authorization`` header, if configured.
        auth_scheme: Scheme word placed before the token in the header.
        timeout: Per-request timeout in seconds.
        max_retries: Number of additional attempts for network errors and 5xx
            gateway errors. Business and auth errors (4xx) are never retried.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_prefix: str | None = None,
        token: str | None = None,
        auth_scheme: str | None = None,
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
            token: API token (``SUBSETTING_API_TOKEN``). Requests are sent
                without the header when no token is configured.
            auth_scheme: Authorization scheme (``SUBSETTING_API_AUTH_SCHEME``).
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
        self.token = token if token is not None else os.getenv("SUBSETTING_API_TOKEN", "")
        self.auth_scheme = (
            auth_scheme or os.getenv("SUBSETTING_API_AUTH_SCHEME", DEFAULT_AUTH_SCHEME)
        ).strip()
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

    @property
    def has_token(self) -> bool:
        """Whether an API token is configured."""
        return bool(self.token.strip())

    # ------------------------------------------------------------------ #
    # Endpoints
    # ------------------------------------------------------------------ #

    async def get_indicators(self) -> list[IndicatorCategory]:
        """List every indicator grouped by stress category, with its datasets.

        Combines ``GET /indicators`` (catalog grouped by category) with
        ``GET /indicator-period`` (datasets per indicator, period and SSP
        scenario) so callers get everything needed to build a cluster request
        from one call.
        """
        categories_payload = await self._request("GET", "/indicators")
        periods_payload = await self._request("GET", "/indicator-period")

        categories = [IndicatorCategory.model_validate(item) for item in categories_payload]
        periods_by_indicator: dict[str, list[IndicatorPeriod]] = {}

        # Index the datasets by indicator id to attach them in one pass.
        for item in periods_payload:
            period = IndicatorPeriod.model_validate(item)
            periods_by_indicator.setdefault(period.indicator, []).append(period)

        # Attach the datasets to their indicator; indicators without datasets
        # keep an empty list and cannot be used in a cluster request.
        for category in categories:
            for indicator in category.indicators:
                indicator.periods = periods_by_indicator.get(indicator.id, [])

        return categories

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

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _url(self, path: str) -> str:
        """Build the absolute URL of an endpoint path such as ``/cluster``."""
        return f"{self.base_url}{self.api_prefix}{path}"

    def _headers(self) -> dict[str, str]:
        """Build the request headers, adding the token only when configured."""
        headers = {"Accept": "application/json"}

        if self.has_token:
            headers["Authorization"] = f"{self.auth_scheme} {self.token.strip()}"

        return headers

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
                response = await self._http.request(method, url, json=json, headers=self._headers())

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

    def _handle_response(self, response: httpx.Response, path: str) -> Any:
        """Map an HTTP response to decoded JSON or to the matching SDK exception.

        Args:
            response: Response received from the API.
            path: Endpoint path, used in error messages.
        """
        body_text = response.text

        # Successful responses are decoded manually because the GET endpoints
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

        # Authentication problems get a hint about the token configuration.
        if response.status_code in (401, 403):
            hint = (
                " No API token configured (SUBSETTING_API_TOKEN)."
                if not self.has_token
                else " Check SUBSETTING_API_TOKEN and SUBSETTING_API_AUTH_SCHEME."
            )
            raise SubsettingAuthError(
                f"Subsetting API refused the request to {path}: {body_text.strip()[:100]}.{hint}",
                status_code=response.status_code,
                response_body=body_text,
            )

        raise SubsettingApiError(
            f"Subsetting API request to {path} failed: {body_text.strip()[:300]}",
            status_code=response.status_code,
            response_body=body_text,
        )
