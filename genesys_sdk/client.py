"""Asynchronous HTTP client for the Genesys PGR REST API.

Replaces the Genesys MCP server: the agent calls the same underlying API
directly, which lets the SDK build validated filters instead of relying on the
language model to assemble them.

Example:
    async with GenesysClient() as client:
        page = await client.list_accessions(
            AccessionFilter.passport(crop_codes=["bean"], with_coordinates=True)
        )
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Any

import httpx

from genesys_sdk.exceptions import (
    GenesysApiError,
    GenesysAuthError,
    GenesysBadRequestError,
    GenesysConnectionError,
    GenesysNotFoundError,
)
from genesys_sdk.models import (
    AccessionDetails,
    AccessionFilter,
    AccessionObservations,
    AccessionOverview,
    AccessionPage,
    Crop,
    DescriptorFilter,
    DescriptorPage,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.sandbox.genesys-pgr.org"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 500
DEFAULT_MAX_ACCESSIONS = 2000

# Gateway statuses worth retrying; the API is read-only for the agent.
_RETRYABLE_STATUSES = frozenset({502, 503, 504})


class GenesysClient:
    """Async client for the Genesys API.

    Attributes:
        base_url: Root URL of the API.
        token: API token sent as ``Authorization: API-Token <token>``.
        timeout: Per-request timeout in seconds.
        max_retries: Additional attempts for network errors and gateway errors.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        token: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        backoff_seconds: float = 1.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure the client from arguments, environment variables or defaults.

        Args:
            base_url: Root URL (``GENESYS_API_URL``).
            token: API token (``GENESYS_API_TOKEN``). Requests are sent without
                the header when no token is configured; the API then serves
                only public data or answers 401.
            timeout: Per-request timeout in seconds (``GENESYS_API_TIMEOUT``).
            max_retries: Additional attempts for retryable failures.
            backoff_seconds: Base delay between retries; doubles each attempt.
            http_client: Pre-configured ``httpx.AsyncClient`` owned by the caller.
        """
        self.base_url = (base_url or os.getenv("GENESYS_API_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.token = token if token is not None else os.getenv("GENESYS_API_TOKEN", "")
        self.timeout = float(
            timeout
            if timeout is not None
            else os.getenv("GENESYS_API_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
        )
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

        self._external_client = http_client is not None
        self._http = http_client or httpx.AsyncClient(timeout=self.timeout)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> GenesysClient:
        """Enter the async context; the client is ready immediately."""
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
        """Release the connection pool unless the HTTP client was injected."""
        # An injected client is owned by the caller.
        if not self._external_client:
            await self._http.aclose()

    @property
    def has_token(self) -> bool:
        """Whether an API token is configured."""
        return bool(self.token.strip())

    # ------------------------------------------------------------------ #
    # Accessions
    # ------------------------------------------------------------------ #

    async def list_accessions(
        self,
        accession_filter: AccessionFilter | None = None,
        *,
        page: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
        filter_code: str | None = None,
        sort: str | None = None,
        direction: str | None = None,
    ) -> AccessionPage:
        """Fetch one page of accessions matching a passport filter.

        Args:
            accession_filter: Passport filter; ignored when ``filter_code`` is given.
            page: Zero-based page index.
            page_size: Records per page (capped at ``MAX_PAGE_SIZE``).
            filter_code: Server-side code of a previously used filter.
            sort: Property to order by.
            direction: Sort direction (``ASC``/``DESC``).

        Raises:
            ValueError: If neither a filter nor a filter code is provided.
        """
        # The API needs either a body filter or a stored filter code.
        if accession_filter is None and filter_code is None:
            raise ValueError("Provide an accession filter or a filter code.")

        params: dict[str, Any] = {"p": page, "l": min(max(page_size, 1), MAX_PAGE_SIZE)}

        # Optional query parameters are only sent when set.
        if filter_code is not None:
            params["f"] = filter_code

        if sort:
            params["s"] = sort

        if direction:
            params["d"] = direction

        body = accession_filter.to_api() if accession_filter is not None else {}
        payload = await self._request("POST", "/api/v2/acn/list", params=params, json=body)

        return AccessionPage.from_api(payload)

    async def iter_accessions(
        self,
        accession_filter: AccessionFilter,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_records: int | None = None,
    ) -> AsyncIterator[AccessionPage]:
        """Iterate over the pages of a filter until exhausted or a record cap is hit.

        The first page is requested with the filter body; following pages reuse
        the ``filterCode`` returned by the server when available.

        Args:
            accession_filter: Passport filter.
            page_size: Records per page.
            max_records: Stop after this many records have been yielded
                (``GENESYS_MAX_ACCESSIONS`` when ``None``). The last page may
                be truncated to respect the cap exactly.

        Yields:
            Pages in order.
        """
        cap = (
            max_records
            if max_records is not None
            else int(os.getenv("GENESYS_MAX_ACCESSIONS", DEFAULT_MAX_ACCESSIONS))
        )
        fetched = 0
        page_index = 0
        filter_code: str | None = None

        # Keep requesting pages while records remain and the cap is not reached.
        while True:
            page = await self.list_accessions(
                accession_filter if filter_code is None else None,
                page=page_index,
                page_size=page_size,
                filter_code=filter_code,
            )

            remaining = cap - fetched

            # Trim the page when it would exceed the cap, then stop.
            if len(page.content) > remaining:
                page.content = page.content[:remaining]
                page.last = True

            fetched += len(page.content)
            yield page

            if page.last or not page.content or fetched >= cap:
                return

            filter_code = page.filter_code or filter_code
            page_index += 1

    async def collect_accessions(
        self,
        accession_filter: AccessionFilter,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        max_records: int | None = None,
    ) -> tuple[list[Any], int]:
        """Fetch accessions into a list, up to a record cap.

        Args:
            accession_filter: Passport filter.
            page_size: Records per page.
            max_records: Record cap; see :meth:`iter_accessions`.

        Returns:
            The accessions fetched and the total number matching the filter on
            the server, so callers can tell whether the cap truncated the result.
        """
        accessions: list[Any] = []
        total = 0

        # Accumulate every page; the total comes from the server metadata.
        async for page in self.iter_accessions(
            accession_filter, page_size=page_size, max_records=max_records
        ):
            accessions.extend(page.content)
            total = page.total_elements

        return accessions, total

    async def accession_overview(
        self,
        accession_filter: AccessionFilter | None = None,
        *,
        filter_code: str | None = None,
        limit: int = 10,
    ) -> AccessionOverview:
        """Get counts per passport field for the accessions matching a filter.

        Args:
            accession_filter: Passport filter; ignored when ``filter_code`` is given.
            filter_code: Server-side code of a previously used filter.
            limit: Number of top values per field.

        Raises:
            ValueError: If neither a filter nor a filter code is provided.
        """
        if accession_filter is None and filter_code is None:
            raise ValueError("Provide an accession filter or a filter code.")

        params: dict[str, Any] = {"limit": limit}

        if filter_code is not None:
            params["f"] = filter_code

        body = accession_filter.to_api() if accession_filter is not None else {}
        payload = await self._request("POST", "/api/v2/acn/overview", params=params, json=body)

        return AccessionOverview.model_validate(payload)

    async def get_accession(self, uuid: str) -> AccessionDetails:
        """Fetch the full record of one accession.

        Args:
            uuid: Accession UUID.

        Raises:
            GenesysNotFoundError: If the UUID is unknown.
        """
        payload = await self._request("GET", f"/api/v2/acn/details/{uuid}")

        return AccessionDetails.from_api(payload)

    async def get_observations(self, uuid: str) -> AccessionObservations:
        """Fetch the trait observations of one accession.

        Args:
            uuid: Accession UUID.
        """
        payload = await self._request("GET", f"/api/v2/acn/{uuid}/observations")

        return AccessionObservations.model_validate(payload or {})

    # ------------------------------------------------------------------ #
    # Reference data
    # ------------------------------------------------------------------ #

    async def list_crops(self) -> list[Crop]:
        """List the crops and crop groups known to Genesys."""
        payload = await self._request("GET", "/api/v2/crop")

        return [Crop.model_validate(item) for item in payload]

    async def list_descriptors(
        self,
        descriptor_filter: DescriptorFilter | None = None,
        *,
        page: int = 0,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> DescriptorPage:
        """Fetch one page of trait descriptors.

        Args:
            descriptor_filter: Descriptor filter; ``None`` lists everything.
            page: Zero-based page index.
            page_size: Records per page.
        """
        params = {"p": page, "l": min(max(page_size, 1), MAX_PAGE_SIZE)}
        body = descriptor_filter.to_api() if descriptor_filter is not None else {}
        payload = await self._request(
            "POST", "/api/v2/descriptor/list/details", params=params, json=body
        )

        return DescriptorPage.model_validate(payload)

    # ------------------------------------------------------------------ #
    # Transport
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        """Build the request headers, adding the token only when configured."""
        headers = {"Accept": "application/json"}

        if self.has_token:
            headers["Authorization"] = f"API-Token {self.token.strip()}"

        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
    ) -> Any:
        """Send a request with retries and translate failures into SDK errors.

        Args:
            method: HTTP method.
            path: Absolute path of the endpoint.
            params: Query parameters.
            json: JSON body for POST requests.
        """
        url = f"{self.base_url}{path}"
        attempt = 0

        # Retry loop for transport errors and gateway statuses only.
        while True:
            try:
                response = await self._http.request(
                    method, url, params=params, json=json, headers=self._headers()
                )

            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    await self._sleep_before_retry(attempt, f"{method} {path}: {exc}")
                    continue

                raise GenesysConnectionError(
                    f"Could not reach the Genesys API at {url}: {exc}"
                ) from exc

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
        """Map an HTTP response to decoded JSON or the matching SDK exception.

        Args:
            response: Response received from the API.
            path: Endpoint path, used in error messages.
        """
        body_text = response.text

        if response.is_success:
            # Some endpoints legitimately return an empty body.
            if not body_text.strip():
                return None

            try:
                return response.json()

            except ValueError as exc:
                raise GenesysApiError(
                    f"Invalid JSON received from {path}: {body_text[:200]}",
                    status_code=response.status_code,
                    response_body=body_text,
                ) from exc

        message = self._error_message(response) or f"Genesys API request to {path} failed."

        # Authentication problems get a hint about the token configuration.
        if response.status_code in (401, 403):
            hint = "" if self.has_token else " No API token configured (GENESYS_API_TOKEN)."
            raise GenesysAuthError(
                message + hint, status_code=response.status_code, response_body=body_text
            )

        if response.status_code == 404:
            raise GenesysNotFoundError(message, status_code=404, response_body=body_text)

        if response.status_code == 400:
            raise GenesysBadRequestError(message, status_code=400, response_body=body_text)

        raise GenesysApiError(message, status_code=response.status_code, response_body=body_text)

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Extract the ``error`` text of an ``ApiErrorException`` body, if present.

        Args:
            response: Failed response.
        """
        try:
            payload = response.json()

        except ValueError:
            return response.text.strip()[:300]

        # The API wraps errors as {"error": ..., "localizedError": ...}.
        if isinstance(payload, dict):
            return str(payload.get("localizedError") or payload.get("error") or "")[:300]

        return str(payload)[:300]
