"""HTTP client for the Genesys PGR API v2 (accession listing).

Only the accession listing endpoint is covered for now::

    POST /api/v2/acn/list?p=<page>&l=<size>&s=<sort>&d=<direction>
    body: AccessionFilter (only the criteria with a value)

Usage::

    client = GenesysClient.from_settings(get_settings())
    page = client.list_accessions(AccessionFilter(crop=["bean"]), page=0, size=100)
    frame = client.list_accessions_dataframe(AccessionFilter(taxonomy=TaxonomyFilter(genus=["Phaseolus"])))
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from core.logger import get_logger
from sdks.genesys.auth import ApiTokenAuth, ClientCredentialsAuth, GenesysAuth
from sdks.genesys.errors import (
    GenesysAuthError,
    GenesysConnectionError,
    GenesysError,
    GenesysRequestError,
)
from sdks.genesys.models import AccessionFilter, AccessionPage, AccessionRecord, Crop

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from core.config import Settings

logger = get_logger(__name__)

# Path of the accession listing endpoint, relative to the base URL.
LIST_ACCESSIONS_PATH = "/api/v2/acn/list"
# Path of the crop catalogue endpoint.
CROPS_PATH = "/api/v2/crop"
# Path of the OAuth2 token endpoint, relative to the base URL.
TOKEN_PATH = "/oauth/token"
# Hard limit imposed by Genesys on the page size of JSON listings.
MAX_PAGE_SIZE = 1000
# HTTP statuses that are worth retrying with a back-off.
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class GenesysClient:
    """Synchronous client for the Genesys API built on ``httpx``."""

    def __init__(
        self,
        base_url: str,
        auth: GenesysAuth,
        *,
        timeout: float = 60.0,
        page_size: int = 500,
        max_records: int = 5000,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create the client.

        Args:
            base_url: API root, e.g. ``https://api.genesys-pgr.org``.
            auth: Authentication strategy.
            timeout: Request timeout in seconds.
            page_size: Default records per page (capped at ``MAX_PAGE_SIZE``).
            max_records: Safety cap on the records fetched by ``iter_accessions``.
            max_retries: Attempts for transient errors (429/5xx/network).
            transport: Optional ``httpx`` transport (used by tests to mock HTTP).
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.page_size = min(max(page_size, 1), MAX_PAGE_SIZE)
        self.max_records = max(max_records, 1)
        self.max_retries = max(max_retries, 1)
        self._crops: list[Crop] | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    # ------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> GenesysClient:
        """Build a client from the application settings.

        The API token takes precedence; OAuth2 client credentials are used when
        no token is configured.

        Args:
            settings: Application settings (``GENESYS_*`` variables).
            **overrides: Keyword arguments forwarded to the constructor
                (``transport``, ``timeout``...), overriding the settings.

        Returns:
            A configured ``GenesysClient``.

        Raises:
            GenesysAuthError: If neither a token nor client credentials are set.
        """
        base_url = settings.genesys_api_url

        # Prefer the personal API token; fall back to OAuth2 client credentials.
        if settings.genesys_api_token:
            auth: GenesysAuth = ApiTokenAuth(settings.genesys_api_token)
        elif settings.genesys_client_id and settings.genesys_client_secret:
            auth = ClientCredentialsAuth(
                client_id=settings.genesys_client_id,
                client_secret=settings.genesys_client_secret,
                token_url=f"{base_url.rstrip('/')}{TOKEN_PATH}",
            )
        else:
            raise GenesysAuthError(
                "Configure GENESYS_API_TOKEN or GENESYS_CLIENT_ID/GENESYS_CLIENT_SECRET in .env."
            )

        kwargs: dict[str, Any] = {
            "timeout": settings.genesys_timeout,
            "page_size": settings.genesys_page_size,
            "max_records": settings.genesys_max_records,
        }
        kwargs.update(overrides)

        return cls(base_url=base_url, auth=auth, **kwargs)

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> GenesysClient:
        """Support ``with GenesysClient(...) as client``."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close the client when leaving the ``with`` block."""
        self.close()

    # -------------------------------------------------------------- public
    def list_accessions(
        self,
        accession_filter: AccessionFilter | None = None,
        *,
        page: int = 0,
        size: int | None = None,
        sort: str | None = None,
        direction: str | None = None,
    ) -> AccessionPage:
        """Fetch one page of accessions matching a filter.

        Args:
            accession_filter: Search criteria; ``None`` or empty lists everything.
            page: Zero-based page number.
            size: Records per page (defaults to the client ``page_size``).
            sort: Property name to order by (API parameter ``s``).
            direction: Sort direction, ``ASC`` or ``DESC`` (API parameter ``d``).

        Returns:
            The parsed page.
        """
        body = accession_filter.to_body() if accession_filter is not None else {}
        params: dict[str, Any] = {"p": page, "l": min(size or self.page_size, MAX_PAGE_SIZE)}

        # Sorting parameters are optional; only send them when given.
        if sort:
            params["s"] = sort
        if direction:
            params["d"] = direction

        payload = self._post_json(LIST_ACCESSIONS_PATH, params=params, body=body)
        page_result = AccessionPage.from_response(payload)

        logger.info(
            "Genesys list_accessions page=%s size=%s -> %s records (total %s)",
            page,
            params["l"],
            len(page_result.content),
            page_result.total_elements,
        )
        return page_result

    def iter_accessions(
        self,
        accession_filter: AccessionFilter | None = None,
        *,
        size: int | None = None,
        max_records: int | None = None,
    ) -> Iterator[AccessionRecord]:
        """Iterate over every accession matching a filter, page by page.

        Args:
            accession_filter: Search criteria.
            size: Records per page (defaults to the client ``page_size``).
            max_records: Stop after this many records (defaults to the client cap).

        Yields:
            ``AccessionRecord`` objects in server order.
        """
        limit = max_records or self.max_records
        yielded = 0
        page_number = 0

        # Never ask for more records than the cap allows (a cap of 20 with a page
        # size of 500 would download 480 useless records). The size stays fixed
        # for the whole iteration so page numbers remain consistent offsets.
        page_size = min(size or self.page_size, MAX_PAGE_SIZE, limit)

        # Request pages until the server says it is the last one or the cap is hit.
        while True:
            page = self.list_accessions(accession_filter, page=page_number, size=page_size)

            for record in page.content:
                # Respect the safety cap even in the middle of a page.
                if yielded >= limit:
                    logger.warning("Genesys iteration stopped at the cap of %s records", limit)
                    return

                yield record
                yielded += 1

            # Stop when the API reports the last page or returns nothing.
            if page.last or not page.content:
                return

            # Also stop once the cap is reached exactly at a page boundary.
            if yielded >= limit:
                logger.warning("Genesys iteration stopped at the cap of %s records", limit)
                return

            page_number += 1

    def count_accessions(self, accession_filter: AccessionFilter | None = None) -> int:
        """Return how many accessions match a filter without downloading them.

        Args:
            accession_filter: Search criteria.

        Returns:
            ``totalElements`` reported by the API for a one-record page.
        """
        return self.list_accessions(accession_filter, page=0, size=1).total_elements

    def list_accessions_dataframe(
        self,
        accession_filter: AccessionFilter | None = None,
        *,
        size: int | None = None,
        max_records: int | None = None,
    ) -> pd.DataFrame:
        """Fetch matching accessions as a flat MCPD-style DataFrame.

        Args:
            accession_filter: Search criteria.
            size: Records per page.
            max_records: Safety cap on the number of rows.

        Returns:
            One row per accession with the columns of ``MCPD_FIELD_MAP``.
            Columns are present even when the result is empty.
        """
        rows = [
            record.flatten()
            for record in self.iter_accessions(accession_filter, size=size, max_records=max_records)
        ]

        # Keep a stable schema so downstream code can rely on the columns.
        from sdks.genesys.models import MCPD_FIELD_MAP  # local import avoids cycles in type checkers

        return pd.DataFrame(rows, columns=list(MCPD_FIELD_MAP))

    # --------------------------------------------------------------- crops
    def list_crops(self, *, refresh: bool = False) -> list[Crop]:
        """Return the crops known by Genesys (``GET /api/v2/crop``), cached.

        Args:
            refresh: ``True`` to ignore the cache and call the API again.

        Returns:
            List of ``Crop`` objects.
        """
        # Crops rarely change; one call per client lifetime is enough.
        if self._crops is None or refresh:
            payload = self._get_json(CROPS_PATH)
            items = payload if isinstance(payload, list) else payload.get("content", [])
            self._crops = [Crop.from_api(item) for item in items if isinstance(item, dict)]
            logger.info("Loaded %s crops from Genesys", len(self._crops))

        return self._crops

    def resolve_crop_codes(self, names: list[str]) -> tuple[list[str], list[str]]:
        """Translate crop names given by a user into Genesys crop codes.

        Each name is compared (case-insensitively) with the ``shortName``,
        ``name`` and ``otherNames`` of every crop.

        Args:
            names: Crop names or codes, e.g. ``["bean", "frijol", "maize"]``.

        Returns:
            ``(resolved_codes, unresolved_names)`` preserving order and without
            duplicates.
        """
        crops = self.list_crops()
        resolved: list[str] = []
        unresolved: list[str] = []

        # Try every name against the whole crop catalogue.
        for name in names:
            code = next((crop.short_name for crop in crops if crop.matches(name)), None)

            if code is None:
                unresolved.append(name)
            elif code not in resolved:
                resolved.append(code)

        return resolved, unresolved

    # ------------------------------------------------------------- internals
    def _post_json(self, path: str, *, params: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON body and decode the JSON answer, with retries.

        Args:
            path: Endpoint path relative to the base URL.
            params: Query string parameters.
            body: JSON body.

        Returns:
            Decoded JSON payload.
        """
        return self._request_json("POST", path, params=params, body=body)

    def _get_json(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        """GET an endpoint and decode the JSON answer, with retries.

        Args:
            path: Endpoint path relative to the base URL.
            params: Query string parameters.

        Returns:
            Decoded JSON payload (object or array).
        """
        return self._request_json("GET", path, params=params or {}, body=None)

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        body: dict[str, Any] | None,
    ) -> Any:
        """Send a request and decode the JSON answer, with retries.

        Args:
            method: HTTP method (``GET`` or ``POST``).
            path: Endpoint path relative to the base URL.
            params: Query string parameters.
            body: JSON body for POST requests; ``None`` for GET.

        Returns:
            Decoded JSON payload.

        Raises:
            GenesysAuthError: On HTTP 401/403.
            GenesysRequestError: On any other non-2xx status after retries.
            GenesysConnectionError: When the API cannot be reached after retries.
        """
        last_error: GenesysError | None = None

        # Retry loop for transient failures; permanent errors break out immediately.
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._http.request(
                    method,
                    path,
                    params=params,
                    json=body,
                    headers=self.auth.headers(self._http),
                )
            except httpx.HTTPError as exc:
                last_error = GenesysConnectionError(f"Could not reach Genesys ({path}): {exc}")
                logger.warning("Genesys request failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                self._sleep_backoff(attempt)
                continue

            # Authentication problems never get better by retrying.
            if response.status_code in (401, 403):
                raise GenesysAuthError(
                    f"Genesys rejected the credentials (HTTP {response.status_code}): {response.text[:300]}"
                )

            # Rate limits and server errors are retried with a back-off.
            if response.status_code in _RETRY_STATUSES:
                last_error = GenesysRequestError(
                    f"Genesys answered HTTP {response.status_code} for {path}",
                    status_code=response.status_code,
                    body=response.text,
                )
                logger.warning(
                    "Genesys transient error %s (attempt %s/%s)",
                    response.status_code,
                    attempt,
                    self.max_retries,
                )
                self._sleep_backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue

            # Any other 4xx is a bad request on our side: report it as is.
            if response.status_code >= 400:
                raise GenesysRequestError(
                    f"Genesys answered HTTP {response.status_code} for {path}",
                    status_code=response.status_code,
                    body=response.text,
                )

            try:
                return response.json()
            except ValueError as exc:
                raise GenesysRequestError(
                    "Genesys returned a non-JSON body", status_code=response.status_code, body=response.text
                ) from exc

        # Every attempt failed with a transient error.
        assert last_error is not None
        raise last_error

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        """Pause before the next retry.

        Args:
            attempt: Current attempt number (1-based).
            retry_after: Optional ``Retry-After`` header value in seconds.
        """
        # Do not sleep after the final attempt; the caller raises right away.
        if attempt >= self.max_retries:
            return

        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt - 1), 10)
        time.sleep(delay)
