"""HTTP client for the Subsetting (climate indicators) API.

Endpoints used (appended to ``SUBSETTING_API_URL``, e.g.
``https://sandbox.genesys-pgr.org/api/subsetting/v1``)::

    GET  /indicators        catalogue of indicators grouped by category
    GET  /indicator-period  periods of every indicator (needed to build requests)
    POST /indicators-data   indicator values for a list of cell ids
    POST /cluster           multivariate clustering of cell ids by indicators

The request/response shapes were taken from the API source code
(``subsets_api/api.py``) and the Angular front-end that consumes it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import pandas as pd

from core.logger import get_logger
from sdks.subsetting.auth import FallbackAuth
from sdks.subsetting.errors import (
    SubsettingAuthError,
    SubsettingConnectionError,
    SubsettingError,
    SubsettingNoDataError,
    SubsettingRequestError,
)
from sdks.subsetting.models import MONTH_COLUMNS, ClusterResult, Indicator, IndicatorPeriod

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from core.config import Settings

logger = get_logger(__name__)

INDICATORS_PATH = "/indicators"
INDICATOR_PERIODS_PATH = "/indicator-period"
INDICATORS_DATA_PATH = "/indicators-data"
CLUSTER_PATH = "/cluster"

# HTTP statuses retried with a back-off.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
# Columns of the long DataFrame returned by ``get_indicators_data``.
DATA_COLUMNS: tuple[str, ...] = ("cellid", "pref_indicator", *MONTH_COLUMNS, "value", "category")


class SubsettingClient:
    """Synchronous client for the Subsetting API built on ``httpx``."""

    def __init__(
        self,
        base_url: str,
        auth: FallbackAuth,
        *,
        timeout: float = 120.0,
        default_period: str = "mean",
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create the client.

        Args:
            base_url: Full API root, e.g. ``https://sandbox.genesys-pgr.org/api/subsetting/v1``.
            auth: Credential strategy with fallback.
            timeout: Request timeout in seconds (clustering can be slow).
            default_period: Period label used when none is given (``"mean"``).
            max_retries: Attempts for transient errors.
            transport: Optional ``httpx`` transport (tests).
        """
        self.base_url = base_url.rstrip("/")
        self.auth = auth
        self.default_period = default_period
        self.max_retries = max(max_retries, 1)
        self._indicators: list[Indicator] | None = None
        self._periods: list[IndicatorPeriod] | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )

    # ------------------------------------------------------------ factory
    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> SubsettingClient:
        """Build a client from the application settings.

        Args:
            settings: Application settings (``SUBSETTING_*`` variables).
            **overrides: Constructor overrides (``transport``, ``timeout``...).

        Returns:
            A configured client.

        Raises:
            SubsettingError: If the URL is missing.
            SubsettingAuthError: If no credential is configured.
        """
        # The URL is mandatory; credentials are validated by FallbackAuth.
        if not settings.subsetting_api_url:
            raise SubsettingError("Configure SUBSETTING_API_URL in .env.")

        auth = FallbackAuth(
            api_token=settings.subsetting_api_token, access_token=settings.subsetting_access_token
        )
        kwargs: dict[str, Any] = {
            "timeout": settings.subsetting_timeout,
            "default_period": settings.subsetting_default_period,
        }
        kwargs.update(overrides)

        return cls(settings.subsetting_api_url, auth, **kwargs)

    # ---------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Release the HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> SubsettingClient:
        """Support ``with SubsettingClient(...) as client``."""
        return self

    def __exit__(self, *_: Any) -> None:
        """Close the client when leaving the ``with`` block."""
        self.close()

    # --------------------------------------------------------- catalogue
    def list_indicators(self, *, refresh: bool = False) -> list[Indicator]:
        """Return every indicator of the catalogue (cached).

        Args:
            refresh: ``True`` to call the API again.

        Returns:
            Flat list of indicators (the API groups them by category).
        """
        if self._indicators is None or refresh:
            payload = self._request_json("GET", INDICATORS_PATH)
            indicators: list[Indicator] = []

            # The API nests indicators under categories; flatten them.
            for group in payload if isinstance(payload, list) else []:
                category = group.get("category")

                for item in group.get("indicators", []) or []:
                    data = dict(item)
                    data.setdefault("category", category)
                    indicators.append(Indicator.model_validate(data))

            self._indicators = indicators
            logger.info("Loaded %s indicators from the Subsetting API", len(indicators))

        return self._indicators

    def list_indicator_periods(self, *, refresh: bool = False) -> list[IndicatorPeriod]:
        """Return every indicator period (cached)."""
        if self._periods is None or refresh:
            payload = self._request_json("GET", INDICATOR_PERIODS_PATH)
            self._periods = [
                IndicatorPeriod.model_validate({**item, "id": str(item.get("id")), "indicator": str(item.get("indicator"))})
                for item in (payload if isinstance(payload, list) else [])
            ]
            logger.info("Loaded %s indicator periods", len(self._periods))

        return self._periods

    def find_indicators(self, query: str) -> list[Indicator]:
        """Search the catalogue by free text (id, name, prefix or category).

        Args:
            query: Text to look for, e.g. ``"drought"`` or ``"t_rain"``.

        Returns:
            Matching indicators (possibly empty).
        """
        return [indicator for indicator in self.list_indicators() if indicator.matches(query)]

    def get_indicator(self, identifier: str, crop: str | None = None) -> Indicator | None:
        """Return the indicator whose id, prefix or name equals ``identifier``.

        Crop-specific indicators share their prefix and name across crops
        (``days_heat`` exists for Beans, Maize, Rice...). When several match,
        the one whose crop equals ``crop`` wins; otherwise the generic (``NC``)
        one, if any; otherwise the first match.

        Args:
            identifier: Indicator id, prefix or name (case-insensitive).
            crop: Crop name used to disambiguate crop-specific indicators.

        Returns:
            The indicator, or ``None`` when nothing matches.
        """
        matches = self.get_indicator_variants(identifier)

        # No match at all.
        if not matches:
            return None

        # A single match needs no disambiguation.
        if len(matches) == 1:
            return matches[0]

        wanted = (crop or "").strip().lower()

        # Prefer the variant computed for the requested crop.
        for indicator in matches:
            if wanted and (indicator.crop or "").strip().lower() == wanted:
                return indicator

        # Then the generic variant.
        for indicator in matches:
            if (indicator.crop or "").strip().upper() == "NC":
                return indicator

        return matches[0]

    def get_indicator_variants(self, identifier: str) -> list[Indicator]:
        """Return every indicator whose id, prefix or name equals ``identifier``.

        Args:
            identifier: Indicator id, prefix or name (case-insensitive).

        Returns:
            Matching indicators (one per crop for crop-specific indicators).
        """
        needle = identifier.strip().lower()

        return [
            indicator
            for indicator in self.list_indicators()
            if needle in (indicator.id.lower(), indicator.pref.lower(), indicator.name.lower())
        ]

    def catalogue_crops(self) -> list[str]:
        """Return the crop names present in the indicator catalogue (excluding ``NC``)."""
        crops = {indicator.crop for indicator in self.list_indicators() if indicator.crop}
        return sorted(crop for crop in crops if crop.strip().upper() != "NC")

    def resolve_periods(self, indicator_ids: list[str], period: str | None = None) -> dict[str, list[str]]:
        """Map indicator ids to the period ids expected by the data endpoints.

        Args:
            indicator_ids: Indicator identifiers (``Indicator.id``).
            period: Period label; the client default (``"mean"``) when omitted.

        Returns:
            ``{indicator_id: [period_id, ...]}``.

        Raises:
            SubsettingError: If an indicator has no period with that label.
        """
        label = (period or self.default_period).strip().lower()
        periods = self.list_indicator_periods()
        resolved: dict[str, list[str]] = {}

        # Collect the period ids of each indicator for the requested label.
        for indicator_id in indicator_ids:
            ids = [p.id for p in periods if p.indicator == str(indicator_id) and p.period.lower() == label]

            if not ids:
                available = sorted({p.period for p in periods if p.indicator == str(indicator_id)})
                raise SubsettingError(
                    f"Indicator '{indicator_id}' has no period '{label}'. Available periods: {available}"
                )

            resolved[str(indicator_id)] = ids

        return resolved

    # -------------------------------------------------------------- data
    def get_indicators_data(
        self,
        cellids: list[int],
        indicator_ids: list[str],
        period: str | None = None,
    ) -> pd.DataFrame:
        """Fetch indicator values for a set of cells.

        Args:
            cellids: Cell ids of the accessions.
            indicator_ids: Indicator identifiers.
            period: Period label (default ``"mean"``).

        Returns:
            Long DataFrame with ``DATA_COLUMNS``: one row per cell and indicator.
        """
        cells = sorted({int(cell) for cell in cellids})
        period_ids = [pid for ids in self.resolve_periods(indicator_ids, period).values() for pid in ids]

        # Nothing to ask for: return the empty schema.
        if not cells or not period_ids:
            return pd.DataFrame(columns=list(DATA_COLUMNS))

        payload = self._request_json(
            "POST", INDICATORS_DATA_PATH, body={"cellid": cells, "indicators": period_ids}
        )
        rows: list[dict[str, Any]] = []

        # Flatten {cellid, data: [...]} into one row per indicator value.
        for entry in payload.get("response", []) if isinstance(payload, dict) else []:
            cellid = entry.get("cellid")

            for item in entry.get("data", []) or []:
                row = {column: item.get(column) for column in DATA_COLUMNS if column != "cellid"}
                row["cellid"] = int(cellid) if cellid is not None else None
                rows.append(row)

        frame = pd.DataFrame(rows, columns=list(DATA_COLUMNS))
        logger.info(
            "Subsetting indicators-data: %s cells x %s periods -> %s rows", len(cells), len(period_ids), len(frame)
        )
        return frame

    def generate_clusters(
        self,
        cellids: list[int],
        indicator_ids: list[str],
        *,
        crop: str = "generic",
        min_clusters: int = 2,
        max_clusters: int = 10,
        months: tuple[int, int] = (1, 12),
        period: str | None = None,
        algorithm: str = "agglomerative",
    ) -> ClusterResult:
        """Run the multivariate clustering of the API.

        Args:
            cellids: Cell ids to cluster.
            indicator_ids: Indicators used as variables.
            crop: Crop name sent in ``cellid_list`` (matters for crop-specific indicators).
            min_clusters: Lower bound of the cluster range explored.
            max_clusters: Upper bound of the cluster range explored.
            months: Inclusive month range ``(first, last)`` used for monthly indicators.
            period: Period label (default ``"mean"``).
            algorithm: ``agglomerative`` (default), ``dbscan`` or ``hdbscan``.

        Returns:
            Parsed ``ClusterResult`` with ``{cellid: cluster}``.

        Raises:
            SubsettingError: On invalid arguments or an empty analysis.
        """
        cells = sorted({int(cell) for cell in cellids})

        # Clustering needs at least three cells to make sense (n_clusters < n_samples).
        if len(cells) < 3:
            raise SubsettingError("At least 3 georeferenced accessions are needed to build clusters.")

        if min_clusters < 2 or max_clusters < min_clusters:
            raise SubsettingError("Cluster range must satisfy 2 <= min_clusters <= max_clusters.")

        periods_by_indicator = self.resolve_periods(indicator_ids, period)
        catalogue = {indicator.id: indicator for indicator in self.list_indicators()}
        data: list[dict[str, Any]] = []

        # One entry per indicator with its period ids, type and crop.
        for indicator_id, period_ids in periods_by_indicator.items():
            indicator = catalogue.get(indicator_id)

            if indicator is None:
                raise SubsettingError(f"Unknown indicator '{indicator_id}'.")

            data.append(
                {
                    "name": indicator.name,
                    "indicator": period_ids,
                    "type": indicator.indicator_type,
                    "crop": indicator.crop or crop,
                }
            )

        body = {
            "cellid_list": [{"crop": crop, "cellids": cells}],
            "data": data,
            "analysis": {
                "algorithm": [algorithm],
                "hyperparameter": {"n_clusters": max_clusters, "min_cluster": min_clusters},
                "summary": False,
            },
            "months": [int(months[0]), int(months[1])],
        }

        payload = self._request_json("POST", CLUSTER_PATH, body=body)
        column = {"agglomerative": "cluster_hac", "dbscan": "cluster_dbscan", "hdbscan": "cluster_hdbscan"}.get(algorithm)
        result = ClusterResult.from_response(payload if isinstance(payload, dict) else {}, column)

        # An empty analysis means the cells had no indicator data.
        if not result.assignments:
            raise SubsettingNoDataError(
                "The Subsetting API returned no clusters: the selected cells have no data for the "
                "chosen indicators (indicator values only exist for cells that hold accessions)."
            )

        logger.info(
            "Subsetting cluster: %s cells -> %s clusters (%s)", len(cells), result.cluster_count, result.cluster_column
        )
        return result

    # --------------------------------------------------------- internals
    def _request_json(self, method: str, path: str, *, body: dict[str, Any] | None = None) -> Any:
        """Send a request handling credential fallback, retries and errors.

        Args:
            method: ``GET`` or ``POST``.
            path: Endpoint path appended to the base URL.
            body: JSON body for POST requests.

        Returns:
            Decoded JSON payload.

        Raises:
            SubsettingAuthError: When every credential is rejected.
            SubsettingRequestError: On other non-2xx statuses.
            SubsettingConnectionError: When the API cannot be reached.
        """
        last_error: SubsettingError | None = None
        attempt = 0

        # Loop until success, a permanent error, or the retry budget is spent.
        while attempt < self.max_retries:
            attempt += 1

            try:
                self._apply_auth_cookies()
                response = self._http.request(method, path, json=body, headers=self.auth.headers())
            except httpx.HTTPError as exc:
                last_error = SubsettingConnectionError(f"Could not reach the Subsetting API ({path}): {exc}")
                logger.warning("Subsetting request failed (attempt %s/%s): %s", attempt, self.max_retries, exc)
                self._sleep_backoff(attempt)
                continue

            # Rejected credential: switch to the next scheme (does not consume a retry).
            if response.status_code in (401, 403):
                if self.auth.has_fallback():
                    self.auth.switch_to_fallback()
                    attempt -= 1
                    continue

                raise SubsettingAuthError(
                    f"The Subsetting API rejected every credential scheme (HTTP {response.status_code}) "
                    f"for {path}: {response.text[:300]}. Check SUBSETTING_API_URL (must include the "
                    "version segment, e.g. .../api/subsetting/v1) and that the tokens belong to the "
                    "same environment (sandbox vs production)."
                )

            # The indicators-data endpoint crashes (500, KeyError 'cellid') when no
            # cell has data: that is a data condition, not a transient failure.
            if response.status_code == 500 and path == INDICATORS_DATA_PATH and self._looks_like_no_data(response.text):
                raise SubsettingNoDataError(
                    "The Subsetting API has no indicator data for the given cells and indicators "
                    "(indicator values only exist for cells that hold accessions)."
                )

            # Transient server-side problems are retried.
            if response.status_code in _RETRY_STATUSES:
                last_error = SubsettingRequestError(
                    f"Subsetting API answered HTTP {response.status_code} for {path}",
                    status_code=response.status_code,
                    body=response.text,
                )
                logger.warning("Subsetting transient error %s (attempt %s/%s)", response.status_code, attempt, self.max_retries)
                self._sleep_backoff(attempt, response.headers.get("Retry-After"))
                continue

            # Other client errors are reported as they are.
            if response.status_code >= 400:
                raise SubsettingRequestError(
                    f"Subsetting API answered HTTP {response.status_code} for {path}",
                    status_code=response.status_code,
                    body=response.text,
                )

            try:
                return response.json()
            except ValueError as exc:
                raise SubsettingRequestError(
                    "Subsetting API returned a non-JSON body", status_code=response.status_code, body=response.text
                ) from exc

        assert last_error is not None
        raise last_error

    def _apply_auth_cookies(self) -> None:
        """Synchronise the client cookie jar with the active auth scheme.

        httpx deprecates per-request cookies, so the ``access_token`` cookie is
        set on (or removed from) the client instance before each request.
        """
        wanted = self.auth.cookies()

        # Drop cookies from a previous scheme that are no longer wanted.
        for name in list(self._http.cookies.keys()):
            if name not in wanted:
                self._http.cookies.delete(name)

        # Set the cookies of the active scheme.
        for name, value in wanted.items():
            self._http.cookies.set(name, value)

    @staticmethod
    def _looks_like_no_data(body: str) -> bool:
        """Whether a 500 body corresponds to the empty-DataFrame crash of the API.

        Args:
            body: Raw response body (Flask HTML/traceback or plain text).

        Returns:
            ``True`` for the known signatures, and also for bodies without any
            detail (the production server hides tracebacks), since the only
            known cause of a 500 on this endpoint is the absence of data.
        """
        text = (body or "").lower()
        signatures = ("keyerror", "'cellid'", "internal server error", "groupby")

        # Empty or generic bodies cannot be distinguished from the no-data case.
        return not text.strip() or any(signature in text for signature in signatures)

    def _sleep_backoff(self, attempt: int, retry_after: str | None = None) -> None:
        """Pause before the next retry (no pause after the last attempt)."""
        if attempt >= self.max_retries:
            return

        delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** (attempt - 1), 10)
        time.sleep(delay)
