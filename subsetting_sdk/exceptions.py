"""Exception hierarchy for the Subsetting API SDK.

Every error raised by the SDK derives from :class:`SubsettingApiError` so callers
can catch a single base class, while still being able to react to the specific
business errors the API reports (no matching data, core collection too small,
unknown indicator name).
"""

from __future__ import annotations

from typing import Any


class SubsettingApiError(Exception):
    """Base error for every failure produced by the Subsetting SDK.

    Attributes:
        message: Human readable description of the failure.
        status_code: HTTP status returned by the API, if the error came from an
            HTTP response. ``None`` for client-side or network errors.
        response_body: Raw body returned by the API, if any. Kept to help with
            debugging since the Flask API returns plain-text error messages.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        """Store the error context and build the base exception message.

        Args:
            message: Human readable description of the failure.
            status_code: HTTP status code associated with the failure.
            response_body: Raw response body associated with the failure.
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_body = response_body

    def __str__(self) -> str:
        """Return the message, prefixed with the HTTP status when available."""
        # Only include the status code when the error originated from an HTTP
        # response; network or validation errors have no status to show.
        if self.status_code is not None:
            return f"[HTTP {self.status_code}] {self.message}"

        return self.message


class SubsettingConnectionError(SubsettingApiError):
    """Raised when the API cannot be reached (DNS, timeout, connection reset)."""


class NoMatchingDataError(SubsettingApiError):
    """Raised when ``/subset`` finds no cell that satisfies the indicator filters.

    The API reports this case as an HTTP 400 with the text
    ``"Bad request! No data matching the selected filters!"``.
    """


class CoreCollectionError(SubsettingApiError):
    """Raised when ``/core-collection`` cannot be computed.

    The API reports this case as an HTTP 422, typically because the requested
    ``amount`` is larger than the number of available cells.
    """


class IndicatorNotFoundError(SubsettingApiError):
    """Raised by the catalog when an indicator name, pref or id cannot be resolved."""


class IndicatorPeriodNotFoundError(SubsettingApiError):
    """Raised by the catalog when no indicator period matches the requested filters."""
