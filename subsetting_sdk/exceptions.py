"""Exception hierarchy for the Subsetting API SDK.

Every error raised by the SDK derives from :class:`SubsettingApiError` so callers
can catch a single base class while still reacting to authentication failures
and catalog lookups that fail.
"""

from __future__ import annotations

from typing import Any


class SubsettingApiError(Exception):
    """Base error for every failure produced by the Subsetting SDK.

    Attributes:
        message: Human readable description of the failure.
        status_code: HTTP status returned by the API, if the error came from an
            HTTP response. ``None`` for client-side or network errors.
        response_body: Raw body returned by the API, if any.
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


class SubsettingAuthError(SubsettingApiError):
    """Raised on HTTP 401/403: missing, invalid or insufficient API token."""


class IndicatorNotFoundError(SubsettingApiError):
    """Raised by the catalog when an indicator name, pref or id cannot be resolved."""


class IndicatorPeriodNotFoundError(SubsettingApiError):
    """Raised by the catalog when no indicator period matches the requested filters."""
