"""Exception hierarchy for the Genesys API SDK."""

from __future__ import annotations

from typing import Any


class GenesysApiError(Exception):
    """Base error for every failure produced by the Genesys SDK.

    Attributes:
        message: Human readable description of the failure.
        status_code: HTTP status returned by the API, when the error came from a
            response; ``None`` for network or client-side errors.
        response_body: Raw response body, kept for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: Any = None,
    ) -> None:
        """Store the error context.

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
        # Network errors carry no status; only HTTP failures show one.
        if self.status_code is not None:
            return f"[HTTP {self.status_code}] {self.message}"

        return self.message


class GenesysConnectionError(GenesysApiError):
    """Raised when the API cannot be reached (DNS, timeout, connection reset)."""


class GenesysAuthError(GenesysApiError):
    """Raised on HTTP 401/403: missing, invalid or insufficient API token."""


class GenesysNotFoundError(GenesysApiError):
    """Raised on HTTP 404, e.g. an unknown accession UUID or DOI."""


class GenesysBadRequestError(GenesysApiError):
    """Raised on HTTP 400, typically a malformed filter."""
