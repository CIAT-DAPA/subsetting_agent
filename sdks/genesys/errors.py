"""Exceptions raised by the Genesys SDK."""


class GenesysError(Exception):
    """Base class of every error raised by the Genesys SDK."""


class GenesysAuthError(GenesysError):
    """Authentication or authorization failure (HTTP 401/403 or token retrieval)."""


class GenesysRequestError(GenesysError):
    """The API answered with an unexpected HTTP status.

    Attributes:
        status_code: HTTP status returned by the API.
        body: Raw response body (truncated) to help debugging.
    """

    def __init__(self, message: str, status_code: int, body: str = "") -> None:
        """Create the error.

        Args:
            message: Human readable description.
            status_code: HTTP status returned by the API.
            body: Raw response body.
        """
        super().__init__(message)
        self.status_code = status_code
        self.body = body[:2000]


class GenesysConnectionError(GenesysError):
    """The API could not be reached (network error or timeout)."""
