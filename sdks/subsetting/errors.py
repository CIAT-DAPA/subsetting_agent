"""Exceptions raised by the Subsetting SDK."""


class SubsettingError(Exception):
    """Base class of every error raised by the Subsetting SDK."""


class SubsettingAuthError(SubsettingError):
    """Both credentials (API token and access token) were rejected."""


class SubsettingRequestError(SubsettingError):
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


class SubsettingConnectionError(SubsettingError):
    """The API could not be reached (network error or timeout)."""


class SubsettingNoDataError(SubsettingError):
    """The API holds no indicator data for the requested cells/indicators."""
