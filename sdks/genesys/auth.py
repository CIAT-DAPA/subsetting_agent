"""Authentication strategies for the Genesys API.

Genesys accepts two schemes (see the ``securitySchemes`` of its OpenAPI spec):

* **API token** - a long-lived token issued to a user, sent as
  ``Authorization: API-Token <token>``.
* **OAuth2 client credentials** - a registered client obtains a short-lived JWT
  from ``/oauth/token`` and sends it as ``Authorization: Bearer <jwt>``.
"""

import time
from abc import ABC, abstractmethod

import httpx

from core.logger import get_logger
from sdks.genesys.errors import GenesysAuthError, GenesysConnectionError

logger = get_logger(__name__)

# Seconds subtracted from the token lifetime so we refresh before it expires.
_REFRESH_MARGIN_SECONDS = 60


class GenesysAuth(ABC):
    """Strategy that produces the ``Authorization`` header for each request."""

    @abstractmethod
    def headers(self, http: httpx.Client) -> dict[str, str]:
        """Return the authorization headers to attach to a request.

        Args:
            http: HTTP client that may be used to obtain a token.

        Returns:
            Headers dictionary (usually just ``Authorization``).
        """


class ApiTokenAuth(GenesysAuth):
    """Authenticate with a personal API token (``Authorization: API-Token ...``)."""

    def __init__(self, token: str) -> None:
        """Store the token.

        Args:
            token: API token issued by Genesys.

        Raises:
            GenesysAuthError: If the token is empty.
        """
        # An empty token would only produce confusing 401 answers later.
        if not token or not token.strip():
            raise GenesysAuthError("The Genesys API token is empty.")

        self._token = token.strip()

    def headers(self, http: httpx.Client) -> dict[str, str]:
        """Return the ``API-Token`` authorization header."""
        return {"Authorization": f"API-Token {self._token}"}


class ClientCredentialsAuth(GenesysAuth):
    """Authenticate as a registered OAuth2 client and cache the resulting JWT."""

    def __init__(self, client_id: str, client_secret: str, token_url: str) -> None:
        """Store the client credentials.

        Args:
            client_id: OAuth2 client id registered in Genesys.
            client_secret: OAuth2 client secret.
            token_url: Absolute URL of the token endpoint (``.../oauth/token``).

        Raises:
            GenesysAuthError: If any credential is empty.
        """
        # Both values are mandatory for the client-credentials grant.
        if not client_id or not client_secret:
            raise GenesysAuthError("Genesys client id and client secret are required.")

        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    def headers(self, http: httpx.Client) -> dict[str, str]:
        """Return a ``Bearer`` header, fetching or refreshing the token when needed."""
        # Refresh when there is no token or it is about to expire.
        if self._access_token is None or time.time() >= self._expires_at:
            self._fetch_token(http)

        return {"Authorization": f"Bearer {self._access_token}"}

    def _fetch_token(self, http: httpx.Client) -> None:
        """Request a new access token from the token endpoint.

        Args:
            http: HTTP client used for the request.

        Raises:
            GenesysAuthError: If the server rejects the credentials or the
                response has no ``access_token``.
            GenesysConnectionError: If the token endpoint cannot be reached.
        """
        logger.info("Requesting Genesys access token for client %s", self._client_id)

        try:
            response = http.post(
                self._token_url,
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
            )
        except httpx.HTTPError as exc:
            raise GenesysConnectionError(f"Could not reach the Genesys token endpoint: {exc}") from exc

        # Any non-2xx answer means the credentials were not accepted.
        if response.status_code >= 400:
            raise GenesysAuthError(
                f"Genesys token request failed with HTTP {response.status_code}: {response.text[:300]}"
            )

        payload = response.json()
        token = payload.get("access_token")

        # A 2xx without a token is malformed; do not continue silently.
        if not token:
            raise GenesysAuthError("Genesys token response did not include 'access_token'.")

        expires_in = float(payload.get("expires_in", 3600))
        self._access_token = token
        self._expires_at = time.time() + max(expires_in - _REFRESH_MARGIN_SECONDS, 0)
