"""Authentication for the Subsetting API with automatic fallback.

The API sits behind the Genesys gateway, which may accept the credential in
several ways. The SDK tries them in cascade, switching to the next one each
time the server answers 401/403, and keeps the first accepted scheme for the
rest of the session:

1. ``Authorization: API-Token <SUBSETTING_API_TOKEN>``
2. ``Authorization: Bearer <SUBSETTING_ACCESS_TOKEN>``
3. Cookie ``access_token=<SUBSETTING_ACCESS_TOKEN>`` (what the web client uses)
"""

from dataclasses import dataclass, field

from core.logger import get_logger
from sdks.subsetting.errors import SubsettingAuthError

logger = get_logger(__name__)

# Name of the cookie read by the Genesys gateway.
ACCESS_TOKEN_COOKIE = "access_token"


@dataclass(frozen=True)
class AuthScheme:
    """One way of presenting a credential to the API.

    Attributes:
        name: Label used in logs (``API-Token``, ``Bearer``, ``Cookie``).
        headers: HTTP headers to send.
        cookies: Cookies to send.
    """

    name: str
    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)


class FallbackAuth:
    """Ordered list of credential schemes; the first accepted one becomes active."""

    def __init__(self, api_token: str | None = None, access_token: str | None = None) -> None:
        """Build the cascade from the configured credentials.

        Args:
            api_token: Personal API token (tried first, as a header).
            access_token: Access token JWT (tried as Bearer header, then as cookie).

        Raises:
            SubsettingAuthError: If no credential is configured.
        """
        self._schemes: list[AuthScheme] = []
        api = (api_token or "").strip()
        access = (access_token or "").strip()

        # Build the cascade only with the credentials that have a value.
        if api:
            self._schemes.append(AuthScheme("API-Token", headers={"Authorization": f"API-Token {api}"}))
        if access:
            self._schemes.append(AuthScheme("Bearer", headers={"Authorization": f"Bearer {access}"}))
            self._schemes.append(AuthScheme("Cookie", cookies={ACCESS_TOKEN_COOKIE: access}))

        if not self._schemes:
            raise SubsettingAuthError(
                "Configure SUBSETTING_API_TOKEN and/or SUBSETTING_ACCESS_TOKEN in .env."
            )

        self._index = 0

    @property
    def scheme(self) -> str:
        """Name of the scheme currently in use."""
        return self._schemes[self._index].name

    def headers(self) -> dict[str, str]:
        """Headers of the active scheme (may be empty for the cookie scheme)."""
        return dict(self._schemes[self._index].headers)

    def cookies(self) -> dict[str, str]:
        """Cookies of the active scheme (empty for header schemes)."""
        return dict(self._schemes[self._index].cookies)

    def has_fallback(self) -> bool:
        """Whether another scheme can still be tried."""
        return self._index + 1 < len(self._schemes)

    def switch_to_fallback(self) -> None:
        """Activate the next scheme after a rejection.

        Raises:
            SubsettingAuthError: If every scheme has already been rejected.
        """
        # Nothing left to try: the caller must surface the failure.
        if not self.has_fallback():
            raise SubsettingAuthError(
                "The Subsetting API rejected every configured credential scheme "
                "(API-Token header, Bearer header and access_token cookie)."
            )

        previous = self.scheme
        self._index += 1
        logger.warning("Subsetting API rejected the %s scheme; switching to %s", previous, self.scheme)
