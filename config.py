"""Application settings.

This module and ``app.py`` are the only places that read environment
variables. Every other module receives its configuration as explicit
parameters, which keeps the SDKs, stores and tools pure and easy to test.

``Settings.from_environment()`` reads the variables documented in
``.env.example``; ``Settings()`` gives the code defaults (used by tests).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from subsetting_sdk.grid import GridSpec


@dataclass(frozen=True)
class GenesysSettings:
    """Configuration of the Genesys REST API client.

    Attributes:
        base_url: Root URL of the API.
        token: API token sent as ``Authorization: API-Token <token>``; empty for
            anonymous access.
        timeout: Per-request timeout in seconds.
        cellid_field: Dotted path of the accession field used as Subsetting cellid.
        max_accessions: Cap on accessions fetched for one passport filter.
    """

    base_url: str = "https://api.sandbox.genesys-pgr.org"
    token: str = ""
    timeout: float = 60.0
    cellid_field: str = "geo.tileIndex"
    max_accessions: int = 2000


@dataclass(frozen=True)
class SubsettingSettings:
    """Configuration of the Subsetting API client.

    Attributes:
        base_url: Root URL of the deployment (without the route prefix).
        api_prefix: Route prefix of the Flask API; empty if the proxy strips it.
        token: API token; empty for anonymous access.
        auth_scheme: Scheme word placed before the token in ``Authorization``.
        timeout: Per-request timeout in seconds (clustering can be slow).
    """

    base_url: str = "https://sandbox.genesys-pgr.org/api/subsetting"
    api_prefix: str = "/api/v1"
    token: str = ""
    auth_scheme: str = "API-Token"
    timeout: float = 120.0


@dataclass(frozen=True)
class StorageSettings:
    """Directories owned by the application.

    Attributes:
        uploads_dir: Durable copies of every attachment.
        document_cache_dir: Markdown conversions of PDFs.
    """

    uploads_dir: Path = Path("data") / "uploads"
    document_cache_dir: Path = Path("data") / "documents"


@dataclass(frozen=True)
class AgentSettings:
    """Configuration of the LLM agent.

    Attributes:
        model: litellm model name.
        api_base: LLM endpoint.
        max_iterations: Maximum LLM calls per turn.
        max_tokens: Maximum tokens per LLM answer.
        temperature: Sampling temperature.
        num_ctx: Context window requested from Ollama.
        max_history_messages: Recent chat messages replayed to the model.
    """

    model: str = "ollama_chat/llama3.1:8b"
    api_base: str = "http://localhost:11434"
    max_iterations: int = 15
    max_tokens: int = 1024
    temperature: float = 0.1
    num_ctx: int = 8192
    max_history_messages: int = 20


@dataclass(frozen=True)
class ServerSettings:
    """Configuration of the Gradio server.

    Attributes:
        host: Bind address.
        port: TCP port.
        log_level: Logging level name.
    """

    host: str = "localhost"
    port: int = 7860
    log_level: str = "INFO"


@dataclass(frozen=True)
class Settings:
    """All application settings, grouped by concern.

    Attributes:
        genesys: Genesys API settings.
        subsetting: Subsetting API settings.
        grid: Grid used to compute cellids for spreadsheet coordinates.
        storage: Directories.
        agent: LLM agent settings.
        server: Gradio server settings.
    """

    genesys: GenesysSettings = field(default_factory=GenesysSettings)
    subsetting: SubsettingSettings = field(default_factory=SubsettingSettings)
    grid: GridSpec = field(default_factory=GridSpec)
    storage: StorageSettings = field(default_factory=StorageSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    server: ServerSettings = field(default_factory=ServerSettings)

    @classmethod
    def from_environment(cls) -> Settings:
        """Build the settings from environment variables, falling back to defaults.

        This is the single place in the project where ``os.getenv`` is called.
        """
        defaults = cls()
        genesys = defaults.genesys
        subsetting = defaults.subsetting
        grid = defaults.grid
        storage = defaults.storage
        agent = defaults.agent
        server = defaults.server

        return cls(
            genesys=GenesysSettings(
                base_url=_env("GENESYS_API_URL", genesys.base_url),
                token=_env("GENESYS_API_TOKEN", genesys.token),
                timeout=float(_env("GENESYS_API_TIMEOUT", genesys.timeout)),
                cellid_field=_env("GENESYS_CELLID_FIELD", genesys.cellid_field),
                max_accessions=int(_env("GENESYS_MAX_ACCESSIONS", genesys.max_accessions)),
            ),
            subsetting=SubsettingSettings(
                base_url=_env("SUBSETTING_API_URL", subsetting.base_url),
                # The prefix may legitimately be empty, so ``None`` is the only
                # value that triggers the default.
                api_prefix=_env("SUBSETTING_API_PREFIX", subsetting.api_prefix, allow_empty=True),
                token=_env("SUBSETTING_API_TOKEN", subsetting.token),
                auth_scheme=_env("SUBSETTING_API_AUTH_SCHEME", subsetting.auth_scheme),
                timeout=float(_env("SUBSETTING_API_TIMEOUT", subsetting.timeout)),
            ),
            grid=GridSpec(
                ncols=int(_env("SUBSETTING_GRID_NCOLS", grid.ncols)),
                nrows=int(_env("SUBSETTING_GRID_NROWS", grid.nrows)),
                xmin=float(_env("SUBSETTING_GRID_XMIN", grid.xmin)),
                ymin=float(_env("SUBSETTING_GRID_YMIN", grid.ymin)),
                cellsize=float(_env("SUBSETTING_GRID_CELLSIZE", grid.cellsize)),
            ),
            storage=StorageSettings(
                uploads_dir=Path(_env("UPLOADS_DIR", storage.uploads_dir)),
                document_cache_dir=Path(_env("DOCUMENT_CACHE_DIR", storage.document_cache_dir)),
            ),
            agent=AgentSettings(
                model=_env("SUBSETTING_AGENT_MODEL", agent.model),
                api_base=_env("SUBSETTING_AGENT_API_BASE", agent.api_base),
                max_iterations=int(_env("SUBSETTING_AGENT_MAX_ITERATIONS", agent.max_iterations)),
                max_tokens=int(_env("SUBSETTING_AGENT_MAX_TOKENS", agent.max_tokens)),
                temperature=float(_env("SUBSETTING_AGENT_TEMPERATURE", agent.temperature)),
                num_ctx=int(_env("SUBSETTING_AGENT_NUM_CTX", agent.num_ctx)),
                max_history_messages=int(
                    _env("SUBSETTING_AGENT_MAX_HISTORY", agent.max_history_messages)
                ),
            ),
            server=ServerSettings(
                host=_env("SUBSETTING_AGENT_HOST", server.host),
                port=int(_env("SUBSETTING_AGENT_PORT", server.port)),
                log_level=_env("SUBSETTING_AGENT_LOG_LEVEL", server.log_level),
            ),
        )


def _env(name: str, default: object, *, allow_empty: bool = False) -> str:
    """Read one environment variable as a string, with a default.

    Args:
        name: Variable name.
        default: Value used when the variable is unset (or empty, unless
            ``allow_empty``).
        allow_empty: Whether an empty value is meaningful and must be kept.
    """
    value = os.getenv(name)

    # Unset variables always fall back; empty ones fall back unless the caller
    # says an empty string is a valid setting (e.g. an empty route prefix).
    if value is None or (value == "" and not allow_empty):
        return str(default)

    return value
