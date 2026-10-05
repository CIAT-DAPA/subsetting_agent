"""Application settings loaded from environment variables and the ``.env`` file.

Every configurable value of the system lives here so that no module needs to
read ``os.environ`` directly. Field names map (case-insensitively) to the
variables documented in ``.env.example``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GridSettings(BaseSettings):
    """Parameters of the base raster grid used to compute accession cell ids.

    Attributes:
        subsetting_grid_ncols: Number of columns of the raster.
        subsetting_grid_nrows: Number of rows of the raster.
        subsetting_grid_xmin: Longitude of the western edge of the raster.
        subsetting_grid_ymin: Latitude of the southern edge of the raster.
        subsetting_grid_cellsize: Size of each square cell, in degrees.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # Global 0.05 degree raster (-180..180, -90..90) that indexes the indicator
    # database (raster_base_complete). Verified against the Subsetting API with
    # scripts/subsetting_cellid_probe.py; the 7198x2000 raster of the source
    # repository is NOT the one the data uses.
    subsetting_grid_ncols: int = 7200
    subsetting_grid_nrows: int = 3600
    subsetting_grid_xmin: float = -180.0
    subsetting_grid_ymin: float = -90.0
    subsetting_grid_cellsize: float = 0.05


class Settings(BaseSettings):
    """Global configuration of the SubsettingAgent application.

    Attributes:
        llm_model: LiteLLM model identifier (e.g. ``ollama_chat/llama3.1:8b``).
        llm_api_base: Base URL of the model server; ``None`` for hosted providers.
        llm_api_key: API key for hosted providers; ``None`` for local models.
        llm_max_tokens: Maximum tokens generated per model call.
        llm_temperature: Sampling temperature of the model.
        llm_num_ctx: Context window requested from Ollama models.
        agent_max_iterations: Maximum LLM/tool iterations per user message.
        agent_max_history_messages: Chat messages kept as conversational memory.
        agent_preview_rows: Rows of the Candidate list shown in every answer.
        app_host: Host where the Gradio application listens.
        app_port: Port where the Gradio application listens.
        tmp_dir: Root folder for per-session temporary files.
        genesys_api_url: Base URL of the Genesys PGR API.
        genesys_api_token: Personal API token (``Authorization: API-Token``); preferred.
        genesys_client_id: OAuth client id for Genesys (used when no token is set).
        genesys_client_secret: OAuth client secret for Genesys (used when no token is set).
        genesys_page_size: Records requested per page (max 1000).
        genesys_max_records: Safety cap on records fetched in one search.
        genesys_timeout: HTTP timeout in seconds for Genesys requests.
        subsetting_api_url: Full URL of the Subsetting API incl. version (``.../api/subsetting/v1``).
        subsetting_api_token: API token tried first (``Authorization: API-Token``).
        subsetting_access_token: JWT tried as ``Bearer`` header and then as ``access_token`` cookie.
        subsetting_timeout: HTTP timeout in seconds (clustering can be slow).
        subsetting_default_period: Indicator period label used by default (``mean``).
        openalex_api_url: Base URL of the OpenAlex API.
        openalex_mailto: Contact email for the OpenAlex polite pool (optional).
        log_level: Logging level name (``DEBUG``, ``INFO``...).
        grid: Nested raster grid settings.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    # --- LLM -----------------------------------------------------------------
    llm_model: str = "ollama_chat/llama3.1:8b"
    llm_api_base: str | None = "http://localhost:11434"
    llm_api_key: str | None = None
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_num_ctx: int = 8192

    # --- Agent ---------------------------------------------------------------
    agent_max_iterations: int = 15
    agent_max_history_messages: int = 20
    agent_preview_rows: int = 15

    # --- Gradio app ----------------------------------------------------------
    app_host: str = "localhost"
    app_port: int = 7860

    # --- Sessions ------------------------------------------------------------
    tmp_dir: Path = Path("tmp")

    # --- External APIs -------------------------------------------------------
    genesys_api_url: str = "https://api.genesys-pgr.org"
    genesys_api_token: str | None = None
    genesys_client_id: str | None = None
    genesys_client_secret: str | None = None
    genesys_page_size: int = 500
    genesys_max_records: int = 5000
    genesys_timeout: float = 60.0
    subsetting_api_url: str | None = None
    subsetting_api_token: str | None = None
    subsetting_access_token: str | None = None
    subsetting_timeout: float = 120.0
    subsetting_default_period: str = "mean"
    openalex_api_url: str = "https://api.openalex.org"
    openalex_mailto: str | None = None

    # --- Logging -------------------------------------------------------------
    log_level: str = "INFO"

    # --- Nested groups -------------------------------------------------------
    grid: GridSettings = Field(default_factory=GridSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton ``Settings`` instance.

    The instance is cached so the ``.env`` file is parsed only once per process.

    Returns:
        The application settings.
    """
    return Settings()
