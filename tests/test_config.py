"""Unit tests for :mod:`config`, the only module that reads the environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Settings


class TestSettings:
    """Settings come from the environment with code defaults as fallback."""

    def test_defaults_without_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no variables set, the documented defaults apply."""
        for name in (
            "GENESYS_API_URL",
            "GENESYS_API_TOKEN",
            "SUBSETTING_API_URL",
            "SUBSETTING_API_PREFIX",
            "SUBSETTING_API_TOKEN",
            "UPLOADS_DIR",
            "SUBSETTING_AGENT_MODEL",
        ):
            monkeypatch.delenv(name, raising=False)

        settings = Settings.from_environment()

        assert settings.genesys.base_url == "https://api.sandbox.genesys-pgr.org"
        assert settings.genesys.token == ""
        assert settings.subsetting.api_prefix == "/api/v1"
        assert settings.subsetting.auth_scheme == "API-Token"
        assert settings.grid.ncols == 7198
        assert settings.storage.uploads_dir == Path("data") / "uploads"
        assert settings.agent.model == "ollama_chat/llama3.1:8b"
        assert settings.server.port == 7860

    def test_environment_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every documented variable is honoured and converted to its type."""
        monkeypatch.setenv("GENESYS_API_URL", "https://g.example/")
        monkeypatch.setenv("GENESYS_API_TOKEN", "g-token")
        monkeypatch.setenv("GENESYS_API_TIMEOUT", "7")
        monkeypatch.setenv("GENESYS_CELLID_FIELD", "tileIndex3min")
        monkeypatch.setenv("GENESYS_MAX_ACCESSIONS", "50")
        monkeypatch.setenv("SUBSETTING_API_URL", "https://s.example")
        monkeypatch.setenv("SUBSETTING_API_TOKEN", "s-token")
        monkeypatch.setenv("SUBSETTING_API_AUTH_SCHEME", "Bearer")
        monkeypatch.setenv("SUBSETTING_API_TIMEOUT", "30")
        monkeypatch.setenv("SUBSETTING_GRID_NCOLS", "360")
        monkeypatch.setenv("SUBSETTING_GRID_CELLSIZE", "1")
        monkeypatch.setenv("UPLOADS_DIR", "/srv/uploads")
        monkeypatch.setenv("DOCUMENT_CACHE_DIR", "/srv/docs")
        monkeypatch.setenv("SUBSETTING_AGENT_MODEL", "ollama_chat/llama3.1:70b")
        monkeypatch.setenv("SUBSETTING_AGENT_MAX_ITERATIONS", "9")
        monkeypatch.setenv("SUBSETTING_AGENT_PORT", "9000")
        monkeypatch.setenv("SUBSETTING_AGENT_LOG_LEVEL", "DEBUG")

        settings = Settings.from_environment()

        assert settings.genesys.base_url == "https://g.example/"
        assert settings.genesys.token == "g-token"
        assert settings.genesys.timeout == 7.0
        assert settings.genesys.cellid_field == "tileIndex3min"
        assert settings.genesys.max_accessions == 50
        assert settings.subsetting.token == "s-token"
        assert settings.subsetting.auth_scheme == "Bearer"
        assert settings.subsetting.timeout == 30.0
        assert settings.grid.ncols == 360 and settings.grid.cellsize == 1.0
        assert settings.storage.uploads_dir == Path("/srv/uploads")
        assert settings.storage.document_cache_dir == Path("/srv/docs")
        assert settings.agent.model == "ollama_chat/llama3.1:70b"
        assert settings.agent.max_iterations == 9
        assert settings.server.port == 9000
        assert settings.server.log_level == "DEBUG"

    def test_empty_prefix_is_kept(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty route prefix is a valid setting (proxy strips it)."""
        monkeypatch.setenv("SUBSETTING_API_PREFIX", "")

        assert Settings.from_environment().subsetting.api_prefix == ""

    def test_empty_values_fall_back_elsewhere(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty strings for other variables fall back to the default."""
        monkeypatch.setenv("SUBSETTING_API_TIMEOUT", "")

        assert Settings.from_environment().subsetting.timeout == 120.0

    def test_settings_are_immutable(self) -> None:
        """Settings cannot be mutated after creation."""
        settings = Settings()

        with pytest.raises(AttributeError):
            settings.agent = None  # type: ignore[misc]
