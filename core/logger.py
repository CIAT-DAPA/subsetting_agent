"""Logging configuration shared by every module of the application."""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent format and level.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ``WARNING``...). Unknown
            names fall back to ``INFO``.
    """
    numeric_level = logging.getLevelName(level.upper())

    # ``getLevelName`` returns a string when the name is unknown; fall back to INFO.
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO

    logging.basicConfig(level=numeric_level, format=_LOG_FORMAT)

    # Third-party libraries are very chatty at DEBUG; keep them at WARNING.
    for noisy_logger in ("httpx", "httpcore", "LiteLLM", "litellm", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` bound to the given name.
    """
    return logging.getLogger(name)
