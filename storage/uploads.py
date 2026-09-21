"""Durable storage of the files the user attaches to the chat.

Gradio keeps uploads in its own temporary cache, which is cleaned on restart and
can be emptied by the operating system. Every attachment is therefore copied
once into a project-owned directory under a stable name,
``YYYYMMDD_HHmmss_<sha256[:12]>.<ext>``, and only that stable path is kept in
the session state. Identical content is stored once (same hash, same file).
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Timestamp format of the stored file name (Windows forbids ':' in file names).
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
HASH_LENGTH = 12


class UploadError(Exception):
    """Raised when an attachment cannot be persisted."""


def content_hash(path: Path) -> str:
    """Return the short SHA-256 digest of a file's content.

    Args:
        path: File to hash.
    """
    digest = hashlib.sha256()

    # Read in blocks so large files do not have to be loaded fully in memory.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()[:HASH_LENGTH]


class UploadStore:
    """Copies attachments into a durable directory under stable names.

    Attributes:
        directory: Where persisted uploads live.
    """

    def __init__(self, directory: str | Path) -> None:
        """Create the store.

        Args:
            directory: Target directory (configured by the application).
        """
        self.directory = Path(directory)

    def find_existing(self, digest: str, suffix: str) -> Path | None:
        """Return an already persisted file with the same content, if any.

        Args:
            digest: Content hash.
            suffix: File extension including the dot.
        """
        if not self.directory.is_dir():
            return None

        # Several timestamps may exist for one hash; the newest name sorts last.
        candidates = sorted(self.directory.glob(f"*_{digest}{suffix.lower()}"))

        return candidates[-1] if candidates else None

    def persist(self, source: str | Path, *, moment: datetime | None = None) -> Path:
        """Copy an attachment into the store and return its stable path.

        Args:
            source: File as handed over by Gradio.
            moment: Timestamp used in the name; defaults to now.

        Returns:
            The persisted path. A file already in the store is returned as is;
            identical content already persisted is reused.

        Raises:
            UploadError: If the source does not exist or cannot be copied.
        """
        source_path = Path(source)

        if not source_path.is_file():
            raise UploadError(f"Attachment not found: {source_path}")

        # Files that already live in the store (a previous turn) need no work.
        if self._is_inside_store(source_path):
            return source_path

        suffix = source_path.suffix.lower()
        digest = content_hash(source_path)
        existing = self.find_existing(digest, suffix)

        # Same content uploaded twice: reuse the first copy.
        if existing is not None:
            return existing

        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = (moment or datetime.now()).strftime(TIMESTAMP_FORMAT)
        target = self.directory / f"{stamp}_{digest}{suffix}"

        try:
            shutil.copy2(source_path, target)

        except OSError as exc:
            raise UploadError(f"Cannot persist attachment {source_path.name}: {exc}") from exc

        logger.info("Persisted upload %s as %s", source_path.name, target.name)

        return target

    def persist_many(self, sources: list[str | Path]) -> tuple[list[Path], list[str]]:
        """Persist several attachments, collecting failures instead of aborting.

        Args:
            sources: Files to persist.

        Returns:
            The persisted paths (deduplicated, in order) and one message per failure.
        """
        persisted: list[Path] = []
        errors: list[str] = []

        # Each file is handled independently so one bad upload does not block the rest.
        for source in sources:
            try:
                path = self.persist(source)

            except UploadError as exc:
                logger.warning("Skipping attachment %s: %s", source, exc)
                errors.append(str(exc))
                continue

            if path not in persisted:
                persisted.append(path)

        return persisted, errors

    def _is_inside_store(self, path: Path) -> bool:
        """Whether a path already points into the store directory.

        Args:
            path: Path to check.
        """
        try:
            return path.resolve().parent == self.directory.resolve()

        except OSError:
            return False
