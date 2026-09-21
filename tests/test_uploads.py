"""Unit tests for the upload store and its wiring in the app."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest

import app as app_module
from app import load_attachment_state, persist_attachments, split_attachments
from config import Settings, StorageSettings
from storage import UploadError, UploadStore, content_hash


def settings_with_uploads(directory: Path) -> Settings:
    """Return default settings whose uploads directory is ``directory``.

    Args:
        directory: Uploads directory for the test.
    """
    return Settings(
        storage=StorageSettings(uploads_dir=directory, document_cache_dir=directory / "docs")
    )


STORED_NAME = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{12}\.(pdf|xlsx|csv)$")


class TestUploadStore:
    """Attachments are copied once under a stable, hash-based name."""

    def test_persist_names_file_by_timestamp_and_hash(self, tmp_path: Path) -> None:
        """The stored name follows YYYYMMDD_HHmmss_<hash>.<ext> and keeps the content."""
        source = tmp_path / "paper.PDF"
        source.write_bytes(b"%PDF-1.4 hello")
        store = UploadStore(tmp_path / "uploads")

        stored = store.persist(source, moment=datetime(2026, 9, 21, 15, 0, 5))

        assert stored.parent == tmp_path / "uploads"
        assert stored.name == f"20260921_150005_{content_hash(source)}.pdf"
        assert STORED_NAME.match(stored.name)
        assert stored.read_bytes() == source.read_bytes()

    def test_same_content_is_stored_once(self, tmp_path: Path) -> None:
        """Two uploads with identical bytes resolve to the same stored file."""
        first = tmp_path / "a.xlsx"
        second = tmp_path / "b.xlsx"
        first.write_bytes(b"same")
        second.write_bytes(b"same")
        store = UploadStore(tmp_path / "uploads")

        stored_first = store.persist(first)
        stored_second = store.persist(second)

        assert stored_first == stored_second
        assert len(list((tmp_path / "uploads").iterdir())) == 1

    def test_already_stored_path_is_returned_as_is(self, tmp_path: Path) -> None:
        """A path inside the store (previous turn) is not copied again."""
        source = tmp_path / "a.csv"
        source.write_bytes(b"id,lat,lon")
        store = UploadStore(tmp_path / "uploads")
        stored = store.persist(source)

        assert store.persist(stored) == stored

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        """A vanished Gradio file is reported, not silently skipped."""
        store = UploadStore(tmp_path / "uploads")

        with pytest.raises(UploadError, match="not found"):
            store.persist(tmp_path / "gone.pdf")

    def test_persist_many_collects_errors(self, tmp_path: Path) -> None:
        """Good files are stored; bad ones are reported."""
        good = tmp_path / "ok.pdf"
        good.write_bytes(b"ok")
        store = UploadStore(tmp_path / "uploads")

        persisted, errors = store.persist_many([good, tmp_path / "missing.pdf", good])

        assert len(persisted) == 1
        assert len(errors) == 1


class TestAppAttachmentState:
    """The app persists new attachments once and keeps stable paths in state."""

    def test_load_attachment_state_tolerates_bad_input(self) -> None:
        """Empty, corrupted or non-object state yields an empty map."""
        assert load_attachment_state("") == {}
        assert load_attachment_state("not json") == {}
        assert load_attachment_state("[1, 2]") == {}
        assert load_attachment_state('{"/g/a.pdf": "/data/x.pdf"}') == {"/g/a.pdf": "/data/x.pdf"}

    def test_persist_attachments_only_new_files(self, tmp_path: Path) -> None:
        """Known Gradio paths are not re-read; new ones are stored and mapped."""
        gradio_dir = tmp_path / "gradio"
        gradio_dir.mkdir()
        old = gradio_dir / "old.pdf"
        new = gradio_dir / "new.xlsx"
        old.write_bytes(b"old")
        new.write_bytes(b"new")
        store = UploadStore(tmp_path / "uploads")
        state = json.dumps({str(old): str(tmp_path / "uploads" / "20260101_000000_abc.pdf")})
        history = [{"role": "user", "content": {"path": str(old)}}]

        # Delete the old Gradio file: it must not be needed anymore.
        old.unlink()

        paths, errors, new_state = persist_attachments(history, [str(new)], state, store)

        mapping = json.loads(new_state)
        assert errors == []
        assert paths[0].endswith("20260101_000000_abc.pdf")
        assert STORED_NAME.match(Path(paths[1]).name)
        assert set(mapping) == {str(old), str(new)}

    def test_persist_attachments_reports_vanished_files(self, tmp_path: Path) -> None:
        """A history file that Gradio no longer has produces an error, not a crash."""
        store = UploadStore(tmp_path / "uploads")
        history = [{"role": "user", "content": {"path": str(tmp_path / "gone.pdf")}}]

        paths, errors, _ = persist_attachments(history, [], "", store)

        assert paths == []
        assert len(errors) == 1

    def test_split_attachments(self) -> None:
        """PDFs and spreadsheets are separated by extension."""
        pdfs, sheets = split_attachments(["/d/a.pdf", "/d/b.XLSX", "/d/c.csv", "/d/d.png"])

        assert pdfs == ["/d/a.pdf"]
        assert sheets == ["/d/b.XLSX", "/d/c.csv"]

    async def test_chat_uses_persisted_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The agent receives the stored copies and the state carries the map."""
        gradio_dir = tmp_path / "gradio"
        gradio_dir.mkdir()
        sheet = gradio_dir / "list.xlsx"
        sheet.write_bytes(b"sheet")
        monkeypatch.setattr(app_module, "SETTINGS", settings_with_uploads(tmp_path / "uploads"))
        captured: dict = {}

        class FakeAgent:
            """Agent double recording the paths it receives."""

            def __init__(self, *args, **kwargs) -> None:
                """Accept any configuration."""
                self.memory = []

            async def chat(self, text, *, document_paths, accession_file_paths, context_json):
                """Record inputs."""
                captured.update(documents=document_paths, sheets=accession_file_paths)
                from types import SimpleNamespace

                return SimpleNamespace(answer="ok", context_json="{}", document_errors=[])

        monkeypatch.setattr(app_module, "SubsettingAgent", FakeAgent)

        answer, context_state, attachments_state = await app_module.chat(
            {"text": "load", "files": [str(sheet)]}, [], "", ""
        )

        assert answer == "ok"
        assert captured["documents"] == []
        assert len(captured["sheets"]) == 1
        assert Path(captured["sheets"][0]).parent == tmp_path / "uploads"
        assert json.loads(attachments_state)[str(sheet)] == captured["sheets"][0]
