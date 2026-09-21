"""Unit tests for the pure helpers of the Gradio app and its chat handler."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as app_module
from app import (
    build_memory_from_history,
    chat,
    collect_accession_file_paths,
    collect_pdf_paths,
    extract_file_paths,
    extract_text,
    format_document_errors,
)
from tests.test_uploads import settings_with_uploads


class TestExtractText:
    """Text is read from strings and text blocks only."""

    def test_string(self) -> None:
        """Plain strings pass through."""
        assert extract_text("hello") == "hello"

    def test_blocks(self) -> None:
        """Text blocks are joined; other blocks are ignored."""
        content = [
            {"type": "text", "text": "hello"},
            {"type": "file", "file": {"path": "/tmp/a.pdf"}},
            {"type": "text", "text": "world"},
        ]

        assert extract_text(content) == "hello world"

    def test_other_shapes(self) -> None:
        """File descriptors and None yield no text."""
        assert extract_text({"path": "/tmp/a.pdf"}) == ""
        assert extract_text(None) == ""


class TestExtractFilePaths:
    """Every shape Gradio uses for files is recognized."""

    def test_shapes(self) -> None:
        """Dicts, nested file blocks, tuples, lists and objects all resolve."""
        assert extract_file_paths({"path": "/a.pdf"}) == ["/a.pdf"]
        assert extract_file_paths({"type": "file", "file": {"path": "/b.pdf"}}) == ["/b.pdf"]
        assert extract_file_paths(("/c.pdf", "alt")) == ["/c.pdf"]
        assert extract_file_paths([{"path": "/d.pdf"}, {"type": "text", "text": "x"}]) == ["/d.pdf"]
        assert extract_file_paths(SimpleNamespace(path="/e.pdf")) == ["/e.pdf"]
        assert extract_file_paths("just text") == []


class TestHistoryHelpers:
    """History is turned into memory and PDF paths."""

    def test_memory_keeps_text_turns_only(self) -> None:
        """Only user/assistant text ends up in memory, capped to the last N."""
        history = [
            {"role": "user", "content": [{"type": "text", "text": "find beans"}]},
            {"role": "user", "content": {"path": "/paper.pdf"}},
            {"role": "assistant", "content": "Found 10."},
            {"role": "system", "content": "ignored"},
            "garbage",
        ]

        memory = build_memory_from_history(history)

        assert memory == [
            {"role": "user", "content": "find beans"},
            {"role": "assistant", "content": "Found 10."},
        ]

    def test_memory_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Older messages are dropped beyond the cap."""
        monkeypatch.setattr(app_module, "MAX_HISTORY_MESSAGES", 2)
        history = [{"role": "user", "content": f"m{i}"} for i in range(5)]

        assert [m["content"] for m in build_memory_from_history(history)] == ["m3", "m4"]

    def test_collect_pdf_paths(self) -> None:
        """PDFs from history and the current message are merged, deduplicated and filtered."""
        history = [
            {"role": "user", "content": {"path": "/old.pdf"}},
            {"role": "user", "content": [{"type": "file", "file": {"path": "/image.png"}}]},
            {"role": "assistant", "content": {"path": "/assistant.pdf"}},
        ]

        paths = collect_pdf_paths(history, ["/new.pdf", {"path": "/old.pdf"}])

        assert paths == ["/old.pdf", "/new.pdf"]

    def test_collect_accession_file_paths(self) -> None:
        """Spreadsheets are separated from PDFs, whatever the turn they came in."""
        history = [{"role": "user", "content": {"path": "/old.csv"}}]

        assert collect_accession_file_paths(history, ["/new.pdf", "/list.XLSX"]) == [
            "/old.csv",
            "/list.XLSX",
        ]
        assert collect_pdf_paths(history, ["/new.pdf", "/list.XLSX"]) == ["/new.pdf"]

    def test_format_document_errors(self) -> None:
        """Errors render as a note; no errors render as nothing."""
        assert format_document_errors([]) == ""
        note = format_document_errors(["PDF is empty: a.pdf"])
        assert "could not be processed" in note
        assert "- PDF is empty: a.pdf" in note


class TestChatHandler:
    """The handler wires history, files and state into the agent."""

    async def test_chat_passes_state_and_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The agent receives memory, persisted PDFs/sheets and context; outputs are returned."""
        monkeypatch.setattr(app_module, "SETTINGS", settings_with_uploads(tmp_path / "uploads"))
        paper = tmp_path / "paper.pdf"
        new_pdf = tmp_path / "new.pdf"
        sheet = tmp_path / "list.xlsx"
        paper.write_bytes(b"paper")
        new_pdf.write_bytes(b"new")
        sheet.write_bytes(b"sheet")
        captured: dict = {}

        class FakeAgent:
            """Minimal agent double recording what it receives."""

            def __init__(self, *args, **kwargs) -> None:
                """Accept any configuration."""
                self.memory = []

            async def chat(self, text, *, document_paths, accession_file_paths, context_json):
                """Record inputs and return a scripted turn."""
                captured.update(
                    text=text,
                    memory=self.memory,
                    documents=document_paths,
                    accession_files=accession_file_paths,
                    context=context_json,
                )
                return SimpleNamespace(
                    answer="done", context_json='{"stage": "passport"}', document_errors=["bad.pdf"]
                )

        monkeypatch.setattr(app_module, "SubsettingAgent", FakeAgent)
        history = [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": {"path": str(paper)}},
        ]

        answer, state, attachments = await chat(
            {"text": "now", "files": [str(new_pdf), str(sheet)]}, history, "{}", ""
        )

        assert answer.startswith("done")
        assert "bad.pdf" in answer
        assert state == '{"stage": "passport"}'
        assert captured["text"] == "now"
        assert [Path(p).suffix for p in captured["documents"]] == [".pdf", ".pdf"]
        assert [Path(p).suffix for p in captured["accession_files"]] == [".xlsx"]
        assert all(Path(p).parent == tmp_path / "uploads" for p in captured["documents"])
        assert set(json.loads(attachments)) == {str(paper), str(new_pdf), str(sheet)}
        assert captured["context"] == "{}"
        assert [m["content"] for m in captured["memory"]] == ["earlier", "ok"]

    async def test_attachment_only_message_gets_default_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A message with a PDF and no text still reaches the agent."""
        monkeypatch.setattr(app_module, "SETTINGS", settings_with_uploads(tmp_path / "uploads"))
        paper = tmp_path / "paper.pdf"
        paper.write_bytes(b"paper")
        captured: dict = {}

        class FakeAgent:
            """Agent double."""

            def __init__(self, *args, **kwargs) -> None:
                """Accept any configuration."""
                self.memory = []

            async def chat(self, text, **kwargs):
                """Record the text."""
                captured["text"] = text
                return SimpleNamespace(answer="ok", context_json="", document_errors=[])

        monkeypatch.setattr(app_module, "SubsettingAgent", FakeAgent)

        await chat({"text": "", "files": [str(paper)]}, [], "", "")

        assert "attached a document" in captured["text"]

    async def test_agent_failure_is_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exception in the agent yields a friendly message and keeps the state."""

        class BrokenAgent:
            """Agent double that always fails."""

            def __init__(self, *args, **kwargs) -> None:
                """Accept any configuration."""
                self.memory = []

            async def chat(self, *args, **kwargs):
                """Fail."""
                raise RuntimeError("boom")

        monkeypatch.setattr(app_module, "SubsettingAgent", BrokenAgent)

        answer, state, attachments = await chat("hello", [], "previous", "{}")

        assert "Something went wrong" in answer
        assert state == "previous"
        assert attachments == "{}"
