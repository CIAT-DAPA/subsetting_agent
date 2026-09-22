"""Unit tests for the evidence recorded by the tools and the results table."""

# ruff: noqa: F811 - the registry/services fixtures are imported from test_tools

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

import app as app_module
from reporting import FIXED_COLUMNS, build_selection_table, to_markdown, write_csv
from tests.test_tools import (  # noqa: F401 - fixtures are picked up by pytest
    GENESYS,
    SUBSETTING,
    mock_catalog,
    registry,
    seeded_context,
    services,
)
from tests.test_uploads import settings_with_uploads
from tools import AccessionContext, Stage, ToolServices
from tools.accession_context import AccessionRecord
from tools.registry import ToolRegistry


def evidence_context() -> AccessionContext:
    """Return a two-accession selection with evidence from every stage."""
    context = AccessionContext()
    context.set_file_selection(
        [
            AccessionRecord(
                uuid="A1",
                accession_number="A1",
                crop="bean",
                country_code="COL",
                latitude=4.5,
                longitude=-74.1,
                cellid=101,
            ),
            AccessionRecord(uuid="A2", accession_number="A2", crop="bean", cellid=102),
        ],
        file_name="list.csv",
        description="Loaded 2 accessions",
    )
    context.add_evidence(["A1"], **{"trait: drought": 5})
    context.add_evidence(None, **{"climate cluster": "0", "prec (mean, months 1-12)": 120.5})
    context.keep(["A1"], stage=Stage.CLIMATE, description="Kept cluster 0 | driest")

    return context


class TestContextEvidence:
    """Evidence is stored per accession and survives serialization."""

    def test_add_evidence_and_labels(self) -> None:
        """Evidence is attached to the requested accessions only, in first-seen order."""
        context = evidence_context()

        assert context.evidence_labels() == [
            "trait: drought",
            "climate cluster",
            "prec (mean, months 1-12)",
        ]
        assert context.accessions["A1"].evidence["trait: drought"] == 5

    def test_unknown_uuid_is_ignored(self) -> None:
        """Annotating a UUID that is not selected does nothing and reports zero."""
        context = evidence_context()

        assert context.add_evidence(["nope"], x=1) == 0

    def test_evidence_round_trips_through_json(self) -> None:
        """The state JSON keeps the evidence for the next turn."""
        context = evidence_context()

        restored = AccessionContext.from_json(context.to_json())

        assert restored.accessions["A1"].evidence == context.accessions["A1"].evidence

    def test_old_state_without_evidence_still_loads(self) -> None:
        """A state saved before this feature (no ``evidence`` key) is accepted."""
        payload = json.loads(evidence_context().to_json())

        for item in payload["accessions"]:
            item.pop("evidence")

        restored = AccessionContext.from_json(json.dumps(payload))

        assert restored.accessions["A1"].evidence == {}


class TestSelectionTable:
    """The table lists passport data, evidence columns and the criteria."""

    def test_columns_and_values(self) -> None:
        """Fixed columns come first, then evidence labels, then the criteria."""
        frame = build_selection_table(evidence_context())

        assert list(frame.columns) == [
            *FIXED_COLUMNS,
            "trait: drought",
            "climate cluster",
            "prec (mean, months 1-12)",
            "criteria",
        ]
        assert len(frame) == 1
        assert frame.iloc[0]["accession_number"] == "A1"
        assert frame.iloc[0]["trait: drought"] == 5
        assert frame.iloc[0]["climate cluster"] == "0"
        assert "passport: Loaded 2 accessions" in frame.iloc[0]["criteria"]
        assert "climate: Kept cluster 0" in frame.iloc[0]["criteria"]

    def test_empty_context_gives_empty_table(self) -> None:
        """No selection produces a table with the fixed columns and no rows."""
        frame = build_selection_table(AccessionContext())

        assert frame.empty
        assert list(frame.columns) == [*FIXED_COLUMNS, "criteria"]

    def test_markdown_preview_is_truncated(self) -> None:
        """The preview shows ``max_rows`` rows, escapes pipes and reports the rest."""
        context = AccessionContext()
        context.set_file_selection(
            [AccessionRecord(uuid=f"U{i}", accession_number=f"U{i}", crop="a|b") for i in range(5)],
            file_name="f.csv",
            description="load",
        )
        frame = build_selection_table(context)

        text = to_markdown(frame, max_rows=2)

        assert text.count("\n| U") == 2
        assert "a\\|b" in text
        assert "3 more rows" in text
        assert "Criteria applied: passport: load" in text
        assert to_markdown(build_selection_table(AccessionContext())) == ""

    def test_write_csv_uses_project_naming(self, tmp_path: Path) -> None:
        """The CSV lands in the exports directory as YYYYMMDD_HHmmss_<hash>_selection.csv."""
        frame = build_selection_table(evidence_context())

        path = write_csv(frame, tmp_path / "exports", moment=datetime(2026, 9, 22, 10, 5, 0))

        assert path.parent == tmp_path / "exports"
        assert path.name.startswith("20260922_100500_")
        assert path.name.endswith("_selection.csv")
        assert "accession_number" in path.read_text(encoding="utf-8").splitlines()[0]
        assert path.read_text(encoding="utf-8").splitlines()[1].startswith("A1,,bean,")


class TestToolsRecordEvidence:
    """Each stage tool writes the values that justify the selection."""

    async def test_trait_filter_records_value(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """The observed value that passed the condition is kept as evidence."""
        services.context = seeded_context({"u-1": 10, "u-2": 20})
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/u-1/observations",
            json={"firstPartyData": [{"drought score": 5}], "thirdPartyData": []},
        )
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/u-2/observations",
            json={"firstPartyData": [{"drought score": 1}], "thirdPartyData": []},
        )

        await registry.execute(
            services, "filter_selection_by_trait", {"descriptor": "drought", "min_value": 3}
        )

        assert services.context.accessions["u-1"].evidence == {"trait: drought": 5}

    async def test_cluster_records_label_and_indicator_means(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Each accession gets its cluster and the mean of every indicator of its cell."""
        services.context = seeded_context({"u-1": 101, "u-2": 103})
        mock_catalog(httpx_mock)
        httpx_mock.add_response(
            url=f"{SUBSETTING}/api/v1/cluster",
            json={
                "data": [
                    {
                        "cellid": 101,
                        "prec_month1": 1.0,
                        "prec_month2": 3.0,
                        "cluster_hac": 0,
                        "crop_name": "bean",
                    },
                    {"cellid": 103, "prec_month1": 9.0, "cluster_hac": 1, "crop_name": "bean"},
                ],
                "summary": [],
            },
        )

        await registry.execute(
            services,
            "cluster_selection_by_climate",
            {"indicators": ["prec"], "month_start": 1, "month_end": 2},
        )
        await registry.execute(services, "pick_cluster", {"cluster": "1"})

        u2 = services.context.accessions["u-2"].evidence
        assert u2["climate cluster"] == "1"
        assert u2["prec (mean, months 1-2)"] == 9.0
        assert u2["cluster kept"] == "1"
        # The dropped accession is gone with its evidence.
        assert "u-1" not in services.context.accessions

    async def test_documents_record_reason(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """The citing reason is attached to the accessions kept from a document."""
        services.context = seeded_context({"u-001": 10, "u-002": 20})

        await registry.execute(
            services,
            "keep_accessions_from_documents",
            {"accession_numbers": ["g001"], "reason": "Table 2 of paper.pdf"},
        )

        assert services.context.accessions["u-001"].evidence == {
            "document evidence": "Table 2 of paper.pdf"
        }


class TestAppResults:
    """The app returns the table, the CSV and the preview after each turn."""

    def test_build_results_from_state(self, tmp_path: Path) -> None:
        """A non-empty selection yields a table, a CSV file and a Markdown preview."""
        table, csv_path, preview = app_module.build_results(
            evidence_context().to_json(), tmp_path / "exports"
        )

        assert table is not None and len(table) == 1
        assert csv_path is not None and Path(csv_path).is_file()
        assert preview.startswith("**Selected accessions (1)**")
        assert "| A1 |" in preview

    def test_build_results_tolerates_empty_or_bad_state(self, tmp_path: Path) -> None:
        """Empty or corrupted state clears the components instead of failing."""
        assert app_module.build_results("", tmp_path) == (None, None, "")
        assert app_module.build_results("not json", tmp_path) == (None, None, "")

    def test_build_reply_orders_answer_preview_and_file(self, tmp_path: Path) -> None:
        """The reply is answer, then preview, then the CSV component; extras only if present."""
        csv_file = tmp_path / "sel.csv"
        csv_file.write_text("a,b\n1,2\n")

        full = app_module.build_reply("Done.", "**Selected accessions (1)** ...", str(csv_file))
        bare = app_module.build_reply("Nothing.", "", None)

        assert full[:2] == ["Done.", "**Selected accessions (1)** ..."]
        assert full[2] == {
            "role": "assistant",
            "content": {"path": str(csv_file), "alt_text": "Selected accessions (CSV)"},
        }
        assert bare == ["Nothing."]

    def test_file_message_survives_the_chat_history_round_trip(self, tmp_path: Path) -> None:
        """The reply can be stored by the Chatbot and read back on the next turn.

        Regression test: a ``gr.File`` component in the reply made Gradio 6.27 fail
        with ``FileData() argument after ** must be a mapping`` on the next message.
        """
        import gradio as gr
        from gradio.components.chatbot import ChatbotDataMessages

        csv_file = tmp_path / "sel.csv"
        csv_file.write_text("a,b\n1,2\n")
        reply = app_module.build_reply("Done.", "**Selected accessions (1)** ...", str(csv_file))
        history = [m if isinstance(m, dict) else {"role": "assistant", "content": m} for m in reply]
        chatbot = gr.Chatbot()

        stored = chatbot.postprocess(history).model_dump()
        restored = chatbot.preprocess(ChatbotDataMessages(root=stored))

        assert restored[2]["content"][0]["type"] == "file"
        assert restored[2]["content"][0]["file"]["path"] == str(csv_file)
        # The rebuilt memory ignores the preview and the file, keeping the answer only.
        assert app_module.build_memory_from_history(restored) == [
            {"role": "assistant", "content": "Done."}
        ]

    def test_preview_messages_are_kept_out_of_memory(self) -> None:
        """Preview and file messages from earlier turns are not replayed to the LLM."""
        history = [
            {"role": "user", "content": "find beans"},
            {"role": "assistant", "content": "Found 3."},
            {"role": "assistant", "content": "**Selected accessions (3)** - preview\n| a |"},
            {"role": "assistant", "content": {"path": "/x/sel.csv"}},
        ]

        memory = app_module.build_memory_from_history(history)

        assert memory == [
            {"role": "user", "content": "find beans"},
            {"role": "assistant", "content": "Found 3."},
        ]

    async def test_chat_appends_preview_and_returns_csv(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The reply carries the answer, the preview and a CSV file message."""
        from types import SimpleNamespace

        settings = settings_with_uploads(tmp_path / "uploads")
        monkeypatch.setattr(app_module, "SETTINGS", settings)
        state_json = evidence_context().to_json()

        class FakeAgent:
            """Agent double returning a fixed selection."""

            def __init__(self, *args, **kwargs) -> None:
                """Accept any configuration."""
                self.memory = []

            async def chat(self, *args, **kwargs):
                """Return the evidence selection."""
                return SimpleNamespace(answer="Done.", context_json=state_json, document_errors=[])

        monkeypatch.setattr(app_module, "SubsettingAgent", FakeAgent)

        reply, state, _ = await app_module.chat("go", [], "", "")

        assert reply[0] == "Done."
        assert reply[1].startswith("**Selected accessions (1)**")
        assert Path(reply[2]["content"]["path"]).is_file()
        assert state == state_json
