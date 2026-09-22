"""Unit tests for the file-based accession source: grid, reader, tools and mode switching."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pytest_httpx import HTTPXMock

import subsetting_agent as agent_module
from accession_files import AccessionFileError, read_accession_file
from document_processing import DocumentStore
from subsetting_agent import SubsettingAgent
from subsetting_sdk import SubsettingClient
from subsetting_sdk.grid import DEFAULT_GRID, GridSpec, cellid_from_coordinates
from tests.test_agent import FakeLLM, llm_response, tool_call
from tests.test_tools import SUBSETTING, mock_catalog
from tools import AccessionContext, Stage, ToolServices, build_registry
from tools.accession_context import AccessionRecord
from tools.registry import FILE_ONLY_TOOLS, GENESYS_ONLY_TOOLS


def write_accessions(path: Path, rows: list[dict], **kwargs) -> Path:
    """Write rows to an Excel or CSV file depending on the suffix.

    Args:
        path: Target file.
        rows: Records to write.
        **kwargs: Forwarded to the pandas writer.
    """
    frame = pd.DataFrame(rows)

    # Excel and delimited files use different writers.
    if path.suffix.lower() in (".xlsx", ".xls"):
        frame.to_excel(path, index=False, **kwargs)

    else:
        frame.to_csv(path, index=False, **kwargs)

    return path


class TestGrid:
    """The grid reproduces R ``raster::cellFromXY`` on the base raster."""

    def test_extent(self) -> None:
        """Default extent matches raster_base.asc."""
        assert DEFAULT_GRID.xmax == pytest.approx(179.9)
        assert DEFAULT_GRID.ymax == pytest.approx(50.0)

    def test_corner_cells(self) -> None:
        """North-west cell is 1, the next row starts at ncols + 1, south-east is the last."""
        grid = DEFAULT_GRID

        assert grid.cellid(49.99, -179.99) == 1
        assert grid.cellid(49.94, -179.99) == grid.ncols + 1
        assert grid.cellid(49.99, -179.94) == 2
        assert grid.cellid(-49.99, 179.89) == grid.ncols * grid.nrows

    def test_center_round_trip(self) -> None:
        """A cell centre maps back to the same cell id."""
        grid = DEFAULT_GRID
        cell = grid.cellid(4.5, -74.1)
        latitude, longitude = grid.cell_center(cell)

        assert grid.cellid(latitude, longitude) == cell

    def test_outside_extent(self) -> None:
        """Points beyond the raster have no cell."""
        assert DEFAULT_GRID.cellid(-60.0, 0.0) is None
        assert DEFAULT_GRID.cellid(60.0, 0.0) is None
        assert DEFAULT_GRID.cellid(0.0, 180.0) is None

    def test_edges_belong_to_first_row_and_column(self) -> None:
        """The northern and western edges are inclusive."""
        assert DEFAULT_GRID.cellid(50.0, -180.0) == 1

    def test_custom_grid(self) -> None:
        """Grid parameters can be replaced without code changes."""
        grid = GridSpec(ncols=360, nrows=180, xmin=-180, ymin=-90, cellsize=1)

        assert grid.ncols == 360 and grid.cellsize == 1.0
        assert cellid_from_coordinates(89.5, -179.5, grid) == 1
        assert cellid_from_coordinates(88.5, -179.5, grid) == 361

    def test_cell_center_out_of_range(self) -> None:
        """Invalid ids are rejected."""
        with pytest.raises(ValueError):
            DEFAULT_GRID.cell_center(0)


class TestReader:
    """The reader detects columns, validates rows and computes cell ids."""

    def test_reads_excel_with_detected_columns(self, tmp_path: Path) -> None:
        """Common headers are detected and every valid row gets a cell id."""
        path = write_accessions(
            tmp_path / "list.xlsx",
            [
                {"Accession number": "G001", "Latitude": 4.5, "Longitude": -74.1, "Crop": "bean"},
                {"Accession number": "G002", "Latitude": -12.0, "Longitude": -77.0, "Crop": "bean"},
            ],
        )

        report = read_accession_file(path, grid=DEFAULT_GRID)

        assert report.columns == {
            "id": "Accession number",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "crop": "Crop",
        }
        assert report.accepted == 2
        assert report.accessions[0].cellid == DEFAULT_GRID.cellid(4.5, -74.1)
        assert report.accessions[0].crop == "bean"

    def test_reads_csv_with_decimal_commas_and_rejections(self, tmp_path: Path) -> None:
        """Decimal commas parse; bad rows are rejected with reasons; duplicates are reported."""
        path = tmp_path / "list.csv"
        path.write_text(
            "ACCENUMB;LAT;LONG\n"
            "A1;4,5;-74,1\n"
            "A2;;-74.1\n"
            ";4.5;-74.1\n"
            "A3;95;-74.1\n"
            "A4;-60;10\n"
            "A1;4.6;-74.2\n",
            encoding="utf-8",
        )

        report = read_accession_file(path, grid=DEFAULT_GRID)

        assert report.total_rows == 6
        assert report.accepted == 1
        assert report.accessions[0].accession_id == "A1"
        assert report.accessions[0].latitude == 4.5
        assert len(report.rejected) == 4
        assert any("invalid coordinates" in r for r in report.rejected)
        assert any("empty identifier" in r for r in report.rejected)
        assert any("outside the indicator grid" in r for r in report.rejected)
        assert report.duplicated_ids == ["A1"]

    def test_explicit_columns_and_default_crop(self, tmp_path: Path) -> None:
        """Explicit column names override detection; default_crop fills missing crops."""
        path = write_accessions(
            tmp_path / "odd.xlsx",
            [{"Code": "X1", "Y": 4.5, "X": -74.1}, {"Code": "X2", "Y": 5.0, "X": -75.0}],
        )

        report = read_accession_file(
            path,
            id_column="code",
            latitude_column="Y",
            longitude_column="X",
            default_crop="Maize",
            grid=DEFAULT_GRID,
        )

        assert report.accepted == 2
        assert all(a.crop == "Maize" for a in report.accessions)

    def test_missing_columns_raise(self, tmp_path: Path) -> None:
        """A file without coordinate columns cannot feed the workflow."""
        path = write_accessions(tmp_path / "bad.xlsx", [{"Accession": "G1", "Country": "COL"}])

        with pytest.raises(AccessionFileError, match="latitude, longitude"):
            read_accession_file(path, grid=DEFAULT_GRID)

    def test_explicit_column_not_found(self, tmp_path: Path) -> None:
        """A wrong explicit column name is reported with the available ones."""
        path = write_accessions(tmp_path / "x.csv", [{"id": "1", "lat": 1, "lon": 1}])

        with pytest.raises(AccessionFileError, match="Available columns"):
            read_accession_file(path, id_column="uuid", grid=DEFAULT_GRID)

    def test_unsupported_and_missing_files(self, tmp_path: Path) -> None:
        """Unsupported types and missing files raise clear errors."""
        text = tmp_path / "list.txt"
        text.write_text("x")

        with pytest.raises(AccessionFileError, match="Unsupported"):
            read_accession_file(text, grid=DEFAULT_GRID)

        with pytest.raises(AccessionFileError, match="not found"):
            read_accession_file(tmp_path / "nope.xlsx", grid=DEFAULT_GRID)

    def test_summary_is_compact(self, tmp_path: Path) -> None:
        """The summary exposes counts and capped examples only."""
        path = write_accessions(tmp_path / "s.csv", [{"id": "1", "lat": 4.5, "lon": -74.1}])

        summary = read_accession_file(path, grid=DEFAULT_GRID).summary()

        assert summary["accepted"] == 1
        assert set(summary) >= {"file_name", "columns", "total_rows", "rejected_count"}


class TestContextSource:
    """The context records where accessions came from."""

    def test_file_selection_round_trip(self) -> None:
        """File selections serialize their source and file name."""
        from tools.accession_context import AccessionRecord

        context = AccessionContext()
        context.set_file_selection(
            [AccessionRecord(uuid="A1", accession_number="A1", cellid=5, crop="bean")],
            file_name="list.xlsx",
            description="loaded",
        )

        restored = AccessionContext.from_json(context.to_json())

        assert restored.source == "file"
        assert restored.source_file == "list.xlsx"
        assert restored.stage is Stage.PASSPORT
        assert restored.summary()["source"] == "file"

    def test_default_source_is_genesys(self) -> None:
        """Legacy state without a source is treated as Genesys."""
        restored = AccessionContext.from_json(json.dumps({"accessions": [], "stage": "empty"}))

        assert restored.source == "genesys"


class TestRegistryModes:
    """Each source gets its own tool set; shared tools are always present."""

    def test_genesys_mode(self) -> None:
        """Genesys mode has passport/trait tools and no file tools."""
        names = set(build_registry("genesys").names())

        assert GENESYS_ONLY_TOOLS <= names
        assert not (FILE_ONLY_TOOLS & names)
        assert {"search_documents", "cluster_selection_by_climate", "describe_selection"} <= names

    def test_file_mode(self) -> None:
        """File mode has file tools and no Genesys tools."""
        names = set(build_registry("file").names())

        assert FILE_ONLY_TOOLS <= names
        assert not (GENESYS_ONLY_TOOLS & names)
        assert {"search_documents", "cluster_selection_by_climate", "describe_selection"} <= names

    def test_unknown_mode(self) -> None:
        """An unknown source is a programming error."""
        with pytest.raises(ValueError):
            build_registry("mcp")


@pytest.fixture
def file_services(tmp_path: Path) -> ToolServices:
    """File-mode services: no Genesys client, one uploaded spreadsheet."""
    path = write_accessions(
        tmp_path / "accessions.xlsx",
        [
            {"Accession": "G001", "Latitude": 4.5, "Longitude": -74.1},
            {"Accession": "G002", "Latitude": 4.5, "Longitude": -74.1},
            {"Accession": "G003", "Latitude": -12.0, "Longitude": -77.0},
        ],
    )

    return ToolServices(
        subsetting=SubsettingClient(
            SUBSETTING, api_prefix="/api/v1", max_retries=0, backoff_seconds=0.0
        ),
        documents=DocumentStore(tmp_path / "cache"),
        accession_files=[path],
    )


class TestFileTools:
    """File tools load the spreadsheet and hand the selection to climate tools."""

    async def test_services_report_file_source(self, file_services: ToolServices) -> None:
        """A spreadsheet switches the source and disables Genesys."""
        assert file_services.source == "file"
        assert not file_services.uses_genesys

        with pytest.raises(RuntimeError, match="not available"):
            file_services.require_genesys()

    async def test_list_and_load(self, file_services: ToolServices) -> None:
        """Loading starts a passport-stage selection with computed cell ids."""
        registry = build_registry("file")

        listed = await registry.execute(file_services, "list_accession_files", {})
        loaded = await registry.execute(file_services, "load_accessions_from_file", {})

        assert listed["count"] == 1
        assert loaded["report"]["accepted"] == 3
        assert loaded["summary"]["source"] == "file"
        assert loaded["summary"]["accession_count"] == 3
        assert "note" in loaded  # no crop column
        assert file_services.context.stage is Stage.PASSPORT
        assert [c.to_api() for c in file_services.context.cellids_by_crop()] == [
            {
                "crop": "unknown",
                "cellids": [DEFAULT_GRID.cellid(4.5, -74.1), DEFAULT_GRID.cellid(-12.0, -77.0)],
            }
        ]

    async def test_load_with_default_crop(self, file_services: ToolServices) -> None:
        """A default crop labels every row and removes the crop note."""
        registry = build_registry("file")

        loaded = await registry.execute(
            file_services, "load_accessions_from_file", {"default_crop": "bean"}
        )

        assert "note" not in loaded
        assert file_services.context.crops() == ["bean"]

    async def test_load_unknown_file_name(self, file_services: ToolServices) -> None:
        """A file name that matches nothing is reported with the available files."""
        registry = build_registry("file")

        result = await registry.execute(
            file_services, "load_accessions_from_file", {"file_name": "other.xlsx"}
        )

        assert "No uploaded file matches" in result["error"]

    async def test_load_without_files(self, tmp_path: Path) -> None:
        """Without uploads the tool explains what is missing."""
        services = ToolServices(
            subsetting=SubsettingClient(SUBSETTING, max_retries=0),
            documents=DocumentStore(tmp_path / "cache"),
        )

        result = await build_registry("file").execute(services, "load_accessions_from_file", {})

        assert "No accession spreadsheet" in result["error"]

    async def test_climate_cluster_on_file_selection(
        self, file_services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Climate tools work on file-loaded cells exactly as on Genesys ones."""
        registry = build_registry("file")
        await registry.execute(file_services, "load_accessions_from_file", {})
        cell_a = DEFAULT_GRID.cellid(4.5, -74.1)
        cell_b = DEFAULT_GRID.cellid(-12.0, -77.0)
        mock_catalog(httpx_mock)
        httpx_mock.add_response(
            url=f"{SUBSETTING}/api/v1/cluster",
            json={
                "data": [
                    {
                        "cellid": cell_a,
                        "prec_month1": 200.0,
                        "cluster_hac": 0,
                        "crop_name": "unknown",
                    },
                    {
                        "cellid": cell_b,
                        "prec_month1": 5.0,
                        "cluster_hac": 1,
                        "crop_name": "unknown",
                    },
                ]
            },
        )

        clustered = await registry.execute(
            file_services, "cluster_selection_by_climate", {"indicators": ["total precipitation"]}
        )
        picked = await registry.execute(file_services, "pick_cluster", {"cluster": "1"})
        sent = json.loads(httpx_mock.get_requests()[-1].content)

        assert sent["cellid_list"] == [{"crop": "unknown", "cellids": [cell_a, cell_b]}]
        assert [c["accessions"] for c in clustered["clusters"]] == [2, 1]
        assert picked["kept"] == 1
        assert file_services.context.uuids() == ["G003"]
        assert file_services.context.stage is Stage.CLIMATE


class TestAgentModeSelection:
    """The agent picks the source from the uploaded files, deterministically."""

    async def test_file_mode_when_spreadsheet_uploaded(
        self, monkeypatch: pytest.MonkeyPatch, file_services: ToolServices
    ) -> None:
        """With a spreadsheet the prompt, tools and state note are in file mode."""
        fake = FakeLLM(
            [
                llm_response(tool_calls=[tool_call("load_accessions_from_file", {})]),
                llm_response("Loaded 3 accessions from your file."),
            ]
        )
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=file_services)

        turn = await agent.chat("load my list", accession_file_paths=file_services.accession_files)

        tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
        assert "load_accessions_from_file" in tool_names
        assert "select_accessions" not in tool_names
        assert "SPREADSHEET" in fake.calls[0]["messages"][0]["content"]
        assert "accession source: spreadsheet" in fake.calls[0]["messages"][-1]["content"]
        assert AccessionContext.from_json(turn.context_json).source == "file"
        assert AccessionContext.from_json(turn.context_json).count == 3

    async def test_genesys_mode_without_spreadsheet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without a spreadsheet the Genesys tools are offered and the note says so."""
        from genesys_sdk import GenesysClient

        services = ToolServices(
            genesys=GenesysClient("https://g.example", token="t", max_retries=0),
            subsetting=SubsettingClient(SUBSETTING, max_retries=0),
            documents=DocumentStore(tmp_path / "cache"),
        )
        fake = FakeLLM([llm_response("Which crop?")])
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=services)

        await agent.chat("find accessions")

        tool_names = {t["function"]["name"] for t in fake.calls[0]["tools"]}
        assert "select_accessions" in tool_names
        assert "load_accessions_from_file" not in tool_names
        assert "GENESYS PGR API" in fake.calls[0]["messages"][0]["content"]
        assert "accession source: Genesys API" in fake.calls[0]["messages"][-1]["content"]

    def test_build_services_skips_genesys_in_file_mode(self, tmp_path: Path) -> None:
        """No Genesys client is created when a spreadsheet is present."""
        agent = SubsettingAgent()
        sheet = tmp_path / "a.xlsx"

        with_file = agent._build_services([sheet])
        without_file = agent._build_services([])

        assert with_file.genesys is None and with_file.source == "file"
        assert without_file.genesys is not None and without_file.source == "genesys"


class TestAutomaticFileLoad:
    """In file mode the spreadsheet is loaded before the model runs."""

    async def test_loads_file_before_first_llm_call(
        self, monkeypatch: pytest.MonkeyPatch, file_services: ToolServices
    ) -> None:
        """The selection exists before the LLM answers and the note reports it."""
        fake = FakeLLM([llm_response("Your file has 3 accessions.")])
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=file_services)

        turn = await agent.chat(
            "cluster my list", accession_file_paths=file_services.accession_files
        )

        user_note = fake.calls[0]["messages"][-1]["content"]
        assert "file loaded automatically: 3 accessions" in user_note
        assert "0 rows rejected" in user_note
        assert "file crop:" in user_note  # the fixture has no crop column
        context = AccessionContext.from_json(turn.context_json)
        assert context.count == 3 and context.stage == Stage.PASSPORT

    async def test_existing_selection_is_not_reloaded(
        self, monkeypatch: pytest.MonkeyPatch, file_services: ToolServices
    ) -> None:
        """A refinement turn continues from the previous selection untouched."""
        previous = AccessionContext()
        previous.set_file_selection(
            [AccessionRecord(uuid="G009", accession_number="G009", cellid=42)],
            file_name="accessions.xlsx",
            description="kept",
        )
        fake = FakeLLM([llm_response("Still 1 accession.")])
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=file_services)

        turn = await agent.chat(
            "now the driest",
            accession_file_paths=file_services.accession_files,
            context_json=previous.to_json(),
        )

        user_note = fake.calls[0]["messages"][-1]["content"]
        assert "file loaded automatically" not in user_note
        assert "1 accessions selected" in user_note
        assert AccessionContext.from_json(turn.context_json).count == 1

    async def test_failed_load_is_reported_in_note(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Undetectable columns produce a failure note instead of an empty selection."""
        path = write_accessions(tmp_path / "odd.csv", [{"code": "A1", "foo": 1, "bar": 2}])
        services = ToolServices(
            subsetting=SubsettingClient(SUBSETTING, max_retries=0),
            documents=DocumentStore(tmp_path / "cache"),
            accession_files=[path],
        )
        fake = FakeLLM([llm_response("Which column has the latitude?")])
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=services)

        turn = await agent.chat("load", accession_file_paths=[path])

        user_note = fake.calls[0]["messages"][-1]["content"]
        assert "file load failed:" in user_note
        assert "load_accessions_from_file with explicit columns" in user_note
        assert AccessionContext.from_json(turn.context_json).is_empty

    async def test_several_files_are_not_loaded_blindly(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two spreadsheets require the user's choice; nothing is loaded."""
        rows = [{"Accession": "G1", "Latitude": 4.5, "Longitude": -74.1}]
        first = write_accessions(tmp_path / "a.csv", rows)
        second = write_accessions(tmp_path / "b.csv", rows)
        services = ToolServices(
            subsetting=SubsettingClient(SUBSETTING, max_retries=0),
            documents=DocumentStore(tmp_path / "cache"),
            accession_files=[first, second],
        )
        fake = FakeLLM([llm_response("Which file?")])
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=services)

        turn = await agent.chat("load", accession_file_paths=[first, second])

        assert "file load skipped: 2 spreadsheets" in fake.calls[0]["messages"][-1]["content"]
        assert AccessionContext.from_json(turn.context_json).is_empty

    def test_empty_selection_error_names_the_stage_1_tool(
        self, file_services: ToolServices, tmp_path: Path
    ) -> None:
        """The hint follows the source: file tool in file mode, Genesys tool otherwise."""
        from tools.services import empty_selection_error

        genesys_services = ToolServices(
            subsetting=SubsettingClient(SUBSETTING, max_retries=0),
            documents=DocumentStore(tmp_path / "cache2"),
        )

        assert "load_accessions_from_file" in empty_selection_error(file_services)["error"]
        assert "select_accessions" in empty_selection_error(genesys_services)["error"]


class TestClimateIsOptional:
    """The Subsetting API is contacted only when a climate tool is called."""

    async def test_file_turn_without_climate_request_makes_no_subsetting_call(
        self, monkeypatch: pytest.MonkeyPatch, file_services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Loading and describing the file never reaches the Subsetting host."""
        fake = FakeLLM(
            [
                llm_response(tool_calls=[tool_call("describe_selection", {})]),
                llm_response("Your file has 3 accessions from 2 sites."),
            ]
        )
        monkeypatch.setattr(agent_module, "acompletion", fake)
        agent = SubsettingAgent(services=file_services)

        turn = await agent.chat(
            "load my accessions", accession_file_paths=file_services.accession_files
        )

        # pytest-httpx fails on unexpected requests; none must have been attempted.
        assert httpx_mock.get_requests() == []
        assert turn.answer == "Your file has 3 accessions from 2 sites."
        assert AccessionContext.from_json(turn.context_json).count == 3
        climate_tools = {"list_climate_indicators", "cluster_selection_by_climate", "pick_cluster"}
        called = {m.get("name") for m in turn.memory if m["role"] == "tool"}
        assert not called & climate_tools

    def test_climate_tool_descriptions_state_the_gate(self) -> None:
        """Both climate tools tell the model to wait for an explicit climate request."""
        registry = build_registry("file")
        descriptions = {
            t["function"]["name"]: t["function"]["description"] for t in registry.openai_tools()
        }

        for name in ("list_climate_indicators", "cluster_selection_by_climate"):
            assert "explicitly asks about climate" in descriptions[name]
