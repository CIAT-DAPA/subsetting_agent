"""Unit tests for the tool layer: session context, tools and registry.

The Genesys and Subsetting clients are mocked with ``pytest-httpx``; the
document store uses a temporary directory and a generated PDF.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from document_processing import DocumentStore
from genesys_sdk import Accession, GenesysClient
from subsetting_sdk import SubsettingClient
from tests.test_document_processing import write_pdf
from tests.test_genesys_sdk import accession_payload, page_payload
from tools import AccessionContext, Stage, ToolServices, build_registry
from tools.genesys_tools import observation_matches
from tools.registry import ToolRegistry

GENESYS = "https://genesys.example"
SUBSETTING = "https://subsetting.example"


@pytest.fixture
def services(tmp_path: Path) -> ToolServices:
    """Services wired to the mocked hosts with a fresh context and store."""
    return ToolServices(
        genesys=GenesysClient(GENESYS, token="t", max_retries=0, backoff_seconds=0.0),
        subsetting=SubsettingClient(
            SUBSETTING, api_prefix="/api/v1", max_retries=0, backoff_seconds=0.0
        ),
        documents=DocumentStore(tmp_path / "cache"),
    )


@pytest.fixture
def registry() -> ToolRegistry:
    """The full tool registry."""
    return build_registry()


def seeded_context(uuids_cells: dict[str, int | None], crop: str = "bean") -> AccessionContext:
    """Build a context with a passport selection already made.

    Args:
        uuids_cells: UUID -> cellid for each accession (``None`` = no coordinates).
        crop: Crop code assigned to every accession.
    """
    accessions = []

    # Build Genesys accessions from the mapping so the context uses real parsing.
    for uuid, cell in uuids_cells.items():
        payload = accession_payload(uuid, tile=cell, tile3=None)
        payload["crop"]["shortName"] = crop
        accessions.append(Accession.from_api(payload))

    context = AccessionContext()
    context.set_passport_selection(
        accessions,
        passport_filter={"crop": [crop]},
        total_matching=len(accessions),
        description="seed",
    )

    return context


def indicators_page_payloads() -> tuple[list[dict], list[dict]]:
    """Minimal catalog payloads for the Subsetting mock."""
    indicators = [
        {
            "category": "Drought stress",
            "checked": False,
            "indicators": [
                {
                    "name": "Total precipitation",
                    "id": "prec",
                    "pref": "prec",
                    "indicator_type": "generic",
                    "crop": "generic",
                    "category": "Drought stress",
                    "checked": False,
                    "unit": "mm",
                }
            ],
        }
    ]
    periods = [
        {
            "id": "64a000000000000000000001",
            "indicator": "prec",
            "period": "1983-2016",
            "ssp": "historical",
        }
    ]

    return indicators, periods


def mock_catalog(httpx_mock: HTTPXMock) -> None:
    """Register the catalog endpoints on the Subsetting mock.

    Args:
        httpx_mock: Active HTTP mock.
    """
    indicators, periods = indicators_page_payloads()
    httpx_mock.add_response(url=f"{SUBSETTING}/api/v1/indicators", json=indicators)
    httpx_mock.add_response(url=f"{SUBSETTING}/api/v1/indicator-period", json=periods)


class TestAccessionContext:
    """The context tracks selection, stages, cells and serialization."""

    def test_passport_selection_and_grouping(self) -> None:
        """Cellids are grouped per crop and deduplicated; no-coordinate rows are skipped."""
        context = seeded_context({"u-1": 10, "u-2": 10, "u-3": 20, "u-4": None})

        assert context.count == 4
        assert context.stage is Stage.PASSPORT
        assert len(context.with_cellid()) == 3
        assert [c.to_api() for c in context.cellids_by_crop()] == [
            {"crop": "bean", "cellids": [10, 20]}
        ]
        assert [r.uuid for r in context.accessions_in_cells([10])] == ["u-1", "u-2"]

    def test_keep_advances_stage_only_forward(self) -> None:
        """Keeping records the step and never moves the stage backwards."""
        context = seeded_context({"u-1": 10, "u-2": 20, "u-3": 30})

        kept = context.keep(["u-1", "u-2", "ghost"], stage=Stage.CLIMATE, description="climate")
        context.keep(["u-1"], stage=Stage.TRAITS, description="trait afterwards")

        assert kept == 2
        assert context.count == 1
        assert context.stage is Stage.CLIMATE
        assert [(s.before, s.after) for s in context.steps] == [(0, 3), (3, 2), (2, 1)]

    def test_truncated_flag(self) -> None:
        """The context knows when the passport stage did not load everything."""
        context = seeded_context({"u-1": 10})
        context.total_matching = 500

        assert context.truncated
        assert context.summary()["truncated"] is True

    def test_clusters_and_sizes(self) -> None:
        """Cluster labels map to accession counts through their cells."""
        context = seeded_context({"u-1": 10, "u-2": 10, "u-3": 20})

        context.set_clusters({0: [10], 1: [20]}, "agglomerative")

        assert context.stage is Stage.CLIMATE
        assert context.cluster_sizes() == {"0": 2, "1": 1}

    def test_json_round_trip(self) -> None:
        """Serialization preserves accessions, steps, clusters and stage."""
        context = seeded_context({"u-1": 10, "u-2": 20})
        context.set_clusters({0: [10], 1: [20]}, "dbscan")
        context.keep(["u-1"], stage=Stage.CLIMATE, description="picked 0")
        context.selected_cluster = "0"

        restored = AccessionContext.from_json(context.to_json())

        assert restored.uuids() == ["u-1"]
        assert restored.stage is Stage.CLIMATE
        assert restored.clusters == {"0": [10], "1": [20]}
        assert restored.selected_cluster == "0"
        assert len(restored.steps) == 2
        assert restored.records()[0].cellid == 10

    def test_from_json_empty(self) -> None:
        """Missing state yields an empty context."""
        assert AccessionContext.from_json(None).is_empty
        assert AccessionContext.from_json("  ").is_empty


class TestRegistry:
    """The registry renders schemas, validates arguments and dispatches."""

    def test_openai_tools_shape(self, registry: ToolRegistry) -> None:
        """Every tool renders as an OpenAI function definition."""
        tools = registry.openai_tools()

        assert len(tools) == 13
        assert all(t["type"] == "function" for t in tools)
        assert all(t["function"]["parameters"]["type"] == "object" for t in tools)
        assert "select_accessions" in registry.describe()

    def test_duplicate_registration_rejected(self, registry: ToolRegistry) -> None:
        """Registering the same name twice is a programming error."""
        with pytest.raises(ValueError, match="already registered"):
            registry.register(registry.tools["search_crops"])

    async def test_unknown_tool(self, registry: ToolRegistry, services: ToolServices) -> None:
        """Unknown names return an error listing valid tools."""
        result = await registry.execute(services, "fly_to_mars", {})

        assert "Unknown tool" in result["error"]
        assert "select_accessions" in result["error"]

    async def test_missing_required_argument(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Missing required arguments are reported without calling the handler."""
        result = await registry.execute(services, "search_documents", {})

        assert "Missing required argument" in result["error"]

    async def test_unknown_arguments_are_dropped(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Extra arguments invented by the model do not break the call."""
        result = await registry.execute(services, "describe_selection", {"sample": 3, "bogus": 1})

        assert result["stage"] == "empty"


class TestPassportTools:
    """Passport tools query Genesys and (re)start the selection."""

    async def test_search_crops(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Crops are filtered by name, case-insensitively."""
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/crop",
            json=[
                {"shortName": "bean", "name": "Bean", "accessionCount": 10},
                {"shortName": "maize", "name": "Maize", "accessionCount": 20},
            ],
        )

        result = await registry.execute(services, "search_crops", {"name": "BEA"})

        assert result["count"] == 1
        assert result["crops"][0]["code"] == "bean"

    async def test_preview_accessions(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Preview sends a filter with coordinates required and returns counts."""
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/overview?limit=8",
            json={
                "accessionCount": 42,
                "overview": {
                    "countryOfOrigin.code3": {"terms": [{"term": "COL", "count": 40}], "total": 42},
                    "storage": {"terms": [{"term": "10", "count": 1}], "total": 42},
                },
            },
        )

        result = await registry.execute(
            services, "preview_accessions", {"crop_codes": ["Bean"], "origin_countries": ["col"]}
        )
        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert sent == {
            "crop": ["bean"],
            "countryOfOrigin": {"code3": ["COL"]},
            "geo": {"referenced": True},
        }
        assert result["matching_accessions"] == 42
        assert "countryOfOrigin.code3" in result["breakdown"]
        assert "storage" not in result["breakdown"]

    async def test_select_accessions_sets_context(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Selecting loads accessions into the context and flags truncation."""
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/list?p=0&l=50",
            json=page_payload(["u-1", "u-2"], number=0, total=10, last=False),
        )

        result = await registry.execute(
            services, "select_accessions", {"crop_codes": ["bean"], "max_records": 2}
        )

        assert services.context.count == 2
        assert services.context.stage is Stage.PASSPORT
        assert result["accession_count"] == 2
        assert "warning" in result
        assert services.context.records()[0].cellid == 555


class TestTraitTools:
    """Trait tools search descriptors and filter by observations."""

    def test_observation_matches_numeric_and_categorical(self) -> None:
        """Keys containing the descriptor are compared with the condition."""
        from genesys_sdk.models import AccessionObservations

        observations = AccessionObservations.model_validate(
            {
                "firstPartyData": [{"Drought tolerance score": 4, "Plant height": 120}],
                "thirdPartyData": [],
            }
        )

        matched, compared = observation_matches(
            observations, descriptor="drought tolerance", min_value=3, max_value=5, equals=None
        )
        not_matched, _ = observation_matches(
            observations, descriptor="plant height", min_value=None, max_value=100, equals=None
        )
        categorical, _ = observation_matches(
            observations, descriptor="drought", min_value=None, max_value=None, equals="4"
        )

        assert matched and compared == ["Drought tolerance score"]
        assert not not_matched
        assert categorical

    def test_observation_matches_nested_value_field(self) -> None:
        """Records mentioning the descriptor in a value expose their 'value' field."""
        from genesys_sdk.models import AccessionObservations

        observations = AccessionObservations.model_validate(
            {
                "firstPartyData": [{"descriptor": "Yield (kg/ha)", "value": 2500}],
                "thirdPartyData": [],
            }
        )

        matched, _ = observation_matches(
            observations, descriptor="yield", min_value=2000, max_value=None, equals=None
        )

        assert matched

    async def test_filter_selection_by_trait(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Only accessions with matching observations are kept; others are counted."""
        services.context = seeded_context({"u-1": 10, "u-2": 20, "u-3": 30})
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/u-1/observations",
            json={"firstPartyData": [{"drought score": 5}], "thirdPartyData": []},
        )
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/acn/u-2/observations",
            json={"firstPartyData": [{"drought score": 1}], "thirdPartyData": []},
        )
        httpx_mock.add_response(url=f"{GENESYS}/api/v2/acn/u-3/observations", json={})

        result = await registry.execute(
            services, "filter_selection_by_trait", {"descriptor": "drought", "min_value": 3}
        )

        assert result["checked"] == 3
        assert result["kept"] == 1
        assert result["without_observations"] == 1
        assert services.context.uuids() == ["u-1"]
        assert services.context.stage is Stage.TRAITS

    async def test_filter_by_trait_requires_selection(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """The trait stage cannot run before the passport stage."""
        result = await registry.execute(
            services, "filter_selection_by_trait", {"descriptor": "x", "min_value": 1}
        )

        assert "select_accessions first" in result["error"]

    async def test_search_trait_descriptors(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Descriptor search sends a title filter and returns compact rows."""
        httpx_mock.add_response(
            url=f"{GENESYS}/api/v2/descriptor/list/details?p=0&l=25",
            json={
                "content": [{"uuid": "d-1", "title": "Drought tolerance", "dataType": "CODED"}],
                "totalElements": 1,
            },
        )

        result = await registry.execute(
            services, "search_trait_descriptors", {"keyword": "drought", "crop_code": "Bean"}
        )
        sent = json.loads(httpx_mock.get_requests()[0].content)

        assert sent == {"crop": ["bean"], "title": {"contains": ["drought"]}}
        assert result["descriptors"][0]["uuid"] == "d-1"


class TestDocumentTools:
    """Document tools wrap the store and can reduce the selection."""

    async def test_search_and_read(
        self, registry: ToolRegistry, services: ToolServices, tmp_path: Path
    ) -> None:
        """Searching returns passages that can then be read in full."""
        pdf = write_pdf(
            tmp_path / "p.pdf",
            [
                [
                    ("Bean drought study", 20),
                    ("Accession G123 was the most drought tolerant landrace. " * 5, 10),
                ]
            ],
        )
        services.documents.add_pdf(pdf)
        services.context = seeded_context({"u-1": 10})

        hits = await registry.execute(services, "search_documents", {"query": "drought tolerant"})
        section = await registry.execute(
            services,
            "read_document_section",
            {
                "document_id": hits["passages"][0]["document_id"],
                "section_index": hits["passages"][0]["section_index"],
            },
        )

        assert hits["count"] >= 1
        assert "G123" in section["text"]
        assert services.context.stage is Stage.DOCUMENTS

    async def test_search_without_documents(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Without uploads the tool asks for a PDF instead of returning nothing."""
        result = await registry.execute(services, "search_documents", {"query": "x"})

        assert "No documents uploaded" in result["error"]

    async def test_keep_from_documents(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Accession numbers cited in a paper reduce the selection; unknown ones are reported."""
        services.context = seeded_context({"u-001": 10, "u-002": 20})

        result = await registry.execute(
            services,
            "keep_accessions_from_documents",
            {"accession_numbers": ["g001", "G999"], "reason": "Table 2 of paper.pdf"},
        )

        assert result["kept"] == 1
        assert result["not_found"] == ["g999"]
        assert services.context.uuids() == ["u-001"]

    async def test_keep_from_documents_refuses_to_empty(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Numbers that match nothing leave the selection untouched."""
        services.context = seeded_context({"u-001": 10})

        result = await registry.execute(
            services,
            "keep_accessions_from_documents",
            {"accession_numbers": ["nope"], "reason": "r"},
        )

        assert "error" in result
        assert services.context.count == 1


class TestClimateTools:
    """Climate tools operate on the cells of the current selection only."""

    async def test_list_indicators(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Indicators are listed with their category."""
        mock_catalog(httpx_mock)

        result = await registry.execute(
            services, "list_climate_indicators", {"category": "drought"}
        )

        assert result["indicators"][0]["name"] == "Total precipitation"

    async def test_cluster_and_pick(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Clustering stores labels; picking a cluster reduces the selection."""
        services.context = seeded_context({"u-1": 101, "u-2": 102, "u-3": 103})
        mock_catalog(httpx_mock)
        httpx_mock.add_response(
            url=f"{SUBSETTING}/api/v1/cluster",
            json={
                "data": [
                    {"cellid": 101, "prec_month1": 1.0, "cluster_hac": 0, "crop_name": "bean"},
                    {"cellid": 102, "prec_month1": 2.0, "cluster_hac": 0, "crop_name": "bean"},
                    {"cellid": 103, "prec_month1": 9.0, "cluster_hac": 1, "crop_name": "bean"},
                ],
                "summary": [{"cluster_hac": 0}],
            },
        )

        clustered = await registry.execute(
            services, "cluster_selection_by_climate", {"indicators": ["prec"], "n_clusters": 2}
        )
        picked = await registry.execute(services, "pick_cluster", {"cluster": "1"})
        sent = json.loads(httpx_mock.get_requests()[-1].content)

        assert sent["analysis"]["hyperparameter"]["n_clusters"] == 2
        assert [c["accessions"] for c in clustered["clusters"]] == [2, 1]
        assert clustered["clusters"][1]["indicators"]["prec"]["mean"] == 9.0
        assert picked["kept"] == 1
        assert services.context.uuids() == ["u-3"]
        assert services.context.selected_cluster == "1"

    async def test_cluster_empty_response_is_error(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """An empty cluster payload is reported, not treated as zero clusters."""
        services.context = seeded_context({"u-1": 101})
        mock_catalog(httpx_mock)
        httpx_mock.add_response(url=f"{SUBSETTING}/api/v1/cluster", json={})

        result = await registry.execute(
            services, "cluster_selection_by_climate", {"indicators": ["prec"]}
        )

        assert "returned no data" in result["error"]

    async def test_unknown_indicator(
        self, registry: ToolRegistry, services: ToolServices, httpx_mock: HTTPXMock
    ) -> None:
        """Unknown indicator names are surfaced as errors from the catalog."""
        services.context = seeded_context({"u-1": 101})
        mock_catalog(httpx_mock)

        result = await registry.execute(
            services, "cluster_selection_by_climate", {"indicators": ["wind"]}
        )

        assert "No indicator matches" in result["error"]

    async def test_climate_requires_selection(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Climate tools refuse to run before a passport selection exists."""
        result = await registry.execute(
            services, "cluster_selection_by_climate", {"indicators": ["prec"]}
        )

        assert "select_accessions first" in result["error"]

    async def test_pick_without_clusters(
        self, registry: ToolRegistry, services: ToolServices
    ) -> None:
        """Picking requires a previous clustering."""
        services.context = seeded_context({"u-1": 101})

        result = await registry.execute(services, "pick_cluster", {"cluster": "0"})

        assert "cluster_selection_by_climate first" in result["error"]

    async def test_describe_selection(self, registry: ToolRegistry, services: ToolServices) -> None:
        """The description lists stage, counts and example accessions."""
        services.context = seeded_context({"u-1": 101, "u-2": 102})

        result = await registry.execute(services, "describe_selection", {"sample": 1})

        assert result["stage"] == "passport"
        assert result["accession_count"] == 2
        assert len(result["accessions"]) == 1
