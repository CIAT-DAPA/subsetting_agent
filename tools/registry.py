"""Registry of the tools exposed to the language model.

Each tool has a name, a description, a JSON schema for its arguments and an
async handler. The registry renders the OpenAI-style ``tools`` list consumed
by litellm, validates the arguments the model produces and dispatches calls,
turning every failure into an ``{"error": ...}`` result the model can act on.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from genesys_sdk.exceptions import GenesysApiError
from subsetting_sdk.exceptions import SubsettingApiError
from tools import document_tools, file_tools, genesys_tools, subsetting_tools
from tools.services import SOURCE_FILE, SOURCE_GENESYS, ToolServices

logger = logging.getLogger(__name__)

ToolHandler = Callable[..., Awaitable[dict[str, Any]]]


@dataclass
class ToolSpec:
    """Definition of one tool.

    Attributes:
        name: Tool name as seen by the model.
        description: What the tool does and when to use it.
        parameters: JSON schema of the arguments (``type: object``).
        handler: Async function ``handler(services, **arguments)``.
        stage: Business stage the tool belongs to, for ordering hints.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    stage: str = "any"

    def to_openai(self) -> dict[str, Any]:
        """Render the tool in the OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    """Build an object schema with the given properties.

    Args:
        properties: JSON schema of each argument.
        required: Names of the mandatory arguments.
    """
    return {"type": "object", "properties": properties, "required": required or []}


def _string_list(description: str) -> dict[str, Any]:
    """Schema of a list of strings.

    Args:
        description: Description shown to the model.
    """
    return {"type": "array", "items": {"type": "string"}, "description": description}


PASSPORT_PROPERTIES: dict[str, Any] = {
    "crop_codes": _string_list("Genesys crop codes, e.g. ['bean']. Find them with search_crops."),
    "genus": _string_list("Genus names, e.g. ['Phaseolus']."),
    "species": _string_list("Species epithets, e.g. ['vulgaris']."),
    "origin_countries": _string_list(
        "ISO-3166 alpha-3 codes of the country of origin, e.g. ['COL','PER']."
    ),
    "institute_codes": _string_list("FAO WIEWS codes of holding genebanks, e.g. ['COL003']."),
    "sample_status": {
        "type": "array",
        "items": {"type": "integer"},
        "description": (
            "MCPD SAMPSTAT codes: 100 wild, 200 weedy, 300 landrace, 400 breeding material, "
            "500 improved cultivar."
        ),
    },
    "available": {"type": "boolean", "description": "Only accessions available for distribution."},
    "text": {
        "type": "string",
        "description": "Free-text keywords; use only when no structured field applies.",
    },
}


# Tools that only make sense when accessions come from the Genesys API.
GENESYS_ONLY_TOOLS = frozenset(
    {
        "search_crops",
        "preview_accessions",
        "select_accessions",
        "search_trait_descriptors",
        "filter_selection_by_trait",
    }
)

# Tools that only make sense when accessions come from an uploaded spreadsheet.
FILE_ONLY_TOOLS = frozenset({"list_accession_files", "load_accessions_from_file"})


def build_registry(source: str = SOURCE_GENESYS) -> ToolRegistry:
    """Create the registry for an accession source.

    Args:
        source: ``"genesys"`` registers passport/trait tools backed by the API;
            ``"file"`` registers the spreadsheet tools instead. Document and
            climate tools are always available.

    Raises:
        ValueError: If the source is unknown.
    """
    if source not in (SOURCE_GENESYS, SOURCE_FILE):
        raise ValueError(f"Unknown accession source '{source}'.")

    full = build_full_registry()
    registry = ToolRegistry()

    # Keep the tools of the requested source plus the shared ones, in order.
    for spec in full.tools.values():
        if source == SOURCE_FILE and spec.name in GENESYS_ONLY_TOOLS:
            continue

        if source == SOURCE_GENESYS and spec.name in FILE_ONLY_TOOLS:
            continue

        registry.register(spec)

    return registry


def build_full_registry() -> ToolRegistry:
    """Create the registry with every tool of every source (used to derive the modes)."""
    registry = ToolRegistry()

    # ---- Stage 1 (file mode): accession spreadsheet -------------------- #
    registry.register(
        ToolSpec(
            name="list_accession_files",
            description="List the accession spreadsheets (Excel/CSV) the user uploaded.",
            parameters=_schema({}),
            handler=file_tools.list_accession_files,
            stage="passport",
        )
    )
    registry.register(
        ToolSpec(
            name="load_accessions_from_file",
            description=(
                "Load the accessions (identifier + coordinates) from the uploaded spreadsheet "
                "and start the selection. Columns are auto-detected; pass them only if the "
                "tool reports it could not find them. Always the FIRST step in file mode."
            ),
            parameters=_schema(
                {
                    "file_name": {
                        "type": "string",
                        "description": "File to load (optional if one).",
                    },
                    "id_column": {"type": "string", "description": "Accession identifier column."},
                    "latitude_column": {"type": "string", "description": "Latitude column."},
                    "longitude_column": {"type": "string", "description": "Longitude column."},
                    "crop_column": {"type": "string", "description": "Crop column, if any."},
                    "default_crop": {
                        "type": "string",
                        "description": "Crop code for all rows when the file has no crop column.",
                    },
                }
            ),
            handler=file_tools.load_accessions_from_file,
            stage="passport",
        )
    )

    # ---- Stage 1 (Genesys mode): passport ------------------------------ #
    registry.register(
        ToolSpec(
            name="search_crops",
            description="Find Genesys crop codes by crop name. Use before filtering by crop.",
            parameters=_schema(
                {"name": {"type": "string", "description": "Crop name, e.g. 'bean'."}}
            ),
            handler=genesys_tools.search_crops,
            stage="passport",
        )
    )
    registry.register(
        ToolSpec(
            name="preview_accessions",
            description=(
                "Count accessions matching passport criteria and show a breakdown by crop, "
                "country and institute, without loading them. Use to refine criteria."
            ),
            parameters=_schema(PASSPORT_PROPERTIES),
            handler=genesys_tools.preview_accessions,
            stage="passport",
        )
    )
    registry.register(
        ToolSpec(
            name="select_accessions",
            description=(
                "Load georeferenced accessions matching passport criteria and start a new "
                "selection. Always the FIRST step of building a subset."
            ),
            parameters=_schema(
                {
                    **PASSPORT_PROPERTIES,
                    "max_records": {
                        "type": "integer",
                        "description": "Maximum accessions to load (default from configuration).",
                    },
                }
            ),
            handler=genesys_tools.select_accessions,
            stage="passport",
        )
    )

    # ---- Stage 2: traits ----------------------------------------------- #
    registry.register(
        ToolSpec(
            name="search_trait_descriptors",
            description="Find trait descriptors (e.g. 'drought', 'yield') and their codes.",
            parameters=_schema(
                {
                    "keyword": {"type": "string", "description": "Word in the descriptor title."},
                    "crop_code": {
                        "type": "string",
                        "description": "Genesys crop code to restrict to.",
                    },
                },
                required=["keyword"],
            ),
            handler=genesys_tools.search_trait_descriptors,
            stage="traits",
        )
    )
    registry.register(
        ToolSpec(
            name="filter_selection_by_trait",
            description=(
                "Keep only selected accessions whose trait observations satisfy a condition. "
                "Numeric traits: min_value/max_value. Categorical traits: equals."
            ),
            parameters=_schema(
                {
                    "descriptor": {
                        "type": "string",
                        "description": "Descriptor title keyword, column name or UUID.",
                    },
                    "min_value": {"type": "number", "description": "Lower bound (numeric traits)."},
                    "max_value": {"type": "number", "description": "Upper bound (numeric traits)."},
                    "equals": {
                        "type": "string",
                        "description": "Expected value (categorical traits).",
                    },
                },
                required=["descriptor"],
            ),
            handler=genesys_tools.filter_selection_by_trait,
            stage="traits",
        )
    )

    # ---- Stage 3: documents -------------------------------------------- #
    registry.register(
        ToolSpec(
            name="list_documents",
            description="List the PDF documents the user uploaded in this conversation.",
            parameters=_schema({}),
            handler=document_tools.list_documents,
            stage="documents",
        )
    )
    registry.register(
        ToolSpec(
            name="search_documents",
            description="Search the uploaded documents for passages about a topic.",
            parameters=_schema(
                {
                    "query": {"type": "string", "description": "Keywords to search for."},
                    "top_k": {"type": "integer", "description": "Number of passages (1-10)."},
                    "document_id": {"type": "string", "description": "Restrict to one document."},
                },
                required=["query"],
            ),
            handler=document_tools.search_documents,
            stage="documents",
        )
    )
    registry.register(
        ToolSpec(
            name="read_document_section",
            description="Read the full text of one section of an uploaded document.",
            parameters=_schema(
                {
                    "document_id": {"type": "string", "description": "Document id."},
                    "section_index": {"type": "integer", "description": "Section index (0-based)."},
                },
                required=["document_id", "section_index"],
            ),
            handler=document_tools.read_document_section,
            stage="documents",
        )
    )
    registry.register(
        ToolSpec(
            name="keep_accessions_from_documents",
            description=(
                "Reduce the selection to accession numbers that a document identifies as "
                "relevant. Pass the numbers you read in the document and cite it in reason."
            ),
            parameters=_schema(
                {
                    "accession_numbers": _string_list("Accession numbers (or UUIDs) to keep."),
                    "reason": {"type": "string", "description": "Why, citing the document."},
                },
                required=["accession_numbers", "reason"],
            ),
            handler=document_tools.keep_accessions_from_documents,
            stage="documents",
        )
    )

    # ---- Stage 4: climate ---------------------------------------------- #
    registry.register(
        ToolSpec(
            name="list_climate_indicators",
            description=(
                "List available climate and soil indicators, optionally by stress category "
                "(drought, flood, heat, photoperiod, soil, crop specific). Use only when the "
                "user explicitly asks about climate or environmental conditions."
            ),
            parameters=_schema(
                {
                    "category": {"type": "string", "description": "Stress category name."},
                    "crop_code": {
                        "type": "string",
                        "description": "Crop for crop-specific indicators.",
                    },
                }
            ),
            handler=subsetting_tools.list_climate_indicators,
            stage="climate",
        )
    )
    registry.register(
        ToolSpec(
            name="cluster_selection_by_climate",
            description=(
                "Group the selected accessions into climate clusters using one or more "
                "indicators, and report each cluster's indicator statistics. Requires an "
                "existing selection. Follow with pick_cluster to keep one cluster. Use only "
                "when the user explicitly asks about climate or environmental conditions."
            ),
            parameters=_schema(
                {
                    "indicators": _string_list("Indicator names or codes."),
                    "month_start": {
                        "type": "integer",
                        "description": "First month 1-12 (default 1).",
                    },
                    "month_end": {
                        "type": "integer",
                        "description": "Last month 1-12 (default 12).",
                    },
                    "algorithm": {
                        "type": "string",
                        "enum": ["agglomerative", "dbscan", "hdbscan"],
                        "description": "Clustering algorithm (default agglomerative).",
                    },
                    "n_clusters": {
                        "type": "integer",
                        "description": "Max clusters, 2-5 (default 5).",
                    },
                },
                required=["indicators"],
            ),
            handler=subsetting_tools.cluster_selection_by_climate,
            stage="climate",
        )
    )
    registry.register(
        ToolSpec(
            name="pick_cluster",
            description="Keep only the accessions of one climate cluster from the last clustering.",
            parameters=_schema(
                {"cluster": {"type": "string", "description": "Cluster label, e.g. '0'."}},
                required=["cluster"],
            ),
            handler=subsetting_tools.pick_cluster,
            stage="climate",
        )
    )

    # ---- Any stage ----------------------------------------------------- #
    registry.register(
        ToolSpec(
            name="describe_selection",
            description=(
                "Describe the current selection: stage, counts, applied steps and example "
                "accessions. Use before answering the user."
            ),
            parameters=_schema(
                {
                    "sample": {
                        "type": "integer",
                        "description": "Example accessions to list (default 10).",
                    }
                }
            ),
            handler=subsetting_tools.describe_selection,
            stage="any",
        )
    )

    return registry


@dataclass
class ToolRegistry:
    """Collection of tools with rendering, validation and dispatch.

    Attributes:
        tools: Registered tools keyed by name.
    """

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        """Add a tool, refusing duplicated names.

        Args:
            spec: Tool definition.
        """
        if spec.name in self.tools:
            raise ValueError(f"Tool '{spec.name}' is already registered.")

        self.tools[spec.name] = spec

    def names(self) -> list[str]:
        """Return the registered tool names in registration order."""
        return list(self.tools)

    def openai_tools(self) -> list[dict[str, Any]]:
        """Render every tool in the OpenAI function-calling format."""
        return [spec.to_openai() for spec in self.tools.values()]

    def describe(self) -> str:
        """Render a one-line-per-tool description for the system prompt."""
        return "\n".join(f"- {spec.name}: {spec.description}" for spec in self.tools.values())

    def validate_arguments(self, spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
        """Check required arguments and drop unknown ones.

        Args:
            spec: Tool definition.
            arguments: Arguments produced by the model.

        Returns:
            Cleaned arguments.

        Raises:
            ValueError: If a required argument is missing.
        """
        properties = spec.parameters.get("properties", {})
        required = spec.parameters.get("required", [])
        missing = [name for name in required if arguments.get(name) in (None, "", [])]

        # Required arguments must be present and non-empty.
        if missing:
            raise ValueError(f"Missing required argument(s) for {spec.name}: {', '.join(missing)}.")

        cleaned: dict[str, Any] = {}

        # Unknown arguments are dropped silently; models sometimes invent them.
        for name, value in arguments.items():
            if name in properties and value is not None:
                cleaned[name] = value

        return cleaned

    async def execute(
        self, services: ToolServices, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a tool by name, converting failures into error results.

        Args:
            services: Shared services for this chat turn.
            name: Tool name requested by the model.
            arguments: Arguments produced by the model.
        """
        spec = self.tools.get(name)

        # Unknown tool names are reported with the valid options.
        if spec is None:
            return {"error": f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}."}

        try:
            cleaned = self.validate_arguments(spec, arguments)
            return await spec.handler(services, **cleaned)

        except ValueError as exc:
            return {"error": str(exc)}

        except (GenesysApiError, SubsettingApiError) as exc:
            logger.warning("Tool %s failed against an API: %s", name, exc)
            return {"error": f"{name} failed: {exc}"}

        except TypeError as exc:
            # Wrong argument types (e.g. a string where a list is expected).
            logger.warning("Tool %s received invalid arguments %s: %s", name, arguments, exc)
            return {"error": f"Invalid arguments for {name}: {exc}"}

        except Exception as exc:  # noqa: BLE001 - the agent loop must never crash on a tool
            logger.exception("Unexpected error in tool %s", name)
            return {"error": f"Unexpected error in {name}: {exc}"}
