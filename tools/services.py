"""Shared dependencies of the tool layer.

A :class:`ToolServices` instance bundles the API clients, the document store,
the user's accession files and the session state so that every tool receives a
single object. Expensive resources (the indicator catalog) are loaded lazily
and cached for the lifetime of the services object, i.e. one chat turn.

The accession source of the conversation is decided once, from the uploaded
files: a spreadsheet with accessions means *file mode* (no Genesys client is
needed); otherwise the agent works in *Genesys mode*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from document_processing.document_store import DocumentStore
from genesys_sdk.client import GenesysClient
from genesys_sdk.models import DEFAULT_CELLID_FIELD
from subsetting_sdk.catalog import IndicatorCatalog
from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.grid import GridSpec
from tools.accession_context import AccessionContext

SOURCE_GENESYS = "genesys"
SOURCE_FILE = "file"


@dataclass
class ToolServices:
    """Clients, stores and state shared by the tools of one chat turn.

    Attributes:
        subsetting: Subsetting API client (always needed).
        documents: Store of the user's uploaded PDFs.
        genesys: Genesys API client; ``None`` in file mode.
        accession_files: Spreadsheets with accession lists uploaded by the user.
        grid: Grid used to compute cellids from spreadsheet coordinates.
        cellid_field: Dotted path of the Genesys accession field used as cellid.
        context: Current accession selection.
        catalog: Indicator catalog; loaded on first use.
        max_trait_lookups: Upper bound of accessions whose observations are
            fetched in one trait filter, to keep the turn responsive.
        trait_concurrency: Number of concurrent observation requests.
    """

    subsetting: SubsettingClient
    documents: DocumentStore
    genesys: GenesysClient | None = None
    accession_files: list[Path] = field(default_factory=list)
    grid: GridSpec = field(default_factory=GridSpec)
    cellid_field: str = DEFAULT_CELLID_FIELD
    context: AccessionContext = field(default_factory=AccessionContext)
    catalog: IndicatorCatalog | None = None
    max_trait_lookups: int = 300
    trait_concurrency: int = 8

    @property
    def source(self) -> str:
        """Accession source of the conversation: ``file`` when a spreadsheet exists."""
        return SOURCE_FILE if self.accession_files else SOURCE_GENESYS

    @property
    def uses_genesys(self) -> bool:
        """Whether the conversation reads accessions from the Genesys API."""
        return self.source == SOURCE_GENESYS

    def require_genesys(self) -> GenesysClient:
        """Return the Genesys client or fail clearly in file mode.

        Raises:
            RuntimeError: If the conversation works from an accession file.
        """
        # Genesys tools are not registered in file mode, so this guards misuse.
        if self.genesys is None:
            raise RuntimeError(
                "The Genesys API is not available: this conversation works from an "
                "uploaded accession file."
            )

        return self.genesys

    async def get_catalog(self) -> IndicatorCatalog:
        """Return the indicator catalog, downloading it on first use."""
        # The catalog is small and rarely changes; one download per turn is enough.
        if self.catalog is None or not self.catalog.is_loaded:
            self.catalog = await IndicatorCatalog.from_client(self.subsetting)

        return self.catalog

    async def aclose(self) -> None:
        """Close the HTTP clients owned by the services."""
        if self.genesys is not None:
            await self.genesys.aclose()

        await self.subsetting.aclose()
