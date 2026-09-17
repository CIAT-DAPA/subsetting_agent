"""Shared dependencies of the tool layer.

A :class:`ToolServices` instance bundles the API clients, the document store
and the session state so that every tool receives a single object. Expensive
resources (the indicator catalog) are loaded lazily and cached for the
lifetime of the services object, i.e. one chat turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from document_processing.document_store import DocumentStore
from genesys_sdk.client import GenesysClient
from subsetting_sdk.catalog import IndicatorCatalog
from subsetting_sdk.client import SubsettingClient
from tools.accession_context import AccessionContext


@dataclass
class ToolServices:
    """Clients, stores and state shared by the tools of one chat turn.

    Attributes:
        genesys: Genesys API client.
        subsetting: Subsetting API client.
        documents: Store of the user's uploaded documents.
        context: Current accession selection.
        catalog: Indicator catalog; loaded on first use.
        max_trait_lookups: Upper bound of accessions whose observations are
            fetched in one trait filter, to keep the turn responsive.
        trait_concurrency: Number of concurrent observation requests.
    """

    genesys: GenesysClient
    subsetting: SubsettingClient
    documents: DocumentStore
    context: AccessionContext = field(default_factory=AccessionContext)
    catalog: IndicatorCatalog | None = None
    max_trait_lookups: int = 300
    trait_concurrency: int = 8

    async def get_catalog(self) -> IndicatorCatalog:
        """Return the indicator catalog, downloading it on first use."""
        # The catalog is small and rarely changes; one download per turn is enough.
        if self.catalog is None or not self.catalog.is_loaded:
            self.catalog = await IndicatorCatalog.from_client(self.subsetting)

        return self.catalog

    async def aclose(self) -> None:
        """Close the HTTP clients owned by the services."""
        await self.genesys.aclose()
        await self.subsetting.aclose()
