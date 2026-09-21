"""SDK for the Genesys Subsetting API (climate indicators and clustering of accessions)."""

from subsetting_sdk.catalog import IndicatorCatalog, normalize_text
from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.exceptions import (
    IndicatorNotFoundError,
    IndicatorPeriodNotFoundError,
    SubsettingApiError,
    SubsettingAuthError,
    SubsettingConnectionError,
)
from subsetting_sdk.grid import DEFAULT_GRID, GridSpec, cellid_from_coordinates
from subsetting_sdk.models import (
    ClusteringAlgorithm,
    ClusteringHyperparameters,
    ClusterRequest,
    ClusterResult,
    ClusterRow,
    CropCellIds,
    Indicator,
    IndicatorCategory,
    IndicatorFilter,
    IndicatorPeriod,
    IndicatorType,
    MonthWindow,
)

__all__ = [
    "DEFAULT_GRID",
    "GridSpec",
    "cellid_from_coordinates",
    "ClusterRequest",
    "ClusterResult",
    "ClusterRow",
    "ClusteringAlgorithm",
    "ClusteringHyperparameters",
    "CropCellIds",
    "Indicator",
    "IndicatorCatalog",
    "IndicatorCategory",
    "IndicatorFilter",
    "IndicatorNotFoundError",
    "IndicatorPeriod",
    "IndicatorPeriodNotFoundError",
    "IndicatorType",
    "MonthWindow",
    "SubsettingApiError",
    "SubsettingAuthError",
    "SubsettingClient",
    "SubsettingConnectionError",
    "normalize_text",
]
