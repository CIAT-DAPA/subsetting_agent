"""Subsetting SDK: climate indicators, indicator data and clustering."""

from sdks.subsetting.auth import FallbackAuth
from sdks.subsetting.client import SubsettingClient
from sdks.subsetting.errors import (
    SubsettingAuthError,
    SubsettingConnectionError,
    SubsettingError,
    SubsettingRequestError,
)
from sdks.subsetting.grid import add_cellid_column, compute_cellid
from sdks.subsetting.models import ClusterResult, Indicator, IndicatorPeriod

__all__ = [
    "FallbackAuth",
    "SubsettingClient",
    "SubsettingAuthError",
    "SubsettingConnectionError",
    "SubsettingError",
    "SubsettingRequestError",
    "add_cellid_column",
    "compute_cellid",
    "ClusterResult",
    "Indicator",
    "IndicatorPeriod",
]
