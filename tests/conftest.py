"""Shared fixtures for the SDK tests.

The fixtures reproduce the JSON shapes emitted by the Flask API so that tests
never touch the network.
"""

from __future__ import annotations

import pytest

from subsetting_sdk.catalog import IndicatorCatalog
from subsetting_sdk.models import IndicatorCategory, IndicatorPeriod

# Period ids use the 24-hex-character form of MongoDB ObjectIds.
PERIOD_PREC_HIST = "64a000000000000000000001"
PERIOD_PREC_SSP = "64a000000000000000000002"
PERIOD_TMAX_HIST = "64a000000000000000000003"
PERIOD_BEAN_OPT_HIST = "64a000000000000000000004"
PERIOD_MAIZE_OPT_HIST = "64a000000000000000000005"
PERIOD_PH_HIST = "64a000000000000000000006"
PERIOD_TEXTURE_HIST = "64a000000000000000000007"


@pytest.fixture
def indicators_payload() -> list[dict]:
    """Raw body of ``GET /indicators``: categories with nested indicators."""
    return [
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
        },
        {
            "category": "Heat stress",
            "checked": False,
            "indicators": [
                {
                    "name": "Average maximum temperature",
                    "id": "tmax",
                    "pref": "tmax",
                    "indicator_type": "generic",
                    "crop": "generic",
                    "category": "Heat stress",
                    "checked": False,
                    "unit": "°C",
                }
            ],
        },
        {
            "category": "Crop specific",
            "checked": False,
            "indicators": [
                {
                    "name": "Days with optimal temperature",
                    "id": "opt_bean",
                    "pref": "opt_bean",
                    "indicator_type": "specific",
                    "crop": "Bean",
                    "category": "Crop specific",
                    "checked": False,
                    "unit": "days",
                },
                {
                    "name": "Days with optimal temperature",
                    "id": "opt_maize",
                    "pref": "opt_maize",
                    "indicator_type": "specific",
                    "crop": "Maize",
                    "category": "Crop specific",
                    "checked": False,
                    "unit": "days",
                },
            ],
        },
        {
            "category": "Soil properties",
            "checked": False,
            "indicators": [
                {
                    "name": "Soil pH",
                    "id": "ph",
                    "pref": "ph",
                    "indicator_type": "extracted",
                    "crop": "generic",
                    "category": "Soil properties",
                    "checked": False,
                    "unit": "-",
                },
                {
                    "name": "Soil texture",
                    "id": "texture",
                    "pref": "texture",
                    "indicator_type": "categorical",
                    "crop": "generic",
                    "category": "Soil properties",
                    "checked": False,
                    "unit": "class",
                },
            ],
        },
    ]


@pytest.fixture
def periods_payload() -> list[dict]:
    """Raw body of ``GET /indicator-period``."""
    return [
        {"id": PERIOD_PREC_HIST, "indicator": "prec", "period": "1983-2016", "ssp": "historical"},
        {"id": PERIOD_PREC_SSP, "indicator": "prec", "period": "2041-2060", "ssp": "ssp245"},
        {"id": PERIOD_TMAX_HIST, "indicator": "tmax", "period": "1983-2016", "ssp": "historical"},
        {
            "id": PERIOD_BEAN_OPT_HIST,
            "indicator": "opt_bean",
            "period": "1983-2016",
            "ssp": "historical",
        },
        {
            "id": PERIOD_MAIZE_OPT_HIST,
            "indicator": "opt_maize",
            "period": "1983-2016",
            "ssp": "historical",
        },
        {"id": PERIOD_PH_HIST, "indicator": "ph", "period": "static", "ssp": "historical"},
        {
            "id": PERIOD_TEXTURE_HIST,
            "indicator": "texture",
            "period": "static",
            "ssp": "historical",
        },
    ]


@pytest.fixture
def catalog(indicators_payload: list[dict], periods_payload: list[dict]) -> IndicatorCatalog:
    """Catalog pre-populated from the raw payloads, without any HTTP call."""
    return IndicatorCatalog(
        categories=[IndicatorCategory.model_validate(item) for item in indicators_payload],
        periods=[IndicatorPeriod.model_validate(item) for item in periods_payload],
    )


@pytest.fixture
def cluster_payload() -> dict:
    """Raw body of ``POST /cluster`` with two crops and two algorithms."""
    return {
        "data": [
            {
                "cellid": 101,
                "prec_month1": 10.0,
                "prec_month2": 12.5,
                "cluster_hac": 0,
                "cluster_dbscan": 0,
                "crop_name": "bean",
            },
            {
                "cellid": 102,
                "prec_month1": 11.0,
                "prec_month2": None,
                "cluster_hac": 0,
                "cluster_dbscan": -1,
                "crop_name": "bean",
            },
            {
                "cellid": 103,
                "prec_month1": 80.0,
                "prec_month2": 90.0,
                "cluster_hac": 1,
                "cluster_dbscan": 1,
                "crop_name": "bean",
            },
            {
                "cellid": 201,
                "prec_month1": 50.0,
                "prec_month2": 55.0,
                "cluster_hac": 0,
                "cluster_dbscan": 0,
                "crop_name": "maize",
            },
        ],
        "calculate": [],
        "quantile": [],
        "summary": [{"cluster_hac": 0, "count": 3}],
        "proportion": [],
    }
