"""Unit tests for :class:`IndicatorCatalog`."""

from __future__ import annotations

import pytest
from pytest_httpx import HTTPXMock

from subsetting_sdk.catalog import IndicatorCatalog, normalize_text
from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.exceptions import IndicatorNotFoundError, IndicatorPeriodNotFoundError
from subsetting_sdk.models import IndicatorType, MonthWindow
from tests.conftest import (
    PERIOD_BEAN_OPT_HIST,
    PERIOD_PREC_HIST,
    PERIOD_PREC_SSP,
    PERIOD_TMAX_HIST,
)


class TestNormalizeText:
    """Normalization makes matching tolerant to case, accents and punctuation."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Total precipitation", "total precipitation"),
            ("Précipitation  totale!", "precipitation totale"),
            ("Drought-stress", "drought stress"),
            ("  ", ""),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        """Each input collapses to its lower-case ASCII word form."""
        assert normalize_text(raw) == expected


class TestLoading:
    """The catalog can be loaded from the API in one call."""

    async def test_from_client_loads_both_listings(
        self, httpx_mock: HTTPXMock, indicators_payload: list[dict], periods_payload: list[dict]
    ) -> None:
        """Indicators and periods are fetched and indexed."""
        base = "https://example.org/api/subsetting"
        httpx_mock.add_response(url=f"{base}/api/v1/indicators", json=indicators_payload)
        httpx_mock.add_response(url=f"{base}/api/v1/indicator-period", json=periods_payload)

        async with SubsettingClient(base, api_prefix="/api/v1", max_retries=0) as client:
            catalog = await IndicatorCatalog.from_client(client)

        assert catalog.is_loaded
        assert len(catalog.indicators) == 6
        assert len(catalog.periods_for("prec")) == 2

    def test_empty_catalog_is_not_loaded(self) -> None:
        """A fresh catalog reports that it has no data."""
        assert not IndicatorCatalog().is_loaded


class TestCategoryLookup:
    """Categories are matched exactly first, then partially."""

    def test_exact_match(self, catalog: IndicatorCatalog) -> None:
        """The full category name resolves its indicators."""
        assert [i.id for i in catalog.list_by_category("Drought stress")] == ["prec"]

    def test_partial_match(self, catalog: IndicatorCatalog) -> None:
        """A keyword contained in the category name is enough."""
        assert [i.id for i in catalog.list_by_category("heat")] == ["tmax"]

    def test_unknown_category_lists_available_ones(self, catalog: IndicatorCatalog) -> None:
        """The error message helps the caller pick a valid category."""
        with pytest.raises(IndicatorNotFoundError, match="Drought stress"):
            catalog.list_by_category("salinity")


class TestIndicatorLookup:
    """Indicators resolve by id, pref, exact name or unambiguous partial name."""

    def test_by_id_and_pref(self, catalog: IndicatorCatalog) -> None:
        """Ids and pref codes match exactly (pref case-insensitively)."""
        assert catalog.find_indicator("tmax").id == "tmax"
        assert catalog.find_indicator("TMAX").id == "tmax"

    def test_by_name_case_insensitive(self, catalog: IndicatorCatalog) -> None:
        """Display names match regardless of case."""
        assert catalog.find_indicator("total PRECIPITATION").id == "prec"

    def test_partial_name(self, catalog: IndicatorCatalog) -> None:
        """A distinctive fragment of the name is enough."""
        assert catalog.find_indicator("maximum temperature").id == "tmax"

    def test_specific_indicator_needs_crop_when_shared(self, catalog: IndicatorCatalog) -> None:
        """A name shared by several crops is ambiguous without a crop."""
        with pytest.raises(IndicatorNotFoundError, match="ambiguous"):
            catalog.find_indicator("Days with optimal temperature")

    def test_specific_indicator_resolved_with_crop(self, catalog: IndicatorCatalog) -> None:
        """Passing the crop picks the matching crop-specific indicator."""
        assert catalog.find_indicator("Days with optimal temperature", crop="bean").id == "opt_bean"
        assert catalog.find_indicator("optimal", crop="Maize").id == "opt_maize"

    def test_generic_indicator_available_for_any_crop(self, catalog: IndicatorCatalog) -> None:
        """Generic indicators are eligible whatever the crop."""
        assert catalog.find_indicator("prec", crop="Cassava").id == "prec"

    def test_unknown_indicator(self, catalog: IndicatorCatalog) -> None:
        """Nothing matching raises a helpful error."""
        with pytest.raises(IndicatorNotFoundError, match="No indicator matches"):
            catalog.find_indicator("wind speed")


class TestPeriodLookup:
    """Periods filter by scenario with tolerant historical labels."""

    def test_all_periods(self, catalog: IndicatorCatalog) -> None:
        """Without a scenario every period of the indicator is returned."""
        assert {p.id for p in catalog.periods_for("prec")} == {PERIOD_PREC_HIST, PERIOD_PREC_SSP}

    @pytest.mark.parametrize("label", ["historical", "Historic", "baseline"])
    def test_historical_spellings(self, catalog: IndicatorCatalog, label: str) -> None:
        """Several spellings select the observed-data period."""
        assert [p.id for p in catalog.periods_for("prec", ssp=label)] == [PERIOD_PREC_HIST]

    def test_future_scenario(self, catalog: IndicatorCatalog) -> None:
        """Projected scenarios match by label."""
        assert [p.id for p in catalog.periods_for("prec", ssp="SSP245")] == [PERIOD_PREC_SSP]

    def test_missing_scenario_lists_available(self, catalog: IndicatorCatalog) -> None:
        """An unavailable scenario raises with the available ones listed."""
        with pytest.raises(IndicatorPeriodNotFoundError, match="ssp245"):
            catalog.periods_for("prec", ssp="ssp585")


class TestBuildFilter:
    """``build_filter`` produces validated filters with the right ids and types."""

    def test_generic_filter_defaults(self, catalog: IndicatorCatalog) -> None:
        """Historical period, full-year window and no range by default."""
        built = catalog.build_filter("total precipitation")

        assert built.type is IndicatorType.GENERIC
        assert built.name == "Total precipitation"
        assert built.indicator_periods == [PERIOD_PREC_HIST]
        assert built.months == MonthWindow(start=1, end=12)
        assert built.range is None
        assert built.crop is None

    def test_generic_filter_with_window_and_range(self, catalog: IndicatorCatalog) -> None:
        """Tuples are accepted for the month window and the range."""
        built = catalog.build_filter("tmax", months=(11, 2), value_range=(25.0, 35.0))

        assert built.indicator_periods == [PERIOD_TMAX_HIST]
        assert built.months.months() == [11, 12, 1, 2]
        assert built.range == (25.0, 35.0)

    def test_specific_filter_uses_catalog_crop_label(self, catalog: IndicatorCatalog) -> None:
        """The crop stored in the catalog wins over the caller's spelling."""
        built = catalog.build_filter("optimal temperature", crop="BEAN")

        assert built.type is IndicatorType.SPECIFIC
        assert built.crop == "Bean"
        assert built.indicator_periods == [PERIOD_BEAN_OPT_HIST]

    def test_all_scenarios_when_ssp_is_none(self, catalog: IndicatorCatalog) -> None:
        """``ssp=None`` includes every available period of the indicator."""
        built = catalog.build_filter("prec", ssp=None)

        assert set(built.indicator_periods) == {PERIOD_PREC_HIST, PERIOD_PREC_SSP}

    def test_period_label_narrows_selection(self, catalog: IndicatorCatalog) -> None:
        """A period label selects exactly that dataset."""
        built = catalog.build_filter("prec", ssp=None, period="2041-2060")

        assert built.indicator_periods == [PERIOD_PREC_SSP]

    def test_unknown_period_label(self, catalog: IndicatorCatalog) -> None:
        """A period label that does not exist raises a period error."""
        with pytest.raises(IndicatorPeriodNotFoundError, match="2100"):
            catalog.build_filter("prec", period="2100")
