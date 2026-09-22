"""In-memory catalog of indicators and indicator periods.

The Subsetting API references indicators through MongoDB ObjectIds of
*indicator periods*. Building a filter by hand requires knowing the indicator
id, its type, its crop and the period id for the wanted scenario. The catalog
downloads both listings once, caches them and resolves human-friendly requests
("total precipitation for bean, historical") into validated
:class:`~subsetting_sdk.models.IndicatorFilter` objects.
"""

from __future__ import annotations

import re
import unicodedata

from subsetting_sdk.client import SubsettingClient
from subsetting_sdk.exceptions import IndicatorNotFoundError, IndicatorPeriodNotFoundError
from subsetting_sdk.models import (
    Indicator,
    IndicatorCategory,
    IndicatorFilter,
    IndicatorPeriod,
    IndicatorType,
    MonthWindow,
)

# Scenario labels that denote observed (non-projected) data. The exact label is
# defined by the database, so several common spellings are accepted.
HISTORICAL_SSP_LABELS = frozenset({"historical", "historic", "baseline", "observed", "none", ""})


def normalize_text(value: str) -> str:
    """Normalize a label for tolerant matching.

    Lower-cases the text, strips accents and collapses every run of
    non-alphanumeric characters into a single space, so that
    ``"Précipitation  totale"`` and ``"precipitation-totale"`` compare equal.

    Args:
        value: Text to normalize.
    """
    stripped = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(char for char in stripped if not unicodedata.combining(char))

    return re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).strip()


class IndicatorCatalog:
    """Cached view of the indicators and their datasets (``SubsettingClient.get_indicators``).

    The catalog must be loaded with :meth:`load` (or :meth:`from_client`) before
    the lookup methods are used.

    Attributes:
        categories: Indicator groups as returned by the API.
        indicators: Every indicator, flattened from the categories.
        periods: Every indicator period.
    """

    def __init__(
        self,
        categories: list[IndicatorCategory] | None = None,
        periods: list[IndicatorPeriod] | None = None,
    ) -> None:
        """Create a catalog, optionally pre-populated (useful for tests).

        Args:
            categories: Indicator categories to start with.
            periods: Indicator periods to attach when not nested in the indicators.
        """
        self.categories: list[IndicatorCategory] = []
        self.indicators: list[Indicator] = []
        self.periods: list[IndicatorPeriod] = []
        self._periods_by_indicator: dict[str, list[IndicatorPeriod]] = {}

        # Populate the derived indexes when initial data is provided.
        if categories is not None or periods is not None:
            self.replace(categories or [], periods)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    async def from_client(cls, client: SubsettingClient) -> IndicatorCatalog:
        """Create and load a catalog from the API in one step.

        Args:
            client: Configured Subsetting client.
        """
        catalog = cls()
        await catalog.load(client)

        return catalog

    async def load(self, client: SubsettingClient) -> None:
        """Download the indicators (with their datasets) and rebuild the indexes.

        Args:
            client: Configured Subsetting client.
        """
        self.replace(await client.get_indicators())

    def replace(
        self,
        categories: list[IndicatorCategory],
        periods: list[IndicatorPeriod] | None = None,
    ) -> None:
        """Replace the catalog content and rebuild the lookup indexes.

        Args:
            categories: Indicator categories; each indicator may carry its periods.
            periods: Extra periods to attach (tests); merged with the nested ones.
        """
        self.categories = list(categories)
        self.indicators = [
            indicator for category in categories for indicator in category.indicators
        ]
        self._periods_by_indicator = {}

        # Periods nested in the indicators are the primary source.
        for indicator in self.indicators:
            for period in indicator.periods:
                self._periods_by_indicator.setdefault(indicator.id, []).append(period)

        # Extra periods are attached to their indicator when not already present.
        for period in periods or []:
            known = self._periods_by_indicator.setdefault(period.indicator, [])

            if all(existing.id != period.id for existing in known):
                known.append(period)

        self.periods = [p for plist in self._periods_by_indicator.values() for p in plist]

    @property
    def is_loaded(self) -> bool:
        """Whether the catalog holds at least one indicator."""
        return bool(self.indicators)

    # ------------------------------------------------------------------ #
    # Lookups
    # ------------------------------------------------------------------ #

    def category_names(self) -> list[str]:
        """Return the stress category names in catalog order."""
        return [category.category for category in self.categories]

    def list_by_category(self, category: str) -> list[Indicator]:
        """Return the indicators of a stress category, matched tolerantly.

        Args:
            category: Category name (e.g. "Drought stress"). Partial matches are
                accepted ("drought" matches "Drought stress").

        Raises:
            IndicatorNotFoundError: If no category matches.
        """
        wanted = normalize_text(category)

        # Prefer an exact normalized match, then fall back to a partial one.
        for group in self.categories:
            if normalize_text(group.category) == wanted:
                return list(group.indicators)

        for group in self.categories:
            if wanted in normalize_text(group.category):
                return list(group.indicators)

        available = ", ".join(self.category_names())
        raise IndicatorNotFoundError(
            f"Unknown indicator category '{category}'. Available: {available}."
        )

    def find_indicator(self, query: str, *, crop: str | None = None) -> Indicator:
        """Resolve an indicator by id, pref or (partial) name.

        Args:
            query: Indicator id, pref code or display name. Names are matched
                case- and accent-insensitively; a partial name is accepted when
                it identifies exactly one indicator.
            crop: Restrict the search to indicators of this crop. Needed to pick
                crop-specific indicators, which share names across crops.

        Raises:
            IndicatorNotFoundError: If nothing matches or the match is ambiguous.
        """
        candidates = self._candidates_for_crop(crop)
        wanted = normalize_text(query)

        # Exact matches on id or pref are unambiguous and resolved first.
        for indicator in candidates:
            if indicator.id == query or indicator.pref.lower() == query.lower():
                return indicator

        exact_names = [ind for ind in candidates if normalize_text(ind.name) == wanted]

        # A single exact name match wins; several mean the name is shared by
        # crops, which the caller must disambiguate with ``crop``.
        if len(exact_names) == 1:
            return exact_names[0]

        if len(exact_names) > 1:
            return self._pick_generic_or_fail(exact_names, query)

        partial = [ind for ind in candidates if wanted and wanted in normalize_text(ind.name)]

        if len(partial) == 1:
            return partial[0]

        if len(partial) > 1:
            return self._pick_generic_or_fail(partial, query)

        raise IndicatorNotFoundError(
            f"No indicator matches '{query}'"
            + (f" for crop '{crop}'" if crop else "")
            + ". Use list_by_category() to see the available indicators."
        )

    def periods_for(self, indicator_id: str, *, ssp: str | None = None) -> list[IndicatorPeriod]:
        """Return the periods of an indicator, optionally filtered by scenario.

        Args:
            indicator_id: Indicator id.
            ssp: Scenario label. ``"historical"`` (or any label in
                ``HISTORICAL_SSP_LABELS``) selects observed data.

        Raises:
            IndicatorPeriodNotFoundError: If the indicator has no matching period.
        """
        periods = self._periods_by_indicator.get(indicator_id, [])

        # Without a scenario every period of the indicator is returned.
        if ssp is None:
            matching = periods

        else:
            matching = [period for period in periods if self._ssp_matches(period.ssp, ssp)]

        if not matching:
            raise IndicatorPeriodNotFoundError(
                f"Indicator '{indicator_id}' has no period"
                + (f" for scenario '{ssp}'" if ssp else "")
                + f". Available scenarios: {sorted({p.ssp for p in periods}) or 'none'}."
            )

        return matching

    def build_filter(
        self,
        indicator: str,
        *,
        crop: str | None = None,
        months: MonthWindow | tuple[int, int] | None = None,
        ssp: str | None = "historical",
        period: str | None = None,
    ) -> IndicatorFilter:
        """Build a validated :class:`IndicatorFilter` from human-friendly inputs.

        Args:
            indicator: Indicator id, pref or name.
            crop: Crop name. Required for crop-specific indicators; also used to
                disambiguate names shared across crops.
            months: Month window; a ``(start, end)`` tuple is accepted. Defaults
                to the full year.
            ssp: Scenario to read. Defaults to historical data; ``None`` selects
                every available period.
            period: Optional period label to narrow the selection further.

        Raises:
            IndicatorNotFoundError, IndicatorPeriodNotFoundError: See the lookups.
        """
        resolved = self.find_indicator(indicator, crop=crop)
        periods = self.periods_for(resolved.id, ssp=ssp)

        # Narrow to one period label when the caller asked for it.
        if period is not None:
            periods = [p for p in periods if normalize_text(p.period) == normalize_text(period)]

            if not periods:
                raise IndicatorPeriodNotFoundError(
                    f"Indicator '{resolved.name}' has no period labelled '{period}'."
                )

        # Accept a plain tuple for convenience and default to the whole year.
        if months is None:
            window = MonthWindow(start=1, end=12)

        elif isinstance(months, tuple):
            window = MonthWindow(start=months[0], end=months[1])

        else:
            window = months

        # Specific indicators must carry the crop they were computed for; the
        # catalog's crop label is authoritative over the caller's spelling.
        filter_crop = resolved.crop if resolved.indicator_type is IndicatorType.SPECIFIC else crop

        return IndicatorFilter(
            type=resolved.indicator_type,
            name=resolved.name,
            indicator_periods=[p.id for p in periods],
            months=window,
            crop=filter_crop,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _candidates_for_crop(self, crop: str | None) -> list[Indicator]:
        """Return the indicators eligible for a crop.

        Generic, extracted and categorical indicators apply to every crop, so
        they are always eligible; specific indicators only when their crop
        matches.

        Args:
            crop: Crop name or ``None`` for no restriction.
        """
        if crop is None:
            return list(self.indicators)

        wanted = normalize_text(crop)
        eligible: list[Indicator] = []

        # Keep non-specific indicators and the specific ones of the wanted crop.
        for indicator in self.indicators:
            if indicator.indicator_type is not IndicatorType.SPECIFIC:
                eligible.append(indicator)

            elif normalize_text(indicator.crop) == wanted:
                eligible.append(indicator)

        return eligible

    @staticmethod
    def _pick_generic_or_fail(matches: list[Indicator], query: str) -> Indicator:
        """Disambiguate several matches, preferring a single non-specific one.

        Args:
            matches: Indicators that matched the query.
            query: Original query, used in the error message.

        Raises:
            IndicatorNotFoundError: If the matches cannot be disambiguated.
        """
        generic = [ind for ind in matches if ind.indicator_type is not IndicatorType.SPECIFIC]

        # Exactly one non-specific match is a safe choice; otherwise the caller
        # must pass a crop to choose among crop-specific variants.
        if len(generic) == 1:
            return generic[0]

        options = ", ".join(f"{ind.name} ({ind.crop})" for ind in matches)
        raise IndicatorNotFoundError(
            f"Indicator '{query}' is ambiguous; specify a crop. Candidates: {options}."
        )

    @staticmethod
    def _ssp_matches(period_ssp: str, wanted_ssp: str) -> bool:
        """Compare scenario labels, treating every historical spelling as equal.

        Args:
            period_ssp: Scenario label stored in the period.
            wanted_ssp: Scenario label requested by the caller.
        """
        actual = normalize_text(period_ssp)
        wanted = normalize_text(wanted_ssp)

        # Both labels refer to observed data: equal regardless of spelling.
        if actual in HISTORICAL_SSP_LABELS and wanted in HISTORICAL_SSP_LABELS:
            return True

        return actual == wanted
