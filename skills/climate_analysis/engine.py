"""Pure logic of the climate skill: indicator resolution, aggregation, crops.

Nothing here talks to the LLM or the session; the skill wraps these helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sdks.subsetting import Indicator, SubsettingClient
from sdks.subsetting.models import MONTH_COLUMNS

# Statistics accepted to aggregate the twelve monthly values of an indicator.
STATISTICS: tuple[str, ...] = ("mean", "sum", "min", "max")

# Spellings the model may use for the statistics.
_STATISTIC_ALIASES: dict[str, str] = {
    "avg": "mean",
    "average": "mean",
    "promedio": "mean",
    "media": "mean",
    "total": "sum",
    "suma": "sum",
    "acumulado": "sum",
    "accumulated": "sum",
    "minimum": "min",
    "minimo": "min",
    "maximum": "max",
    "maximo": "max",
}

# Indicators whose monthly values are naturally added up rather than averaged.
SUM_BY_DEFAULT_PREFIXES: frozenset[str] = frozenset({"t_rain", "cdd", "ndws", "ndwl", "days_heat", "days_cold", "days_optm", "nvpd4"})


class ClimateError(ValueError):
    """Raised when a climate request cannot be fulfilled with the given arguments."""


@dataclass(frozen=True)
class ResolvedIndicator:
    """An indicator chosen for the analysis.

    Attributes:
        indicator: Catalogue entry.
        requested_as: Text the model used to refer to it.
    """

    indicator: Indicator
    requested_as: str

    @property
    def label(self) -> str:
        """Short label used in criteria texts (the prefix)."""
        return self.indicator.pref


# --------------------------------------------------------------- statistics
def normalise_statistic(statistic: str | None, pref: str) -> str:
    """Map a statistic spelling to a supported key, with per-indicator defaults.

    Args:
        statistic: ``mean``/``sum``/``min``/``max`` or a synonym; ``None`` for default.
        pref: Indicator prefix, used to pick the default (sum for counts/rainfall).

    Returns:
        One of ``STATISTICS``.

    Raises:
        ClimateError: If the statistic is unknown.
    """
    # Default: totals for rainfall/day-count indicators, averages otherwise.
    if not statistic or not str(statistic).strip():
        return "sum" if pref.lower() in SUM_BY_DEFAULT_PREFIXES else "mean"

    key = str(statistic).strip().lower()

    if key in STATISTICS:
        return key
    if key in _STATISTIC_ALIASES:
        return _STATISTIC_ALIASES[key]

    raise ClimateError(f"Unknown statistic '{statistic}'. Use one of {list(STATISTICS)}.")


def normalise_months(months: Any) -> tuple[int, int]:
    """Interpret a month range argument.

    Args:
        months: ``None`` (whole year), ``[first, last]``, ``"3-8"`` or a single month.

    Returns:
        Inclusive ``(first, last)`` range within 1..12.

    Raises:
        ClimateError: If the value cannot be interpreted.
    """
    # Whole year by default.
    if months is None or months == "" or months == []:
        return (1, 12)

    values: list[int]

    # Text forms: "3-8", "3,8", "3".
    if isinstance(months, str):
        parts = [part for part in re.split(r"[-,;:\s]+", months.strip()) if part]
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ClimateError(f"Invalid months '{months}'. Use [first, last] such as [3, 8].") from exc
    elif isinstance(months, (list, tuple)):
        try:
            values = [int(item) for item in months]
        except (TypeError, ValueError) as exc:
            raise ClimateError(f"Invalid months '{months}'. Use [first, last] such as [3, 8].") from exc
    else:
        values = [int(months)]

    if len(values) == 1:
        values = [values[0], values[0]]

    first, last = values[0], values[-1]

    if not (1 <= first <= 12 and 1 <= last <= 12):
        raise ClimateError("Months must be between 1 and 12.")

    return (first, last)


def month_columns(months: tuple[int, int]) -> list[str]:
    """Return the ``monthN`` columns of an inclusive range (wrapping over December)."""
    first, last = months

    # A wrapped range such as (10, 3) covers Oct..Dec + Jan..Mar.
    if first <= last:
        numbers = list(range(first, last + 1))
    else:
        numbers = list(range(first, 13)) + list(range(1, last + 1))

    return [f"month{number}" for number in numbers]


def describe_window(statistic: str, months: tuple[int, int]) -> str:
    """Render ``mean(m1-12)`` style text for criteria."""
    first, last = months
    span = f"m{first}" if first == last else f"m{first}-{last}"
    return f"{statistic}({span})"


# ---------------------------------------------------------------- aggregate
def aggregate_indicator(
    data: pd.DataFrame,
    indicator: Indicator,
    statistic: str,
    months: tuple[int, int],
) -> pd.Series:
    """Reduce the long ``indicators-data`` table to one value per cell for an indicator.

    Args:
        data: Long DataFrame (``cellid``, ``pref_indicator``, ``month1..12``, ``value``, ``category``).
        indicator: Indicator to aggregate.
        statistic: One of ``STATISTICS`` (ignored for extracted/categorical indicators).
        months: Inclusive month range.

    Returns:
        Series indexed by ``cellid`` with the aggregated value (float or category).
    """
    rows = data[data["pref_indicator"].astype(str).str.lower() == indicator.pref.lower()]

    # No rows for this indicator: return an empty series with the right name.
    if rows.empty:
        return pd.Series(dtype="float64", name=indicator.pref)

    kind = (indicator.indicator_type or "").lower()

    # Categorical indicators carry a class code in ``category``.
    if kind == "categorical":
        series = rows.groupby("cellid")["category"].first()
        return series.rename(indicator.pref)

    # Extracted (soil) indicators carry a single value.
    if kind == "extracted":
        series = pd.to_numeric(rows.groupby("cellid")["value"].first(), errors="coerce")
        return series.rename(indicator.pref)

    # Monthly indicators: aggregate the selected months, ignoring missing months.
    columns = [column for column in month_columns(months) if column in rows.columns]
    numeric = rows[["cellid", *columns]].copy()

    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    per_cell = numeric.groupby("cellid")[columns].mean()  # duplicates per cell (several periods) averaged
    reducers = {"mean": per_cell.mean(axis=1), "sum": per_cell.sum(axis=1, min_count=1), "min": per_cell.min(axis=1), "max": per_cell.max(axis=1)}

    return reducers[statistic].rename(indicator.pref)


# --------------------------------------------------------------- indicators
def _normalise_text(text: str) -> str:
    """Lower-case text without separators for fuzzy comparisons."""
    return re.sub(r"[\s_\-\.]+", "", str(text).strip().lower())


def infer_crop(frame: pd.DataFrame, catalogue_crops: list[str]) -> str | None:
    """Guess the catalogue crop of the accessions from their crop columns.

    Args:
        frame: Candidate list.
        catalogue_crops: Crop names known by the Subsetting catalogue.

    Returns:
        The single matching crop name, or ``None`` when none or several match.
    """
    columns = [column for column in frame.columns if _normalise_text(column) in ("cropname", "cropcode", "crop", "cultivo")]

    # Without crop columns there is nothing to infer from.
    if not columns:
        return None

    values: set[str] = set()

    # Collect every distinct crop value present in the list.
    for column in columns:
        values.update(_normalise_text(value) for value in frame[column].dropna().astype(str).unique())

    matches: set[str] = set()

    # A catalogue crop matches when its name is contained in a value or vice versa.
    for crop in catalogue_crops:
        key = _normalise_text(crop)
        singular = key[:-1] if key.endswith("s") else key

        for value in values:
            if key in value or value in key or singular in value:
                matches.add(crop)

    return matches.pop() if len(matches) == 1 else None


def resolve_indicators(
    client: SubsettingClient,
    names: list[str],
    crop: str | None,
) -> list[ResolvedIndicator]:
    """Translate indicator names/prefixes/ids into catalogue entries.

    Args:
        client: Subsetting client (catalogue access).
        names: Identifiers given by the model.
        crop: Crop used to pick crop-specific variants.

    Returns:
        Resolved indicators, without duplicates.

    Raises:
        ClimateError: When a name is unknown, ambiguous, or a crop-specific
            indicator is requested without a resolvable crop.
    """
    resolved: list[ResolvedIndicator] = []
    seen: set[str] = set()

    for name in names:
        variants = client.get_indicator_variants(name)

        # Unknown identifier: try a free-text search to suggest candidates.
        if not variants:
            candidates = client.find_indicators(name)

            if len(candidates) == 1:
                variants = candidates
            else:
                suggestions = sorted({f"{c.pref} ({c.name})" for c in candidates})[:8]
                raise ClimateError(
                    f"Indicator '{name}' was not found in the catalogue. "
                    f"Candidates: {suggestions or 'none'}. Use action='list_indicators' to explore."
                )

        specific = [v for v in variants if (v.crop or "").strip().upper() != "NC"]

        # Crop-specific indicator without a crop: the user must say which crop.
        if specific and len(variants) > 1 and not crop:
            crops = sorted(v.crop for v in specific if v.crop)
            raise ClimateError(
                f"Indicator '{name}' is crop-specific and exists for {crops}. "
                "Tell which crop to use (argument 'crop')."
            )

        indicator = client.get_indicator(name, crop=crop)

        # The crop given does not have this specific indicator.
        if indicator is not None and specific and crop and (indicator.crop or "").strip().lower() != crop.strip().lower() and (indicator.crop or "").strip().upper() != "NC":
            crops = sorted(v.crop for v in specific if v.crop)
            raise ClimateError(f"Indicator '{name}' is not available for crop '{crop}'. Available crops: {crops}.")

        if indicator is None or indicator.id in seen:
            continue

        seen.add(indicator.id)
        resolved.append(ResolvedIndicator(indicator=indicator, requested_as=name))

    return resolved


def summarise_catalogue(indicators: list[Indicator], query: str | None = None) -> list[dict[str, Any]]:
    """Group the catalogue by prefix for the model, optionally filtered by text.

    Args:
        indicators: Catalogue entries.
        query: Free text matched against id, name, prefix and category.

    Returns:
        One entry per prefix with category, type, unit, name and available crops.
    """
    grouped: dict[str, dict[str, Any]] = {}
    categories = categories_for_query(query) if query else []

    for indicator in indicators:
        in_category = any(
            category in (indicator.category or "").lower()
            or (category == "crop-specific" and (indicator.indicator_type or "").lower() == "specific")
            for category in categories
        )

        # Skip entries that match neither the text nor the categories it implies.
        if query and not (indicator.matches(query) or in_category):
            continue

        entry = grouped.setdefault(
            indicator.pref,
            {
                "pref": indicator.pref,
                "name": indicator.name,
                "category": indicator.category,
                "type": indicator.indicator_type,
                "unit": indicator.unit,
                "crops": [],
            },
        )

        crop = (indicator.crop or "").strip()

        # Generic indicators apply to every crop; specific ones list their crops.
        if crop and crop.upper() != "NC" and crop not in entry["crops"]:
            entry["crops"].append(crop)

    for entry in grouped.values():
        entry["crops"] = sorted(entry["crops"]) or ["all (generic)"]

    return sorted(grouped.values(), key=lambda item: (item["category"] or "", item["pref"]))


# ------------------------------------------------------------- by query
# User vocabulary (Spanish/English) mapped to catalogue categories/prefixes.
QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "drought": ("drought", "sequia", "sequía", "seco", "dry", "aridez", "arid", "lluvia", "rain", "precipit"),
    "heat": ("heat", "calor", "temperatura", "temperature", "termico", "térmico", "vpd", "calido", "cálido"),
    "flooding": ("flood", "inundacion", "inundación", "anegamiento", "waterlogging", "exceso de agua", "humedad"),
    "photoperiod": ("photoperiod", "fotoperiodo", "daylength", "radiacion", "radiación", "radiation", "luz"),
    "soil": ("soil", "suelo", "suelos", "edaf", "ph", "textura", "texture", "salinidad", "salinity", "carbono", "carbon", "fertilidad"),
    "crop-specific": ("crop-specific", "specific", "cultivo", "dias de calor", "días de calor", "dias frios", "días fríos", "optimo", "óptimo"),
}

# Maximum indicators chosen automatically from a free-text need.
MAX_AUTO_INDICATORS = 5


def categories_for_query(query: str) -> list[str]:
    """Translate a free-text need into catalogue category keywords.

    Args:
        query: Text such as ``"sequía"``, ``"drought tolerance"`` or ``"suelos ácidos"``.

    Returns:
        Category keywords (``drought``, ``heat``, ``flooding``, ``photoperiod``,
        ``soil``, ``crop-specific``) found in the text; empty when none matches.
    """
    text = query.strip().lower()
    found: list[str] = []

    # Every category whose synonyms appear in the text is selected.
    for category, synonyms in QUERY_SYNONYMS.items():
        if any(synonym in text for synonym in synonyms):
            found.append(category)

    return found


def select_indicators_by_query(
    client: SubsettingClient,
    query: str,
    crop: str | None,
    limit: int = MAX_AUTO_INDICATORS,
) -> list[ResolvedIndicator]:
    """Choose indicators for a need expressed in words (step 2 of the business rules).

    Generic indicators of the matching categories are preferred; crop-specific
    ones are added only when a crop is known. Exact prefixes/names inside the
    query (``"CDD"``) are honoured too.

    Args:
        client: Subsetting client (catalogue access).
        query: Free text describing the need.
        crop: Crop used for crop-specific indicators.
        limit: Maximum number of indicators returned.

    Returns:
        Resolved indicators (at least one).

    Raises:
        ClimateError: If nothing in the catalogue matches the text.
    """
    catalogue = client.list_indicators()
    categories = categories_for_query(query)
    chosen: list[Indicator] = []
    seen: set[str] = set()

    def add(indicator: Indicator) -> None:
        """Append an indicator once (by prefix for generic ones)."""
        key = indicator.id

        if key not in seen and len(chosen) < limit:
            seen.add(key)
            chosen.append(indicator)

    # 1) Prefixes or names written literally in the query.
    for token in re.split(r"[\s,;/]+", query):
        for indicator in client.get_indicator_variants(token) if token else []:
            is_generic = (indicator.crop or "").strip().upper() == "NC"
            if is_generic or (crop and (indicator.crop or "").strip().lower() == crop.strip().lower()):
                add(indicator)

    # 2) Indicators of the matching categories (generic first, then crop-specific for the crop).
    for category in categories:
        for indicator in catalogue:
            in_category = category in (indicator.category or "").lower() or (
                category == "crop-specific" and (indicator.indicator_type or "").lower() == "specific"
            )
            if not in_category:
                continue

            is_generic = (indicator.crop or "").strip().upper() == "NC"

            if is_generic and (indicator.indicator_type or "").lower() != "specific":
                add(indicator)

        for indicator in catalogue:
            in_category = category in (indicator.category or "").lower() or (
                category == "crop-specific" and (indicator.indicator_type or "").lower() == "specific"
            )
            if in_category and crop and (indicator.crop or "").strip().lower() == crop.strip().lower():
                add(indicator)

    # 3) Fallback: free-text match on name/category.
    if not chosen:
        for indicator in catalogue:
            if indicator.matches(query) and (indicator.crop or "").strip().upper() == "NC":
                add(indicator)

    if not chosen:
        raise ClimateError(
            f"No indicator in the catalogue matches '{query}'. Try terms such as drought/sequía, "
            "heat/calor, flooding/inundación, photoperiod/fotoperiodo, soil/suelo, or name the "
            "indicators (CDD, t_rain, TX, PHIHOX...). Use action='list_indicators' to explore."
        )

    return [ResolvedIndicator(indicator=indicator, requested_as=query) for indicator in chosen]
