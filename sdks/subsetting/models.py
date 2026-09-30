"""Pydantic models of the Subsetting API responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Indicator types returned by the API and how they must be requested.
INDICATOR_TYPES: tuple[str, ...] = ("generic", "specific", "extracted", "categorical")
# Month columns of an indicator value.
MONTH_COLUMNS: tuple[str, ...] = tuple(f"month{month}" for month in range(1, 13))


class Indicator(BaseModel):
    """A climate/agro-climatic indicator (``GET /indicators``).

    Attributes:
        id: Indicator identifier (used to look up its periods).
        name: Human readable name.
        pref: Short prefix used as column name in data responses (e.g. ``t_rain``).
        indicator_type: ``generic``, ``specific``, ``extracted`` or ``categorical``.
        crop: Crop the indicator was computed for (``generic`` for all).
        category: Stress/category group (e.g. ``Drought``).
        unit: Measurement unit.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    name: str
    pref: str
    indicator_type: str
    crop: str | None = None
    category: str | None = None
    unit: str | None = None

    def matches(self, text: str) -> bool:
        """Whether a free-text query refers to this indicator.

        Args:
            text: Query typed by a user or the model.

        Returns:
            ``True`` when the query appears in the id, name, prefix or category.
        """
        needle = text.strip().lower()

        # Empty queries never match.
        if not needle:
            return False

        haystack = [self.id, self.name, self.pref, self.category or ""]
        return any(needle in item.lower() or item.lower() in needle for item in haystack if item)


class IndicatorPeriod(BaseModel):
    """A period of an indicator (``GET /indicator-period``).

    Attributes:
        id: Period identifier expected by ``/cluster`` and ``/indicators-data``.
        indicator: Indicator id the period belongs to.
        period: ``"mean"`` or a year such as ``"1983"``.
        ssp: Climate scenario label.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    indicator: str
    period: str
    ssp: str | None = None


class ClusterResult(BaseModel):
    """Outcome of ``POST /cluster``.

    Attributes:
        assignments: ``{cellid: cluster}`` for the clustering column used.
        cluster_column: Name of the column read from the response (``cluster_hac``...).
        indicator_values: Per cell values of the indicators used in the analysis.
        summary: ``summary`` block of the response (per cluster/indicator statistics).
        calculate: ``calculate`` block (min/max/mean/sd per month).
        raw: Full decoded response for callers that need more detail.
    """

    model_config = ConfigDict(extra="ignore")

    assignments: dict[int, int] = Field(default_factory=dict)
    cluster_column: str | None = None
    indicator_values: dict[int, dict[str, Any]] = Field(default_factory=dict)
    summary: list[dict[str, Any]] = Field(default_factory=list)
    calculate: list[dict[str, Any]] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def cluster_count(self) -> int:
        """Number of distinct clusters found."""
        return len(set(self.assignments.values()))

    @classmethod
    def from_response(cls, payload: dict[str, Any], preferred_column: str | None = None) -> ClusterResult:
        """Parse the cluster endpoint response.

        Args:
            payload: Decoded JSON body.
            preferred_column: Cluster column to read (``cluster_hac``, ``cluster_dbscan``,
                ``cluster_hdbscan``). The first one present is used when omitted.

        Returns:
            The parsed result; empty when the API returned no analysis.
        """
        rows = payload.get("data") or []
        assignments: dict[int, int] = {}
        values: dict[int, dict[str, Any]] = {}
        column: str | None = None

        # Detect the cluster column from the first row.
        if rows:
            candidates = [key for key in rows[0] if str(key).startswith("cluster")]

            if preferred_column and preferred_column in candidates:
                column = preferred_column
            elif candidates:
                column = candidates[0]

        # One assignment per cell; the remaining keys are indicator values.
        for row in rows:
            cellid = row.get("cellid")

            if cellid is None or column is None or row.get(column) is None:
                continue

            cell = int(cellid)
            assignments[cell] = int(row[column])
            values[cell] = {
                key: value
                for key, value in row.items()
                if key not in ("cellid", "crop_name") and not str(key).startswith("cluster")
            }

        return cls(
            assignments=assignments,
            cluster_column=column,
            indicator_values=values,
            summary=payload.get("summary") or [],
            calculate=payload.get("calculate") or [],
            raw=payload,
        )
