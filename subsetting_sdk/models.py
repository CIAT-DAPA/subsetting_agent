"""Typed request and response models for the Subsetting API.

The models mirror the JSON contracts implemented by the Flask API in
``CIAT-DAPA/subsets_genebank_accessions`` (``src/subsets_api/api.py``). They give
the agent a validated, documented surface instead of hand-built dictionaries.

Terminology:
    cellid: Identifier of a ~5 km grid cell. The API works with cellids, never
        with accession identifiers. Each accession returned by the Genesys MCP
        carries its own cellid.
    indicator: A climate or soil variable (e.g. "Total precipitation").
    indicator period: A concrete dataset of an indicator for a time period and a
        climate scenario (SSP). Filters reference indicator periods by id.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class IndicatorType(str, Enum):
    """How the API stores and filters the values of an indicator.

    Attributes:
        GENERIC: Monthly series (``month1``..``month12``) shared by every crop.
            Filtering aggregates the selected months with a sum for
            "Total precipitation" and an average for every other indicator.
        SPECIFIC: Monthly series computed for one crop only (e.g. days with
            optimal temperature for bean). Applies only to cellids of that crop.
        EXTRACTED: A single numeric ``value`` per cell (soil properties).
        CATEGORICAL: A single categorical ``value_c`` per cell (soil texture).
    """

    GENERIC = "generic"
    SPECIFIC = "specific"
    EXTRACTED = "extracted"
    CATEGORICAL = "categorical"


class ClusteringAlgorithm(str, Enum):
    """Clustering algorithms supported by ``POST /cluster``.

    Each algorithm adds its own label column to the result rows.
    """

    AGGLOMERATIVE = "agglomerative"
    DBSCAN = "dbscan"
    HDBSCAN = "hdbscan"

    @property
    def result_column(self) -> str:
        """Return the column name the API uses for this algorithm's labels."""
        return _ALGORITHM_COLUMNS[self]


# Column emitted by the API for each algorithm (see clustering_analysis.py).
_ALGORITHM_COLUMNS: dict[ClusteringAlgorithm, str] = {
    ClusteringAlgorithm.AGGLOMERATIVE: "cluster_hac",
    ClusteringAlgorithm.DBSCAN: "cluster_dbscan",
    ClusteringAlgorithm.HDBSCAN: "cluster_hdbscan",
}


# --------------------------------------------------------------------------- #
# Catalog models (GET /indicators, GET /indicator-period)
# --------------------------------------------------------------------------- #


class IndicatorPeriod(BaseModel):
    """One dataset of an indicator for a period and scenario (``GET /indicator-period``).

    Attributes:
        id: MongoDB ObjectId of the period. Filters reference this value.
        indicator: Id of the indicator the period belongs to.
        period: Time span label (e.g. "1983-2016" or a future window).
        ssp: Climate scenario label (e.g. "historical", "ssp245").
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    indicator: str
    period: str
    ssp: str


class Indicator(BaseModel):
    """One indicator as listed by ``GET /indicators``.

    Attributes:
        id: Internal identifier of the indicator.
        name: Display name (e.g. "Total precipitation").
        pref: Short code used as column prefix in cluster results (e.g. "prec").
        indicator_type: Storage/filtering type of the indicator.
        crop: Crop the indicator belongs to. Generic indicators use a generic
            crop label (as stored by the API), specific ones name the crop.
        category: Stress group the indicator belongs to (e.g. "Drought stress").
        unit: Measurement unit reported by the API.
        periods: Datasets available for the indicator (period x scenario). Filled
            by the client from ``GET /indicator-period``.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    name: str
    pref: str
    indicator_type: IndicatorType
    crop: str
    category: str
    unit: str
    periods: list[IndicatorPeriod] = Field(default_factory=list)


class IndicatorCategory(BaseModel):
    """Group of indicators sharing a stress category, as returned by ``GET /indicators``.

    Attributes:
        category: Name of the stress group.
        indicators: Indicators that belong to the group.
    """

    model_config = ConfigDict(extra="ignore")

    category: str
    indicators: list[Indicator]


# --------------------------------------------------------------------------- #
# Request building blocks
# --------------------------------------------------------------------------- #


class MonthWindow(BaseModel):
    """Inclusive window of months used to aggregate monthly indicators.

    The window may wrap around the end of the year: ``MonthWindow(start=11, end=2)``
    means November, December, January and February, matching the behaviour of
    ``restrict_months_list`` in the API.

    Attributes:
        start: First month of the window (1-12).
        end: Last month of the window (1-12).
    """

    start: int = Field(ge=1, le=12)
    end: int = Field(ge=1, le=12)

    def to_api(self) -> list[int]:
        """Serialize the window to the ``[low, up]`` list expected by the API."""
        return [self.start, self.end]

    def months(self) -> list[int]:
        """Expand the window to the ordered list of month numbers it covers."""
        # A window whose start is after its end wraps around the year, so the
        # months are listed from ``start`` to December and then from January
        # to ``end``.
        if self.start > self.end:
            return list(range(self.start, 13)) + list(range(1, self.end + 1))

        return list(range(self.start, self.end + 1))


class CropCellIds(BaseModel):
    """Cellids of the accessions selected for one crop.

    Attributes:
        crop: Crop name, as used by Genesys and the Subsetting API.
        cellids: Distinct cellids of the selected accessions of that crop.
    """

    crop: str
    cellids: list[int]

    @field_validator("cellids")
    @classmethod
    def _dedupe_cellids(cls, value: list[int]) -> list[int]:
        """Remove duplicated cellids while keeping the original order."""
        return list(dict.fromkeys(value))

    def to_api(self) -> dict[str, Any]:
        """Serialize to the ``{"crop": ..., "cellids": [...]}`` item used by the API."""
        return {"crop": self.crop, "cellids": self.cellids}


class IndicatorFilter(BaseModel):
    """One indicator to include in the multivariate analysis of ``/cluster``.

    Attributes:
        type: Storage type of the indicator; decides how the API aggregates it.
        name: Indicator display name. The API uses it to choose sum vs. average
            for generic indicators, so it must match the catalog name exactly.
        indicator_periods: Ids of the indicator periods to read values from.
        months: Month window for monthly indicators. Ignored by the API for
            extracted and categorical indicators, but always sent for safety.
        crop: Crop name; required for crop-specific indicators.
    """

    type: IndicatorType
    name: str
    indicator_periods: list[str] = Field(min_length=1)
    months: MonthWindow = Field(default_factory=lambda: MonthWindow(start=1, end=12))
    crop: str | None = None

    @model_validator(mode="after")
    def _check_specific_has_crop(self) -> IndicatorFilter:
        """Validate that crop-specific indicators name their crop.

        The API restricts specific indicators to the cellids of ``crop``, so a
        specific filter without a crop would silently match nothing.
        """
        if self.type is IndicatorType.SPECIFIC and not self.crop:
            raise ValueError("A crop-specific indicator filter requires 'crop'.")

        return self

    def to_api(self) -> dict[str, Any]:
        """Serialize to the dictionary consumed by the API.

        The API reads ``indicator`` (list of period ids), ``months``, ``type``,
        ``name`` and, for specific indicators, ``crop``.
        """
        payload: dict[str, Any] = {
            "type": self.type.value,
            "name": self.name,
            "indicator": self.indicator_periods,
            "months": self.months.to_api(),
        }

        # Only specific indicators carry a crop; sending it for the others is
        # harmless but keeps the payload identical to the web client's.
        if self.crop:
            payload["crop"] = self.crop

        return payload


class ClusteringHyperparameters(BaseModel):
    """Hyperparameters accepted by ``/cluster``, with the defaults used by the Genesys UI.

    Attributes:
        n_clusters: Upper bound of clusters for agglomerative clustering; the
            API selects the best k in ``[2, n_clusters]`` by silhouette score.
        epsilon: DBSCAN neighbourhood radius.
        minpts: DBSCAN minimum points per neighbourhood.
        min_cluster_size: HDBSCAN minimum cluster size.
    """

    n_clusters: int = Field(default=5, ge=2)
    epsilon: float = Field(default=10.0, gt=0)
    minpts: int = Field(default=20, ge=1)
    min_cluster_size: int = Field(default=10, ge=2)

    def to_api(self) -> dict[str, Any]:
        """Serialize to the ``hyperparameter`` object of the cluster request."""
        return self.model_dump()


class ClusterRequest(BaseModel):
    """Full body of ``POST /cluster``.

    Attributes:
        cellid_list: Selected cellids grouped by crop.
        filters: Indicators to include in the multivariate analysis.
        months: Month window applied to every monthly indicator.
        algorithms: Clustering algorithms to run. Each adds a label column.
        hyperparameters: Hyperparameters shared by the algorithms.
    """

    cellid_list: list[CropCellIds] = Field(min_length=1)
    filters: list[IndicatorFilter] = Field(min_length=1)
    months: MonthWindow = Field(default_factory=lambda: MonthWindow(start=1, end=12))
    algorithms: list[ClusteringAlgorithm] = Field(
        default_factory=lambda: [ClusteringAlgorithm.AGGLOMERATIVE], min_length=1
    )
    hyperparameters: ClusteringHyperparameters = Field(default_factory=ClusteringHyperparameters)

    def to_api(self) -> dict[str, Any]:
        """Serialize to the JSON body of the request."""
        return {
            "cellid_list": [item.to_api() for item in self.cellid_list],
            "data": [item.to_api() for item in self.filters],
            "months": self.months.to_api(),
            "analysis": {
                "algorithm": [algorithm.value for algorithm in self.algorithms],
                "hyperparameter": self.hyperparameters.to_api(),
            },
        }


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class ClusterRow(BaseModel):
    """One cell in the ``data`` array of the ``/cluster`` response.

    The API returns a flat row with the cellid, one column per indicator month
    (``<pref>_month<N>``) or value (``<pref>_value``), one label column per
    algorithm and the crop name. Indicator columns are kept in ``values``.

    Attributes:
        cellid: Grid cell identifier.
        crop: Crop the row belongs to (``crop_name`` in the API).
        labels: Cluster label per algorithm result column (e.g. ``cluster_hac``).
            DBSCAN/HDBSCAN use ``-1`` for noise points.
        values: Indicator columns used for the analysis.
    """

    cellid: int
    crop: str
    labels: dict[str, int]
    values: dict[str, float | None]

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> ClusterRow:
        """Split a flat API row into cellid, crop, labels and indicator values.

        Args:
            row: Raw dictionary from the ``data`` array of the response.
        """
        labels: dict[str, int] = {}
        values: dict[str, float | None] = {}

        # Route each column by name: cluster columns become labels, the crop
        # and cellid are lifted to attributes, everything else is a value.
        for key, value in row.items():
            if key in ("cellid", "crop_name"):
                continue

            if key.startswith("cluster"):
                labels[key] = int(value)

            else:
                values[key] = value

        return cls(
            cellid=int(row["cellid"]), crop=str(row["crop_name"]), labels=labels, values=values
        )


class ClusterResult(BaseModel):
    """Response of ``POST /cluster``.

    Attributes:
        rows: One entry per cell with its labels and indicator values.
        summary: Per-cluster statistics as returned by the API, kept raw.
        calculate: Min/max/mean/sd per indicator and cluster, kept raw.
        quantile: Quartiles per indicator and cluster, kept raw.
        proportion: Category proportions per cluster, kept raw.
    """

    model_config = ConfigDict(extra="ignore")

    rows: list[ClusterRow] = Field(default_factory=list)
    summary: list[dict[str, Any]] = Field(default_factory=list)
    calculate: list[dict[str, Any]] = Field(default_factory=list)
    quantile: list[dict[str, Any]] = Field(default_factory=list)
    proportion: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> ClusterResult:
        """Build the result from the raw API response.

        Args:
            payload: Decoded JSON body of the response. An empty dictionary is
                accepted because the API returns ``{}`` when the analysis fails
                internally.
        """
        return cls(
            rows=[ClusterRow.from_api(row) for row in payload.get("data", [])],
            summary=payload.get("summary", []) or [],
            calculate=payload.get("calculate", []) or [],
            quantile=payload.get("quantile", []) or [],
            proportion=payload.get("proportion", []) or [],
        )

    def clusters(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.AGGLOMERATIVE,
        *,
        crop: str | None = None,
        include_noise: bool = False,
    ) -> dict[int, list[int]]:
        """Group cellids by cluster label for one algorithm.

        Args:
            algorithm: Algorithm whose labels are used for grouping.
            crop: When given, only rows of this crop are considered
                (case-insensitive), because the API clusters each crop separately.
            include_noise: Whether to keep the ``-1`` noise label produced by
                DBSCAN/HDBSCAN.

        Returns:
            Mapping ``{cluster_label: [cellid, ...]}`` with sorted labels.
        """
        column = algorithm.result_column
        grouped: dict[int, list[int]] = defaultdict(list)

        # Assign each row to its cluster, skipping rows that do not belong to
        # the requested crop, that lack the algorithm column, or that are noise.
        for row in self.rows:
            if crop is not None and row.crop.lower() != crop.lower():
                continue

            if column not in row.labels:
                continue

            label = row.labels[column]

            if label == -1 and not include_noise:
                continue

            grouped[label].append(row.cellid)

        return dict(sorted(grouped.items()))

    def cluster_statistics(
        self,
        algorithm: ClusteringAlgorithm = ClusteringAlgorithm.AGGLOMERATIVE,
        *,
        crop: str | None = None,
    ) -> dict[int, dict[str, dict[str, float]]]:
        """Summarize the indicator values of each cluster.

        Value columns are named ``<pref>_month<N>`` or ``<pref>_value``; all the
        values sharing a prefix are pooled per cluster, which gives the typical
        monthly level (or the single value) of each indicator in the cluster.

        Args:
            algorithm: Algorithm whose labels define the clusters.
            crop: When given, only rows of this crop are considered.

        Returns:
            ``{cluster_label: {indicator_prefix: {"mean", "min", "max", "n"}}}``,
            noise rows excluded.
        """
        column = algorithm.result_column
        pooled: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        # Pool the numeric values of every indicator per cluster.
        for row in self.rows:
            if crop is not None and row.crop.lower() != crop.lower():
                continue

            label = row.labels.get(column)

            if label is None or label == -1:
                continue

            for key, value in row.values.items():
                if value is None:
                    continue

                prefix = key.rsplit("_", 1)[0] if "_" in key else key
                pooled[label][prefix].append(float(value))

        statistics: dict[int, dict[str, dict[str, float]]] = {}

        # Reduce each pool to compact descriptive statistics.
        for label, indicators in sorted(pooled.items()):
            statistics[label] = {}

            for prefix, values in indicators.items():
                statistics[label][prefix] = {
                    "mean": round(sum(values) / len(values), 3),
                    "min": round(min(values), 3),
                    "max": round(max(values), 3),
                    "n": float(len(values)),
                }

        return statistics
