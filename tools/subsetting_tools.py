"""Tools backed by the Subsetting API: climate indicators and clustering.

These tools operate exclusively on the cellids of the accessions already in the
selection, which is how the business order (climate last) is enforced by data.
"""

from __future__ import annotations

from typing import Any

from subsetting_sdk.exceptions import IndicatorNotFoundError, IndicatorPeriodNotFoundError
from subsetting_sdk.models import (
    ClusteringAlgorithm,
    ClusteringHyperparameters,
    ClusterRequest,
    IndicatorFilter,
    MonthWindow,
)
from tools.accession_context import Stage
from tools.services import ToolServices


def _month_window(month_start: int | None, month_end: int | None) -> MonthWindow:
    """Build a month window from optional tool arguments (full year by default).

    Args:
        month_start: First month (1-12) or ``None``.
        month_end: Last month (1-12) or ``None``.
    """
    return MonthWindow(start=int(month_start or 1), end=int(month_end or 12))


async def list_climate_indicators(
    services: ToolServices, *, category: str | None = None, crop_code: str | None = None
) -> dict[str, Any]:
    """List the available climate/soil indicators, optionally by stress category.

    Args:
        services: Shared services.
        category: Stress category (drought, flood, heat, photoperiod, soil, crop specific).
        crop_code: Include crop-specific indicators of this crop only.
    """
    catalog = await services.get_catalog()

    try:
        indicators = catalog.list_by_category(category) if category else list(catalog.indicators)

    except IndicatorNotFoundError as exc:
        return {"error": str(exc), "categories": catalog.category_names()}

    listed = []

    # Hide crop-specific indicators of other crops to keep the list short.
    for indicator in indicators:
        if indicator.indicator_type.value == "specific" and crop_code:
            if indicator.crop.lower() != crop_code.lower():
                continue

        listed.append(
            {
                "name": indicator.name,
                "code": indicator.pref,
                "category": indicator.category,
                "type": indicator.indicator_type.value,
                "crop": indicator.crop,
                "unit": indicator.unit,
            }
        )

    return {"categories": catalog.category_names(), "count": len(listed), "indicators": listed}


async def _build_filters(
    services: ToolServices,
    indicators: list[str],
    window: MonthWindow,
) -> tuple[list[IndicatorFilter], str | None]:
    """Resolve indicator names into API filters for the crops of the selection.

    Args:
        services: Shared services.
        indicators: Indicator names, codes or ids.
        window: Month window applied to monthly indicators.

    Returns:
        The filters and an error message when a name cannot be resolved.
    """
    catalog = await services.get_catalog()
    crops = services.context.crops()
    filters: list[IndicatorFilter] = []

    # Resolve each indicator; crop-specific ones are tried per selected crop.
    for name in indicators:
        try:
            crop_hint = crops[0] if len(crops) == 1 else None
            filters.append(catalog.build_filter(name, crop=crop_hint, months=window))

        except (IndicatorNotFoundError, IndicatorPeriodNotFoundError) as exc:
            return [], str(exc)

    return filters, None


async def cluster_selection_by_climate(
    services: ToolServices,
    *,
    indicators: list[str],
    month_start: int | None = None,
    month_end: int | None = None,
    algorithm: str = "agglomerative",
    n_clusters: int = 5,
) -> dict[str, Any]:
    """Group the selected accessions into climate clusters.

    Args:
        services: Shared services.
        indicators: Indicator names or codes to characterize the sites.
        month_start: First month of the window (1-12); default January.
        month_end: Last month of the window (1-12); default December.
        algorithm: ``agglomerative`` (default), ``dbscan`` or ``hdbscan``.
        n_clusters: Maximum number of clusters for agglomerative clustering (2-5 typical).
    """
    context = services.context

    if context.is_empty:
        return {"error": "No accessions selected yet. Run select_accessions first."}

    cellid_list = context.cellids_by_crop()

    if not cellid_list:
        return {"error": "None of the selected accessions has a grid cell (no coordinates)."}

    if not indicators:
        return {"error": "Provide at least one indicator."}

    try:
        chosen = ClusteringAlgorithm(algorithm.lower())

    except ValueError:
        return {"error": f"Unknown algorithm '{algorithm}'. Use agglomerative, dbscan or hdbscan."}

    window = _month_window(month_start, month_end)
    filters, error = await _build_filters(services, indicators, window)

    if error:
        return {"error": error}

    request = ClusterRequest(
        cellid_list=cellid_list,
        filters=filters,
        months=window,
        algorithms=[chosen],
        hyperparameters=ClusteringHyperparameters(n_clusters=max(2, int(n_clusters))),
    )
    result = await services.subsetting.cluster(request)

    # An empty result means the API could not run the analysis (too few cells,
    # missing indicator data...). Report it instead of pretending there are no clusters.
    if not result.rows:
        return {
            "error": (
                "The clustering returned no data. Check that the selection has enough "
                "distinct sites (cells) and that the indicators have data for them."
            )
        }

    clusters = result.clusters(chosen, include_noise=False)
    statistics = result.cluster_statistics(chosen)
    context.set_clusters(clusters, chosen.value)

    cluster_report = []

    # Describe each cluster with its size, examples and indicator statistics so
    # the model can compare clusters against the user's thresholds.
    for label, cells in clusters.items():
        members = context.accessions_in_cells(cells)
        cluster_report.append(
            {
                "cluster": label,
                "sites": len(cells),
                "accessions": len(members),
                "countries": _top_countries(members),
                "examples": [m.label() for m in members[:3]],
                "indicators": statistics.get(label, {}),
            }
        )

    return {
        "algorithm": chosen.value,
        "indicators": [f.name for f in filters],
        "months": [window.start, window.end],
        "clusters": cluster_report,
        "hint": (
            "Indicator statistics are per cluster (mean/min/max of the monthly values in the "
            "window). Use pick_cluster with a cluster label to keep only its accessions."
        ),
    }


def _top_countries(members: list[Any], limit: int = 3) -> list[str]:
    """Return the most frequent countries of origin among cluster members.

    Args:
        members: Accession records of the cluster.
        limit: Number of countries to return.
    """
    counts: dict[str, int] = {}

    for member in members:
        key = member.country_code or "unknown"
        counts[key] = counts.get(key, 0) + 1

    ranked = sorted(counts.items(), key=lambda item: -item[1])

    return [f"{code} ({count})" for code, count in ranked[:limit]]


async def pick_cluster(services: ToolServices, *, cluster: int | str) -> dict[str, Any]:
    """Keep only the accessions of one climate cluster.

    Args:
        services: Shared services.
        cluster: Cluster label from ``cluster_selection_by_climate``.
    """
    context = services.context
    label = str(cluster)

    if not context.clusters:
        return {"error": "No clustering has been run. Call cluster_selection_by_climate first."}

    if label not in context.clusters:
        return {"error": f"Unknown cluster '{label}'. Available: {list(context.clusters)}."}

    members = context.accessions_in_cells(context.clusters[label])
    context.keep(
        [m.uuid for m in members],
        stage=Stage.CLIMATE,
        description=f"Kept climate cluster {label} ({context.cluster_algorithm})",
    )
    context.selected_cluster = label

    return {"cluster": label, "kept": len(members), "summary": context.summary()}


async def describe_selection(services: ToolServices, *, sample: int = 10) -> dict[str, Any]:
    """Describe the current selection: stage, counts, steps and example accessions.

    Args:
        services: Shared services.
        sample: Number of example accessions to include.
    """
    context = services.context
    summary = context.summary(sample=max(1, min(int(sample), 50)))
    summary["accessions"] = [
        {
            "accession_number": r.accession_number,
            "institute": r.institute_code,
            "crop": r.crop,
            "taxon": r.taxon_name,
            "origin": r.country_code,
            "doi": r.doi,
        }
        for r in context.records()[: max(1, min(int(sample), 50))]
    ]

    return summary
