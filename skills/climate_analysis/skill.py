"""Skill that filters or clusters the Candidate list by climate indicators.

Workflow (business rules):

1. The ``cellid`` of every accession is computed when the list is loaded.
2. The model identifies the indicators to use (``list_indicators`` helps map a
   need such as "drought tolerance" to catalogue indicators); the chosen
   indicators are remembered in ``state.climate_indicators``.
3. Either ``filter`` (keep accessions meeting conditions; ``criteria_climate``
   records criteria and values, ``cluster_climate`` = 0) or ``cluster`` (no
   filtering; ``cluster_climate`` gets the cluster of each accession).
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from core.config import get_settings
from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState
from sdks.subsetting import SubsettingAuthError, SubsettingClient, SubsettingConnectionError, SubsettingError, SubsettingNoDataError, SubsettingRequestError
from skills.arguments import as_dict_list, as_int, as_list
from skills.base import Skill
from skills.climate_analysis.engine import (
    ClimateError,
    ResolvedIndicator,
    aggregate_indicator,
    describe_window,
    infer_crop,
    normalise_months,
    normalise_statistic,
    resolve_indicators,
    summarise_catalogue,
)
from skills.passport_filter.engine import OPERATORS, Condition, FilterError, evaluate_condition, normalise_operator

logger = get_logger(__name__)

# Columns of the Candidate list written by this skill.
CELLID_COLUMN = "cellid"
CRITERIA_COLUMN = "criteria_climate"
CLUSTER_COLUMN = "cluster_climate"
# Separator between criteria applied in successive turns.
_CRITERIA_SEPARATOR = " | "
# Business rule: the cluster range explored by the API.
MIN_CLUSTERS, MAX_CLUSTERS = 2, 10

ClientFactory = Callable[[], SubsettingClient]


def _default_client_factory() -> SubsettingClient:
    """Build a Subsetting client from the application settings."""
    return SubsettingClient.from_settings(get_settings())


class ClimateAnalysisSkill(Skill):
    """Filter or cluster the Candidate list using climate/agro-climatic indicators."""

    name = "climate_analysis"
    description = (
        "Work with climate and agro-climatic indicators of the accession collecting sites "
        "(Subsetting API). action='list_indicators' explores the catalogue (drought, heat, "
        "flooding, photoperiod, soil, crop-specific) and remembers the chosen indicators; "
        "action='filter' keeps the accessions whose indicator values satisfy conditions and "
        "records them in criteria_climate; action='cluster' groups the accessions (2-10 "
        "clusters) by the indicators and writes cluster_climate. Requires the 'cellid' "
        "computed when the list was loaded."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_indicators", "filter", "cluster"],
                "description": "What to do.",
            },
            "query": {
                "type": "string",
                "description": "list_indicators: free text such as 'drought', 'heat', 'soil', 'CDD'.",
            },
            "indicators": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Indicator prefixes/names/ids, e.g. ['CDD','t_rain','TX']. For "
                    "list_indicators: the indicators to remember for later calls. For cluster: "
                    "the variables to cluster by (defaults to the remembered ones)."
                ),
            },
            "conditions": {
                "type": "array",
                "description": (
                    "filter: [{\"indicator\": <pref>, \"operator\": <op>, \"value\": <v>, "
                    "\"statistic\": mean|sum|min|max (optional), \"months\": [first,last] (optional)}]. "
                    "Operators: equals, not_equals, gt, gte, lt, lte, between, in, not_in."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "indicator": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {},
                        "statistic": {"type": "string"},
                        "months": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["indicator", "operator"],
                },
            },
            "logic": {"type": "string", "enum": ["and", "or"], "description": "filter: combine conditions (default and)."},
            "months": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Inclusive month range [first, last] for monthly indicators (default [1, 12]).",
            },
            "period": {
                "type": "string",
                "description": "Indicator period: 'mean' (default), 'min', 'max', '1983-2016', '2021-2040', '2041-2060'...",
            },
            "crop": {
                "type": "string",
                "description": "Crop for crop-specific indicators (Beans, Maize, Rice...). Inferred from the list when omitted.",
            },
            "min_clusters": {"type": "integer", "description": "cluster: lower bound (default 2)."},
            "max_clusters": {"type": "integer", "description": "cluster: upper bound (default 10)."},
        },
        "required": ["action"],
    }

    def __init__(self, client_factory: ClientFactory | None = None) -> None:
        """Create the skill.

        Args:
            client_factory: Callable returning a ``SubsettingClient`` (lazy default from settings).
        """
        self._client_factory = client_factory or _default_client_factory
        self._client: SubsettingClient | None = None

    # ---------------------------------------------------------------- run
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        action: str = "",
        query: str | None = None,
        indicators: Any = None,
        conditions: Any = None,
        logic: str = "and",
        months: Any = None,
        period: str | None = None,
        crop: str | None = None,
        min_clusters: Any = None,
        max_clusters: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Dispatch the requested action, translating SDK errors into formal messages.

        Args:
            state: Session state (Candidate list, remembered indicators).
            paths: Session folders (unused).
            action: ``list_indicators``, ``filter`` or ``cluster``.
            query: Free text for ``list_indicators``.
            indicators: Indicator identifiers.
            conditions: Filter conditions.
            logic: ``and``/``or`` for the filter.
            months: Month range for monthly indicators.
            period: Indicator period label.
            crop: Crop for crop-specific indicators.
            min_clusters: Lower bound of the cluster range.
            max_clusters: Upper bound of the cluster range.
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result dictionary.
        """
        mode = (action or "").strip().lower()

        try:
            if mode == "list_indicators":
                return self._list_indicators(state, query, indicators, crop)

            # Data actions need a loaded list with cell ids.
            guard = self._check_data(state)
            if guard is not None:
                return guard

            if mode == "filter":
                return self._filter(state, conditions, logic, months, period, crop)
            if mode == "cluster":
                return self._cluster(state, indicators, months, period, crop, min_clusters, max_clusters)

            return self.error(f"Unknown action '{action}'. Use 'list_indicators', 'filter' or 'cluster'.")
        except (ClimateError, FilterError) as exc:
            return self.error(str(exc))
        except SubsettingNoDataError as exc:
            return self.error(
                f"{exc} Suggest other indicators or check that the accessions are georeferenced in "
                "areas covered by the climate database."
            )
        except SubsettingAuthError:
            return self.error(
                "The climate service rejected the configured credentials (the access token may have "
                "expired). The administrator must renew SUBSETTING_ACCESS_TOKEN."
            )
        except SubsettingConnectionError:
            return self.error("The climate service could not be reached right now. Suggest retrying later.")
        except (SubsettingRequestError, SubsettingError) as exc:
            return self.error(f"The climate service returned an error: {exc}")

    # ------------------------------------------------------------ helpers
    def _get_client(self) -> SubsettingClient:
        """Return the Subsetting client, creating it on first use."""
        if self._client is None:
            self._client = self._client_factory()

        return self._client

    def _check_data(self, state: SessionState) -> dict[str, Any] | None:
        """Validate that a list with ``cellid`` is loaded.

        Returns:
            An error result when the data is missing, ``None`` otherwise.
        """
        if not state.has_data or state.candidate_list is None:
            return self.error("No accession list is loaded yet. Load accessions (file or Genesys) first.")

        if CELLID_COLUMN not in state.candidate_list.columns:
            return self.error(
                "The Candidate list has no 'cellid' column: the coordinates were not detected when "
                "loading. Reload the list indicating latitude_column and longitude_column."
            )

        if state.candidate_list[CELLID_COLUMN].notna().sum() == 0:
            return self.error("No accession of the Candidate list has a valid cellid (missing coordinates).")

        return None

    def _resolve_crop(self, state: SessionState, crop: str | None) -> str | None:
        """Return the crop to use for crop-specific indicators.

        Args:
            state: Session state (used to infer the crop from the list and to cache it).
            crop: Crop given by the model.

        Returns:
            Crop name, or ``None`` when it cannot be determined.
        """
        # An explicit crop always wins and is remembered.
        if crop and str(crop).strip():
            state.extras["climate_crop"] = str(crop).strip()
            return state.extras["climate_crop"]

        if state.extras.get("climate_crop"):
            return state.extras["climate_crop"]

        inferred = infer_crop(state.candidate_list, self._get_client().catalogue_crops()) if state.candidate_list is not None else None

        if inferred:
            state.extras["climate_crop"] = inferred

        return inferred

    def _cells(self, state: SessionState) -> list[int]:
        """Distinct valid cell ids of the Candidate list."""
        return sorted({int(cell) for cell in state.candidate_list[CELLID_COLUMN].dropna()})

    @staticmethod
    def _append_criteria(existing: pd.Series, texts: pd.Series) -> pd.Series:
        """Chain new criteria texts to the existing ``criteria_climate`` values."""
        previous = existing.astype("string").fillna("").str.strip()
        return previous.where(previous == "", previous + _CRITERIA_SEPARATOR).astype(str) + texts.astype(str)

    # --------------------------------------------------- list_indicators
    def _list_indicators(self, state: SessionState, query: str | None, indicators: Any, crop: str | None) -> dict[str, Any]:
        """Explore the catalogue and remember the indicators the model selects.

        Args:
            state: Session state.
            query: Free text filter.
            indicators: Indicators to remember for later calls.
            crop: Crop to resolve crop-specific indicators.

        Returns:
            Tool result with catalogue entries and the remembered indicators.
        """
        client = self._get_client()
        catalogue = summarise_catalogue(client.list_indicators(), query)
        remembered: list[str] = []

        selected = as_list(indicators)

        # Remember the selection so filter/cluster can run without repeating it.
        if selected:
            resolved = resolve_indicators(client, [str(item) for item in selected], self._resolve_crop(state, crop) if state.has_data else crop)
            state.climate_indicators = [item.indicator.id for item in resolved]
            remembered = [item.label for item in resolved]
            state.log_activity(self.name, f"Selected climate indicators: {remembered}")

        message = f"{len(catalogue)} indicator(s) in the catalogue" + (f" matching '{query}'" if query else "")

        if remembered:
            message += f". Remembered for the next steps: {remembered}"

        return self.ok(
            message + ".",
            indicators=catalogue,
            remembered=remembered,
            crops=client.catalogue_crops(),
            periods=sorted({p.period for p in client.list_indicator_periods()}),
        )

    # -------------------------------------------------------------- filter
    def _filter(
        self,
        state: SessionState,
        raw_conditions: Any,
        logic: str,
        months: Any,
        period: str | None,
        crop: str | None,
    ) -> dict[str, Any]:
        """Keep the accessions whose indicator values satisfy the conditions.

        Args:
            state: Session state.
            raw_conditions: Conditions sent by the model.
            logic: ``and``/``or``.
            months: Default month range for conditions without their own.
            period: Indicator period label.
            crop: Crop for crop-specific indicators.

        Returns:
            Tool result with counts and criteria.
        """
        conditions = as_dict_list(raw_conditions)

        if not conditions:
            raise ClimateError("action='filter' needs at least one condition {indicator, operator, value}.")

        client = self._get_client()
        crop_name = self._resolve_crop(state, crop)
        default_months = normalise_months(months)
        names = [str(item.get("indicator") or item.get("pref") or item.get("name") or "") for item in conditions]

        if any(not name for name in names):
            raise ClimateError("Every condition needs an 'indicator'.")

        resolved = resolve_indicators(client, names, crop_name)
        by_name = {item.requested_as: item for item in resolved}
        cells = self._cells(state)
        data = client.get_indicators_data(cells, [item.indicator.id for item in resolved], period=period)
        state.extras["climate_data"] = data

        # One aggregated column per condition, evaluated with the passport operators.
        wide = pd.DataFrame(index=pd.Index(cells, name=CELLID_COLUMN))
        parsed: list[tuple[Condition, str]] = []
        prefs: dict[str, str] = {}  # condition column -> indicator prefix (short label for values)

        for index, (raw, name) in enumerate(zip(conditions, names)):
            item = by_name.get(name) or resolved[min(index, len(resolved) - 1)]
            statistic = normalise_statistic(raw.get("statistic"), item.label)
            window = normalise_months(raw.get("months")) if raw.get("months") not in (None, "", []) else default_months
            column = f"{item.label}_{index}"
            wide[column] = aggregate_indicator(data, item.indicator, statistic, window).reindex(wide.index)

            operator = normalise_operator(raw.get("operator") or raw.get("op"))
            value = raw.get("value", raw.get("values"))

            if operator not in ("is_null", "not_null") and value is None:
                raise ClimateError(f"Condition on '{item.label}' needs a value.")

            if operator in ("in", "not_in", "between") and not isinstance(value, list):
                value = as_list(value) or []

            condition = Condition(column=column, operator=operator, value=value)
            label = f"{item.label} {describe_window(statistic, window)}" if (item.indicator.indicator_type or "").lower() in ("generic", "specific") else item.label
            parsed.append((condition, label))
            prefs[column] = item.label

        # Evaluate every condition on the per-cell table and combine them.
        masks = [evaluate_condition(wide, condition).astype(bool) for condition, _ in parsed]
        combined = masks[0]

        for mask in masks[1:]:
            combined = (combined & mask) if (logic or "and").lower() == "and" else (combined | mask)

        has_data = wide.notna().any(axis=1)
        passing_cells = set(wide.index[combined & has_data])
        criteria_text = (" AND " if (logic or "and").lower() == "and" else " OR ").join(
            f"{label} {OPERATORS[c.operator]} {c.value if c.value is not None else ''}".strip() for c, label in parsed
        )

        candidate = state.candidate_list
        before = len(candidate)
        cell_series = candidate[CELLID_COLUMN]
        keep = cell_series.notna() & cell_series.astype("Int64").isin(list(passing_cells))
        without_cell = int(cell_series.isna().sum())
        without_data = int((cell_series.notna() & ~cell_series.astype("Int64").isin(list(wide.index[has_data]))).sum())
        after = int(keep.sum())

        # Nothing passes: keep the list and explain with the value ranges.
        if after == 0:
            ranges = {label: {"min": float(wide[c.column].min()), "max": float(wide[c.column].max())} for c, label in parsed if wide[c.column].notna().any()}
            state.log_activity(self.name, f"Climate filter '{criteria_text}' matched no accessions; list unchanged")
            return self.error(
                f"No accession satisfies '{criteria_text}' (period {period or client.default_period}). "
                f"The Candidate list was left unchanged ({before} accessions). Observed ranges: {ranges}.",
                criteria=criteria_text,
                accessions_before=before,
                without_cellid=without_cell,
                without_climate_data=without_data,
                value_ranges=ranges,
            )

        filtered = candidate.loc[keep].copy()

        # Per-row text: criteria plus the actual values of that accession's cell.
        def row_text(cell: Any) -> str:
            values = ", ".join(f"{prefs[c.column]}={_fmt(wide.at[int(cell), c.column])}" for c, _ in parsed)
            return f"{criteria_text} [{values}]"

        texts = filtered[CELLID_COLUMN].map(row_text)
        filtered[CRITERIA_COLUMN] = self._append_criteria(filtered[CRITERIA_COLUMN], texts)
        filtered[CLUSTER_COLUMN] = 0
        state.update_candidate_list(filtered)
        state.climate_indicators = [item.indicator.id for item in resolved]

        description = (
            f"Filtered Candidate list by climate '{criteria_text}' (period {period or client.default_period}): "
            f"{before} -> {after} accessions ({without_cell} without cellid, {without_data} without climate data)"
        )
        state.log_activity(self.name, description)

        return self.ok(
            f"{description}. Criteria and values recorded in '{CRITERIA_COLUMN}'; '{CLUSTER_COLUMN}' set to 0.",
            criteria=criteria_text,
            accessions_before=before,
            accessions_after=after,
            without_cellid=without_cell,
            without_climate_data=without_data,
            indicators=[item.label for item in resolved],
            period=period or client.default_period,
            crop=crop_name,
        )

    # ------------------------------------------------------------- cluster
    def _cluster(
        self,
        state: SessionState,
        indicators: Any,
        months: Any,
        period: str | None,
        crop: str | None,
        min_clusters: Any,
        max_clusters: Any,
    ) -> dict[str, Any]:
        """Group the accessions by climate indicators without filtering.

        Args:
            state: Session state.
            indicators: Indicator identifiers (defaults to the remembered ones).
            months: Month range used by the API for monthly indicators.
            period: Indicator period label.
            crop: Crop for crop-specific indicators and for the API ``cellid_list``.
            min_clusters: Lower bound (default 2).
            max_clusters: Upper bound (default 10).

        Returns:
            Tool result with cluster sizes and the API summary.
        """
        client = self._get_client()
        crop_name = self._resolve_crop(state, crop)
        names = [str(item) for item in (as_list(indicators) or [])]

        # Fall back to the indicators remembered from list_indicators/filter.
        if not names and state.climate_indicators:
            names = list(state.climate_indicators)

        if not names:
            raise ClimateError(
                "action='cluster' needs the indicators to cluster by (argument 'indicators'), or a "
                "previous action='list_indicators' that remembered them."
            )

        resolved = resolve_indicators(client, names, crop_name)
        window = normalise_months(months)
        low = as_int(min_clusters) or MIN_CLUSTERS
        high = as_int(max_clusters) or MAX_CLUSTERS
        low, high = max(MIN_CLUSTERS, low), min(MAX_CLUSTERS, max(high, low))
        cells = self._cells(state)

        result = client.generate_clusters(
            cells,
            [item.indicator.id for item in resolved],
            crop=crop_name or "generic",
            min_clusters=low,
            max_clusters=high,
            months=window,
            period=period,
        )

        candidate = state.candidate_list.copy()
        labels = [item.label for item in resolved]
        cell_series = candidate[CELLID_COLUMN]
        assigned = cell_series.map(lambda cell: result.assignments.get(int(cell)) if pd.notna(cell) else None)
        has_cluster = assigned.notna()

        # Per-row text: indicators used and the values of that cell in the analysis.
        def row_text(cell: Any) -> str:
            values = result.indicator_values.get(int(cell), {}) if pd.notna(cell) else {}
            summary = _summarise_values(values, labels)
            return f"cluster by {', '.join(labels)} (period {period or client.default_period}, months {window[0]}-{window[1]}) [{summary}]"

        texts = cell_series.where(has_cluster).map(lambda cell: row_text(cell) if pd.notna(cell) else "")
        candidate.loc[has_cluster, CRITERIA_COLUMN] = self._append_criteria(
            candidate.loc[has_cluster, CRITERIA_COLUMN], texts[has_cluster]
        )
        candidate.loc[has_cluster, CLUSTER_COLUMN] = assigned[has_cluster].astype(int)
        state.update_candidate_list(candidate)
        state.climate_indicators = [item.indicator.id for item in resolved]

        sizes = assigned[has_cluster].astype(int).value_counts().sort_index()
        cluster_sizes = {int(k): int(v) for k, v in sizes.items()}
        unassigned = int((~has_cluster).sum())

        description = (
            f"Clustered Candidate list by climate indicators {labels} into {result.cluster_count} clusters "
            f"(period {period or client.default_period}, months {window[0]}-{window[1]}); "
            f"{int(has_cluster.sum())} accessions assigned, {unassigned} without cluster"
        )
        state.log_activity(self.name, description)

        return self.ok(
            f"{description}. Cluster of each accession written in '{CLUSTER_COLUMN}'; indicators and values in '{CRITERIA_COLUMN}'.",
            clusters=result.cluster_count,
            cluster_sizes=cluster_sizes,
            assigned=int(has_cluster.sum()),
            unassigned=unassigned,
            indicators=labels,
            period=period or client.default_period,
            months=list(window),
            crop=crop_name,
            summary=result.summary,
        )


def _fmt(value: Any) -> str:
    """Format a value for criteria texts (2 decimals for floats)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _summarise_values(values: dict[str, Any], labels: list[str]) -> str:
    """Reduce the per-cell analysis columns (``pref_monthN``/``pref_value``) to one number per indicator."""
    parts: list[str] = []

    for label in labels:
        numbers = [v for k, v in values.items() if str(k).lower().startswith(label.lower() + "_") and isinstance(v, (int, float)) and not pd.isna(v)]
        categories = [v for k, v in values.items() if str(k).lower() == f"{label.lower()}_category"]

        if numbers:
            parts.append(f"{label}={sum(numbers) / len(numbers):.2f}")
        elif categories:
            parts.append(f"{label}={categories[0]}")

    return ", ".join(parts)
