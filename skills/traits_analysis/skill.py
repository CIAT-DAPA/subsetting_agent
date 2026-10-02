"""Skill that annotates and groups the Candidate list by trait (phenotypic) data.

Business rules:

* Traits never remove accessions from the Candidate list: the skill only adds
  trait columns and fills ``cluster_traits`` / ``criteria_traits``.
* Local mode: trait values are columns of the uploaded file (``detect`` lists
  the candidates). Genesys mode (or a local list with a ``UUID`` column):
  ``fetch`` downloads characterization/evaluation data from Genesys datasets
  and adds ``trait_<columnName>`` columns (mean per accession for numeric
  descriptors, mode for categorical ones).
* ``group``: one trait with a condition -> ``cluster_traits`` 1 (meets) / 0
  (does not meet); several traits -> groups by combination (terciles for
  numeric traits, categories for categorical ones). ``criteria_traits`` keeps
  the criteria and the values of every accession.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd

from core.config import get_settings
from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState
from sdks.genesys import AccessionFilter, GenesysAuthError, GenesysClient, GenesysConnectionError, GenesysError, GenesysRequestError, InstituteFilter, observations_to_dataframe
from sdks.genesys.traits import Descriptor
from skills.arguments import as_dict_list, as_int, as_list
from skills.base import Skill
from skills.passport_filter.engine import OPERATORS, Condition, FilterError, evaluate_condition, normalise_operator
from skills.traits_analysis.engine import (
    TRAIT_PREFIX,
    TraitsError,
    aggregate_observations,
    build_groups,
    describe_trait_column,
    detect_trait_columns,
    resolve_trait_column,
    select_descriptors,
    to_numeric_series,
)

logger = get_logger(__name__)

# Columns of the Candidate list written by this skill.
CRITERIA_COLUMN = "criteria_traits"
CLUSTER_COLUMN = "cluster_traits"
_CRITERIA_SEPARATOR = " | "
# Accession identifier columns (Genesys flattening and common local spellings).
UUID_COLUMNS = ("UUID", "uuid", "accession_uuid")
NUMBER_COLUMNS = ("ACCENUMB", "accessionNumber", "accession_number", "acceNumb")
INSTITUTE_COLUMNS = ("INSTCODE", "instituteCode", "institute_code", "instCode")
# Download limits: descriptors when nothing is selected, uuids per request, rows per dataset.
DEFAULT_MAX_DESCRIPTORS = 10
UUID_CHUNK = 200
MAX_ROWS_PER_DATASET = 50000

ClientFactory = Callable[[], GenesysClient]


def _default_client_factory() -> GenesysClient:
    """Build a Genesys client from the application settings."""
    return GenesysClient.from_settings(get_settings())


class TraitsAnalysisSkill(Skill):
    """Annotate and group the Candidate list by trait data (local columns or Genesys datasets)."""

    name = "traits_analysis"
    description = (
        "Work with TRAITS (phenotypic / characterization / evaluation data) of the Candidate "
        "list without removing accessions. action='detect' lists the trait columns present in "
        "an uploaded file; action='fetch' downloads trait data from Genesys datasets for the "
        "accessions of the list (optionally filtered by query='iron zinc' or traits=[...]) and "
        "adds trait_<name> columns; action='group' creates subsets: ONE trait with a condition "
        "(Fe.Mean >= 60) -> cluster_traits 1/0 (meets / does not meet), SEVERAL traits -> "
        "groups by combination (terciles low/medium/high for numeric traits, categories for "
        "categorical ones). criteria_traits records the criteria and values."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["detect", "fetch", "group"], "description": "What to do."},
            "query": {
                "type": "string",
                "description": "fetch: free text to choose descriptors (e.g. 'iron zinc', 'seed color', 'yield').",
            },
            "traits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Trait column names (local columns or Genesys descriptor names such as 'Fe.Mean'). "
                "fetch: descriptors to download; group: columns to combine into groups.",
            },
            "conditions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "trait": {"type": "string"},
                        "operator": {"type": "string", "enum": list(OPERATORS)},
                        "value": {"description": "Scalar or list (for in/between)."},
                    },
                    "required": ["trait", "operator"],
                },
                "description": "group: conditions such as [{\"trait\": \"Fe.Mean\", \"operator\": \"gte\", \"value\": 60}].",
            },
            "max_descriptors": {"type": "integer", "description": "fetch: maximum descriptors downloaded when none is selected (default 10)."},
        },
        "required": ["action"],
    }

    def __init__(self, client_factory: ClientFactory | None = None) -> None:
        """Create the skill.

        Args:
            client_factory: Callable returning a ``GenesysClient`` (lazy default from settings).
        """
        self._client_factory = client_factory or _default_client_factory
        self._client: GenesysClient | None = None

    # ---------------------------------------------------------------- run
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        action: str = "",
        query: str | None = None,
        traits: Any = None,
        conditions: Any = None,
        max_descriptors: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Dispatch the requested action, translating errors into formal messages.

        Args:
            state: Session state (Candidate list).
            paths: Session folders (unused).
            action: ``detect``, ``fetch`` or ``group``.
            query: Free text to pick descriptors.
            traits: Trait names.
            conditions: Grouping conditions.
            max_descriptors: Cap of descriptors downloaded without selection.
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result dictionary.
        """
        mode = (action or "").strip().lower()

        # Every action needs a loaded list.
        if not state.has_data or state.candidate_list is None:
            return self.error("No accession list is loaded yet. Load accessions (file or Genesys) first.")

        try:
            if mode == "detect":
                return self._detect(state)
            if mode == "fetch":
                return self._fetch(state, query, as_list(traits), as_int(max_descriptors))
            if mode == "group":
                return self._group(state, as_list(traits), as_dict_list(conditions))

            return self.error(f"Unknown action '{action}'. Use 'detect', 'fetch' or 'group'.")
        except (TraitsError, FilterError) as exc:
            return self.error(str(exc))
        except GenesysAuthError:
            return self.error("Genesys rejected the configured credentials. The administrator must check GENESYS_API_TOKEN.")
        except GenesysConnectionError:
            return self.error("Genesys could not be reached right now. Suggest retrying later.")
        except (GenesysRequestError, GenesysError) as exc:
            return self.error(f"Genesys returned an error: {exc}")

    # ------------------------------------------------------------ helpers
    def _get_client(self) -> GenesysClient:
        """Return the Genesys client, creating it on first use."""
        if self._client is None:
            self._client = self._client_factory()

        return self._client

    @staticmethod
    def _find_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
        """Return the first existing column among the candidates (case-insensitive)."""
        lowered = {str(column).lower(): str(column) for column in frame.columns}

        for candidate in candidates:
            if candidate.lower() in lowered:
                return lowered[candidate.lower()]

        return None

    @staticmethod
    def _append_criteria(existing: pd.Series, texts: pd.Series) -> pd.Series:
        """Chain new criteria texts to the existing ``criteria_traits`` values."""
        previous = existing.astype("string").fillna("").str.strip()
        return previous.where(previous == "", previous + _CRITERIA_SEPARATOR).astype(str) + texts.astype(str)

    # -------------------------------------------------------------- detect
    def _detect(self, state: SessionState) -> dict[str, Any]:
        """List the columns of the Candidate list that look like traits."""
        found = detect_trait_columns(state.candidate_list)

        if not found:
            return self.ok(
                "No trait columns were found in the list (only passport data). In Genesys mode, or if the "
                "list has a UUID column, use action='fetch' to download traits from Genesys datasets.",
                trait_columns=[],
                subsets_created=False,
                next_step="traits_analysis action='fetch' to download traits, or ask the user which columns are traits",
            )

        state.log_activity(self.name, f"Detected {len(found)} trait column(s): {', '.join(item.column for item in found)}")
        return self.ok(
            f"{len(found)} trait column(s) found. NO SUBSET HAS BEEN CREATED YET: use action='group' with these columns.",
            trait_columns=[item.to_dict() for item in found],
            subsets_created=False,
            next_step="traits_analysis action='group' with traits=[...] or conditions=[...]",
        )

    # --------------------------------------------------------------- fetch
    def _accession_filter(self, frame: pd.DataFrame) -> tuple[AccessionFilter, str, str]:
        """Build the Genesys filter that selects the accessions of the Candidate list.

        Args:
            frame: Candidate list.

        Returns:
            ``(filter, key_column, key_kind)`` where ``key_kind`` is ``uuid`` or ``accessionNumber``.

        Raises:
            TraitsError: When the list has neither UUID nor accession numbers.
        """
        uuid_column = self._find_column(frame, UUID_COLUMNS)

        # UUIDs are the exact key used by Genesys; accession numbers (+ institute) are the fallback.
        if uuid_column is not None and frame[uuid_column].notna().any():
            uuids = frame[uuid_column].dropna().astype(str).unique().tolist()
            return AccessionFilter(uuid=uuids), uuid_column, "uuid"

        number_column = self._find_column(frame, NUMBER_COLUMNS)
        if number_column is None or frame[number_column].notna().sum() == 0:
            raise TraitsError(
                "The list has no UUID nor accession number column, so its accessions cannot be matched in Genesys. "
                "Load the list from Genesys or include UUID / ACCENUMB (+ INSTCODE) columns."
            )

        numbers = frame[number_column].dropna().astype(str).unique().tolist()
        institute_column = self._find_column(frame, INSTITUTE_COLUMNS)
        institutes = frame[institute_column].dropna().astype(str).unique().tolist() if institute_column else []
        institute = InstituteFilter(code=institutes) if institutes else None
        return AccessionFilter(accession_numbers=numbers, institute=institute), number_column, "accessionNumber"

    def _fetch(self, state: SessionState, query: str | None, trait_names: list[Any] | None, max_descriptors: int | None) -> dict[str, Any]:
        """Download trait data from Genesys and add ``trait_*`` columns to the Candidate list."""
        frame = state.candidate_list
        client = self._get_client()
        accession_filter, key_column, key_kind = self._accession_filter(frame)
        keys = frame[key_column].dropna().astype(str).unique().tolist()
        # UUID lists are chunked by the SDK; accession-number filters go in one body.
        dataset_uuids = client.find_datasets_for_uuids(keys) if key_kind == "uuid" else client.find_datasets(accession_filter)

        if not dataset_uuids:
            return self.ok(
                "Genesys holds no trait datasets for the accessions of the Candidate list.",
                datasets=0, subsets_created=False, next_step="inform the user; traits cannot be downloaded for this list",
            )

        # Collect the descriptors of every dataset (de-duplicated by uuid) and select the requested ones.
        descriptors_by_dataset: dict[str, list[Descriptor]] = {}
        catalogue: dict[str, Descriptor] = {}
        titles: dict[str, str] = {}
        for dataset_uuid in dataset_uuids:
            summary = client.get_dataset(dataset_uuid)
            titles[dataset_uuid] = summary.title or dataset_uuid
            descriptors = client.list_dataset_descriptors(dataset_uuid)
            descriptors_by_dataset[dataset_uuid] = descriptors
            for descriptor in descriptors:
                catalogue.setdefault(descriptor.uuid, descriptor)

        names = [str(item) for item in trait_names or [] if str(item).strip()]
        selected = select_descriptors(list(catalogue.values()), query, names)

        if not selected:
            return self.error(
                f"No descriptor matches '{query or ', '.join(names)}'. Available descriptors: "
                + "; ".join(f"{d.label} ({d.title}, {d.data_type}, {d.uom or '-'})" for d in list(catalogue.values())[:40]),
                available=[d.label for d in catalogue.values()],
            )

        # Without an explicit selection, cap the download to keep it fast.
        cap = max_descriptors or DEFAULT_MAX_DESCRIPTORS
        if not names and not (query and str(query).strip()) and len(selected) > cap:
            selected = selected[:cap]

        selected_uuids = {descriptor.uuid for descriptor in selected}
        rows: list[dict[str, Any]] = []

        # Download dataset by dataset, in chunks of accessions, only the selected descriptors it holds.
        for dataset_uuid, descriptors in descriptors_by_dataset.items():
            fields = [descriptor.uuid for descriptor in descriptors if descriptor.uuid in selected_uuids]
            if not fields:
                continue

            for start in range(0, len(keys), UUID_CHUNK):
                chunk = keys[start : start + UUID_CHUNK]
                chunk_filter = AccessionFilter(uuid=chunk) if key_kind == "uuid" else AccessionFilter(accession_numbers=chunk, institute=accession_filter.institute)
                rows.extend(client.iter_dataset_data([dataset_uuid], fields, chunk_filter, max_rows=MAX_ROWS_PER_DATASET))

        observations = observations_to_dataframe(rows, selected)
        aggregated = aggregate_observations(observations, selected)
        merged = self._merge_traits(frame, aggregated, key_column, key_kind)

        trait_columns = [f"{TRAIT_PREFIX}{descriptor.label}" for descriptor in selected if f"{TRAIT_PREFIX}{descriptor.label}" in merged.columns]
        with_data = int(merged[trait_columns].notna().any(axis=1).sum()) if trait_columns else 0
        state.update_candidate_list(merged)
        state.extras["trait_descriptors"] = {f"{TRAIT_PREFIX}{d.label}": {"title": d.title, "type": d.data_type, "uom": d.uom} for d in selected}
        state.log_activity(
            self.name,
            f"Downloaded {len(selected)} trait(s) from {len(dataset_uuids)} Genesys dataset(s): {with_data} of {len(merged)} accessions have data",
        )

        coverage = {column: int(merged[column].notna().sum()) for column in trait_columns}
        summaries = [describe_trait_column(merged, column).to_dict() for column in trait_columns]
        return self.ok(
            f"{len(selected)} trait column(s) added to the Candidate list ({', '.join(trait_columns)}); "
            f"{with_data} of {len(merged)} accessions have trait data, {len(merged) - with_data} have none. "
            "NO SUBSET HAS BEEN CREATED YET: use action='group' to create subsets with these traits.",
            datasets=[{"uuid": uuid, "title": titles[uuid]} for uuid in dataset_uuids],
            traits=[{"column": f"{TRAIT_PREFIX}{d.label}", "title": d.title, "type": d.data_type, "uom": d.uom} for d in selected],
            coverage=coverage,
            trait_columns=summaries,
            accessions_with_data=with_data,
            accessions_without_data=len(merged) - with_data,
            subsets_created=False,
            next_step="traits_analysis action='group' with conditions=[...] (one trait) or traits=[...] (several traits)",
        )

    @staticmethod
    def _merge_traits(frame: pd.DataFrame, aggregated: pd.DataFrame, key_column: str, key_kind: str) -> pd.DataFrame:
        """Attach the aggregated ``trait_*`` columns to the Candidate list by accession key.

        Args:
            frame: Candidate list.
            aggregated: One row per accession with ``uuid``/``accessionNumber`` and trait columns.
            key_column: Column of ``frame`` holding the key.
            key_kind: ``uuid`` or ``accessionNumber``.

        Returns:
            Candidate list with the trait columns (existing ``trait_*`` columns are refreshed).
        """
        result = frame.copy()
        trait_columns = [column for column in aggregated.columns if str(column).startswith(TRAIT_PREFIX)]

        if not trait_columns:
            return result

        lookup = aggregated.dropna(subset=[key_kind]).drop_duplicates(subset=[key_kind]).set_index(aggregated[key_kind].astype(str))
        keys = result[key_column].astype("string").fillna("").astype(str)

        # Columns downloaded again replace the previous values.
        for column in trait_columns:
            result[column] = keys.map(lookup[column]).astype("object")

        return result

    # --------------------------------------------------------------- group
    def _group(self, state: SessionState, trait_names: list[Any] | None, raw_conditions: list[dict[str, Any]]) -> dict[str, Any]:
        """Fill ``cluster_traits`` / ``criteria_traits`` from traits and conditions (no filtering)."""
        frame = state.candidate_list.copy()
        names = [str(item) for item in trait_names or [] if str(item).strip()]
        columns = [resolve_trait_column(name, frame) for name in names]

        # Conditions are evaluated with the passport engine on the resolved trait columns.
        flags: dict[str, pd.Series] = {}
        for raw in raw_conditions:
            trait = raw.get("trait") or raw.get("column") or raw.get("indicator")
            if not trait:
                raise TraitsError("Every condition needs a 'trait'.")
            column = resolve_trait_column(str(trait), frame)
            operator = normalise_operator(str(raw.get("operator", "equals")))
            value = raw.get("value")
            condition = Condition(column=column, operator=operator, value=value)
            evaluated = self._evaluate(frame, condition)
            text = f"{column} {OPERATORS[operator]} {value}" if value is not None else f"{column} {OPERATORS[operator]}"
            flags[text] = evaluated
            if column not in columns:
                columns.append(column)

        if not columns:
            raise TraitsError("Indicate at least one trait (traits=[...]) or one condition (conditions=[...]).")

        # One trait with one condition -> binary subset; otherwise combination groups.
        if len(flags) == 1 and len(columns) == 1:
            text, mask = next(iter(flags.items()))
            codes = mask.map(lambda value: pd.NA if pd.isna(value) else int(bool(value))).astype("Int64")
            labels = codes.map(lambda code: pd.NA if pd.isna(code) else ("meets" if code == 1 else "does not meet")).astype("object")
            description = f"'{text}' -> cluster_traits 1 (meets) / 0 (does not meet)"
        else:
            labels, codes = build_groups(frame, [column for column in columns if not any(column in key for key in flags)], flags)
            description = f"groups by {', '.join(columns)}" + (f" with conditions {list(flags)}" if flags else "")

        # Criteria text: criterion + the trait values of the accession.
        def row_text(index: Any) -> str:
            """Compose the criteria_traits text of one accession."""
            values = ", ".join(f"{column}={_fmt(frame.at[index, column])}" for column in columns)
            if pd.isna(labels.at[index]):
                return f"{description}: no trait data ({values})"
            return f"{description}: {labels.at[index]} [{values}]"

        texts = pd.Series([row_text(index) for index in frame.index], index=frame.index)
        frame[CRITERIA_COLUMN] = self._append_criteria(frame[CRITERIA_COLUMN], texts)
        frame[CLUSTER_COLUMN] = codes
        state.update_candidate_list(frame)

        sizes = codes.dropna().astype(int).value_counts().sort_index()
        group_summary = [{"cluster_traits": int(code), "label": str(labels[codes == code].iloc[0]), "accessions": int(size)} for code, size in sizes.items()]
        assigned = int(codes.notna().sum())
        state.log_activity(self.name, f"Trait subsets created ({description}): {len(sizes)} group(s), {assigned} accessions assigned, {len(frame) - assigned} without trait data")

        return self.ok(
            f"Trait subsets created: {len(sizes)} group(s) written to cluster_traits ({assigned} accessions assigned, "
            f"{len(frame) - assigned} without trait data). No accession was removed.",
            groups=group_summary,
            assigned=assigned,
            without_data=len(frame) - assigned,
            columns=columns,
            subsets_created=True,
        )

    @staticmethod
    def _evaluate(frame: pd.DataFrame, condition: Condition) -> pd.Series:
        """Evaluate a condition; numeric comparisons are done on coerced numbers, missing values stay NA."""
        series = frame[condition.column]
        numeric_ops = {"gt", "gte", "lt", "lte", "between"}

        # Numeric operators need numbers; the passport engine handles the rest.
        if condition.operator in numeric_ops:
            working = frame.assign(**{condition.column: to_numeric_series(series)})
            mask = evaluate_condition(working, condition)
        else:
            mask = evaluate_condition(frame, condition)

        result = mask.astype("object")
        result[series.isna()] = pd.NA
        return result


def _fmt(value: Any) -> str:
    """Format a value for criteria texts (2 decimals for floats)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "NA"
    try:
        if pd.isna(value):
            return "NA"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
