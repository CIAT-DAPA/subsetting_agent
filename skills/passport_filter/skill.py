"""Skill that filters the Candidate list by passport data.

Three actions are offered to the model:

* ``filter``   - keep the accessions that satisfy one or more conditions and
  record the applied criteria in ``criteria_passport``.
* ``describe`` - summarise columns (type, nulls, frequent values) so the model
  can choose valid columns and values before filtering.
* ``reset``    - rebuild the Candidate list from the Original list.
"""

from typing import Any

import pandas as pd

from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState
from skills.arguments import as_dict_list, as_int, as_list
from skills.base import Skill
from skills.passport_filter.engine import (
    OPERATORS,
    Condition,
    FilterError,
    apply_conditions,
    build_condition,
    describe_columns,
    describe_conditions,
    resolve_column,
)

logger = get_logger(__name__)

# Column of the Candidate list where the applied passport criteria are recorded.
CRITERIA_COLUMN = "criteria_passport"
# Separator between criteria applied in successive turns.
_CRITERIA_SEPARATOR = " | "
# Maximum columns described when the model does not name any.
_MAX_DESCRIBED_COLUMNS = 25


class PassportFilterSkill(Skill):
    """Filter, describe or reset the Candidate list using passport data columns."""

    name = "passport_filter"
    description = (
        "Work with the passport data of the Candidate list. action='filter' keeps only "
        "the accessions that satisfy the given conditions (column/operator/value) and "
        "records them in criteria_passport; action='describe' lists the columns with "
        "their types and most frequent values (use it before filtering when the values "
        "are unknown); action='reset' restores the Candidate list from the Original list. "
        "Filters from successive calls are chained (each call narrows the current list)."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["filter", "describe", "reset"],
                "description": "What to do. Defaults to 'filter'.",
            },
            "conditions": {
                "type": "array",
                "description": (
                    "Conditions for action='filter'. Each item: {\"column\": <name>, "
                    "\"operator\": <op>, \"value\": <value>}. Operators: equals, not_equals, "
                    "in, not_in, contains, not_contains, starts_with, gt, gte, lt, lte, "
                    "between (value=[min,max]), is_null, not_null. Column names may be MCPD "
                    "codes (ORIGCTY, SAMPSTAT) or plain words (country, latitude)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {},
                    },
                    "required": ["column", "operator"],
                },
            },
            "logic": {
                "type": "string",
                "enum": ["and", "or"],
                "description": "How to combine the conditions of this call. Defaults to 'and'.",
            },
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to describe for action='describe' (all when omitted).",
            },
            "top": {
                "type": "integer",
                "description": "Number of frequent values per column for action='describe' (default 10).",
            },
        },
        "required": [],
    }

    # ---------------------------------------------------------------- run
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        action: str = "filter",
        conditions: Any = None,
        logic: str = "and",
        columns: Any = None,
        top: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Dispatch the requested action.

        Args:
            state: Session state holding the Candidate list.
            paths: Session folders (unused, kept for the skill contract).
            action: ``filter``, ``describe`` or ``reset``.
            conditions: Filter conditions (list of dicts or JSON text).
            logic: ``and``/``or`` combination for this call.
            columns: Columns to describe.
            top: Frequent values per column when describing.
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result dictionary.
        """
        # Every action needs data loaded first.
        if not state.has_data or state.candidate_list is None:
            return self.error(
                "No accession list is loaded yet. Ask the user to upload a file or to search "
                "accessions in Genesys before filtering by passport data."
            )

        mode = (action or "filter").strip().lower()

        # Route by action; unknown actions are reported with the valid options.
        if mode == "filter":
            return self._filter(state, conditions, logic)
        if mode == "describe":
            return self._describe(state, columns, top)
        if mode == "reset":
            return self._reset(state)

        return self.error(f"Unknown action '{action}'. Use 'filter', 'describe' or 'reset'.")

    # -------------------------------------------------------------- filter
    def _filter(self, state: SessionState, raw_conditions: Any, logic: str) -> dict[str, Any]:
        """Apply the conditions to the Candidate list.

        Args:
            state: Session state.
            raw_conditions: Conditions as sent by the model.
            logic: ``and``/``or``.

        Returns:
            Tool result with counts and the recorded criteria.
        """
        candidate = state.candidate_list
        data_columns = [str(column) for column in candidate.columns]

        try:
            raw_list = as_dict_list(raw_conditions)
        except ValueError as exc:
            return self.error(f"Invalid conditions: {exc}")

        # Without conditions there is nothing to filter by.
        if not raw_list:
            return self.error(
                "action='filter' needs at least one condition "
                "({column, operator, value}). Use action='describe' to see the columns."
            )

        try:
            parsed: list[Condition] = [build_condition(raw, data_columns) for raw in raw_list]
            mask = apply_conditions(candidate, parsed, logic)
        except FilterError as exc:
            return self.error(str(exc), operators=list(OPERATORS))

        criteria_text = describe_conditions(parsed, logic)
        before = len(candidate)
        after = int(mask.sum())

        # Never leave the user with an empty list: report and keep the current one.
        if after == 0:
            hints = describe_columns(candidate, [condition.column for condition in parsed], top=8)
            state.log_activity(
                self.name, f"Passport filter '{criteria_text}' matched no accessions; list unchanged"
            )
            return self.error(
                f"No accession satisfies '{criteria_text}'. The Candidate list was left unchanged "
                f"({before} accessions). Check the frequent values of the columns involved and "
                "suggest the user to adjust the criteria.",
                criteria=criteria_text,
                accessions_before=before,
                column_hints=hints,
            )

        filtered = candidate.loc[mask].copy()
        filtered[CRITERIA_COLUMN] = self._append_criteria(filtered[CRITERIA_COLUMN], criteria_text)
        state.update_candidate_list(filtered)

        description = f"Filtered Candidate list by passport data '{criteria_text}': {before} -> {after} accessions"
        state.log_activity(self.name, description)

        return self.ok(
            f"{description}. The criteria were recorded in the column '{CRITERIA_COLUMN}'.",
            criteria=criteria_text,
            accessions_before=before,
            accessions_after=after,
            removed=before - after,
            conditions=[condition.describe() for condition in parsed],
        )

    @staticmethod
    def _append_criteria(existing: pd.Series, criteria_text: str) -> pd.Series:
        """Append the new criteria to whatever each row already recorded.

        Args:
            existing: Current ``criteria_passport`` values (may be NA).
            criteria_text: Criteria applied in this call.

        Returns:
            Updated series.
        """
        previous = existing.astype("string").fillna("").str.strip()

        # Rows without previous criteria get the text alone; others are chained.
        return previous.where(previous == "", previous + _CRITERIA_SEPARATOR).astype(str) + criteria_text

    # ------------------------------------------------------------ describe
    def _describe(self, state: SessionState, raw_columns: Any, raw_top: Any) -> dict[str, Any]:
        """Summarise the columns of the Candidate list.

        Args:
            state: Session state.
            raw_columns: Columns requested by the model (optional).
            raw_top: Number of frequent values per column (optional).

        Returns:
            Tool result with one summary per column.
        """
        candidate = state.candidate_list
        data_columns = [str(column) for column in candidate.columns]
        requested = as_list(raw_columns)
        top = as_int(raw_top) or 10

        try:
            # Resolve requested names; describe every data column when none given.
            if requested:
                selected = [resolve_column(str(name), data_columns) for name in requested]
            else:
                selected = [column for column in data_columns if not column.startswith(("criteria_", "cluster_"))]
                selected = selected[:_MAX_DESCRIBED_COLUMNS]
        except FilterError as exc:
            return self.error(str(exc))

        summaries = describe_columns(candidate, selected, top=top)

        return self.ok(
            f"Described {len(summaries)} columns of the Candidate list ({len(candidate)} accessions).",
            accessions=len(candidate),
            total_columns=len(data_columns),
            columns=summaries,
        )

    # --------------------------------------------------------------- reset
    def _reset(self, state: SessionState) -> dict[str, Any]:
        """Restore the Candidate list from the Original list.

        Args:
            state: Session state.

        Returns:
            Tool result with the restored row count.
        """
        before = state.candidate_count

        try:
            state.reset_candidate_list()
        except ValueError as exc:
            return self.error(str(exc))

        after = state.candidate_count
        description = f"Reset Candidate list from the Original list: {before} -> {after} accessions"
        state.log_activity(self.name, description)

        return self.ok(
            f"{description}. All previous filters and annotations were cleared.",
            accessions_before=before,
            accessions_after=after,
        )
