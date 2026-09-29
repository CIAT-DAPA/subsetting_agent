"""Pure filtering logic over a pandas DataFrame of passport data.

This module has no knowledge of sessions or the LLM: it resolves column names,
evaluates conditions and describes columns. The skill wraps it.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Operators accepted in a condition and their human-readable symbol.
OPERATORS: dict[str, str] = {
    "equals": "=",
    "not_equals": "!=",
    "in": "in",
    "not_in": "not in",
    "contains": "contains",
    "not_contains": "not contains",
    "starts_with": "starts with",
    "gt": ">",
    "gte": ">=",
    "lt": "<",
    "lte": "<=",
    "between": "between",
    "is_null": "is empty",
    "not_null": "is not empty",
}

# Alternative spellings the model may use for the operators.
_OPERATOR_ALIASES: dict[str, str] = {
    "eq": "equals",
    "==": "equals",
    "=": "equals",
    "is": "equals",
    "ne": "not_equals",
    "!=": "not_equals",
    "<>": "not_equals",
    "not": "not_equals",
    "not_in": "not_in",
    "notin": "not_in",
    "like": "contains",
    "includes": "contains",
    "startswith": "starts_with",
    "sw": "starts_with",
    ">": "gt",
    ">=": "gte",
    "ge": "gte",
    "<": "lt",
    "<=": "lte",
    "le": "lte",
    "range": "between",
    "null": "is_null",
    "empty": "is_null",
    "is_empty": "is_null",
    "notnull": "not_null",
    "not_empty": "not_null",
    "is_not_null": "not_null",
}

# Operators that need no value.
_UNARY_OPERATORS = {"is_null", "not_null"}
# Operators that need a list value.
_LIST_OPERATORS = {"in", "not_in", "between"}
# Operators evaluated numerically.
_NUMERIC_OPERATORS = {"gt", "gte", "lt", "lte", "between"}

# Common synonyms for MCPD passport columns (lower-case, no separators).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ORIGCTY": ("country", "countryoforigin", "origin", "pais", "paisdeorigen", "origcty", "iso3"),
    "INSTCODE": ("institute", "institutecode", "instituto", "holdinginstitute", "wiews", "instcode"),
    "ACCENUMB": ("accession", "accessionnumber", "accnumber", "acceno", "accenumb", "id", "numero"),
    "GENUS": ("genus", "genero"),
    "SPECIES": ("species", "especie", "specificepithet"),
    "SAMPSTAT": ("sampstat", "biologicalstatus", "status", "samplestatus", "estadobiologico"),
    "DECLATITUDE": ("latitude", "lat", "latitud", "declatitude", "decimallatitude"),
    "DECLONGITUDE": ("longitude", "lon", "long", "longitud", "declongitude", "decimallongitude"),
    "ELEVATION": ("elevation", "altitude", "altitud", "elevacion"),
    "ACQDATE": ("acquisitiondate", "acqdate", "dateacquired"),
    "COLLDATE": ("collectingdate", "colldate", "collectiondate", "fechacolecta"),
    "COLLSITE": ("collectingsite", "collsite", "site", "sitio", "locality", "localidad"),
    "CROPNAME": ("crop", "cropname", "cultivo"),
    "STORAGE": ("storage", "storagetype", "almacenamiento"),
    "MLSSTAT": ("mls", "mlsstatus", "mlsstat"),
    "AVAILABLE": ("available", "availability", "disponible"),
    "DOI": ("doi",),
    "ACCENAME": ("accessionname", "name", "nombre", "accename"),
    "DONORCODE": ("donor", "donorcode", "donante"),
    "TAXONNAME": ("taxon", "taxonname", "taxonomy", "scientificname"),
}


class FilterError(ValueError):
    """Raised when a condition cannot be evaluated (bad column, operator or value)."""


@dataclass(frozen=True)
class Condition:
    """A single normalised filter condition.

    Attributes:
        column: Real column name of the DataFrame.
        operator: One of the keys of ``OPERATORS``.
        value: Scalar, list or ``None`` depending on the operator.
    """

    column: str
    operator: str
    value: Any

    def describe(self) -> str:
        """Render the condition as readable text (``ORIGCTY in [COL, PER]``)."""
        symbol = OPERATORS[self.operator]

        # Unary operators carry no value.
        if self.operator in _UNARY_OPERATORS:
            return f"{self.column} {symbol}"

        # List operators are rendered with brackets.
        if isinstance(self.value, list):
            rendered = ", ".join(str(item) for item in self.value)
            return f"{self.column} {symbol} [{rendered}]"

        return f"{self.column} {symbol} {self.value}"


# -------------------------------------------------------------- normalise
def _normalise_key(text: str) -> str:
    """Lower-case a name, strip accents and drop separators so spellings compare."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[\s_\-\.]+", "", ascii_text.strip().lower())


def resolve_column(name: str, columns: list[str]) -> str:
    """Find the real column that a user/model provided name refers to.

    Resolution order: exact name, case/separator-insensitive match, MCPD alias,
    then substring match. Suggestions are included in the error otherwise.

    Args:
        name: Column name as given.
        columns: Real columns of the DataFrame.

    Returns:
        The matching column name.

    Raises:
        FilterError: If no column matches.
    """
    # Exact match wins immediately.
    if name in columns:
        return name

    key = _normalise_key(name)
    normalised = {_normalise_key(column): column for column in columns}

    # Same name ignoring case and separators.
    if key in normalised:
        return normalised[key]

    # MCPD synonyms: "country" -> ORIGCTY when that column exists.
    for canonical, aliases in COLUMN_ALIASES.items():
        if key == _normalise_key(canonical) or key in aliases:
            canonical_key = _normalise_key(canonical)

            if canonical_key in normalised:
                return normalised[canonical_key]

    # Unique substring match ("lat" inside "declatitude").
    partial = [column for norm, column in normalised.items() if key and (key in norm or norm in key)]

    if len(partial) == 1:
        return partial[0]

    suggestions = difflib.get_close_matches(key, list(normalised), n=3, cutoff=0.5)
    suggested = [normalised[item] for item in suggestions] or partial[:3]

    raise FilterError(
        f"Column '{name}' does not exist in the Candidate list. "
        f"Did you mean {suggested}? Available columns: {columns}"
    )


def normalise_operator(operator: str | None) -> str:
    """Map an operator spelling to one of the supported keys.

    Args:
        operator: Operator as given (``"="``, ``"eq"``, ``"in"``...).

    Returns:
        Canonical operator key.

    Raises:
        FilterError: If the operator is unknown.
    """
    key = str(operator or "equals").strip().lower().replace(" ", "_")

    if key in OPERATORS:
        return key

    if key in _OPERATOR_ALIASES:
        return _OPERATOR_ALIASES[key]

    raise FilterError(f"Unknown operator '{operator}'. Supported operators: {list(OPERATORS)}")


def build_condition(raw: dict[str, Any], columns: list[str]) -> Condition:
    """Validate and normalise a raw condition dictionary.

    Args:
        raw: ``{"column": ..., "operator": ..., "value": ...}`` from the model.
        columns: Real columns of the DataFrame.

    Returns:
        A ``Condition`` ready to evaluate.

    Raises:
        FilterError: On missing column, unknown operator or invalid value shape.
    """
    column_name = raw.get("column") or raw.get("field") or raw.get("name")

    # A condition without a column cannot be evaluated.
    if not column_name:
        raise FilterError(f"Condition {raw} has no 'column'.")

    column = resolve_column(str(column_name), columns)
    operator = normalise_operator(raw.get("operator") or raw.get("op"))
    value = raw.get("value", raw.get("values"))

    # Unary operators ignore any value.
    if operator in _UNARY_OPERATORS:
        return Condition(column=column, operator=operator, value=None)

    # Everything else needs a value.
    if value is None or (isinstance(value, str) and not value.strip()):
        raise FilterError(f"Condition on '{column}' with operator '{operator}' needs a value.")

    # List operators accept a list or a delimited string.
    if operator in _LIST_OPERATORS:
        if isinstance(value, str):
            value = [part.strip() for part in re.split(r"[,;|]", value.strip("[]")) if part.strip()]
        elif not isinstance(value, list):
            value = [value]

        if operator == "between" and len(value) != 2:
            raise FilterError("Operator 'between' needs exactly two values: [min, max].")

    elif isinstance(value, list):
        # A scalar operator with a list value is treated as "in" for convenience.
        if len(value) == 1:
            value = value[0]
        else:
            operator = "in" if operator == "equals" else operator
            if operator != "in":
                raise FilterError(f"Operator '{operator}' expects a single value, got a list.")

    return Condition(column=column, operator=operator, value=value)


# ---------------------------------------------------------------- evaluate
def _text_series(series: pd.Series) -> pd.Series:
    """Return the series as trimmed lower-case text (NaN stays NaN)."""
    return series.astype("string").str.strip().str.lower()


def _numeric_series(series: pd.Series, column: str) -> pd.Series:
    """Convert a series to numbers, failing when nothing is numeric.

    Args:
        series: Column values (often text loaded from Excel).
        column: Column name for the error message.

    Returns:
        Float series with ``NaN`` where conversion failed.

    Raises:
        FilterError: If the column has values but none is numeric.
    """
    numeric = pd.to_numeric(series, errors="coerce")

    # A column with values but no numbers cannot be compared numerically.
    if series.notna().any() and numeric.notna().sum() == 0:
        raise FilterError(f"Column '{column}' has no numeric values; use a text operator instead.")

    return numeric


def evaluate_condition(frame: pd.DataFrame, condition: Condition) -> pd.Series:
    """Evaluate one condition and return a boolean mask.

    Args:
        frame: Candidate list.
        condition: Normalised condition.

    Returns:
        Boolean ``Series`` aligned with ``frame``.
    """
    series = frame[condition.column]
    operator = condition.operator
    value = condition.value

    # Null checks look at emptiness only.
    if operator == "is_null":
        return series.isna() | (series.astype("string").str.strip() == "")
    if operator == "not_null":
        return ~(series.isna() | (series.astype("string").str.strip() == ""))

    # Numeric comparisons.
    if operator in _NUMERIC_OPERATORS:
        numeric = _numeric_series(series, condition.column)

        if operator == "between":
            low, high = (_to_float(item) for item in value)
            return numeric.between(min(low, high), max(low, high))

        target = _to_float(value)
        comparisons = {"gt": numeric > target, "gte": numeric >= target, "lt": numeric < target, "lte": numeric <= target}
        return comparisons[operator].fillna(False)

    # Text comparisons (case- and space-insensitive).
    text = _text_series(series)

    if operator in ("in", "not_in"):
        targets = {str(item).strip().lower() for item in value}
        mask = text.isin(targets).fillna(False)
        return ~mask if operator == "not_in" else mask

    target = str(value).strip().lower()

    if operator == "equals":
        # Numeric-looking targets also match numerically ("300" vs "300.0").
        mask = (text == target).fillna(False)
        numeric_target = _try_float(target)
        if numeric_target is not None:
            numeric = pd.to_numeric(series, errors="coerce")
            mask = mask | (numeric == numeric_target).fillna(False)
        return mask.astype(bool)
    if operator == "not_equals":
        return ~(text == target).fillna(False)
    if operator == "contains":
        return text.str.contains(re.escape(target), na=False)
    if operator == "not_contains":
        return ~text.str.contains(re.escape(target), na=False)
    if operator == "starts_with":
        return text.str.startswith(target, na=False)

    raise FilterError(f"Operator '{operator}' is not implemented.")  # pragma: no cover


def _to_float(value: Any) -> float:
    """Convert a scalar to float or raise a ``FilterError``."""
    number = _try_float(value)

    if number is None:
        raise FilterError(f"Value '{value}' is not numeric.")

    return number


def _try_float(value: Any) -> float | None:
    """Convert a scalar to float, returning ``None`` when impossible."""
    try:
        return float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None


def apply_conditions(frame: pd.DataFrame, conditions: list[Condition], logic: str = "and") -> pd.Series:
    """Combine the masks of several conditions.

    Args:
        frame: Candidate list.
        conditions: Conditions to evaluate (at least one).
        logic: ``"and"`` or ``"or"``.

    Returns:
        Boolean mask of the rows that satisfy the combination.

    Raises:
        FilterError: If ``logic`` is unknown or no conditions are given.
    """
    if not conditions:
        raise FilterError("At least one condition is required.")

    logic = (logic or "and").strip().lower()

    if logic not in ("and", "or"):
        raise FilterError(f"Unknown logic '{logic}'. Use 'and' or 'or'.")

    mask = evaluate_condition(frame, conditions[0]).astype(bool)

    # Fold the remaining conditions into the mask.
    for condition in conditions[1:]:
        current = evaluate_condition(frame, condition).astype(bool)
        mask = (mask & current) if logic == "and" else (mask | current)

    return mask


def describe_conditions(conditions: list[Condition], logic: str = "and") -> str:
    """Render a list of conditions as one readable expression."""
    joiner = " AND " if (logic or "and").lower() == "and" else " OR "
    return joiner.join(condition.describe() for condition in conditions)


# ---------------------------------------------------------------- describe
def describe_columns(frame: pd.DataFrame, columns: list[str] | None = None, top: int = 10) -> list[dict[str, Any]]:
    """Summarise columns so the model knows which values exist.

    Args:
        frame: Candidate list.
        columns: Columns to describe (all when ``None``).
        top: Number of most frequent values to include.

    Returns:
        One dictionary per column with type, null count, distinct count and top values.
    """
    selected = columns or [str(column) for column in frame.columns]
    summaries: list[dict[str, Any]] = []

    # Build a compact summary for each requested column.
    for column in selected:
        series = frame[column]
        non_null = series.dropna()
        numeric = pd.to_numeric(non_null, errors="coerce")
        is_numeric = len(non_null) > 0 and numeric.notna().mean() > 0.9

        summary: dict[str, Any] = {
            "column": column,
            "type": "numeric" if is_numeric else "text",
            "non_null": int(non_null.shape[0]),
            "nulls": int(series.isna().sum()),
            "distinct": int(non_null.nunique()),
        }

        # Numeric columns get a range; text columns get their frequent values.
        if is_numeric:
            summary["min"] = float(numeric.min())
            summary["max"] = float(numeric.max())
        else:
            counts = non_null.astype(str).str.strip().value_counts().head(top)
            summary["top_values"] = {str(key): int(count) for key, count in counts.items()}

        summaries.append(summary)

    return summaries
