"""Pure trait logic over the Candidate list (no sessions, no LLM, no HTTP).

Responsibilities:

* detect which columns of a local list look like traits (anything that is not
  passport data nor an annotation column);
* resolve trait column names given by the model (with or without the
  ``trait_`` prefix, case-insensitive, substring);
* select Genesys descriptors by name or free-text query;
* aggregate observation rows (several observations per accession, list values)
  into one value per accession: mean for numeric traits, mode for categorical;
* build groups from one or several trait columns (terciles for numeric
  columns, categories for categorical columns, met/not met for conditions).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any

import pandas as pd

from core.state import CANDIDATE_EXTRA_COLUMNS
from sdks.genesys.models import MCPD_FIELD_MAP
from sdks.genesys.traits import Descriptor

# Prefix of the columns added to the Candidate list with Genesys trait data.
TRAIT_PREFIX = "trait_"
# Labels of the terciles used to group numeric traits.
TERCILE_LABELS = ("low", "medium", "high")
# Columns that are never considered traits in a local list.
NON_TRAIT_COLUMNS: frozenset[str] = frozenset(
    {column.lower() for column in MCPD_FIELD_MAP}
    | {column.lower() for column in CANDIDATE_EXTRA_COLUMNS}
    | {"cellid", "latitude", "longitude", "lat", "lon", "long", "country", "pais", "país", "instituto", "institute", "genero", "género", "especie"}
)
# Maximum categories listed when describing a categorical trait column.
MAX_CATEGORIES = 12


class TraitsError(ValueError):
    """Raised when trait arguments cannot be interpreted."""


@dataclass
class TraitColumn:
    """Description of one trait column of the Candidate list.

    Attributes:
        column: Real column name.
        numeric: ``True`` when most values are numbers.
        non_null: Number of accessions with a value.
        summary: Range (numeric) or list of categories (categorical).
    """

    column: str
    numeric: bool
    non_null: int
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise for a tool result."""
        return {"column": self.column, "type": "numeric" if self.numeric else "categorical", "non_null": self.non_null, **self.summary}


# ----------------------------------------------------------------- helpers
def _normalise_text(text: str) -> str:
    """Lower-case, accent-free, alphanumeric version of a text for matching."""
    stripped = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", stripped.lower())


def to_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a column to numbers (``NaN`` where impossible)."""
    return pd.to_numeric(series, errors="coerce")


def is_numeric_column(series: pd.Series, threshold: float = 0.8) -> bool:
    """Decide whether a column is numeric.

    Args:
        series: Column values.
        threshold: Minimum share of non-null values that must parse as numbers.

    Returns:
        ``True`` when the column holds numbers (at least ``threshold`` of them).
    """
    values = series.dropna()

    if values.empty:
        return False

    numeric = to_numeric_series(values)
    return numeric.notna().sum() / len(values) >= threshold


def describe_trait_column(frame: pd.DataFrame, column: str) -> TraitColumn:
    """Summarise one trait column (range or categories).

    Args:
        frame: Candidate list.
        column: Column to describe.

    Returns:
        A ``TraitColumn`` with type, coverage and summary.
    """
    series = frame[column]
    numeric = is_numeric_column(series)

    # Numeric traits are summarised with min/mean/max; categorical with their categories.
    if numeric:
        values = to_numeric_series(series).dropna()
        summary = {"min": _round(values.min()), "mean": _round(values.mean()), "max": _round(values.max())}
    else:
        counts = series.dropna().astype(str).value_counts()
        summary = {"categories": counts.head(MAX_CATEGORIES).index.tolist(), "n_categories": int(len(counts))}

    return TraitColumn(column=column, numeric=numeric, non_null=int(series.notna().sum()), summary=summary)


def detect_trait_columns(frame: pd.DataFrame) -> list[TraitColumn]:
    """Return the columns of a list that look like traits.

    A column is a trait candidate when it is not passport data (MCPD names and
    common aliases), not an annotation column and not completely empty.

    Args:
        frame: Candidate list.

    Returns:
        Trait column descriptions in column order.
    """
    found: list[TraitColumn] = []

    for column in frame.columns:
        name = str(column)
        lowered = name.lower()

        # Skip passport, annotation and empty columns; ``trait_`` columns always count.
        if not lowered.startswith(TRAIT_PREFIX) and (lowered in NON_TRAIT_COLUMNS or frame[column].notna().sum() == 0):
            continue

        found.append(describe_trait_column(frame, name))

    return found


def resolve_trait_column(name: str, frame: pd.DataFrame) -> str:
    """Find the real trait column for a name given by the model.

    Matching order: exact, case-insensitive, with/without ``trait_`` prefix,
    normalised text, substring of the normalised name.

    Args:
        name: Column name as written by the model or the user.
        frame: Candidate list.

    Returns:
        The real column name.

    Raises:
        TraitsError: When no column matches.
    """
    columns = [str(column) for column in frame.columns]
    wanted = str(name).strip()

    if not wanted:
        raise TraitsError("A trait column name is required.")

    if wanted in columns:
        return wanted

    lowered = {column.lower(): column for column in columns}
    for candidate in (wanted.lower(), f"{TRAIT_PREFIX}{wanted.lower()}", wanted.lower().removeprefix(TRAIT_PREFIX)):
        if candidate in lowered:
            return lowered[candidate]

    normalised = {_normalise_text(column.removeprefix(TRAIT_PREFIX) if column.lower().startswith(TRAIT_PREFIX) else column): column for column in columns}
    key = _normalise_text(wanted.removeprefix(TRAIT_PREFIX) if wanted.lower().startswith(TRAIT_PREFIX) else wanted)

    if key in normalised:
        return normalised[key]

    # Substring match (e.g. "iron" -> "trait_Fe.Mean" would not match, but "Fe" -> "trait_Fe.Mean" does).
    partial = [column for norm, column in normalised.items() if key and key in norm]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise TraitsError(f"The trait '{name}' is ambiguous; candidates: {partial}.")

    traits = [item.column for item in detect_trait_columns(frame)]
    raise TraitsError(f"Trait column '{name}' not found. Available trait columns: {traits or 'none'}.")


# ------------------------------------------------------------- descriptors
def select_descriptors(descriptors: list[Descriptor], query: str | None = None, names: list[str] | None = None) -> list[Descriptor]:
    """Choose descriptors by explicit names or by free text.

    Args:
        descriptors: Candidate descriptors (already de-duplicated by uuid).
        query: Free text (e.g. "iron zinc", "seed color"); every word is tried.
        names: Column names, titles or uuids given by the model.

    Returns:
        Matching descriptors (all of them when no name and no query is given).
    """
    selected: list[Descriptor] = []

    # Explicit names win: match uuid, columnName or title (normalised).
    for name in names or []:
        key = _normalise_text(str(name).removeprefix(TRAIT_PREFIX))
        for descriptor in descriptors:
            if descriptor in selected:
                continue
            if key and key in {_normalise_text(descriptor.uuid), _normalise_text(descriptor.column_name or ""), _normalise_text(descriptor.title or "")}:
                selected.append(descriptor)
            elif key and (key in _normalise_text(descriptor.column_name or "") or key in _normalise_text(descriptor.title or "")):
                selected.append(descriptor)

    # A query matches any of its words against title/columnName/description.
    if query and str(query).strip():
        words = [word for word in re.split(r"[\s,;/]+", str(query)) if len(word) > 1]
        for descriptor in descriptors:
            if descriptor not in selected and any(descriptor.matches(word) for word in words):
                selected.append(descriptor)

    if not names and not (query and str(query).strip()):
        return list(descriptors)

    return selected


# ------------------------------------------------------------- aggregation
def aggregate_values(values: Any, numeric: bool) -> Any:
    """Collapse one or several observations into a single value.

    Args:
        values: Scalar, list of scalars, or ``None``.
        numeric: ``True`` for numeric traits (mean), ``False`` for categorical (mode).

    Returns:
        The aggregated value or ``None`` when nothing usable is present.
    """
    items = values if isinstance(values, list) else [values]
    cleaned = [item for item in items if item is not None and not (isinstance(item, float) and pd.isna(item)) and str(item).strip() != ""]

    if not cleaned:
        return None

    # Numeric: mean of the parsable numbers; categorical: most frequent value.
    if numeric:
        numbers = pd.to_numeric(pd.Series(cleaned), errors="coerce").dropna()
        return None if numbers.empty else float(numbers.mean())

    return Counter(str(item) for item in cleaned).most_common(1)[0][0]


def aggregate_observations(frame: pd.DataFrame, descriptors: list[Descriptor]) -> pd.DataFrame:
    """Build one row per accession with ``trait_<columnName>`` columns.

    Args:
        frame: Output of ``observations_to_dataframe`` (``uuid``, ``accessionNumber``, one column per descriptor label).
        descriptors: Descriptors present in the frame (decide numeric vs categorical).

    Returns:
        DataFrame with ``uuid``, ``accessionNumber`` and the aggregated trait columns.
    """
    if frame.empty:
        return pd.DataFrame(columns=["uuid", "accessionNumber"] + [f"{TRAIT_PREFIX}{d.label}" for d in descriptors])

    key = "uuid" if "uuid" in frame.columns and frame["uuid"].notna().any() else "accessionNumber"
    records: list[dict[str, Any]] = []

    # Each accession may appear in several rows (datasets/pages); pool its observations.
    for identifier, rows in frame.groupby(key, dropna=True, sort=False):
        record: dict[str, Any] = {"uuid": rows["uuid"].dropna().iloc[0] if "uuid" in rows and rows["uuid"].notna().any() else None,
                                  "accessionNumber": rows["accessionNumber"].dropna().iloc[0] if "accessionNumber" in rows and rows["accessionNumber"].notna().any() else None}
        record[key] = identifier

        for descriptor in descriptors:
            if descriptor.label not in rows.columns:
                continue
            pooled: list[Any] = []
            for value in rows[descriptor.label].tolist():
                pooled.extend(value if isinstance(value, list) else [value])
            record[f"{TRAIT_PREFIX}{descriptor.label}"] = aggregate_values(pooled, descriptor.is_numeric or not descriptor.is_categorical and _mostly_numeric(pooled))

        records.append(record)

    return pd.DataFrame.from_records(records)


def _mostly_numeric(values: list[Any]) -> bool:
    """Whether a pool of raw values is numeric (used when the descriptor type is unknown)."""
    cleaned = [value for value in values if value is not None and str(value).strip() != ""]
    return bool(cleaned) and is_numeric_column(pd.Series(cleaned))


# ------------------------------------------------------------------ groups
def tercile_labels(series: pd.Series) -> pd.Series:
    """Split a numeric column into low/medium/high terciles.

    Args:
        series: Numeric values (``NaN`` allowed).

    Returns:
        Labels aligned with ``series`` (``NaN`` where the value is missing).
    """
    numeric = to_numeric_series(series)
    valid = numeric.dropna()

    if valid.empty:
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="object")

    # Rank-based cut copes with repeated values that would break qcut.
    ranks = valid.rank(method="average", pct=True)
    bins = pd.cut(ranks, bins=[0, 1 / 3, 2 / 3, 1.0], labels=list(TERCILE_LABELS), include_lowest=True)
    labels = pd.Series([pd.NA] * len(series), index=series.index, dtype="object")
    labels.loc[valid.index] = bins.astype(str).values
    return labels


def build_groups(frame: pd.DataFrame, columns: list[str], flags: dict[str, pd.Series] | None = None) -> tuple[pd.Series, pd.Series]:
    """Group accessions by the combination of trait values.

    Numeric columns contribute their tercile, categorical columns their value and
    conditions (``flags``) contribute ``met``/``not met``.

    Args:
        frame: Candidate list.
        columns: Trait columns to combine (terciles/categories).
        flags: Optional ``{"<condition text>": boolean mask}`` entries.

    Returns:
        ``(labels, codes)``: human-readable group label and integer group number
        (1..n, ``NA`` when any trait value is missing).
    """
    parts: list[pd.Series] = []

    # Every column becomes a text part of the group label.
    for column in columns:
        series = frame[column]
        if is_numeric_column(series):
            part = tercile_labels(series).map(lambda value, c=column: f"{c}={value}" if not pd.isna(value) else pd.NA)
        else:
            part = series.map(lambda value, c=column: f"{c}={value}" if not pd.isna(value) and str(value).strip() else pd.NA)
        parts.append(part.astype("object"))

    # Conditions contribute met / not met; missing values stay NA.
    for text, mask in (flags or {}).items():
        parts.append(mask.map(lambda value, t=text: pd.NA if pd.isna(value) else (f"{t}: met" if value else f"{t}: not met")).astype("object"))

    if not parts:
        raise TraitsError("At least one trait or condition is needed to build groups.")

    combined = pd.concat(parts, axis=1)
    complete = combined.notna().all(axis=1)
    labels = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    labels.loc[complete] = combined.loc[complete].astype(str).agg(" & ".join, axis=1)

    ordered = sorted(labels.dropna().unique(), key=_group_sort_key)
    code_of = {label: index + 1 for index, label in enumerate(ordered)}
    codes = labels.map(lambda label: code_of.get(label, pd.NA)).astype("Int64")
    return labels, codes


def _group_sort_key(label: str) -> tuple:
    """Sort group labels low < medium < high, then alphabetically."""
    order = {name: index for index, name in enumerate(TERCILE_LABELS)}
    return tuple((order.get(token.split("=")[-1], 99), token) for token in str(label).split(" & "))


def _round(value: Any) -> Any:
    """Round floats for readable summaries."""
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return value
