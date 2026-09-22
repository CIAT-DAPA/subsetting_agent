"""Build the results table of the current selection and export it.

The table is produced deterministically from the :class:`AccessionContext`
after every chat turn: one row per selected accession with its passport data,
the evidence recorded by the tools (trait value, climate cluster, indicator
means, citing document) and the criteria applied so far. The model never
writes the table; it only summarises it.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from tools.accession_context import AccessionContext

# Passport columns shown first, in this order.
FIXED_COLUMNS: list[str] = [
    "accession_number",
    "institute_code",
    "crop",
    "taxon_name",
    "country_code",
    "latitude",
    "longitude",
    "cellid",
]

# Column holding the plain-words criteria that produced the selection.
CRITERIA_COLUMN = "criteria"


def build_selection_table(context: AccessionContext) -> pd.DataFrame:
    """Return one row per selected accession with passport data and evidence.

    Args:
        context: Current selection.

    Returns:
        A DataFrame with :data:`FIXED_COLUMNS`, one column per evidence label
        found in the selection, and a final ``criteria`` column that repeats the
        applied steps (``stage: description``) so every row is self-explanatory
        when pasted elsewhere.
    """
    labels = context.evidence_labels()
    criteria = "; ".join(f"{step.stage}: {step.description}" for step in context.steps)
    rows: list[dict[str, object]] = []

    # Each record becomes a row; missing evidence stays empty rather than failing.
    for record in context.records():
        row: dict[str, object] = {column: getattr(record, column) for column in FIXED_COLUMNS}

        for label in labels:
            row[label] = record.evidence.get(label)

        row[CRITERIA_COLUMN] = criteria
        rows.append(row)

    return pd.DataFrame(rows, columns=[*FIXED_COLUMNS, *labels, CRITERIA_COLUMN])


def to_markdown(frame: pd.DataFrame, *, max_rows: int = 20) -> str:
    """Render the first rows of the table as a Markdown table for the chat.

    Args:
        frame: Table built by :func:`build_selection_table`.
        max_rows: Rows shown; the rest are summarised in a trailing line.

    Returns:
        Markdown text, empty when the table has no rows.
    """
    if frame.empty:
        return ""

    # The criteria column repeats the same text on every row; the chat shows it once.
    shown = frame.drop(columns=[CRITERIA_COLUMN], errors="ignore").head(max_rows)
    shown = shown.dropna(axis=1, how="all")
    shown = shown.where(shown.notna(), "")
    header = "| " + " | ".join(str(column) for column in shown.columns) + " |"
    separator = "|" + "|".join(" --- " for _ in shown.columns) + "|"
    lines = [header, separator]

    # Pipes inside values would break the table, so they are escaped.
    for _, values in shown.iterrows():
        cells = [str(value).replace("|", "\\|") for value in values.tolist()]
        lines.append("| " + " | ".join(cells) + " |")

    remaining = len(frame) - len(shown)

    if remaining > 0:
        lines.append(f"\n... {remaining} more rows in the full table and the CSV below.")

    criteria = frame[CRITERIA_COLUMN].iloc[0] if CRITERIA_COLUMN in frame else ""

    if criteria:
        lines.append(f"\nCriteria applied: {criteria}")

    return "\n".join(lines)


def write_csv(frame: pd.DataFrame, directory: Path, *, moment: datetime | None = None) -> Path:
    """Write the table as UTF-8 CSV under ``directory``.

    The file name follows the project convention ``YYYYMMDD_HHmmss_<hash>_selection.csv``,
    where the hash is computed from the table content so identical selections map
    to identical names.

    Args:
        frame: Table to write.
        directory: Exports directory (created if needed).
        moment: Timestamp for the name; ``now`` when omitted.

    Returns:
        Path of the written file.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (moment or datetime.now()).strftime("%Y%m%d_%H%M%S")
    content = frame.to_csv(index=False)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    path = directory / f"{stamp}_{digest}_selection.csv"
    path.write_text(content, encoding="utf-8")

    return path
