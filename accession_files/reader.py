"""Reader for user-provided accession lists (Excel or CSV).

When the user uploads a spreadsheet with accession identifiers and collecting
coordinates, the agent works from that list instead of Genesys. The reader
detects the relevant columns from their headers (or uses the columns given by
the caller), validates coordinates and computes the Subsetting API ``cellid``
of every row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from subsetting_sdk.catalog import normalize_text
from subsetting_sdk.grid import GridSpec

SUPPORTED_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls", ".csv", ".tsv"})

# Header candidates per role, normalized (lower case, alphanumeric words). The
# first match in list order wins, so the most specific names come first.
ID_HEADERS = [
    "accession number",
    "accession id",
    "accession",
    "acc number",
    "acc numb",
    "accenumb",
    "acce numb",
    "acc id",
    "accid",
    "acc",
    "genesys id",
    "uuid",
    "doi",
    "id",
    "code",
]
LATITUDE_HEADERS = [
    "latitude",
    "declatitude",
    "dec latitude",
    "geo lat",
    "lat",
    "y",
]
LONGITUDE_HEADERS = [
    "longitude",
    "declongitude",
    "dec longitude",
    "geo lon",
    "geo long",
    "long",
    "lon",
    "lng",
    "x",
]
CROP_HEADERS = ["crop code", "crop name", "cropname", "crop", "cultivo"]
TAXON_HEADERS = ["taxon name", "taxonname", "taxon", "species", "scientific name", "genus"]
COUNTRY_HEADERS = ["country of origin", "origin country", "origcty", "orig cty", "country", "iso3"]


class AccessionFileError(Exception):
    """Raised when an accession file cannot be read or lacks required columns."""


@dataclass
class FileAccession:
    """One accession read from a user file.

    Attributes:
        accession_id: Identifier as written in the file.
        latitude: Collecting-site latitude.
        longitude: Collecting-site longitude.
        cellid: Subsetting API grid cell.
        crop: Crop code or name, if the file had one (or a default was given).
        taxon_name: Taxon, if the file had one.
        country_code: Country of origin, if the file had one.
        row_number: 1-based row in the file (header excluded), for messages.
    """

    accession_id: str
    latitude: float
    longitude: float
    cellid: int
    crop: str | None = None
    taxon_name: str | None = None
    country_code: str | None = None
    row_number: int = 0


@dataclass
class AccessionFileReport:
    """Outcome of reading an accession file.

    Attributes:
        file_name: Name of the file.
        columns: Column used for each role (``id``, ``latitude``, ``longitude``,
            optional ``crop``, ``taxon``, ``country``).
        total_rows: Data rows in the file.
        accepted: Rows converted into accessions.
        rejected: Rows dropped, with the reason per row (capped).
        duplicated_ids: Identifiers that appeared more than once (first kept).
        accessions: Accepted accessions.
    """

    file_name: str
    columns: dict[str, str]
    total_rows: int
    accepted: int
    rejected: list[str] = field(default_factory=list)
    duplicated_ids: list[str] = field(default_factory=list)
    accessions: list[FileAccession] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a compact, JSON-serializable summary for the LLM."""
        return {
            "file_name": self.file_name,
            "columns": self.columns,
            "total_rows": self.total_rows,
            "accepted": self.accepted,
            "rejected_count": len(self.rejected),
            "rejected_examples": self.rejected[:5],
            "duplicated_ids": self.duplicated_ids[:10],
        }


def _read_table(path: Path) -> pd.DataFrame:
    """Load a spreadsheet or delimited file into a DataFrame of strings.

    Args:
        path: File to read.

    Raises:
        AccessionFileError: If the file is missing, unsupported or unreadable.
    """
    if not path.is_file():
        raise AccessionFileError(f"Accession file not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise AccessionFileError(
            f"Unsupported accession file type '{suffix}'. Use .xlsx, .xls, .csv or .tsv."
        )

    try:
        # Everything is read as text; numeric parsing happens per role later,
        # so identifiers such as "0012" keep their leading zeros.
        if suffix == ".csv":
            return pd.read_csv(path, dtype=str, sep=None, engine="python")

        if suffix == ".tsv":
            return pd.read_csv(path, dtype=str, sep="\t")

        return pd.read_excel(path, dtype=str)

    except Exception as exc:
        raise AccessionFileError(f"Cannot read accession file {path.name}: {exc}") from exc


def _find_column(columns: list[str], candidates: list[str], explicit: str | None) -> str | None:
    """Pick the column for a role.

    Args:
        columns: Column names of the file.
        candidates: Normalized header names accepted for the role, best first.
        explicit: Column name requested by the caller; matched tolerantly.

    Returns:
        The matching column name, or ``None``.

    Raises:
        AccessionFileError: If an explicit column does not exist.
    """
    normalized = {normalize_text(column): column for column in columns}

    # An explicit request must exist; otherwise the caller made a mistake.
    if explicit:
        match = normalized.get(normalize_text(explicit))

        if match is None:
            raise AccessionFileError(
                f"Column '{explicit}' not found. Available columns: {', '.join(columns)}."
            )

        return match

    # Exact normalized matches first, in candidate priority order.
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    # Then headers that contain a candidate as a whole word (e.g. "Latitude (dd)").
    for candidate in candidates:
        for key, column in normalized.items():
            if f" {candidate} " in f" {key} ":
                return column

    return None


def _parse_coordinate(value: Any, low: float, high: float) -> float | None:
    """Parse a coordinate cell, accepting decimal commas.

    Args:
        value: Raw cell value.
        low: Minimum valid value.
        high: Maximum valid value.

    Returns:
        The coordinate, or ``None`` when missing, unparsable or out of range.
    """
    if value is None:
        return None

    text = str(value).strip().replace(",", ".")

    if not text or text.lower() in ("nan", "none", "null", "na"):
        return None

    try:
        number = float(text)

    except ValueError:
        return None

    # NaN and out-of-range values are treated as missing.
    if math.isnan(number) or number < low or number > high:
        return None

    return number


def _clean_text(value: Any) -> str | None:
    """Strip a text cell, returning ``None`` for blanks and NaN markers.

    Args:
        value: Raw cell value.
    """
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in ("nan", "none", "null"):
        return None

    return text


def read_accession_file(
    path: str | Path,
    *,
    id_column: str | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    crop_column: str | None = None,
    default_crop: str | None = None,
    grid: GridSpec | None = None,
    max_rejected_messages: int = 50,
) -> AccessionFileReport:
    """Read an accession list and compute the grid cell of every row.

    Args:
        path: Excel or CSV file uploaded by the user.
        id_column: Column holding the accession identifier (auto-detected when
            ``None``).
        latitude_column: Latitude column (auto-detected when ``None``).
        longitude_column: Longitude column (auto-detected when ``None``).
        crop_column: Crop column (auto-detected when ``None``; optional).
        default_crop: Crop assigned to rows without a crop value.
        grid: Grid definition; the environment-configured grid when ``None``.
        max_rejected_messages: Cap on the number of rejection messages kept.

    Raises:
        AccessionFileError: If the file cannot be read or a required column is
            missing.
    """
    file_path = Path(path)
    frame = _read_table(file_path)
    columns = [str(column) for column in frame.columns]
    frame.columns = columns
    grid = grid or GridSpec.from_environment()

    id_col = _find_column(columns, ID_HEADERS, id_column)
    lat_col = _find_column(columns, LATITUDE_HEADERS, latitude_column)
    lon_col = _find_column(columns, LONGITUDE_HEADERS, longitude_column)
    crop_col = _find_column(columns, CROP_HEADERS, crop_column)
    taxon_col = _find_column(columns, TAXON_HEADERS, None)
    country_col = _find_column(columns, COUNTRY_HEADERS, None)

    missing = [
        role
        for role, column in (("id", id_col), ("latitude", lat_col), ("longitude", lon_col))
        if column is None
    ]

    # Without identifier and coordinates the file cannot feed the workflow.
    if missing:
        raise AccessionFileError(
            f"Could not detect the {', '.join(missing)} column(s) in {file_path.name}. "
            f"Available columns: {', '.join(columns)}. Specify them explicitly."
        )

    used_columns = {"id": id_col, "latitude": lat_col, "longitude": lon_col}

    # Optional roles are reported only when present.
    if crop_col:
        used_columns["crop"] = crop_col

    if taxon_col:
        used_columns["taxon"] = taxon_col

    if country_col:
        used_columns["country"] = country_col

    accessions: list[FileAccession] = []
    rejected: list[str] = []
    seen_ids: set[str] = set()
    duplicated: list[str] = []

    # Validate row by row; each rejection is explained for the user.
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        values = dict(zip(columns, row, strict=False))
        accession_id = _clean_text(values.get(id_col))
        latitude = _parse_coordinate(values.get(lat_col), -90.0, 90.0)
        longitude = _parse_coordinate(values.get(lon_col), -180.0, 180.0)

        if accession_id is None:
            _reject(rejected, max_rejected_messages, f"row {index}: empty identifier")
            continue

        if latitude is None or longitude is None:
            _reject(
                rejected,
                max_rejected_messages,
                f"row {index} ({accession_id}): invalid coordinates",
            )
            continue

        if accession_id in seen_ids:
            duplicated.append(accession_id)
            continue

        cellid = grid.cellid(latitude, longitude)

        if cellid is None:
            _reject(
                rejected,
                max_rejected_messages,
                f"row {index} ({accession_id}): outside the indicator grid",
            )
            continue

        seen_ids.add(accession_id)
        crop = _clean_text(values.get(crop_col)) if crop_col else None

        accessions.append(
            FileAccession(
                accession_id=accession_id,
                latitude=latitude,
                longitude=longitude,
                cellid=cellid,
                crop=crop or default_crop,
                taxon_name=_clean_text(values.get(taxon_col)) if taxon_col else None,
                country_code=_clean_text(values.get(country_col)) if country_col else None,
                row_number=index,
            )
        )

    return AccessionFileReport(
        file_name=file_path.name,
        columns=used_columns,
        total_rows=len(frame),
        accepted=len(accessions),
        rejected=rejected,
        duplicated_ids=list(dict.fromkeys(duplicated)),
        accessions=accessions,
    )


def _reject(messages: list[str], cap: int, message: str) -> None:
    """Append a rejection message unless the cap was reached.

    Args:
        messages: Accumulated messages.
        cap: Maximum number of messages kept.
        message: Message to add.
    """
    if len(messages) < cap:
        messages.append(message)
