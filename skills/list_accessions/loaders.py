"""Readers for local accession files (Excel and CSV).

In local mode every column of the file is considered passport data, so the
loaders keep all columns untouched except for a light clean-up of the headers.
"""

import csv
from pathlib import Path

import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

# File extensions handled by each reader.
EXCEL_EXTENSIONS: tuple[str, ...] = (".xlsx", ".xlsm", ".xls")
CSV_EXTENSIONS: tuple[str, ...] = (".csv", ".txt", ".tsv")
SUPPORTED_EXTENSIONS: tuple[str, ...] = EXCEL_EXTENSIONS + CSV_EXTENSIONS

# Lower-case column names commonly used for coordinates in passport data.
LATITUDE_ALIASES: tuple[str, ...] = (
    "latitude",
    "decimallatitude",
    "declatitude",
    "lat",
    "latitud",
    "y",
)
LONGITUDE_ALIASES: tuple[str, ...] = (
    "longitude",
    "decimallongitude",
    "declongitude",
    "lon",
    "long",
    "lng",
    "longitud",
    "x",
)


class UnsupportedFileError(ValueError):
    """Raised when the file extension is not one of the supported formats."""


def normalize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Clean column headers without changing their meaning.

    Leading/trailing spaces are removed, unnamed columns produced by empty
    Excel headers get a stable name and duplicated names receive a suffix.

    Args:
        dataframe: Table as read from the file.

    Returns:
        The same table with cleaned headers.
    """
    cleaned: list[str] = []
    seen: dict[str, int] = {}

    # Walk the headers in order so duplicates get deterministic suffixes.
    for position, column in enumerate(dataframe.columns):
        name = str(column).strip()

        # pandas names empty headers "Unnamed: N"; make that explicit and stable.
        if not name or name.lower().startswith("unnamed"):
            name = f"column_{position + 1}"

        # Suffix repeated names so every column stays addressable.
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0

        cleaned.append(name)

    dataframe = dataframe.copy()
    dataframe.columns = cleaned
    return dataframe


def _detect_csv_delimiter(path: Path) -> str:
    """Guess the delimiter of a delimited text file.

    Args:
        path: File to inspect.

    Returns:
        The detected delimiter, defaulting to ``,`` when detection fails.
    """
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        sample = handle.read(20_000)

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # The sniffer gives up on very regular or very small files; assume comma.
        return ","


def read_accession_file(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Read an accession table from an Excel or delimited text file.

    Args:
        path: File to read.
        sheet_name: Excel sheet to read; the first sheet when ``None``.

    Returns:
        A DataFrame with cleaned headers and fully empty rows removed.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        UnsupportedFileError: If the extension is not supported.
        ValueError: If the requested sheet does not exist or the file is empty.
    """
    path = Path(path)

    # Fail early with a clear message instead of a pandas stack trace.
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    # Route to the right reader according to the extension.
    if suffix in EXCEL_EXTENSIONS:
        dataframe = _read_excel(path, sheet_name)
    elif suffix in CSV_EXTENSIONS:
        dataframe = pd.read_csv(
            path, sep=_detect_csv_delimiter(path), encoding="utf-8-sig", dtype=str
        )
    else:
        raise UnsupportedFileError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    dataframe = normalize_columns(dataframe)
    dataframe = dataframe.dropna(how="all").reset_index(drop=True)

    # A table without rows cannot become an Original list.
    if dataframe.empty:
        raise ValueError(f"The file {path.name} does not contain any accession rows.")

    logger.info("Read %s rows and %s columns from %s", *dataframe.shape, path.name)
    return dataframe


def _read_excel(path: Path, sheet_name: str | None) -> pd.DataFrame:
    """Read one sheet of an Excel workbook.

    Args:
        path: Workbook to read.
        sheet_name: Sheet to read; the first one when ``None``.

    Returns:
        The sheet as a DataFrame (all cells read as text).

    Raises:
        ValueError: If ``sheet_name`` is not present in the workbook.
    """
    workbook = pd.ExcelFile(path)

    # Validate the sheet name up front to give the model a useful error.
    if sheet_name is not None and sheet_name not in workbook.sheet_names:
        raise ValueError(
            f"Sheet '{sheet_name}' not found in {path.name}. "
            f"Available sheets: {workbook.sheet_names}"
        )

    target = sheet_name if sheet_name is not None else workbook.sheet_names[0]
    return workbook.parse(target, dtype=str)


def detect_coordinate_columns(columns: list[str]) -> dict[str, str | None]:
    """Find the columns that most likely hold latitude and longitude.

    Args:
        columns: Column names of the accession table.

    Returns:
        ``{"latitude": <name or None>, "longitude": <name or None>}``.
    """
    lowered = {str(column).strip().lower().replace("_", "").replace(" ", ""): column for column in columns}
    result: dict[str, str | None] = {"latitude": None, "longitude": None}

    # Aliases are ordered from most to least specific; keep the first match.
    for alias in LATITUDE_ALIASES:
        if alias in lowered:
            result["latitude"] = str(lowered[alias])
            break

    for alias in LONGITUDE_ALIASES:
        if alias in lowered:
            result["longitude"] = str(lowered[alias])
            break

    return result
