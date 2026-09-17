"""Tools for conversations whose accessions come from a user spreadsheet.

In file mode the passport stage is replaced by loading the user's Excel/CSV:
identifiers and coordinates are read, the Subsetting grid cell of every row is
computed, and the selection starts from there. No Genesys call is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from accession_files.reader import AccessionFileError, read_accession_file
from subsetting_sdk.catalog import normalize_text
from tools.accession_context import AccessionRecord
from tools.services import ToolServices


def _resolve_file(services: ToolServices, file_name: str | None) -> Path:
    """Pick the accession file to load.

    Args:
        services: Shared services holding the uploaded files.
        file_name: Name (or distinctive part) of the wanted file; when ``None``
            and only one file exists, that file is used.

    Raises:
        AccessionFileError: If no file matches or the choice is ambiguous.
    """
    files = services.accession_files

    if not files:
        raise AccessionFileError("No accession spreadsheet was uploaded in this conversation.")

    # A single upload needs no name; several require the user to pick one.
    if file_name is None:
        if len(files) == 1:
            return files[0]

        raise AccessionFileError(
            "Several accession files were uploaded; specify file_name. Available: "
            + ", ".join(path.name for path in files)
        )

    wanted = normalize_text(file_name)
    matches = [path for path in files if wanted and wanted in normalize_text(path.name)]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise AccessionFileError(
            f"No uploaded file matches '{file_name}'. Available: "
            + ", ".join(path.name for path in files)
        )

    raise AccessionFileError(
        f"'{file_name}' matches several files: " + ", ".join(path.name for path in matches)
    )


async def list_accession_files(services: ToolServices) -> dict[str, Any]:
    """List the accession spreadsheets the user uploaded.

    Args:
        services: Shared services.
    """
    return {
        "count": len(services.accession_files),
        "files": [
            {"file_name": path.name, "size_kb": round(path.stat().st_size / 1024, 1)}
            for path in services.accession_files
            if path.exists()
        ],
    }


async def load_accessions_from_file(
    services: ToolServices,
    *,
    file_name: str | None = None,
    id_column: str | None = None,
    latitude_column: str | None = None,
    longitude_column: str | None = None,
    crop_column: str | None = None,
    default_crop: str | None = None,
) -> dict[str, Any]:
    """Read the user's accession spreadsheet and start the selection from it.

    Args:
        services: Shared services.
        file_name: Which uploaded file to read (optional when only one exists).
        id_column: Accession identifier column (auto-detected when omitted).
        latitude_column: Latitude column (auto-detected when omitted).
        longitude_column: Longitude column (auto-detected when omitted).
        crop_column: Crop column (auto-detected when omitted; optional).
        default_crop: Crop assigned to rows without one; needed for crop-specific
            climate indicators.
    """
    try:
        path = _resolve_file(services, file_name)
        report = read_accession_file(
            path,
            id_column=id_column,
            latitude_column=latitude_column,
            longitude_column=longitude_column,
            crop_column=crop_column,
            default_crop=default_crop,
        )

    except AccessionFileError as exc:
        return {"error": str(exc)}

    # A file with no usable rows must not silently produce an empty selection.
    if not report.accessions:
        return {
            "error": (
                f"No valid accession rows were found in {report.file_name}. "
                "Check the identifier and coordinate columns."
            ),
            "report": report.summary(),
        }

    records = [
        AccessionRecord(
            uuid=item.accession_id,
            accession_number=item.accession_id,
            crop=item.crop,
            taxon_name=item.taxon_name,
            country_code=item.country_code,
            latitude=item.latitude,
            longitude=item.longitude,
            cellid=item.cellid,
        )
        for item in report.accessions
    ]
    description = (
        f"Loaded {len(records)} accessions from file {report.file_name} (columns: {report.columns})"
    )
    services.context.set_file_selection(
        records, file_name=report.file_name, description=description
    )

    result = {"report": report.summary(), "summary": services.context.summary()}

    # Crop-specific indicators need a crop; tell the model when none is known.
    if all(record.crop is None for record in records):
        result["note"] = (
            "The file has no crop column and no default_crop was given. Generic climate "
            "indicators work; crop-specific ones need a crop. Ask the user for the crop if needed."
        )

    return result
