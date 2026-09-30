"""Cell id computation on the base raster grid.

The indicator database is indexed by the cell id of a base raster. The id is
computed exactly like ``raster::cellFromXY`` in R, which is what the data
pipeline used: cells are numbered from 1, row by row, starting at the
north-west corner (top-left). Row 1 is the northernmost row, i.e. the one that
touches ``ymax = ymin + nrows * cellsize``.
"""

from __future__ import annotations

import math

import pandas as pd

from core.config import GridSettings


def compute_cellid(latitude: float, longitude: float, grid: GridSettings) -> int | None:
    """Return the 1-based cell id of a coordinate, or ``None`` when outside the grid.

    Args:
        latitude: Latitude in decimal degrees.
        longitude: Longitude in decimal degrees.
        grid: Base raster parameters.

    Returns:
        The cell id, or ``None`` for missing/invalid/out-of-extent coordinates.
    """
    # Missing or non-numeric coordinates cannot be located on the grid.
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None

    if math.isnan(lat) or math.isnan(lon):
        return None

    xmin = grid.subsetting_grid_xmin
    ymin = grid.subsetting_grid_ymin
    size = grid.subsetting_grid_cellsize
    ncols = grid.subsetting_grid_ncols
    nrows = grid.subsetting_grid_nrows
    xmax = xmin + ncols * size
    ymax = ymin + nrows * size

    # Points outside the raster extent have no cell (the eastern/northern edges are excluded).
    if lon < xmin or lon >= xmax or lat <= ymin or lat > ymax:
        return None

    col = int(math.floor((lon - xmin) / size))
    row = int(math.floor((ymax - lat) / size))

    # Points exactly on the northern edge fall in row 0, as in raster::cellFromXY.
    row = min(max(row, 0), nrows - 1)
    col = min(max(col, 0), ncols - 1)

    return row * ncols + col + 1


def add_cellid_column(
    frame: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    grid: GridSettings,
    output_column: str = "cellid",
) -> pd.DataFrame:
    """Append a cell id column computed from two coordinate columns.

    Args:
        frame: Table with the coordinates (values may be text).
        latitude_column: Name of the latitude column.
        longitude_column: Name of the longitude column.
        grid: Base raster parameters.
        output_column: Name of the new column.

    Returns:
        A copy of ``frame`` with ``output_column`` as nullable integers.
    """
    result = frame.copy()
    latitudes = pd.to_numeric(result[latitude_column], errors="coerce")
    longitudes = pd.to_numeric(result[longitude_column], errors="coerce")

    # Compute one id per row; ``None`` marks rows without a usable location.
    ids = [compute_cellid(lat, lon, grid) for lat, lon in zip(latitudes, longitudes)]
    result[output_column] = pd.array(ids, dtype="Int64")

    return result
