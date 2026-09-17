"""Grid used by the Subsetting API to locate collecting sites.

The API stores indicator values per grid cell (``cellid``). The grid comes from
``data/builder_indicators/raster_base.asc`` in the ``subsets_genebank_accessions``
repository::

    NCOLS 7198  NROWS 2000  XLLCORNER -180  YLLCORNER -50  CELLSIZE 0.05

Cell ids follow R ``raster::cellFromXY``: 1-based, row-major, starting at the
top-left corner (north-west). The grid parameters can be overridden with
environment variables in case the deployed raster differs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GridSpec:
    """Definition of a regular latitude/longitude grid.

    Attributes:
        ncols: Number of columns.
        nrows: Number of rows.
        xmin: Longitude of the western edge.
        ymin: Latitude of the southern edge.
        cellsize: Cell size in degrees (square cells).
    """

    ncols: int = 7198
    nrows: int = 2000
    xmin: float = -180.0
    ymin: float = -50.0
    cellsize: float = 0.05

    @property
    def xmax(self) -> float:
        """Longitude of the eastern edge."""
        return self.xmin + self.ncols * self.cellsize

    @property
    def ymax(self) -> float:
        """Latitude of the northern edge."""
        return self.ymin + self.nrows * self.cellsize

    @classmethod
    def from_environment(cls) -> GridSpec:
        """Build the grid from ``SUBSETTING_GRID_*`` variables, with raster defaults."""
        defaults = cls()

        return cls(
            ncols=int(os.getenv("SUBSETTING_GRID_NCOLS", defaults.ncols)),
            nrows=int(os.getenv("SUBSETTING_GRID_NROWS", defaults.nrows)),
            xmin=float(os.getenv("SUBSETTING_GRID_XMIN", defaults.xmin)),
            ymin=float(os.getenv("SUBSETTING_GRID_YMIN", defaults.ymin)),
            cellsize=float(os.getenv("SUBSETTING_GRID_CELLSIZE", defaults.cellsize)),
        )

    def contains(self, latitude: float, longitude: float) -> bool:
        """Whether a point falls inside the grid extent.

        The eastern and southern edges are exclusive, matching ``cellFromXY``,
        except that points exactly on the northern/western edge belong to the
        first row/column.

        Args:
            latitude: Decimal degrees.
            longitude: Decimal degrees.
        """
        return self.xmin <= longitude < self.xmax and self.ymin < latitude <= self.ymax

    def cellid(self, latitude: float, longitude: float) -> int | None:
        """Return the 1-based cell id of a point, or ``None`` when outside the grid.

        Args:
            latitude: Decimal degrees.
            longitude: Decimal degrees.
        """
        # Points outside the raster have no indicator data.
        if not self.contains(latitude, longitude):
            return None

        row = math.floor((self.ymax - latitude) / self.cellsize) + 1
        col = math.floor((longitude - self.xmin) / self.cellsize) + 1

        # A point exactly on the northern edge would compute row 0; clamp both
        # indexes to the valid range to absorb floating point edge cases.
        row = min(max(row, 1), self.nrows)
        col = min(max(col, 1), self.ncols)

        return (row - 1) * self.ncols + col

    def cell_center(self, cellid: int) -> tuple[float, float]:
        """Return the ``(latitude, longitude)`` of a cell centre.

        Args:
            cellid: 1-based cell id.

        Raises:
            ValueError: If the id is outside the grid.
        """
        if cellid < 1 or cellid > self.ncols * self.nrows:
            raise ValueError(f"cellid {cellid} is outside the grid.")

        row = (cellid - 1) // self.ncols + 1
        col = (cellid - 1) % self.ncols + 1
        latitude = self.ymax - (row - 0.5) * self.cellsize
        longitude = self.xmin + (col - 0.5) * self.cellsize

        return latitude, longitude


DEFAULT_GRID = GridSpec()


def cellid_from_coordinates(
    latitude: float, longitude: float, grid: GridSpec | None = None
) -> int | None:
    """Compute the Subsetting API cell id of a collecting site.

    Args:
        latitude: Decimal degrees.
        longitude: Decimal degrees.
        grid: Grid definition; the environment-configured grid when ``None``.
    """
    return (grid or GridSpec.from_environment()).cellid(latitude, longitude)
