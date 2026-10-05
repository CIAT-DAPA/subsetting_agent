"""Diagnose which cell-id convention the Subsetting database uses.

The indicator database is indexed by ``cellid`` but the raster that generated
those ids (``raster_base_complete.asc``) is not available. This script computes
the cell ids of real accession coordinates under several plausible conventions,
sends all of them in one ``indicators-data`` request and reports which
convention produced ids that hold data.

Usage (from the project root, with the Subsetting credentials in ``.env``)::

    uv run python scripts/subsetting_cellid_probe.py --from-csv tests/fixtures/candidate_list_beans_col.csv
    uv run python scripts/subsetting_cellid_probe.py --from-csv <csv> --indicator CDD --all-periods
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from sdks.subsetting import SubsettingClient, SubsettingError, SubsettingNoDataError  # noqa: E402
from sdks.subsetting.client import INDICATORS_DATA_PATH  # noqa: E402


@dataclass(frozen=True)
class Hypothesis:
    """One candidate raster convention.

    Attributes:
        name: Short label shown in the report.
        ncols: Number of columns.
        nrows: Number of rows.
        xmin: Western edge.
        ymin: Southern edge.
        size: Cell size in degrees.
        one_based: Whether ids start at 1 (R) or 0 (Python/GDAL).
        top_down: Whether row 0/1 is the northern row (raster) or the southern one.
    """

    name: str
    ncols: int
    nrows: int
    xmin: float
    ymin: float
    size: float
    one_based: bool = True
    top_down: bool = True

    def cellid(self, lat: float, lon: float) -> int | None:
        """Compute the cell id of a coordinate under this convention."""
        ymax = self.ymin + self.nrows * self.size
        xmax = self.xmin + self.ncols * self.size

        # Outside the extent: no cell.
        if lon < self.xmin or lon >= xmax or lat <= self.ymin or lat > ymax:
            return None

        col = int(math.floor((lon - self.xmin) / self.size))
        row = int(math.floor((ymax - lat) / self.size)) if self.top_down else int(math.floor((lat - self.ymin) / self.size))
        row = min(max(row, 0), self.nrows - 1)
        col = min(max(col, 0), self.ncols - 1)

        return row * self.ncols + col + (1 if self.one_based else 0)


def build_hypotheses() -> list[Hypothesis]:
    """Return the conventions worth testing."""
    return [
        Hypothesis("A: 7198x2000 ymin=-50, 1-based, top-down (legacy repo raster)", 7198, 2000, -180, -50, 0.05),
        Hypothesis("B: 7198x2000 ymin=-50, 0-based, top-down", 7198, 2000, -180, -50, 0.05, one_based=False),
        Hypothesis("C: 7200x2000 ymin=-50, 1-based, top-down", 7200, 2000, -180, -50, 0.05),
        Hypothesis("D: 7200x2000 ymin=-50, 0-based, top-down", 7200, 2000, -180, -50, 0.05, one_based=False),
        Hypothesis("E: 7200x3600 global (-90..90), 1-based, top-down (CONFIRMED 2026-09-29)", 7200, 3600, -180, -90, 0.05),
        Hypothesis("F: 7200x3600 global (-90..90), 0-based, top-down", 7200, 3600, -180, -90, 0.05, one_based=False),
        Hypothesis("G: 7198x2000 ymin=-50, 1-based, bottom-up", 7198, 2000, -180, -50, 0.05, top_down=False),
        Hypothesis("H: 7198x2000 ymin=-50, 0-based, bottom-up", 7198, 2000, -180, -50, 0.05, one_based=False, top_down=False),
        Hypothesis("I: 7200x2800 (-90..50), 1-based, top-down", 7200, 2800, -180, -90, 0.05),
        Hypothesis("J: 7198x2000 ymin=-60 (ymax 40), 1-based, top-down", 7198, 2000, -180, -60, 0.05),
    ]


def parse_args() -> argparse.Namespace:
    """Define and parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Probe the cell-id convention of the Subsetting database.")
    parser.add_argument("--from-csv", required=True, help="CSV with DECLATITUDE/DECLONGITUDE columns")
    parser.add_argument("--indicator", default="CDD", help="Indicator id/pref used for the probe (default CDD)")
    parser.add_argument("--all-periods", action="store_true", help="Send every period of the indicator, not only 'mean'")
    parser.add_argument("--max-rows", type=int, default=200, help="Coordinates used from the CSV")
    return parser.parse_args()


def main() -> int:
    """Run the probe and print which hypotheses hold data."""
    load_dotenv()
    settings = get_settings()
    setup_logging(settings.log_level)
    args = parse_args()

    frame = pd.read_csv(args.from_csv).dropna(subset=["DECLATITUDE", "DECLONGITUDE"]).head(args.max_rows)
    coordinates = [(float(a), float(b)) for a, b in zip(frame["DECLATITUDE"], frame["DECLONGITUDE"])]
    print(f"{len(coordinates)} coordinates from {Path(args.from_csv).name}")

    hypotheses = build_hypotheses()
    ids_by_hypothesis: dict[str, set[int]] = {}

    # Compute the candidate ids of every coordinate under every convention.
    for hypothesis in hypotheses:
        ids = {hypothesis.cellid(lat, lon) for lat, lon in coordinates}
        ids.discard(None)
        ids_by_hypothesis[hypothesis.name] = ids  # type: ignore[arg-type]

    all_ids = sorted(set().union(*ids_by_hypothesis.values()))
    print(f"{len(all_ids)} distinct candidate cell ids across {len(hypotheses)} hypotheses")

    try:
        with SubsettingClient.from_settings(settings) as client:
            indicator = client.get_indicator(args.indicator)
            if indicator is None:
                raise SubsettingError(f"Indicator '{args.indicator}' not found")

            periods = client.list_indicator_periods()
            if args.all_periods:
                period_ids = [p.id for p in periods if p.indicator == indicator.id]
            else:
                period_ids = client.resolve_periods([indicator.id])[indicator.id]
            print(f"Indicator {indicator.pref} ({indicator.id}); {len(period_ids)} period id(s)")

            try:
                payload = client._request_json(  # noqa: SLF001 - diagnostic access
                    "POST", INDICATORS_DATA_PATH, body={"cellid": all_ids, "indicators": period_ids}
                )
            except SubsettingNoDataError:
                payload = {"response": []}

    except SubsettingError as exc:
        print(f"Subsetting error: {exc}")
        return 1

    found = {int(entry["cellid"]) for entry in payload.get("response", []) if entry.get("cellid") is not None}
    print(f"\nCells with data returned by the API: {len(found)}")

    # Report how many ids of each hypothesis were found in the database.
    for hypothesis in hypotheses:
        ids = ids_by_hypothesis[hypothesis.name]
        hits = len(ids & found)
        marker = "  <== MATCH" if hits and hits >= 0.5 * len(ids) else ""
        print(f"  {hypothesis.name}: {hits}/{len(ids)} cells with data{marker}")

    if not found:
        print(
            "\nNo hypothesis returned data. Either the database only covers other crops/cells, or the "
            "raster differs from every convention tested (a header of raster_base_complete.asc would settle it)."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
