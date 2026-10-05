"""Manual smoke test of the Subsetting SDK against the real API.

Not part of the pytest suite. Requires ``SUBSETTING_API_URL`` and
``SUBSETTING_API_TOKEN`` (and/or ``SUBSETTING_ACCESS_TOKEN``) in ``.env``::

    uv run python scripts/subsetting_smoke.py --list-indicators
    uv run python scripts/subsetting_smoke.py --data --indicators CDD t_rain --coords "3.5,-76.35 -12.0,-77.0"
    uv run python scripts/subsetting_smoke.py --cluster --indicators CDD t_rain --from-genesys \\
        --genus Phaseolus --species vulgaris --country COL --institute COL003 --limit 30
    uv run python scripts/subsetting_smoke.py --data --indicators CDD t_rain --from-csv tests/fixtures/candidate_list_beans_col.csv

Coordinates can be given as ``lat,lon`` pairs separated by spaces or ``;``
(negative values are fine) or taken from real accessions in Genesys with
``--from-genesys`` (indicator data only exists for cells that hold accessions).
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from sdks.genesys import AccessionFilter, CountryFilter, GenesysClient, InstituteFilter, TaxonomyFilter  # noqa: E402
from sdks.subsetting import (  # noqa: E402
    SubsettingClient,
    SubsettingError,
    SubsettingRequestError,
    compute_cellid,
)


def parse_args() -> argparse.Namespace:
    """Define and parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test for the Subsetting SDK.")
    parser.add_argument("--list-indicators", action="store_true", help="Print the indicator catalogue")
    parser.add_argument("--data", action="store_true", help="Fetch indicator values for the coordinates")
    parser.add_argument("--cluster", action="store_true", help="Cluster the coordinates by the indicators")
    parser.add_argument("--indicators", nargs="*", default=[], help="Indicator ids, prefixes or names")
    parser.add_argument(
        "--coords",
        default="",
        help='lat,lon pairs separated by spaces or ";", e.g. "3.5,-76.35 -12.0,-77.0" (quote the string)',
    )
    parser.add_argument("--crop", default="generic", help="Crop name for crop-specific indicators")
    parser.add_argument("--period", default=None, help="Indicator period label (default from .env)")
    # Coordinates taken from real accessions in Genesys.
    parser.add_argument("--from-genesys", action="store_true", help="Take coordinates from Genesys accessions")
    parser.add_argument("--genus", nargs="*", default=None, help="Genesys filter: genera")
    parser.add_argument("--species", nargs="*", default=None, help="Genesys filter: specific epithets")
    parser.add_argument("--country", nargs="*", default=None, help="Genesys filter: ISO3 country of origin")
    parser.add_argument("--institute", nargs="*", default=None, help="Genesys filter: WIEWS institute codes")
    parser.add_argument("--limit", type=int, default=30, help="Genesys: maximum accessions to download")
    # Coordinates taken from a CSV exported by the agent (Candidate/Original list).
    parser.add_argument(
        "--from-csv", default=None, help="CSV with DECLATITUDE/DECLONGITUDE (or latitude/longitude) columns"
    )
    parser.add_argument("--max-cells", type=int, default=0, help="Use at most this many cells (0 = all)")
    return parser.parse_args()


def coordinates_from_csv(path: str) -> list[tuple[float, float]]:
    """Read coordinates from a CSV exported by the agent.

    Args:
        path: CSV file with latitude/longitude columns (MCPD names or plain).

    Returns:
        ``(latitude, longitude)`` of the rows with both values.

    Raises:
        SubsettingError: If the coordinate columns cannot be found.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    lowered = {str(column).lower(): column for column in frame.columns}
    latitude = lowered.get("declatitude") or lowered.get("latitude") or lowered.get("lat")
    longitude = lowered.get("declongitude") or lowered.get("longitude") or lowered.get("lon")

    if not latitude or not longitude:
        raise SubsettingError(f"No coordinate columns found in {path}. Columns: {list(frame.columns)}")

    frame = frame.dropna(subset=[latitude, longitude])
    print(f"CSV {Path(path).name}: {len(frame)} georeferenced rows")

    return [(float(lat), float(lon)) for lat, lon in zip(frame[latitude], frame[longitude])]


def parse_coordinates(text: str) -> list[tuple[float, float]]:
    """Parse ``lat,lon`` pairs from a string.

    Args:
        text: Pairs separated by whitespace or ``;`` (e.g. ``"3.5,-76.35 -12,-77"``).

    Returns:
        List of ``(latitude, longitude)`` tuples.

    Raises:
        SubsettingError: If a pair is malformed.
    """
    pairs: list[tuple[float, float]] = []

    # Accept both separators so the argument never needs escaping tricks.
    for token in text.replace(";", " ").split():
        parts = token.split(",")

        if len(parts) != 2:
            raise SubsettingError(f"Malformed coordinate '{token}'. Use lat,lon such as 3.5,-76.35")

        pairs.append((float(parts[0]), float(parts[1])))

    return pairs


def coordinates_from_genesys(args: argparse.Namespace) -> list[tuple[float, float]]:
    """Download accessions from Genesys and return their coordinates.

    Args:
        args: Parsed arguments (Genesys filter and limit).

    Returns:
        ``(latitude, longitude)`` of the georeferenced accessions found.
    """
    accession_filter = AccessionFilter(
        taxonomy=TaxonomyFilter(genus=args.genus, species=args.species),
        country_of_origin=CountryFilter(code3=args.country),
        institute=InstituteFilter(code=args.institute),
    )

    if accession_filter.is_empty():
        raise SubsettingError("--from-genesys needs at least one of --genus/--species/--country/--institute")

    print("Genesys filter:", json.dumps(accession_filter.to_body()))

    with GenesysClient.from_settings(get_settings()) as genesys:
        frame = genesys.list_accessions_dataframe(accession_filter, max_records=args.limit)

    frame = frame.dropna(subset=["DECLATITUDE", "DECLONGITUDE"])
    print(f"Genesys returned {len(frame)} georeferenced accessions")

    return [(float(lat), float(lon)) for lat, lon in zip(frame["DECLATITUDE"], frame["DECLONGITUDE"])]


def resolve_indicator_ids(client: SubsettingClient, names: list[str]) -> list[str]:
    """Translate ids/prefixes/names into indicator ids, failing on unknown ones."""
    ids: list[str] = []

    # Exact lookup first, then free-text search as a fallback.
    for name in names:
        indicator = client.get_indicator(name)

        if indicator is None:
            candidates = client.find_indicators(name)
            if len(candidates) != 1:
                raise SubsettingError(
                    f"Indicator '{name}' not found or ambiguous. Candidates: {[c.pref for c in candidates]}"
                )
            indicator = candidates[0]

        ids.append(indicator.id)

    return ids


def main() -> int:
    """Run the requested smoke checks."""
    load_dotenv()
    settings = get_settings()
    setup_logging(settings.log_level)
    args = parse_args()

    try:
        with SubsettingClient.from_settings(settings) as client:
            print("Base URL :", client.base_url)

            # Catalogue listing.
            if args.list_indicators:
                indicators = client.list_indicators()
                print(f"{len(indicators)} indicators (category | id | pref | type | crop | unit | name):")
                for indicator in sorted(indicators, key=lambda i: (i.category or "", i.pref)):
                    print(
                        f"  {indicator.category} | {indicator.id} | {indicator.pref} | {indicator.indicator_type}"
                        f" | {indicator.crop} | {indicator.unit} | {indicator.name}"
                    )
                periods = client.list_indicator_periods()
                labels = sorted({p.period for p in periods})
                print(f"{len(periods)} indicator periods; labels: {labels}")

            # Coordinates: explicit pairs and/or real accessions from Genesys.
            coordinates = parse_coordinates(args.coords)
            if args.from_genesys:
                coordinates.extend(coordinates_from_genesys(args))
            if args.from_csv:
                coordinates.extend(coordinates_from_csv(args.from_csv))

            # Optional cap to keep the requests small during smoke tests.
            if args.max_cells and len(coordinates) > args.max_cells:
                coordinates = coordinates[: args.max_cells]

            cells: list[int] = []
            for lat, lon in coordinates:
                cell = compute_cellid(lat, lon, settings.grid)
                if len(coordinates) <= 20:
                    print(f"  ({lat}, {lon}) -> cellid {cell}")
                if cell is not None:
                    cells.append(cell)
            print(f"{len(cells)} cells from {len(coordinates)} coordinates")

            if args.data or args.cluster:
                ids = resolve_indicator_ids(client, args.indicators)
                print("Indicator ids:", ids, "| distinct cells:", len(set(cells)))

            if args.data:
                frame = client.get_indicators_data(cells, ids, period=args.period)
                print(f"indicators-data -> {len(frame)} rows")
                print(frame.head(20).to_string(index=False))

            if args.cluster:
                result = client.generate_clusters(cells, ids, crop=args.crop, period=args.period)
                print(f"cluster -> {result.cluster_count} clusters via {result.cluster_column}")
                print(json.dumps(result.assignments, indent=2))
                print("summary:", json.dumps(result.summary[:5], indent=2, default=str))

            print("Accepted auth scheme:", client.auth.scheme)

    except SubsettingRequestError as exc:
        # Show the server body (Flask traceback when debug is on) to diagnose quickly.
        print(f"Subsetting error: {exc}")
        print("Response body:", exc.body[:1500])
        return 1
    except SubsettingError as exc:
        print(f"Subsetting error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
