"""Manual smoke test of the Subsetting SDK against the real API.

Not part of the pytest suite. Requires ``SUBSETTING_API_URL`` and
``SUBSETTING_API_TOKEN`` (and/or ``SUBSETTING_ACCESS_TOKEN``) in ``.env``::

    uv run python scripts/subsetting_smoke.py --list-indicators
    uv run python scripts/subsetting_smoke.py --data --indicators cdd t_rain --coords 3.5,-76.35 4.7,-74.1
    uv run python scripts/subsetting_smoke.py --cluster --indicators cdd t_rain --coords 3.5,-76.35 4.7,-74.1 -12.0,-77.0 19.4,-99.1
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from sdks.subsetting import SubsettingClient, SubsettingError, compute_cellid  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Define and parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test for the Subsetting SDK.")
    parser.add_argument("--list-indicators", action="store_true", help="Print the indicator catalogue")
    parser.add_argument("--data", action="store_true", help="Fetch indicator values for the coordinates")
    parser.add_argument("--cluster", action="store_true", help="Cluster the coordinates by the indicators")
    parser.add_argument("--indicators", nargs="*", default=[], help="Indicator ids, prefixes or names")
    parser.add_argument("--coords", nargs="*", default=[], help="lat,lon pairs, e.g. 3.5,-76.35")
    parser.add_argument("--crop", default="generic", help="Crop name for crop-specific indicators")
    parser.add_argument("--period", default=None, help="Indicator period label (default from .env)")
    return parser.parse_args()


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

            # Cell ids from coordinates.
            cells: list[int] = []
            for pair in args.coords:
                lat, lon = (float(part) for part in pair.split(","))
                cell = compute_cellid(lat, lon, settings.grid)
                print(f"  ({lat}, {lon}) -> cellid {cell}")
                if cell is not None:
                    cells.append(cell)

            if args.data or args.cluster:
                ids = resolve_indicator_ids(client, args.indicators)
                print("Indicator ids:", ids, "| auth scheme:", client.auth.scheme)

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

    except SubsettingError as exc:
        print(f"Subsetting error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
