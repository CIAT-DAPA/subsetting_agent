"""Manual smoke test of the Genesys SDK against the real API.

It is NOT part of the pytest suite. Run it from the project root once the
``.env`` file contains ``GENESYS_API_TOKEN`` (or client credentials)::

    uv run python scripts/genesys_smoke.py
    uv run python scripts/genesys_smoke.py --genus Phaseolus --species vulgaris --country COL --limit 20
    uv run python scripts/genesys_smoke.py --list-crops

The script prints the request body that is sent, the total number of matching
accessions and a preview of the flattened MCPD table.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running the script directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from sdks.genesys import (  # noqa: E402
    AccessionFilter,
    CountryFilter,
    GenesysClient,
    GenesysError,
    InstituteFilter,
    TaxonomyFilter,
)


def parse_args() -> argparse.Namespace:
    """Define and parse the command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Smoke test for the Genesys SDK.")
    parser.add_argument("--crop", nargs="*", default=None, help="Crop names or codes, resolved against the Genesys catalogue")
    parser.add_argument("--genus", nargs="*", default=["Phaseolus"], help="Genera")
    parser.add_argument("--species", nargs="*", default=None, help="Specific epithets")
    parser.add_argument("--country", nargs="*", default=None, help="ISO3 country of origin codes")
    parser.add_argument("--institute", nargs="*", default=None, help="FAO WIEWS institute codes")
    parser.add_argument("--text", default=None, help="Full-text keywords")
    parser.add_argument("--limit", type=int, default=10, help="Maximum records to download")
    parser.add_argument(
        "--list-crops", action="store_true", help="Print the Genesys crop catalogue and exit"
    )
    return parser.parse_args()


def build_filter(args: argparse.Namespace) -> AccessionFilter:
    """Translate the command line arguments into an ``AccessionFilter``.

    Args:
        args: Parsed arguments.

    Returns:
        The filter to send to Genesys.
    """
    return AccessionFilter(
        crop=args.crop,
        text=args.text,
        taxonomy=TaxonomyFilter(genus=args.genus, species=args.species),
        country_of_origin=CountryFilter(code3=args.country),
        institute=InstituteFilter(code=args.institute),
    )


def main() -> int:
    """Run the smoke test.

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    load_dotenv()
    settings = get_settings()
    setup_logging(settings.log_level)
    args = parse_args()

    # Catalogue mode: show the real crop codes so filters can be written correctly.
    if args.list_crops:
        try:
            with GenesysClient.from_settings(settings) as client:
                crops = client.list_crops()
        except GenesysError as exc:
            print(f"Genesys error: {exc}")
            return 1

        print(f"{len(crops)} crops in Genesys (shortName | name | otherNames):")
        for crop in sorted(crops, key=lambda item: item.short_name):
            print(f"  {crop.short_name} | {crop.name} | {', '.join(crop.other_names)}")
        return 0

    accession_filter = build_filter(args)

    print("Base URL :", settings.genesys_api_url)
    print("Body sent:", json.dumps(accession_filter.to_body(), indent=2))

    try:
        with GenesysClient.from_settings(settings) as client:
            total = client.count_accessions(accession_filter)
            print(f"Matching accessions: {total}")

            frame = client.list_accessions_dataframe(accession_filter, max_records=args.limit)
    except GenesysError as exc:
        # Report SDK errors in plain language and fail the script.
        print(f"Genesys error: {exc}")
        return 1

    print(f"Downloaded {len(frame)} records. Preview:")
    columns = ["INSTCODE", "ACCENUMB", "GENUS", "SPECIES", "ORIGCTY", "SAMPSTAT", "DECLATITUDE", "DECLONGITUDE"]
    print(frame[columns].head(15).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
