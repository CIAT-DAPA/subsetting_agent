"""Manual smoke test of the Genesys trait endpoints against the real API.

Not part of the pytest suite. Uses the Genesys credentials in ``.env``::

    uv run python scripts/genesys_traits_smoke.py --from-csv tests/fixtures/candidate_list_beans_col.csv --limit 30
    uv run python scripts/genesys_traits_smoke.py --uuids <uuid1> <uuid2>
    uv run python scripts/genesys_traits_smoke.py --dataset <dataset uuid> --limit 30

It runs the four-step trait workflow (accessions -> datasets -> descriptors ->
observations) and prints RAW rows of the untyped responses so the parser in
``sdks/genesys/traits.py`` can be adjusted to the real shapes.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from core.config import get_settings  # noqa: E402
from core.logger import setup_logging  # noqa: E402
from sdks.genesys import (  # noqa: E402
    AccessionFilter,
    GenesysClient,
    GenesysError,
    accession_observations_to_rows,
    observations_to_dataframe,
)


def parse_args() -> argparse.Namespace:
    """Define and parse the command line arguments."""
    parser = argparse.ArgumentParser(description="Smoke test for the Genesys trait endpoints.")
    parser.add_argument("--from-csv", default=None, help="CSV exported by the agent (needs a UUID column)")
    parser.add_argument("--uuids", nargs="*", default=[], help="Accession UUIDs")
    parser.add_argument("--dataset", default=None, help="Inspect one dataset UUID directly")
    parser.add_argument("--limit", type=int, default=30, help="Accessions taken from the CSV")
    parser.add_argument("--rows", type=int, default=3, help="Raw rows printed per untyped response")
    return parser.parse_args()


def accession_uuids(args: argparse.Namespace) -> list[str]:
    """Collect accession UUIDs from the arguments."""
    uuids = list(args.uuids)

    # The agent exports carry a UUID column in Genesys mode.
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        column = next((c for c in frame.columns if c.lower() == "uuid"), None)

        if column is None:
            raise GenesysError(f"No UUID column in {args.from_csv}")

        uuids.extend(frame[column].dropna().astype(str).head(args.limit).tolist())

    return [uuid for uuid in dict.fromkeys(uuids) if uuid]


def main() -> int:
    """Run the trait workflow and print the findings."""
    load_dotenv()
    settings = get_settings()
    setup_logging(settings.log_level)
    args = parse_args()

    try:
        with GenesysClient.from_settings(settings) as client:
            print("Base URL :", client.base_url)
            uuids = accession_uuids(args)
            print(f"{len(uuids)} accession UUID(s)")

            # Step 1: datasets for the accessions (or the one given explicitly).
            datasets = [args.dataset] if args.dataset else client.find_datasets_for_uuids(uuids)
            print(f"Step 1 - datasets found: {len(datasets)}")

            if not datasets:
                print("No datasets hold trait data for these accessions. Try other accessions.")
                return 0

            for dataset_uuid in datasets[:5]:
                summary = client.get_dataset(dataset_uuid)
                print(f"  dataset {dataset_uuid}: '{summary.title}' crops={summary.crops} accessions={summary.accession_count} descriptors={summary.descriptor_count}")

            dataset_uuid = datasets[0]

            # Step 2: descriptors and accession references of the first dataset.
            descriptors = client.list_dataset_descriptors(dataset_uuid)
            print(f"\nStep 2 - {len(descriptors)} descriptor(s) in dataset {dataset_uuid} (uuid | columnName | dataType | category | uom | title):")
            for descriptor in descriptors[:25]:
                print(f"  {descriptor.uuid} | {descriptor.column_name} | {descriptor.data_type} | {descriptor.category} | {descriptor.uom} | {descriptor.title}")

            refs = []
            for index, ref in enumerate(client.iter_dataset_accessions(dataset_uuid, size=200)):
                refs.append(ref)
                if index >= 199:
                    break
            matched = sum(1 for ref in refs if ref.uuid)
            print(f"\nStep 2b - first {len(refs)} accession refs of the dataset ({matched} matched to Genesys accessions)")
            for ref in refs[: args.rows]:
                print("  RAW ref:", json.dumps(ref.model_dump(by_alias=True), default=str)[:400])

            # Step 3: observation rows for the accessions of interest.
            fields = [d.uuid for d in descriptors[:10]]
            accession_filter = AccessionFilter(uuid=uuids) if uuids and not args.dataset else None
            page = client.get_dataset_data([dataset_uuid], fields, accession_filter, page=0, size=50)
            print(f"\nStep 3 - dataset/data: {len(page.content)} row(s) in page 0, total {page.total_elements}")
            for row in page.content[: args.rows]:
                print("  RAW row:", json.dumps(row, default=str)[:800])

            frame = observations_to_dataframe(page.content, descriptors[:10])
            print("\nParsed table (first rows):")
            print(frame.head(args.rows).to_string(index=False)[:2000])

            # Step 3-alt: observations endpoint for one accession.
            if uuids:
                observations = client.get_accession_observations(uuids[0])
                print(f"\nStep 3-alt - acn/{uuids[0]}/observations keys: {list(observations)}")
                for key in ("firstPartyData", "thirdPartyData"):
                    items = observations.get(key) or []
                    print(f"  {key}: {len(items)} item(s)")
                    for item in items[: args.rows]:
                        print("  RAW:", json.dumps(item, default=str)[:800])

                # The per-accession payload is converted into the same row shape as dataset/data.
                alt_rows = accession_observations_to_rows(observations, dataset_uuid=dataset_uuid)
                alt_frame = observations_to_dataframe(alt_rows, descriptors[:10])
                print(f"  Parsed observations restricted to dataset {dataset_uuid}: {len(alt_frame)} row(s)")
                if not alt_frame.empty:
                    print(alt_frame.head(args.rows).to_string(index=False)[:1500])

            # Step 4: descriptor metadata.
            if descriptors:
                descriptor = client.get_descriptor(descriptors[0].uuid)
                print(f"\nStep 4 - descriptor {descriptor.uuid}: {descriptor.title} ({descriptor.data_type}, {descriptor.uom}); terms={len(descriptor.terms)}")

    except GenesysError as exc:
        print(f"Genesys error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
