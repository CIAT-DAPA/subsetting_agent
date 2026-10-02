"""Trait (characterization & evaluation) data of the Genesys API v2.

The Genesys spec defines a four step workflow to reach phenotypic observations:

1. ``POST /api/v2/dataset/accessions-datasets`` - dataset UUIDs holding data of
   the accessions matched by an ``AccessionFilter``.
2. ``GET /api/v2/dataset/{uuid}/descriptors`` - descriptors (traits) of a dataset;
   ``GET /api/v2/dataset/{uuid}`` gives the dataset summary and
   ``GET /api/v2/dataset/accessions/{uuid}`` the accessions it contains.
3. ``POST /api/v2/dataset/data?datasetUuids=&fields=`` - observation rows
   (``PageObject`` whose rows the spec leaves untyped);
   ``GET /api/v2/acn/{uuid}/observations`` - observations of one accession.
4. ``GET /api/v2/descriptor/{uuid}`` - descriptor metadata.

Row parsing is isolated in :func:`observations_to_dataframe` so it can be
adjusted once real responses are inspected (``scripts/genesys_traits_smoke.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from core.logger import get_logger
from sdks.genesys.models import AccessionFilter

logger = get_logger(__name__)

ACCESSIONS_DATASETS_PATH = "/api/v2/dataset/accessions-datasets"
DATASET_PATH = "/api/v2/dataset/{uuid}"
DATASET_DESCRIPTORS_PATH = "/api/v2/dataset/{uuid}/descriptors"
DATASET_ACCESSIONS_PATH = "/api/v2/dataset/accessions/{uuid}"
DATASET_DATA_PATH = "/api/v2/dataset/data"
ACCESSION_OBSERVATIONS_PATH = "/api/v2/acn/{uuid}/observations"
DESCRIPTOR_PATH = "/api/v2/descriptor/{uuid}"

# Hard limit of the JSON page size in Genesys.
MAX_PAGE_SIZE = 1000

# Descriptor data types considered numeric / categorical.
NUMERIC_TYPES = frozenset({"NUMERIC"})
CATEGORICAL_TYPES = frozenset({"SCALE", "CODED", "BOOLEAN", "TEXT"})

# Keys an observation row may use to identify the accession (checked in order).
# Real ``/dataset/data`` rows (verified 2026-10-02) look like
# ``{"accession": "<uuid>", "accessionNumber": "G7236", "doi": "...", "<descriptor uuid>": [55.9]}``.
ACCESSION_KEY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "uuid": ("accession", "uuid", "accessionUuid", "accession.uuid", "accessionRef.accession.uuid"),
    "doi": ("doi", "accessionRef.doi", "accession.doi"),
    "accessionNumber": ("accessionNumber", "acceNumb", "accessionRef.acceNumb", "accession.accessionNumber"),
    "instituteCode": ("instituteCode", "instCode", "accessionRef.instCode", "accession.instituteCode"),
}


# ------------------------------------------------------------------ models
class VocabularyTerm(BaseModel):
    """A coded value of a SCALE/CODED descriptor."""

    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    title: str | None = None
    description: str | None = None


class Descriptor(BaseModel):
    """A trait descriptor (``TranslatedDescriptorDTO``).

    Attributes:
        uuid: Identifier used in ``fields`` of the data endpoint.
        title: Human readable name.
        column_name: Short column name used in dataset tables (``columnName``).
        data_type: ``NUMERIC``, ``TEXT``, ``SCALE``, ``CODED``, ``BOOLEAN`` or ``DATE``.
        category: ``CHARACTERIZATION``, ``EVALUATION``, ``ABIOTICSTRESS``, ``BIOTICSTRESS``...
        uom: Unit of measurement.
        crop: Crop the descriptor belongs to.
        min_value: Lower bound for numeric descriptors.
        max_value: Upper bound for numeric descriptors.
        description: Methodology text.
        terms: Coded values for categorical descriptors.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    uuid: str
    title: str | None = None
    column_name: str | None = Field(default=None, alias="columnName")
    data_type: str | None = Field(default=None, alias="dataType")
    category: str | None = None
    uom: str | None = None
    crop: str | None = None
    min_value: float | None = Field(default=None, alias="minValue")
    max_value: float | None = Field(default=None, alias="maxValue")
    description: str | None = None
    terms: list[VocabularyTerm] = Field(default_factory=list)

    @property
    def label(self) -> str:
        """Column label used when observations are merged into a table."""
        return self.column_name or self.title or self.uuid

    @property
    def is_numeric(self) -> bool:
        """Whether values are numbers."""
        return (self.data_type or "").upper() in NUMERIC_TYPES

    @property
    def is_categorical(self) -> bool:
        """Whether values are codes/classes/text."""
        return (self.data_type or "").upper() in CATEGORICAL_TYPES

    def matches(self, text: str) -> bool:
        """Whether a free-text query refers to this descriptor.

        Args:
            text: Query typed by a user or the model.

        Returns:
            ``True`` when the query appears in the title, column name, category,
            description or uuid (case-insensitive).
        """
        needle = text.strip().lower()

        # Empty queries never match.
        if not needle:
            return False

        haystack = [self.uuid, self.title or "", self.column_name or "", self.category or "", self.description or ""]
        return any(needle in item.lower() for item in haystack if item)


class DatasetSummary(BaseModel):
    """Dataset metadata (``TranslatedDatasetDTO`` / ``DatasetInfo``)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    uuid: str
    title: str | None = None
    description: str | None = None
    crops: list[str] = Field(default_factory=list)
    accession_count: int | None = Field(default=None, alias="accessionCount")
    descriptor_count: int | None = Field(default=None, alias="descriptorCount")
    descriptors: list[Descriptor] = Field(default_factory=list)
    published: bool | None = None


class DatasetAccessionRef(BaseModel):
    """An accession referenced by a dataset (``DatasetAccessionRefDTO``)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    doi: str | None = None
    inst_code: str | None = Field(default=None, alias="instCode")
    acce_numb: str | None = Field(default=None, alias="acceNumb")
    genus: str | None = None
    species: str | None = None
    accession: dict[str, Any] | None = None

    @property
    def uuid(self) -> str | None:
        """UUID of the matched Genesys accession, when the reference was matched."""
        return (self.accession or {}).get("uuid")


class TraitPage(BaseModel):
    """One page of observation rows (``PageObject``)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    content: list[dict[str, Any]] = Field(default_factory=list)
    number: int = 0
    size: int = 0
    total_elements: int = Field(default=0, alias="totalElements")
    total_pages: int = Field(default=0, alias="totalPages")
    last: bool = True


# ----------------------------------------------------------------- parsing
def _dig(data: dict[str, Any], dotted: str) -> Any:
    """Read a nested value with a dotted path; ``None`` when missing."""
    current: Any = data

    for segment in dotted.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]

    return current


def _first_present(row: dict[str, Any], candidates: tuple[str, ...]) -> Any:
    """Return the first candidate key holding a scalar in the row (nested paths allowed).

    Dictionaries are skipped (e.g. ``accession`` may be an object in some
    responses and a uuid string in others).
    """
    for key in candidates:
        value = _dig(row, key)

        if value not in (None, "") and not isinstance(value, (dict, list)):
            return value

    return None


def unwrap_values(value: Any) -> Any:
    """Simplify the list-valued observations of the API.

    Args:
        value: Raw value (``[55.9]``, ``["Yellow", "Red"]`` or a scalar).

    Returns:
        ``None`` for empty lists, the single element for one-element lists,
        otherwise the value unchanged (several observations are kept).
    """
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return value[0]

    return value


def observations_to_dataframe(rows: list[dict[str, Any]], descriptors: list[Descriptor]) -> pd.DataFrame:
    """Turn raw observation rows into a flat table.

    The spec types the rows as ``object``; this function is tolerant: accession
    identifiers are read from any of the known keys and descriptor values are
    read either by descriptor ``uuid`` or by ``columnName``. Unknown extra keys
    are kept as they are so nothing is lost.

    Args:
        rows: Raw rows from ``/dataset/data`` (or observation lists).
        descriptors: Descriptors requested, used to name the columns.

    Returns:
        DataFrame with ``uuid``, ``doi``, ``accessionNumber``, ``instituteCode``
        and one column per descriptor label (plus any unrecognised keys).
    """
    by_uuid = {descriptor.uuid: descriptor for descriptor in descriptors}
    by_column = {(descriptor.column_name or "").lower(): descriptor for descriptor in descriptors if descriptor.column_name}
    records: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        record: dict[str, Any] = {
            key: _first_present(row, candidates) for key, candidates in ACCESSION_KEY_CANDIDATES.items()
        }

        # Values may be nested under a sub-object; flatten one level of dicts.
        flat: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
                    flat.setdefault(sub_key, sub_value)
            else:
                flat[key] = value

        # Map descriptor values by uuid or by column name; lists are unwrapped.
        for key, value in flat.items():
            descriptor = by_uuid.get(key) or by_column.get(str(key).lower())

            if descriptor is not None:
                record[descriptor.label] = unwrap_values(value)
            elif key not in record and "." not in key and key not in ("accession", "accessionRef", "sources"):
                record.setdefault(key, unwrap_values(value))

        records.append(record)

    columns = list(ACCESSION_KEY_CANDIDATES) + [descriptor.label for descriptor in descriptors]
    frame = pd.DataFrame(records)

    # Guarantee the expected columns exist even when the rows were empty.
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA

    return frame[columns + [column for column in frame.columns if column not in columns]]


def accession_observations_to_rows(payload: dict[str, Any], dataset_uuid: str | None = None) -> list[dict[str, Any]]:
    """Convert an ``AccessionObservations`` answer into ``/dataset/data``-like rows.

    Real shape (verified 2026-10-02)::

        {"firstPartyData": [{"accession": "<uuid>", "accessionNumber": "G4680", "doi": "...",
          "instituteCode": "COL003",
          "sources": {"<source uuid>": [{"f": "<descriptor uuid>", "v": [18.0], "d": "<dataset uuid>", ...}]}}],
         "thirdPartyData": [...]}

    Args:
        payload: Decoded answer of ``GET /api/v2/acn/{uuid}/observations``.
        dataset_uuid: Keep only observations of this dataset (all when ``None``).

    Returns:
        One row per party entry with the accession identifiers and the values
        of every descriptor (``{"<descriptor uuid>": [values...]}``).
    """
    rows: list[dict[str, Any]] = []

    # First and third party data share the same structure.
    for party in ("firstPartyData", "thirdPartyData"):
        for entry in payload.get(party, []) or []:
            if not isinstance(entry, dict):
                continue

            row: dict[str, Any] = {
                key: entry.get(key)
                for key in ("accession", "accessionNumber", "doi", "instituteCode", "genus")
                if entry.get(key) is not None
            }
            row["party"] = party
            values: dict[str, list[Any]] = {}

            # Every source holds a list of observations keyed by descriptor ("f").
            for observations in (entry.get("sources") or {}).values():
                for observation in observations or []:
                    if not isinstance(observation, dict):
                        continue

                    if dataset_uuid and observation.get("d") != dataset_uuid:
                        continue

                    descriptor_uuid = observation.get("f")
                    raw = observation.get("v")

                    if not descriptor_uuid or raw is None:
                        continue

                    values.setdefault(str(descriptor_uuid), []).extend(raw if isinstance(raw, list) else [raw])

            row.update(values)
            rows.append(row)

    return rows


# ------------------------------------------------------------------- mixin
class TraitsMixin:
    """Trait endpoints of the Genesys API; mixed into ``GenesysClient``.

    Relies on the host class providing ``_request_json(method, path, params, body)``.
    """

    # --------------------------------------------------------- step 1
    def find_datasets(self, accession_filter: AccessionFilter) -> list[str]:
        """Return the UUIDs of the datasets holding data for the filtered accessions.

        Args:
            accession_filter: Accession criteria (typically ``uuid`` or ``accessionNumbers``).

        Returns:
            Dataset UUIDs (possibly empty).
        """
        payload = self._request_json("POST", ACCESSIONS_DATASETS_PATH, params={}, body=accession_filter.to_body())  # type: ignore[attr-defined]
        uuids = [str(item) for item in payload] if isinstance(payload, list) else []
        logger.info("Genesys accessions-datasets -> %s dataset(s)", len(uuids))
        return uuids

    def find_datasets_for_uuids(self, accession_uuids: list[str], chunk_size: int = 500) -> list[str]:
        """Convenience: datasets for a list of accession UUIDs, chunked to keep bodies small."""
        found: list[str] = []
        unique = [uuid for uuid in dict.fromkeys(accession_uuids) if uuid]

        # Query in chunks and merge the dataset ids preserving order.
        for start in range(0, len(unique), chunk_size):
            chunk = unique[start : start + chunk_size]
            for dataset in self.find_datasets(AccessionFilter(uuid=chunk)):
                if dataset not in found:
                    found.append(dataset)

        return found

    # --------------------------------------------------------- step 2
    def get_dataset(self, dataset_uuid: str) -> DatasetSummary:
        """Return the dataset summary with its inline descriptors."""
        payload = self._request_json("GET", DATASET_PATH.format(uuid=dataset_uuid), params={}, body=None)  # type: ignore[attr-defined]
        data = dict(payload) if isinstance(payload, dict) else {}
        data["descriptors"] = [Descriptor.model_validate(item) for item in data.get("descriptors", []) or [] if isinstance(item, dict) and item.get("uuid")]
        return DatasetSummary.model_validate(data)

    def list_dataset_descriptors(self, dataset_uuid: str) -> list[Descriptor]:
        """Return the descriptors (traits) of a dataset."""
        payload = self._request_json("GET", DATASET_DESCRIPTORS_PATH.format(uuid=dataset_uuid), params={}, body=None)  # type: ignore[attr-defined]
        items = payload if isinstance(payload, list) else []
        return [Descriptor.model_validate(item) for item in items if isinstance(item, dict) and item.get("uuid")]

    def iter_dataset_accessions(self, dataset_uuid: str, size: int = 500) -> Iterator[DatasetAccessionRef]:
        """Iterate over the accessions referenced by a dataset."""
        page_number = 0

        # Walk the pages until the server reports the last one.
        while True:
            payload = self._request_json(  # type: ignore[attr-defined]
                "GET", DATASET_ACCESSIONS_PATH.format(uuid=dataset_uuid), params={"p": page_number, "l": min(size, MAX_PAGE_SIZE)}, body=None
            )
            page = payload if isinstance(payload, dict) else {}

            for item in page.get("content", []) or []:
                if isinstance(item, dict):
                    yield DatasetAccessionRef.model_validate(item)

            if page.get("last", True) or not page.get("content"):
                return

            page_number += 1

    # --------------------------------------------------------- step 3
    def get_dataset_data(
        self,
        dataset_uuids: list[str],
        descriptor_uuids: list[str],
        accession_filter: AccessionFilter | None = None,
        *,
        page: int = 0,
        size: int = 500,
    ) -> TraitPage:
        """Fetch one page of observation rows.

        Args:
            dataset_uuids: Datasets to read.
            descriptor_uuids: Descriptors (traits) to include (``fields``).
            accession_filter: Restrict rows to these accessions.
            page: Zero-based page.
            size: Rows per page (max 1000).

        Returns:
            The page with raw rows.
        """
        body: dict[str, Any] = {"filters": {}, "select": []}

        # Only send the accession filter when it carries criteria.
        if accession_filter is not None and not accession_filter.is_empty():
            body["filters"] = {"accession": accession_filter.to_body()}

        params = {"datasetUuids": dataset_uuids, "fields": descriptor_uuids, "p": page, "l": min(size, MAX_PAGE_SIZE)}
        payload = self._request_json("POST", DATASET_DATA_PATH, params=params, body=body)  # type: ignore[attr-defined]
        result = TraitPage.model_validate(payload if isinstance(payload, dict) else {})
        logger.info("Genesys dataset/data page=%s -> %s rows (total %s)", page, len(result.content), result.total_elements)
        return result

    def iter_dataset_data(
        self,
        dataset_uuids: list[str],
        descriptor_uuids: list[str],
        accession_filter: AccessionFilter | None = None,
        *,
        size: int = 500,
        max_rows: int = 20000,
    ) -> Iterator[dict[str, Any]]:
        """Iterate over every observation row, page by page, up to ``max_rows``."""
        yielded = 0
        page_number = 0

        while True:
            page = self.get_dataset_data(dataset_uuids, descriptor_uuids, accession_filter, page=page_number, size=size)

            for row in page.content:
                if yielded >= max_rows:
                    logger.warning("Genesys dataset/data iteration stopped at %s rows", max_rows)
                    return
                yield row
                yielded += 1

            if page.last or not page.content:
                return

            page_number += 1

    def get_accession_observations(self, accession_uuid: str) -> dict[str, Any]:
        """Return the raw ``AccessionObservations`` of one accession (first/third party)."""
        payload = self._request_json("GET", ACCESSION_OBSERVATIONS_PATH.format(uuid=accession_uuid), params={}, body=None)  # type: ignore[attr-defined]
        return payload if isinstance(payload, dict) else {}

    # --------------------------------------------------------- step 4
    def get_descriptor(self, descriptor_uuid: str) -> Descriptor:
        """Return the metadata of a descriptor."""
        payload = self._request_json("GET", DESCRIPTOR_PATH.format(uuid=descriptor_uuid), params={}, body=None)  # type: ignore[attr-defined]
        return Descriptor.model_validate(payload if isinstance(payload, dict) else {"uuid": descriptor_uuid})
