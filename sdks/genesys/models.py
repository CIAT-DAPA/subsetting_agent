"""Pydantic models of the Genesys API v2 used by the SDK.

The filter models mirror the ``AccessionFilter`` schema of the Genesys OpenAPI
specification. Every property is optional: the API treats a missing or ``null``
property as "do not apply this criterion", so :meth:`AccessionFilter.to_body`
serialises **only** the properties that carry a value.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- base
class _FilterModel(BaseModel):
    """Common behaviour of every filter model."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    def to_body(self) -> dict[str, Any]:
        """Serialise the filter keeping only properties with a meaningful value.

        ``None`` values, empty strings, empty lists and nested filters that end
        up empty after pruning are all removed, so the request body contains
        exactly the criteria the caller set.

        Returns:
            A JSON-serialisable dictionary.
        """
        raw = self.model_dump(by_alias=True, exclude_none=True)
        return _prune_empty(raw)

    def is_empty(self) -> bool:
        """Whether the filter carries no criteria at all."""
        return not self.to_body()


def _prune_empty(value: Any) -> Any:
    """Recursively drop empty containers and blank strings from a structure.

    Args:
        value: Any JSON-like value (dict, list, scalar).

    Returns:
        The cleaned value. Dictionaries and lists may come back empty; callers
        decide whether an empty container should itself be removed.
    """
    # Dictionaries: prune each entry and keep only the non-empty ones.
    if isinstance(value, dict):
        cleaned = {}

        for key, item in value.items():
            pruned = _prune_empty(item)

            # Skip None, "", [], {} but keep False and 0 (they are valid criteria).
            if pruned is None or pruned == "" or pruned == [] or pruned == {}:
                continue

            cleaned[key] = pruned

        return cleaned

    # Lists: prune items and drop the empty ones.
    if isinstance(value, list):
        items = [_prune_empty(item) for item in value]
        return [item for item in items if item is not None and item != "" and item != {} and item != []]

    # Blank strings are treated as "no value".
    if isinstance(value, str) and not value.strip():
        return ""

    return value


# ------------------------------------------------------------------ primitives
class StringFilter(_FilterModel):
    """Text criteria (``StringFilter`` in the spec).

    Attributes:
        eq: Exact matches (any of).
        contains: Substrings that must appear.
        sw: Prefixes ("starts with").
        ge: Lower bound (lexicographic).
        le: Upper bound (lexicographic).
    """

    eq: list[str] | None = None
    contains: list[str] | None = None
    sw: list[str] | None = None
    ge: str | None = None
    le: str | None = None


class NumberFilter(_FilterModel):
    """Numeric range criteria (``NumberFilterDouble`` in the spec).

    Attributes:
        eq: Exact values (any of).
        ge: Greater than or equal.
        gt: Strictly greater than.
        lt: Strictly lower than.
        le: Lower than or equal.
    """

    eq: list[float] | None = None
    ge: float | None = None
    gt: float | None = None
    lt: float | None = None
    le: float | None = None


class TemporalFilter(_FilterModel):
    """Date/time range criteria (``TemporalFilterInstant`` in the spec)."""

    eq: list[str] | None = None
    ge: str | None = None
    gt: str | None = None
    lt: str | None = None
    le: str | None = None


# --------------------------------------------------------------- sub filters
class CountryFilter(_FilterModel):
    """Criteria on a country (``CountryFilter``).

    Attributes:
        code3: Three-letter ISO-3166 codes.
        region: Region codes.
        text: Full-text keywords (serialised as ``_text``).
    """

    code3: list[str] | None = None
    region: list[str] | None = None
    text: str | None = Field(default=None, alias="_text")


class InstituteFilter(_FilterModel):
    """Criteria on the holding institute (``InstituteFilter``).

    Attributes:
        code: FAO WIEWS institute codes.
        country: Country of the institute.
        full_name: Institute name criteria (serialised as ``fullName``).
        accessions: ``True`` to match only institutes with accessions in Genesys.
        text: Full-text keywords (serialised as ``_text``).
    """

    code: list[str] | None = None
    country: CountryFilter | None = None
    full_name: StringFilter | None = Field(default=None, alias="fullName")
    accessions: bool | None = None
    text: str | None = Field(default=None, alias="_text")


class TaxonomyFilter(_FilterModel):
    """Criteria on the taxonomy (``TaxonomyFilter``).

    Attributes:
        genus: Genera (e.g. ``["Phaseolus"]``).
        species: Specific epithets (e.g. ``["vulgaris"]``).
        genus_species: Full ``Genus species`` names (serialised as ``genusSpecies``).
        subtaxa: Subtaxa criteria.
        taxon_name: Full taxon name criteria (serialised as ``taxonName``).
        family: Botanical families.
        synonyms: Whether to include taxonomic synonyms.
    """

    genus: list[str] | None = None
    species: list[str] | None = None
    genus_species: list[str] | None = Field(default=None, alias="genusSpecies")
    subtaxa: StringFilter | None = None
    taxon_name: StringFilter | None = Field(default=None, alias="taxonName")
    family: list[str] | None = None
    synonyms: bool | None = None


class GeoFilter(_FilterModel):
    """Criteria on the collecting site coordinates (``AccessionGeoFilter``).

    Attributes:
        latitude: Latitude range in decimal degrees.
        longitude: Longitude range in decimal degrees.
        elevation: Elevation range in metres.
        referenced: ``True`` to keep only georeferenced accessions.
    """

    latitude: NumberFilter | None = None
    longitude: NumberFilter | None = None
    elevation: NumberFilter | None = None
    referenced: bool | None = None


class CollectFilter(_FilterModel):
    """Criteria on the collecting event (``AccessionCollectFilter``).

    Attributes:
        coll_date: Collecting date criteria (serialised as ``collDate``).
        coll_numb: Collecting number criteria (serialised as ``collNumb``).
        coll_miss_id: Collecting mission identifiers (serialised as ``collMissId``).
        coll_site: Collecting site text criteria (serialised as ``collSite``).
    """

    coll_date: StringFilter | None = Field(default=None, alias="collDate")
    coll_numb: StringFilter | None = Field(default=None, alias="collNumb")
    coll_miss_id: list[str] | None = Field(default=None, alias="collMissId")
    coll_site: StringFilter | None = Field(default=None, alias="collSite")


# ------------------------------------------------------------ main filter
class AccessionFilter(_FilterModel):
    """Search criteria for accessions (``AccessionFilter`` in the spec).

    Only the business-relevant properties of the schema are modelled; audit
    fields (``createdBy``, ``version``...) are left out on purpose. Attribute
    names follow Python conventions and are serialised with the exact JSON
    names expected by the API.

    Attributes:
        text: Full-text keywords across the accession database (``_text``).
        crop: Genesys crop codes (e.g. ``["bean"]``).
        crop_name: Crop name criteria (``cropName``).
        institute: Holding institute criteria.
        accession_number: Accession number text criteria (``accessionNumber``).
        accession_numbers: Exact accession numbers (``accessionNumbers``).
        doi: Accession DOIs.
        uuid: Accession UUIDs.
        taxonomy: Taxonomy criteria.
        samp_stat: MCPD biological status codes (``sampStat``).
        country_of_origin: Country of origin criteria (``countryOfOrigin``).
        geo: Coordinates criteria.
        coll: Collecting event criteria.
        storage: MCPD storage type codes.
        historic: ``True`` for historical records, ``False`` for active (API default).
        available: Availability for distribution.
        mls_status: Inclusion in the Multilateral System (``mlsStatus``).
        sgsv: Backed up in the Svalbard Global Seed Vault.
        in_trust: Held in trust (``inTrust``).
        has_doi: Has a DOI (``hasDoi``).
        has_dataset: Linked to a dataset (``hasDataset``).
        has_subset: Included in a subset (``hasSubset``).
        images: Has images.
        genotyped: Has genotypic data.
        dupl_site: Safety-duplication institute codes (``duplSite``).
        donor_code: Donor institute codes (``donorCode``).
        donor_name: Donor name criteria (``donorName``).
        breeder_code: Breeder institute codes (``breederCode``).
        curation_type: Curation types (``curationType``).
        acquisition_date: Acquisition date criteria (``acquisitionDate``).
        collection: Collection names.
        lists: Accession list UUIDs.
        subsets: Subset UUIDs.
        datasets: Dataset UUIDs.
        seq_no: Sequence number range (``seqNo``).
        pdci: Passport Data Completeness Index range.
        last_modified_date: Last modification range (``lastModifiedDate``).
        not_: Negated criteria (``NOT``).
        and_: Criteria that must also match (``AND``).
        or_: Alternative criteria (``OR``).
    """

    text: str | None = Field(default=None, alias="_text")
    crop: list[str] | None = None
    crop_name: StringFilter | None = Field(default=None, alias="cropName")
    institute: InstituteFilter | None = None
    accession_number: StringFilter | None = Field(default=None, alias="accessionNumber")
    accession_numbers: list[str] | None = Field(default=None, alias="accessionNumbers")
    doi: list[str] | None = None
    uuid: list[str] | None = None
    taxonomy: TaxonomyFilter | None = None
    samp_stat: list[int] | None = Field(default=None, alias="sampStat")
    country_of_origin: CountryFilter | None = Field(default=None, alias="countryOfOrigin")
    geo: GeoFilter | None = None
    coll: CollectFilter | None = None
    storage: list[int] | None = None
    historic: bool | None = None
    available: bool | None = None
    mls_status: bool | None = Field(default=None, alias="mlsStatus")
    sgsv: bool | None = None
    in_trust: bool | None = Field(default=None, alias="inTrust")
    has_doi: bool | None = Field(default=None, alias="hasDoi")
    has_dataset: bool | None = Field(default=None, alias="hasDataset")
    has_subset: bool | None = Field(default=None, alias="hasSubset")
    images: bool | None = None
    genotyped: bool | None = None
    dupl_site: list[str] | None = Field(default=None, alias="duplSite")
    donor_code: list[str] | None = Field(default=None, alias="donorCode")
    donor_name: StringFilter | None = Field(default=None, alias="donorName")
    breeder_code: list[str] | None = Field(default=None, alias="breederCode")
    curation_type: list[str] | None = Field(default=None, alias="curationType")
    acquisition_date: StringFilter | None = Field(default=None, alias="acquisitionDate")
    collection: list[str] | None = None
    lists: list[str] | None = None
    subsets: list[str] | None = None
    datasets: list[str] | None = None
    seq_no: NumberFilter | None = Field(default=None, alias="seqNo")
    pdci: NumberFilter | None = None
    last_modified_date: TemporalFilter | None = Field(default=None, alias="lastModifiedDate")
    not_: AccessionFilter | None = Field(default=None, alias="NOT")
    and_: AccessionFilter | None = Field(default=None, alias="AND")
    or_: AccessionFilter | None = Field(default=None, alias="OR")


# ------------------------------------------------------------- responses
# Mapping from flattened MCPD-style column names to the dotted path inside an
# ``AccessionDTO``. Lists are joined with ";" when flattened.
MCPD_FIELD_MAP: dict[str, str] = {
    "INSTCODE": "instituteCode",
    "INSTNAME": "institute.fullName",
    "INSTCTY": "institute.countryCode3",
    "ACCENUMB": "accessionNumber",
    "DOI": "doi",
    "UUID": "uuid",
    "GENUS": "taxonomy.genus",
    "SPECIES": "taxonomy.species",
    "SPAUTHOR": "taxonomy.spAuthor",
    "SUBTAXA": "taxonomy.subtaxa",
    "SUBTAUTHOR": "taxonomy.subtAuthor",
    "TAXONNAME": "taxonomy.taxonName",
    "CROPNAME": "cropName",
    "CROPCODE": "crop.shortName",
    "SAMPSTAT": "sampStat",
    "ACQDATE": "acquisitionDate",
    "ACCENAME": "accessionName",
    "ORIGCTY": "origCty",
    "ORIGCTYNAME": "countryOfOrigin.name",
    "COLLSITE": "coll.collSite",
    "DECLATITUDE": "geo.latitude",
    "DECLONGITUDE": "geo.longitude",
    "COORDUNCERT": "geo.uncertainty",
    "COORDDATUM": "geo.datum",
    "GEOREFMETH": "geo.method",
    "ELEVATION": "geo.elevation",
    "COLLDATE": "coll.collDate",
    "COLLSRC": "coll.collSrc",
    "COLLNUMB": "coll.collNumb",
    "COLLCODE": "coll.collCode",
    "COLLNAME": "coll.collName",
    "COLLMISSID": "coll.collMissId",
    "DONORCODE": "donorCode",
    "DONORNAME": "donorName",
    "DONORNUMB": "donorNumb",
    "BREDCODE": "breederCode",
    "BREDNAME": "breederName",
    "ANCEST": "ancest",
    "DUPLSITE": "duplSite",
    "STORAGE": "storage",
    "MLSSTAT": "mlsStatus",
    "AVAILABLE": "available",
    "HISTORIC": "historic",
    "INSVALBARD": "inSvalbard",
    "INTRUST": "inTrust",
    "ACCEURL": "acceUrl",
    "CURATION": "curationType",
    "PDCI": "pdci.score",
    "LASTMODIFIED": "lastModifiedDate",
}


def _dig(data: dict[str, Any], dotted_path: str) -> Any:
    """Read a nested value using a dotted path.

    Args:
        data: Nested dictionary (an ``AccessionDTO``).
        dotted_path: Path such as ``"taxonomy.genus"``.

    Returns:
        The value, or ``None`` when any segment is missing.
    """
    current: Any = data

    # Walk one segment at a time; stop as soon as the path breaks.
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return None

        current = current[segment]

    return current


class AccessionRecord(BaseModel):
    """One accession as returned by the API (``AccessionDTO``).

    The DTO has dozens of nested properties; the SDK keeps the raw dictionary
    and offers :meth:`flatten` to produce MCPD-style columns.

    Attributes:
        raw: The ``AccessionDTO`` dictionary exactly as returned by Genesys.
    """

    model_config = ConfigDict(extra="forbid")

    raw: dict[str, Any]

    @property
    def accession_number(self) -> str | None:
        """Accession number (``ACCENUMB``)."""
        return self.raw.get("accessionNumber")

    @property
    def institute_code(self) -> str | None:
        """Holding institute code (``INSTCODE``)."""
        return self.raw.get("instituteCode")

    def flatten(self) -> dict[str, Any]:
        """Convert the nested DTO into a flat MCPD-style record.

        Returns:
            Dictionary with the keys of :data:`MCPD_FIELD_MAP`; list values
            are joined with ``";"`` so the record fits in one table row.
        """
        record: dict[str, Any] = {}

        # Resolve each MCPD column from its dotted path in the DTO.
        for column, path in MCPD_FIELD_MAP.items():
            value = _dig(self.raw, path)

            # Multi-valued properties (storage, duplSite, collCode...) become one cell.
            if isinstance(value, list):
                value = ";".join(str(item) for item in value) if value else None

            record[column] = value

        return record


class AccessionPage(BaseModel):
    """One page of accessions (``FilteredPageAccessionDTOAccessionFilter``).

    Attributes:
        content: Accessions of the page.
        number: Zero-based page number.
        size: Requested page size.
        total_elements: Total accessions matching the filter.
        total_pages: Total number of pages.
        last: Whether this is the last page.
        filter_code: Server-side code identifying the filter (reusable via ``f``).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    content: list[AccessionRecord] = Field(default_factory=list)
    number: int = 0
    size: int = 0
    total_elements: int = Field(default=0, alias="totalElements")
    total_pages: int = Field(default=0, alias="totalPages")
    last: bool = True
    filter_code: str | None = Field(default=None, alias="filterCode")

    @classmethod
    def from_response(cls, payload: dict[str, Any]) -> AccessionPage:
        """Build a page from the raw JSON answer of ``/acn/list``.

        Args:
            payload: Decoded JSON body.

        Returns:
            The parsed page; ``content`` items are wrapped as ``AccessionRecord``.
        """
        records = [AccessionRecord(raw=item) for item in payload.get("content", []) or []]
        data = {key: value for key, value in payload.items() if key != "content"}
        return cls(content=records, **data)


# ----------------------------------------------------------------- crops
class Crop(BaseModel):
    """A crop of the Genesys catalogue (``CropDTO``).

    Attributes:
        short_name: Code used in ``AccessionFilter.crop`` (``shortName``).
        name: Display name.
        other_names: Alternative names in several languages.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    short_name: str = Field(alias="shortName")
    name: str | None = None
    other_names: list[str] = Field(default_factory=list, alias="otherNames")

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> Crop:
        """Build a crop from the API payload.

        Args:
            item: ``CropDTO`` dictionary.

        Returns:
            The parsed crop. ``otherNames`` may arrive as a list or as a
            comma-separated string; both are normalised to a list.
        """
        other = item.get("otherNames") or []

        # Some Genesys versions serialise otherNames as one string.
        if isinstance(other, str):
            other = [part.strip() for part in other.split(",") if part.strip()]

        return cls(shortName=str(item.get("shortName", "")), name=item.get("name"), otherNames=list(other))

    def matches(self, text: str) -> bool:
        """Whether a user-provided name refers to this crop.

        Args:
            text: Crop name or code typed by the user.

        Returns:
            ``True`` if ``text`` equals the code, the name or any other name
            (case-insensitive, ignoring surrounding spaces).
        """
        needle = text.strip().lower()

        # An empty string never matches anything.
        if not needle:
            return False

        candidates = [self.short_name, self.name or "", *self.other_names]
        return any(candidate.strip().lower() == needle for candidate in candidates)
