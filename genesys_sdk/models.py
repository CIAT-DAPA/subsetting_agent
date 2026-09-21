"""Typed request filters and response models for the Genesys API.

The models follow the OpenAPI document "Genesys API for MCP" (version
2026.2.0). Request filters serialize with the exact property names of the API
and omit unset fields, because Genesys treats an absent filter property as
"do not filter on this".
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Dotted path of the accession field that is the Subsetting API ``cellid`` by
# default (alternative: ``tileIndex3min``). The app can override it.
DEFAULT_CELLID_FIELD = "geo.tileIndex"


def read_path(data: dict[str, Any] | None, path: str) -> Any:
    """Read a nested value from a dictionary using a dotted path.

    Args:
        data: Dictionary to read from.
        path: Dotted path such as ``"geo.tileIndex"``.

    Returns:
        The value, or ``None`` when any segment is missing.
    """
    current: Any = data

    # Walk the path one segment at a time; stop as soon as a segment is
    # missing or the current value is not a dictionary.
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None

        current = current.get(segment)

    return current


# --------------------------------------------------------------------------- #
# Request filters
# --------------------------------------------------------------------------- #


class _ApiFilter(BaseModel):
    """Base class for request filters: alias-based, omit-unset serialization."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    def to_api(self) -> dict[str, Any]:
        """Serialize using API property names and dropping unset/None fields."""
        return self.model_dump(by_alias=True, exclude_none=True, exclude_unset=True)


class StringFilter(_ApiFilter):
    """Genesys ``StringFilter``: equality, containment, prefix and bounds.

    Attributes:
        eq: Exact matches (any of).
        contains: Substrings (any of).
        sw: Prefixes (any of).
        ge: Lower bound (inclusive), lexicographic.
        le: Upper bound (inclusive), lexicographic.
    """

    eq: list[str] | None = None
    contains: list[str] | None = None
    sw: list[str] | None = None
    ge: str | None = None
    le: str | None = None


class NumberFilter(_ApiFilter):
    """Genesys ``NumberFilterDouble``: equality and numeric bounds.

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

    @classmethod
    def between(cls, low: float | None, high: float | None) -> NumberFilter | None:
        """Build an inclusive range filter, or ``None`` when both bounds are missing.

        Args:
            low: Lower bound, or ``None`` for unbounded.
            high: Upper bound, or ``None`` for unbounded.
        """
        # No bounds means no filter, which the API expects as an absent field.
        if low is None and high is None:
            return None

        return cls(ge=low, le=high)


class TaxonomyFilter(_ApiFilter):
    """Genesys ``TaxonomyFilter``.

    Attributes:
        genus: Genus names (any of).
        species: Species epithets (any of).
        genus_species: ``"Genus species"`` strings (any of).
        taxon_name: Full taxon name filter.
        family: Family names (any of).
    """

    genus: list[str] | None = None
    species: list[str] | None = None
    genus_species: list[str] | None = Field(default=None, alias="genusSpecies")
    taxon_name: StringFilter | None = Field(default=None, alias="taxonName")
    family: list[str] | None = None


class CountryFilter(_ApiFilter):
    """Genesys ``CountryFilter``.

    Attributes:
        code3: ISO-3166 alpha-3 codes (any of).
        region: Region codes (any of).
    """

    code3: list[str] | None = None
    region: list[str] | None = None


class InstituteFilter(_ApiFilter):
    """Genesys ``InstituteFilter``.

    Attributes:
        code: FAO WIEWS institute codes (any of).
        country: Country of the institute.
        full_name: Institute name filter.
    """

    code: list[str] | None = None
    country: CountryFilter | None = None
    full_name: StringFilter | None = Field(default=None, alias="fullName")


class GeoFilter(_ApiFilter):
    """Genesys ``AccessionGeoFilter`` (collecting-site coordinates).

    Attributes:
        latitude: Latitude bounds in decimal degrees.
        longitude: Longitude bounds in decimal degrees.
        elevation: Elevation bounds in metres.
        referenced: ``True`` to keep only accessions with coordinates.
        tile_index: Grid tile indexes (any of).
        tile_index_3min: 3-arc-minute grid tile indexes (any of).
    """

    latitude: NumberFilter | None = None
    longitude: NumberFilter | None = None
    elevation: NumberFilter | None = None
    referenced: bool | None = None
    tile_index: list[int] | None = Field(default=None, alias="tileIndex")
    tile_index_3min: list[int] | None = Field(default=None, alias="tileIndex3min")


class CollectFilter(_ApiFilter):
    """Genesys ``AccessionCollectFilter`` (collecting event).

    Attributes:
        coll_date: Collecting date filter (MCPD ``COLLDATE`` strings).
        coll_site: Collecting site description filter.
        coll_miss_id: Collecting mission identifiers (any of).
    """

    coll_date: StringFilter | None = Field(default=None, alias="collDate")
    coll_site: StringFilter | None = Field(default=None, alias="collSite")
    coll_miss_id: list[str] | None = Field(default=None, alias="collMissId")


class AccessionFilter(_ApiFilter):
    """Genesys ``AccessionFilter``: the passport-data filter of ``/acn/list``.

    Only the properties useful to the agent are modelled; see the OpenAPI
    document for the full list.

    Attributes:
        crop: Genesys crop codes (``CropInfo.shortName``), any of.
        crop_name: Free-form crop name filter.
        taxonomy: Taxonomy filter.
        country_of_origin: Country of origin filter.
        institute: Holding institute filter.
        samp_stat: MCPD ``SAMPSTAT`` biological status codes (any of).
        geo: Collecting-site coordinates filter.
        coll: Collecting event filter.
        available: Whether the accession can be requested.
        historic: ``False`` (default in Genesys) keeps active accessions only.
        mls_status: Whether the accession is in the Multilateral System.
        sgsv: Whether the accession is backed up in Svalbard.
        in_trust: Whether the accession is held in trust (Article 15).
        images: Whether the accession has images.
        has_doi: Whether the accession has a DOI.
        genotyped: Whether genotype data exists.
        doi: DOIs (any of).
        accession_numbers: Accession numbers (any of).
        uuid: Accession UUIDs (any of).
        donor_code: Donor institute WIEWS codes (any of).
        text: Full-text keywords; use only when no structured field applies.
        and_: Nested filter combined with AND.
        or_: Nested filter combined with OR.
        not_: Nested filter negated.
        null: Fields that must be null.
        not_null: Fields that must not be null.
    """

    crop: list[str] | None = None
    crop_name: StringFilter | None = Field(default=None, alias="cropName")
    taxonomy: TaxonomyFilter | None = None
    country_of_origin: CountryFilter | None = Field(default=None, alias="countryOfOrigin")
    institute: InstituteFilter | None = None
    samp_stat: list[int] | None = Field(default=None, alias="sampStat")
    geo: GeoFilter | None = None
    coll: CollectFilter | None = None
    available: bool | None = None
    historic: bool | None = None
    mls_status: bool | None = Field(default=None, alias="mlsStatus")
    sgsv: bool | None = None
    in_trust: bool | None = Field(default=None, alias="inTrust")
    images: bool | None = None
    has_doi: bool | None = Field(default=None, alias="hasDoi")
    genotyped: bool | None = None
    doi: list[str] | None = None
    accession_numbers: list[str] | None = Field(default=None, alias="accessionNumbers")
    uuid: list[str] | None = None
    donor_code: list[str] | None = Field(default=None, alias="donorCode")
    text: str | None = Field(default=None, alias="_text")
    and_: AccessionFilter | None = Field(default=None, alias="AND")
    or_: AccessionFilter | None = Field(default=None, alias="OR")
    not_: AccessionFilter | None = Field(default=None, alias="NOT")
    null: list[str] | None = Field(default=None, alias="NULL")
    not_null: list[str] | None = Field(default=None, alias="NOTNULL")

    @classmethod
    def passport(
        cls,
        *,
        crop_codes: list[str] | None = None,
        crop_name: str | None = None,
        genus: list[str] | None = None,
        species: list[str] | None = None,
        taxon_name: str | None = None,
        origin_countries: list[str] | None = None,
        institute_codes: list[str] | None = None,
        sample_status: list[int] | None = None,
        available: bool | None = None,
        with_coordinates: bool | None = None,
        latitude: tuple[float | None, float | None] | None = None,
        longitude: tuple[float | None, float | None] | None = None,
        elevation: tuple[float | None, float | None] | None = None,
        text: str | None = None,
    ) -> AccessionFilter:
        """Build a filter from domain-level passport criteria.

        Every argument is optional; omitted criteria are not applied.

        Args:
            crop_codes: Genesys crop codes (e.g. ``["bean"]``).
            crop_name: Substring of the crop name as reported by the genebank.
            genus: Genus names.
            species: Species epithets.
            taxon_name: Substring of the full taxon name.
            origin_countries: ISO-3166 alpha-3 codes of the country of origin.
            institute_codes: FAO WIEWS codes of the holding institutes.
            sample_status: MCPD ``SAMPSTAT`` codes (100 wild, 300 landrace, ...).
            available: Whether the material can be requested.
            with_coordinates: ``True`` keeps only geo-referenced accessions,
                which is required before any climate analysis.
            latitude: ``(min, max)`` latitude bounds.
            longitude: ``(min, max)`` longitude bounds.
            elevation: ``(min, max)`` elevation bounds in metres.
            text: Full-text keywords, as a last resort.
        """
        taxonomy = None

        # Only build a taxonomy filter when some taxonomic criterion was given.
        if genus or species or taxon_name:
            taxonomy = TaxonomyFilter(
                genus=genus or None,
                species=species or None,
                taxon_name=StringFilter(contains=[taxon_name]) if taxon_name else None,
            )

        geo = None
        lat_filter = NumberFilter.between(*latitude) if latitude else None
        lon_filter = NumberFilter.between(*longitude) if longitude else None
        elev_filter = NumberFilter.between(*elevation) if elevation else None

        # Same idea for the geo block: absent unless a geo criterion exists.
        if with_coordinates is not None or lat_filter or lon_filter or elev_filter:
            geo = GeoFilter(
                referenced=with_coordinates,
                latitude=lat_filter,
                longitude=lon_filter,
                elevation=elev_filter,
            )

        return cls(
            crop=crop_codes or None,
            crop_name=StringFilter(contains=[crop_name]) if crop_name else None,
            taxonomy=taxonomy,
            country_of_origin=CountryFilter(code3=origin_countries) if origin_countries else None,
            institute=InstituteFilter(code=institute_codes) if institute_codes else None,
            samp_stat=sample_status or None,
            geo=geo,
            available=available,
            text=text or None,
        )


class DescriptorFilter(_ApiFilter):
    """Genesys ``DescriptorFilter`` for ``/descriptor/list/details``.

    Attributes:
        crop: Crop codes (any of).
        category: Descriptor categories (any of).
        title: Descriptor title filter.
        text: Full-text keywords.
    """

    crop: list[str] | None = None
    category: list[str] | None = None
    title: StringFilter | None = None
    text: str | None = Field(default=None, alias="_text")


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class _ApiModel(BaseModel):
    """Base class for response models: tolerant to extra fields."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class Crop(_ApiModel):
    """A Genesys crop or crop group (``GET /crop``).

    Attributes:
        short_name: Genesys crop code, used in ``AccessionFilter.crop``.
        name: English name.
        title: Name in the requested language.
        accession_count: Number of accessions of the crop in Genesys.
        annex1: Whether the crop is in Annex 1 of the Plant Treaty.
    """

    short_name: str = Field(alias="shortName")
    name: str
    title: str | None = None
    accession_count: int | None = Field(default=None, alias="accessionCount")
    annex1: bool | None = None


class CountryInfo(_ApiModel):
    """Country reference data."""

    code3: str | None = None
    name: str | None = None


class InstituteInfo(_ApiModel):
    """Holding institute (genebank) reference data."""

    code: str | None = None
    acronym: str | None = None
    full_name: str | None = Field(default=None, alias="fullName")
    country_code3: str | None = Field(default=None, alias="countryCode3")


class TaxonomyInfo(_ApiModel):
    """Taxonomic classification of an accession."""

    genus: str | None = None
    species: str | None = None
    subtaxa: str | None = None
    taxon_name: str | None = Field(default=None, alias="taxonName")
    genus_species: str | None = Field(default=None, alias="genusSpecies")


class GeoInfo(_ApiModel):
    """Collecting-site coordinates of an accession.

    Attributes:
        longitude: Decimal degrees.
        latitude: Decimal degrees.
        elevation: Metres above sea level.
        uncertainty: Coordinate uncertainty in metres.
        tile_index: Grid tile index assigned by Genesys.
        referenced: Whether the coordinates are geo-referenced.
    """

    longitude: float | None = None
    latitude: float | None = None
    elevation: float | None = None
    uncertainty: float | None = None
    tile_index: int | None = Field(default=None, alias="tileIndex")
    referenced: bool | None = None


class CollectInfo(_ApiModel):
    """Collecting event of an accession."""

    coll_date: str | None = Field(default=None, alias="collDate")
    coll_site: str | None = Field(default=None, alias="collSite")
    coll_miss_id: str | None = Field(default=None, alias="collMissId")
    coll_src: int | None = Field(default=None, alias="collSrc")


class Accession(_ApiModel):
    """An accession record (``AccessionDTO``), reduced to the agent's needs.

    The raw payload is kept in ``raw`` so the configurable cellid field can be
    resolved whatever its location in the document.
    """

    uuid: str
    id: int | None = None
    doi: str | None = None
    accession_number: str | None = Field(default=None, alias="accessionNumber")
    accession_name: str | None = Field(default=None, alias="accessionName")
    institute_code: str | None = Field(default=None, alias="instituteCode")
    institute: InstituteInfo | None = None
    crop_name: str | None = Field(default=None, alias="cropName")
    crop: Crop | None = None
    genus: str | None = None
    taxonomy: TaxonomyInfo | None = None
    orig_cty: str | None = Field(default=None, alias="origCty")
    country_of_origin: CountryInfo | None = Field(default=None, alias="countryOfOrigin")
    samp_stat: int | None = Field(default=None, alias="sampStat")
    available: bool | None = None
    mls_status: bool | None = Field(default=None, alias="mlsStatus")
    historic: bool | None = None
    geo: GeoInfo | None = None
    tile_index_3min: int | None = Field(default=None, alias="tileIndex3min")
    coll: CollectInfo | None = None
    donor_code: str | None = Field(default=None, alias="donorCode")
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> Accession:
        """Parse an ``AccessionDTO`` keeping the raw payload.

        Args:
            payload: Decoded JSON object of one accession.
        """
        accession = cls.model_validate(payload)
        accession.raw = payload

        return accession

    @property
    def crop_code(self) -> str | None:
        """Return the Genesys crop code, falling back to the reported crop name."""
        # The curated crop code is preferred; the genebank's free-text crop
        # name is the fallback when the accession is not mapped to a crop.
        if self.crop is not None:
            return self.crop.short_name

        return self.crop_name

    @property
    def latitude(self) -> float | None:
        """Return the collecting-site latitude, if any."""
        return self.geo.latitude if self.geo else None

    @property
    def longitude(self) -> float | None:
        """Return the collecting-site longitude, if any."""
        return self.geo.longitude if self.geo else None

    @property
    def has_coordinates(self) -> bool:
        """Whether both latitude and longitude are present."""
        return self.latitude is not None and self.longitude is not None

    @property
    def country_code(self) -> str | None:
        """Return the ISO3 code of the country of origin."""
        # ``countryOfOrigin`` is the resolved reference; ``origCty`` the raw MCPD value.
        if self.country_of_origin and self.country_of_origin.code3:
            return self.country_of_origin.code3

        return self.orig_cty

    @property
    def taxon_name(self) -> str | None:
        """Return the full taxon name, falling back to the genus."""
        if self.taxonomy and self.taxonomy.taxon_name:
            return self.taxonomy.taxon_name

        return self.genus

    def cellid(self, field: str = DEFAULT_CELLID_FIELD) -> int | None:
        """Return the Subsetting API cellid of the collecting site.

        Args:
            field: Dotted path of the field to read (``geo.tileIndex`` by default).

        Returns:
            The integer cellid, or ``None`` when the accession has no value.
        """
        value = read_path(self.raw, field)

        # The field may be missing or explicitly null for non-georeferenced
        # accessions; both mean "no cell".
        if value is None:
            return None

        return int(value)


class AccessionPage(_ApiModel):
    """One page of ``POST /acn/list``.

    Attributes:
        content: Accessions of the page.
        total_elements: Total number of matching accessions.
        total_pages: Total number of pages for the requested page size.
        number: Zero-based index of this page.
        size: Requested page size.
        filter_code: Server-side code that can replay this filter.
        last: Whether this is the final page.
    """

    content: list[Accession] = Field(default_factory=list)
    total_elements: int = Field(default=0, alias="totalElements")
    total_pages: int = Field(default=0, alias="totalPages")
    number: int = 0
    size: int = 0
    filter_code: str | None = Field(default=None, alias="filterCode")
    last: bool = True

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> AccessionPage:
        """Parse the page, building each accession with its raw payload.

        Args:
            payload: Decoded JSON body of the response.
        """
        page = cls.model_validate({**payload, "content": []})
        page.content = [Accession.from_api(item) for item in payload.get("content", [])]

        return page

    @property
    def has_next(self) -> bool:
        """Whether another page exists after this one."""
        return not self.last


class TermCount(_ApiModel):
    """A value and its count inside an overview field."""

    term: str | None = None
    count: int = 0


class OverviewField(_ApiModel):
    """Distribution of one passport field over the filtered accessions."""

    terms: list[TermCount] = Field(default_factory=list)
    total: int = 0
    other: int = 0
    missing: int = 0


class AccessionOverview(_ApiModel):
    """Response of ``POST /acn/overview``.

    Attributes:
        filter_code: Server-side code of the applied filter.
        accession_count: Number of matching accessions.
        overview: Distribution per passport field (e.g. ``countryOfOrigin.code3``).
        suggestions: Alternative filter values suggested by Genesys, kept raw.
    """

    filter_code: str | None = Field(default=None, alias="filterCode")
    accession_count: int = Field(default=0, alias="accessionCount")
    overview: dict[str, OverviewField] = Field(default_factory=dict)
    suggestions: dict[str, Any] = Field(default_factory=dict)


class VocabularyTerm(_ApiModel):
    """A term of a controlled vocabulary used by a coded descriptor."""

    code: str | None = None
    title: str | None = None
    description: str | None = None


class Descriptor(_ApiModel):
    """A trait descriptor (``TranslatedDescriptorDTO``).

    Attributes:
        uuid: Descriptor identifier.
        title: Descriptor name.
        description: Methodology or definition.
        data_type: Data type (numeric, coded, text, ...).
        uom: Unit of measurement.
        crop: Crop code.
        category: Descriptor category (e.g. agronomic, biotic stress).
        column_name: Column name used in dataset files.
        min_value: Minimum valid value for numeric descriptors.
        max_value: Maximum valid value for numeric descriptors.
        terms: Controlled vocabulary for coded descriptors.
    """

    uuid: str
    title: str | None = None
    description: str | None = None
    data_type: str | None = Field(default=None, alias="dataType")
    uom: str | None = None
    crop: str | None = None
    category: str | None = None
    column_name: str | None = Field(default=None, alias="columnName")
    min_value: float | None = Field(default=None, alias="minValue")
    max_value: float | None = Field(default=None, alias="maxValue")
    terms: list[VocabularyTerm] = Field(default_factory=list)


class DescriptorPage(_ApiModel):
    """One page of ``POST /descriptor/list/details``."""

    content: list[Descriptor] = Field(default_factory=list)
    total_elements: int = Field(default=0, alias="totalElements")
    total_pages: int = Field(default=0, alias="totalPages")
    number: int = 0
    last: bool = True


class AccessionObservations(_ApiModel):
    """Trait observations of one accession (``GET /acn/{uuid}/observations``).

    The API returns loosely typed records, so they are kept as dictionaries.

    Attributes:
        first_party_data: Observations published by the holding genebank.
        third_party_data: Observations published by other providers.
    """

    first_party_data: list[dict[str, Any]] = Field(default_factory=list, alias="firstPartyData")
    third_party_data: list[dict[str, Any]] = Field(default_factory=list, alias="thirdPartyData")

    @property
    def all_records(self) -> list[dict[str, Any]]:
        """Return every observation record regardless of its provider."""
        return [*self.first_party_data, *self.third_party_data]

    @property
    def is_empty(self) -> bool:
        """Whether the accession has no observation data at all."""
        return not self.first_party_data and not self.third_party_data


class AccessionDetails(_ApiModel):
    """Response of ``GET /acn/details/{uuid}``, reduced to the accession record."""

    details: Accession

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> AccessionDetails:
        """Parse the details, keeping the accession's raw payload.

        Args:
            payload: Decoded JSON body of the response.
        """
        return cls(details=Accession.from_api(payload.get("details", {})))
