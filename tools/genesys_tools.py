"""Tools backed by the Genesys API: passport selection and trait filtering.

Every tool is an ``async`` function that receives the shared
:class:`~tools.services.ToolServices` plus the arguments produced by the LLM,
and returns a JSON-serializable dictionary kept deliberately compact.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from genesys_sdk.models import (
    AccessionFilter,
    AccessionObservations,
    DescriptorFilter,
    StringFilter,
)
from subsetting_sdk.catalog import normalize_text
from tools.accession_context import Stage
from tools.services import ToolServices

logger = logging.getLogger(__name__)

# MCPD SAMPSTAT codes, exposed so the LLM can map user words to codes.
SAMPLE_STATUS_CODES = {
    100: "Wild",
    110: "Natural",
    120: "Semi-natural/wild",
    130: "Semi-natural/sown",
    200: "Weedy",
    300: "Traditional cultivar / landrace",
    400: "Breeding / research material",
    410: "Breeder's line",
    411: "Synthetic population",
    412: "Hybrid",
    413: "Founder stock / base population",
    414: "Inbred line",
    415: "Segregating population",
    416: "Clonal selection",
    420: "Genetic stock",
    421: "Mutant",
    422: "Cytogenetic stock",
    423: "Other genetic stock",
    500: "Advanced or improved cultivar",
    600: "GMO",
    999: "Other",
}


def _upper_list(values: list[str] | None) -> list[str] | None:
    """Upper-case a list of codes, returning ``None`` for an empty input.

    Args:
        values: Codes as given by the LLM.
    """
    cleaned = [v.strip().upper() for v in (values or []) if v and v.strip()]

    return cleaned or None


def _clean_list(values: list[str] | None) -> list[str] | None:
    """Strip a list of strings, returning ``None`` for an empty input.

    Args:
        values: Strings as given by the LLM.
    """
    cleaned = [v.strip() for v in (values or []) if v and v.strip()]

    return cleaned or None


def build_passport_filter(arguments: dict[str, Any]) -> tuple[AccessionFilter, str]:
    """Translate tool arguments into a Genesys filter and a description.

    Args:
        arguments: Arguments of ``preview_accessions`` / ``select_accessions``.

    Returns:
        The filter and a plain-words description used in the step history.
    """
    crop_codes = _clean_list(arguments.get("crop_codes"))
    genus = _clean_list(arguments.get("genus"))
    species = _clean_list(arguments.get("species"))
    countries = _upper_list(arguments.get("origin_countries"))
    institutes = _upper_list(arguments.get("institute_codes"))
    sample_status = [int(v) for v in arguments.get("sample_status") or []] or None
    available = arguments.get("available")
    text = (arguments.get("text") or "").strip() or None

    accession_filter = AccessionFilter.passport(
        crop_codes=[c.lower() for c in crop_codes] if crop_codes else None,
        genus=genus,
        species=species,
        origin_countries=countries,
        institute_codes=institutes,
        sample_status=sample_status,
        available=available,
        # Climate analysis needs coordinates, so selections always require them.
        with_coordinates=True,
        text=text,
    )

    parts = []

    # Build a readable description from the criteria that were actually set.
    if crop_codes:
        parts.append(f"crop={','.join(crop_codes)}")

    if genus:
        parts.append(f"genus={','.join(genus)}")

    if species:
        parts.append(f"species={','.join(species)}")

    if countries:
        parts.append(f"origin={','.join(countries)}")

    if institutes:
        parts.append(f"institute={','.join(institutes)}")

    if sample_status:
        parts.append(f"sampstat={','.join(map(str, sample_status))}")

    if available is not None:
        parts.append(f"available={available}")

    if text:
        parts.append(f"text='{text}'")

    description = (
        "Passport filter: " + ("; ".join(parts) if parts else "no criteria") + " (with coordinates)"
    )

    return accession_filter, description


async def search_crops(services: ToolServices, *, name: str = "") -> dict[str, Any]:
    """Find Genesys crop codes by name.

    Args:
        services: Shared services.
        name: Full or partial crop name; empty lists every crop.
    """
    crops = await services.genesys.list_crops()
    wanted = normalize_text(name)
    matches = []

    # Keep crops whose code or names contain the query (or all when empty).
    for crop in crops:
        haystack = " ".join(filter(None, [crop.short_name, crop.name, crop.title or ""]))

        if not wanted or wanted in normalize_text(haystack):
            matches.append(
                {"code": crop.short_name, "name": crop.name, "accessions": crop.accession_count}
            )

    return {"query": name, "count": len(matches), "crops": matches[:30]}


async def preview_accessions(services: ToolServices, **arguments: Any) -> dict[str, Any]:
    """Count accessions matching passport criteria without fetching them.

    Args:
        services: Shared services.
        **arguments: Passport criteria (see :func:`build_passport_filter`).
    """
    accession_filter, description = build_passport_filter(arguments)
    overview = await services.genesys.accession_overview(accession_filter, limit=8)
    breakdown: dict[str, list[dict[str, Any]]] = {}

    # Keep only the distributions that help a user refine the query.
    for field_name, values in overview.overview.items():
        if any(
            key in field_name
            for key in ("crop", "countryOfOrigin", "institute.code", "sampStat", "taxonomy.genus")
        ):
            breakdown[field_name] = [{"value": t.term, "count": t.count} for t in values.terms[:8]]

    return {
        "description": description,
        "matching_accessions": overview.accession_count,
        "breakdown": breakdown,
        "hint": (
            "Use select_accessions with the same criteria to load them, or narrow the criteria "
            "if the count is far above the loading cap (GENESYS_MAX_ACCESSIONS)."
        ),
    }


async def select_accessions(services: ToolServices, **arguments: Any) -> dict[str, Any]:
    """Fetch accessions matching passport criteria and start a new selection.

    Args:
        services: Shared services.
        **arguments: Passport criteria (see :func:`build_passport_filter`) and an
            optional ``max_records`` cap.
    """
    accession_filter, description = build_passport_filter(arguments)
    max_records = arguments.get("max_records")
    accessions, total = await services.genesys.collect_accessions(
        accession_filter, max_records=int(max_records) if max_records else None
    )

    services.context.set_passport_selection(
        accessions,
        passport_filter=accession_filter.to_api(),
        total_matching=total,
        description=description,
    )
    summary = services.context.summary()

    # Warn explicitly when the cap truncated the result so the agent can tell the user.
    if services.context.truncated:
        summary["warning"] = (
            f"Only {services.context.count} of {total} matching accessions were loaded. "
            "Narrow the passport criteria for a complete selection."
        )

    return summary


async def search_trait_descriptors(
    services: ToolServices, *, keyword: str, crop_code: str | None = None
) -> dict[str, Any]:
    """Find trait descriptors (characterization/evaluation variables) by keyword.

    Args:
        services: Shared services.
        keyword: Word(s) to look for in the descriptor title.
        crop_code: Restrict to descriptors of this crop.
    """
    descriptor_filter = DescriptorFilter(
        crop=[crop_code.lower()] if crop_code else None,
        title=StringFilter(contains=[keyword]) if keyword else None,
    )
    page = await services.genesys.list_descriptors(descriptor_filter, page_size=25)

    descriptors = [
        {
            "uuid": d.uuid,
            "title": d.title,
            "crop": d.crop,
            "category": d.category,
            "data_type": d.data_type,
            "unit": d.uom,
            "column_name": d.column_name,
            "terms": [{"code": t.code, "title": t.title} for t in d.terms[:10]],
        }
        for d in page.content
    ]

    return {"keyword": keyword, "total": page.total_elements, "descriptors": descriptors}


def _flatten(record: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a nested observation record into ``(dotted_key, value)`` pairs.

    Args:
        record: Dictionary, list or scalar from the observations payload.
        prefix: Key prefix accumulated during recursion.
    """
    pairs: list[tuple[str, Any]] = []

    # Dictionaries contribute their keys; lists are indexed; scalars are leaves.
    if isinstance(record, dict):
        for key, value in record.items():
            pairs.extend(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))

    elif isinstance(record, list):
        for index, value in enumerate(record):
            pairs.extend(_flatten(value, f"{prefix}[{index}]"))

    else:
        pairs.append((prefix, record))

    return pairs


def _value_matches(
    value: Any,
    *,
    min_value: float | None,
    max_value: float | None,
    equals: str | None,
) -> bool:
    """Check whether an observed value satisfies the requested condition.

    Args:
        value: Observed value (number, string or other).
        min_value: Lower bound for numeric comparison.
        max_value: Upper bound for numeric comparison.
        equals: Expected value for categorical comparison (case-insensitive).
    """
    # Categorical match: compare normalized strings.
    if equals is not None:
        return normalize_text(str(value)) == normalize_text(equals)

    # Numeric match: the value must parse as a number inside the bounds.
    try:
        number = float(value)

    except (TypeError, ValueError):
        return False

    if min_value is not None and number < min_value:
        return False

    if max_value is not None and number > max_value:
        return False

    return True


def observation_matches(
    observations: AccessionObservations,
    *,
    descriptor: str,
    min_value: float | None,
    max_value: float | None,
    equals: str | None,
) -> tuple[bool, list[str]]:
    """Decide whether an accession's observations satisfy a trait condition.

    Observation records are loosely typed, so a record matches when any of its
    keys contains the descriptor keyword (title, column name or UUID) and the
    associated value satisfies the condition.

    Args:
        observations: Observations of one accession.
        descriptor: Descriptor keyword, column name or UUID.
        min_value: Lower bound for numeric traits.
        max_value: Upper bound for numeric traits.
        equals: Expected value for categorical traits.

    Returns:
        Whether the accession matches and the keys that were compared.
    """
    wanted = normalize_text(descriptor)
    compared: list[str] = []

    # Scan every flattened key of every record for the descriptor keyword.
    for record in observations.all_records:
        pairs = _flatten(record)

        # A record that mentions the descriptor UUID/name in any value is also
        # considered: its "value"-like fields are then compared.
        mentions_descriptor = any(
            isinstance(v, str) and wanted and wanted in normalize_text(v) for _, v in pairs
        )

        for key, value in pairs:
            key_matches = wanted and wanted in normalize_text(key)
            value_field = key.rsplit(".", 1)[-1].lower() in ("value", "val", "observation")

            if not key_matches and not (mentions_descriptor and value_field):
                continue

            compared.append(key)

            if _value_matches(value, min_value=min_value, max_value=max_value, equals=equals):
                return True, compared

    return False, compared


async def filter_selection_by_trait(
    services: ToolServices,
    *,
    descriptor: str,
    min_value: float | None = None,
    max_value: float | None = None,
    equals: str | None = None,
) -> dict[str, Any]:
    """Reduce the current selection to accessions whose observations satisfy a trait condition.

    Args:
        services: Shared services.
        descriptor: Descriptor title keyword, column name or UUID.
        min_value: Lower bound for numeric traits.
        max_value: Upper bound for numeric traits.
        equals: Expected value for categorical traits.
    """
    context = services.context

    # Traits come after passport: there must be a selection to reduce.
    if context.is_empty:
        return {"error": "No accessions selected yet. Run select_accessions first."}

    if min_value is None and max_value is None and equals is None:
        return {
            "error": (
                "Provide min_value/max_value for numeric traits or equals for categorical ones."
            )
        }

    total_before = context.count
    records = context.records()[: services.max_trait_lookups]
    semaphore = asyncio.Semaphore(services.trait_concurrency)

    async def lookup(uuid: str) -> tuple[str, AccessionObservations | None]:
        """Fetch observations for one accession, tolerating failures.

        Args:
            uuid: Accession UUID.
        """
        async with semaphore:
            try:
                return uuid, await services.genesys.get_observations(uuid)

            except Exception as exc:  # noqa: BLE001 - one failure must not abort the batch
                logger.warning("Observations lookup failed for %s: %s", uuid, exc)
                return uuid, None

    results = await asyncio.gather(*(lookup(record.uuid) for record in records))

    kept: list[str] = []
    without_data = 0
    compared_keys: set[str] = set()

    # Classify each accession: matched, no observations, or observed but not matching.
    for uuid, observations in results:
        if observations is None or observations.is_empty:
            without_data += 1
            continue

        matched, compared = observation_matches(
            observations,
            descriptor=descriptor,
            min_value=min_value,
            max_value=max_value,
            equals=equals,
        )
        compared_keys.update(compared)

        if matched:
            kept.append(uuid)

    condition = equals if equals is not None else f"[{min_value}, {max_value}]"
    description = f"Trait '{descriptor}' {condition} (checked {len(records)} accessions)"
    context.keep(kept, stage=Stage.TRAITS, description=description)

    result: dict[str, Any] = {
        "checked": len(records),
        "not_checked": total_before - len(records),
        "without_observations": without_data,
        "kept": len(kept),
        "compared_fields_sample": sorted(compared_keys)[:10],
        "summary": context.summary(),
    }

    # Help the model recover when the descriptor keyword matched nothing at all.
    if not compared_keys:
        result["warning"] = (
            f"No observation field mentioned '{descriptor}'. Use search_trait_descriptors to find "
            "the exact descriptor title or column name."
        )

    # Accessions beyond the lookup cap were dropped without being checked.
    if total_before > len(records):
        result["note"] = (
            f"Only the first {len(records)} of {total_before} accessions were checked "
            "(GENESYS trait lookup cap). Narrow the passport selection to check all of them."
        )

    return result
