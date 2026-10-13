"""Builds the human-readable report of a trait grouping.

Composed from the Candidate list after ``group`` ran: the trait columns used,
the group of every accession and the descriptor metadata remembered by
``fetch`` (title and unit). Written in the user's language.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.reporting import describe_differences, fmt_number, mode_profile, numeric_profile, report_strings
from skills.traits_analysis.engine import TRAIT_PREFIX, is_numeric_column, to_numeric_series

# Bilingual sentences specific to traits.
_TRAIT_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "numeric_range": "numeric trait measured in {unit}; in the Candidate list it ranges from {min} to {max} (mean {mean}), so it separates the accessions with low and high values",
        "numeric_range_nounit": "numeric trait; in the Candidate list it ranges from {min} to {max} (mean {mean}), so it separates the accessions with low and high values",
        "categorical": "categorical trait with {n} classes in the Candidate list ({classes}); it separates the accessions by class",
        "condition": "Condition applied: {condition}.",
        "meets": "meets the condition",
        "not_meets": "does not meet the condition",
    },
    "es": {
        "numeric_range": "rasgo numérico medido en {unit}; en la lista candidata va de {min} a {max} (promedio {mean}), por lo que separa las accesiones con valores bajos y altos",
        "numeric_range_nounit": "rasgo numérico; en la lista candidata va de {min} a {max} (promedio {mean}), por lo que separa las accesiones con valores bajos y altos",
        "categorical": "rasgo categórico con {n} clases en la lista candidata ({classes}); separa las accesiones por clase",
        "condition": "Condición aplicada: {condition}.",
        "meets": "cumple la condición",
        "not_meets": "no cumple la condición",
    },
}


def trait_display_name(column: str, descriptors: dict[str, dict[str, Any]]) -> tuple[str, str | None]:
    """Return the full name and unit of a trait column.

    Args:
        column: Column of the Candidate list.
        descriptors: ``state.extras["trait_descriptors"]`` (``{column: {title, type, uom}}``).

    Returns:
        ``(name, unit)``; the name falls back to the column without the ``trait_`` prefix.
    """
    info = descriptors.get(column) or {}
    name = info.get("title") or column.removeprefix(TRAIT_PREFIX)
    return str(name), info.get("uom")


def build_traits_report(
    *,
    language: str | None,
    crop: str | None,
    candidate: pd.DataFrame,
    columns: list[str],
    cluster_column: str,
    criteria_column: str,
    descriptors: dict[str, dict[str, Any]],
    conditions: list[str],
    binary: bool,
) -> str:
    """Compose the markdown report of a trait grouping.

    Args:
        language: User language.
        crop: Crop of the list (``None`` when unknown).
        candidate: Candidate list after ``group``.
        columns: Trait columns used.
        cluster_column: Column holding the group of every accession.
        criteria_column: Column where criteria/reasons were written.
        descriptors: Descriptor metadata from ``fetch`` (may be empty for local columns).
        conditions: Human-readable conditions applied (may be empty).
        binary: ``True`` when the grouping is meets / does not meet.

    Returns:
        Markdown text with the sections of the report.
    """
    lang = language if language in ("es", "en") else "en"
    texts = report_strings(lang)
    own = _TRAIT_STRINGS[lang]
    names = {column: trait_display_name(column, descriptors)[0] for column in columns}
    units = {column: trait_display_name(column, descriptors)[1] for column in columns}

    assigned = candidate[candidate[cluster_column].notna()]
    groups = sorted(int(code) for code in assigned[cluster_column].unique())

    # Per-group statistics of every trait.
    profiles: dict[int, dict[str, Any]] = {}
    for group in groups:
        rows = assigned[assigned[cluster_column].astype(int) == group]
        profile: dict[str, Any] = {}
        for column in columns:
            if is_numeric_column(candidate[column]):
                profile[column] = numeric_profile(rows[column])
            else:
                profile[column] = mode_profile(rows[column])
        profiles[group] = profile

    lines = [texts["title_traits"], "", texts["overview"], ""]
    lines.append(f"- {texts['crop']}: {crop or texts['crop_unknown']}.")
    lines.append(f"- {texts['traits']}: " + "; ".join(f"{names[c]}" + (f" ({units[c]})" if units[c] else "") for c in columns) + ".")
    lines.append(f"- {texts['purpose_binary'] if binary else texts['purpose_traits']}")
    for condition in conditions:
        lines.append(f"- {own['condition'].format(condition=condition)}")
    if not binary and any(is_numeric_column(candidate[c]) for c in columns):
        lines.append(f"- {texts['tercile_note']}")
    lines.append(f"- {texts['groups_found'].format(n=len(groups), assigned=len(assigned))}.")

    lines += ["", texts["interpretation"], ""]
    for column in columns:
        if is_numeric_column(candidate[column]):
            stats = numeric_profile(to_numeric_series(candidate[column]))
            if stats is None:
                continue
            key = "numeric_range" if units[column] else "numeric_range_nounit"
            lines.append(f"- {names[column]}: " + own[key].format(unit=units[column], min=fmt_number(stats['min']), max=fmt_number(stats['max']), mean=fmt_number(stats['mean'])) + ".")
        else:
            classes = candidate[column].dropna().astype(str).value_counts()
            lines.append(f"- {names[column]}: " + own["categorical"].format(n=len(classes), classes=", ".join(classes.head(8).index)) + ".")

    lines += ["", texts["profiles"], ""]
    for group in groups:
        size = int((assigned[cluster_column].astype(int) == group).sum())
        parts = []
        for column in columns:
            stats = profiles[group][column]
            if isinstance(stats, dict):
                parts.append(f"{names[column]} {texts['mean_min_max'].format(mean=fmt_number(stats['mean'], units[column]), min=fmt_number(stats['min']), max=fmt_number(stats['max']))}")
            elif stats is not None:
                parts.append(f"{names[column]} {texts['mostly'].format(value=stats)}")
        label = ""
        if binary:
            label = f" – {own['meets'] if group == 1 else own['not_meets']}"
        lines.append(f"- {texts['group'].format(code=group)}{label} ({size} {texts['accessions']}): " + "; ".join(parts) + ".")

    lines += ["", texts["differences"], ""]
    for sentence in describe_differences(profiles, names, units, texts):
        lines.append(f"- {sentence}.")

    lines += ["", texts["left_out"], ""]
    left_out = int(candidate[cluster_column].isna().sum())
    if left_out == 0:
        lines.append(f"- {texts['none_left_out']}")
    else:
        lines.append(f"- {texts['no_trait_data'].format(n=left_out)}.")
        lines.append(f"- {texts['reason_column'].format(column=criteria_column)}")

    return "\n".join(lines)
