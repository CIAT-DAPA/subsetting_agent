"""Shared helpers to write the analysis reports attached to cluster answers.

The reports are built in Python (never by the LLM) from the real results of the
analysis, in the user's language, and appended to the answer by the formatter
through ``SessionState.turn_report``. Each skill composes its own sections with
the helpers below; this module only knows about formatting and the common
bilingual strings.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.language import DEFAULT_LANGUAGE

# Bilingual strings shared by the reports.
REPORT_STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "title_climate": "### Environmental clustering report",
        "title_traits": "### Trait grouping report",
        "overview": "**General description**",
        "interpretation": "**What the indicators reveal**",
        "profiles": "**Profile of each group**",
        "differences": "**Differences between groups**",
        "left_out": "**Accessions left out**",
        "crop": "Crop analysed",
        "crop_unknown": "not specified (generic indicators)",
        "indicators": "Indicators used",
        "traits": "Traits used",
        "groups_found": "{n} groups were found among {assigned} accessions",
        "group": "Group {code}",
        "accessions": "accessions",
        "mean_min_max": "mean {mean}, from {min} to {max}",
        "highest": "{name}: highest in group {high} ({high_value}) and lowest in group {low} ({low_value})",
        "same": "{name}: similar in all groups ({value})",
        "mostly": "mostly {value}",
        "none_left_out": "All accessions received a group.",
        "no_coordinates": "{n} accessions have no coordinates, so no climate cell could be assigned to them",
        "no_climate_data": "{n} accessions (in {cells} collecting sites) have no values in the climate database for the indicators used; this is a coverage gap of the database, not an error",
        "no_trait_data": "{n} accessions have no values for the traits used, so they received no group",
        "reason_column": "The reason is written in the column '{column}' of the Candidate list.",
        "purpose_climate": "The collecting sites of the accessions were grouped by the similarity of their environmental conditions, so that each group represents a distinct type of environment and the accessions collected there can be chosen as candidates adapted to it.",
        "purpose_traits": "The accessions were grouped by the similarity of their trait values, so that each group gathers material with a comparable phenotypic profile.",
        "purpose_binary": "The accessions were split into those that meet the condition (group 1) and those that do not (group 0).",
        "tercile_note": "Numeric traits were divided into three levels with the same number of accessions each: low, medium and high.",
        "no_interpretation": "higher values indicate more {name}",
    },
    "es": {
        "title_climate": "### Informe de la agrupación por ambientes",
        "title_traits": "### Informe de la agrupación por rasgos",
        "overview": "**Descripción general**",
        "interpretation": "**Qué revelan los indicadores**",
        "profiles": "**Perfil de cada grupo**",
        "differences": "**Diferencias entre grupos**",
        "left_out": "**Accesiones no agrupadas**",
        "crop": "Cultivo analizado",
        "crop_unknown": "no especificado (indicadores genéricos)",
        "indicators": "Indicadores utilizados",
        "traits": "Rasgos utilizados",
        "groups_found": "Se encontraron {n} grupos entre {assigned} accesiones",
        "group": "Grupo {code}",
        "accessions": "accesiones",
        "mean_min_max": "promedio {mean}, entre {min} y {max}",
        "highest": "{name}: más alto en el grupo {high} ({high_value}) y más bajo en el grupo {low} ({low_value})",
        "same": "{name}: similar en todos los grupos ({value})",
        "mostly": "principalmente {value}",
        "none_left_out": "Todas las accesiones recibieron un grupo.",
        "no_coordinates": "{n} accesiones no tienen coordenadas, por lo que no se les pudo asignar una celda climática",
        "no_climate_data": "{n} accesiones (en {cells} sitios de colecta) no tienen valores en la base de datos climática para los indicadores usados; es un vacío de cobertura de la base de datos, no un error",
        "no_trait_data": "{n} accesiones no tienen valores para los rasgos usados, por lo que no recibieron grupo",
        "reason_column": "La razón quedó escrita en la columna '{column}' de la lista candidata.",
        "purpose_climate": "Los sitios de colecta de las accesiones se agruparon según la similitud de sus condiciones ambientales, de modo que cada grupo representa un tipo de ambiente distinto y las accesiones colectadas allí pueden elegirse como candidatas adaptadas a él.",
        "purpose_traits": "Las accesiones se agruparon según la similitud de los valores de sus rasgos, de modo que cada grupo reúne material con un perfil fenotípico comparable.",
        "purpose_binary": "Las accesiones se separaron entre las que cumplen la condición (grupo 1) y las que no la cumplen (grupo 0).",
        "tercile_note": "Los rasgos numéricos se dividieron en tres niveles con el mismo número de accesiones cada uno: bajo, medio y alto.",
        "no_interpretation": "valores más altos indican más {name}",
    },
}


def report_strings(language: str | None) -> dict[str, str]:
    """Return the report strings of a language (English fallback)."""
    return REPORT_STRINGS.get(language or DEFAULT_LANGUAGE, REPORT_STRINGS[DEFAULT_LANGUAGE])


def fmt_number(value: Any, unit: str | None = None) -> str:
    """Format a number with at most two decimals and an optional unit.

    Args:
        value: Number (or anything else, returned as text).
        unit: Unit appended after a space when given.

    Returns:
        Readable text such as ``"12.5 mm"``.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if pd.isna(number):
        return "NA"

    text = f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.2f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip() if unit else text


def numeric_profile(values: pd.Series) -> dict[str, float] | None:
    """Mean/min/max of a numeric series (``None`` when nothing is numeric)."""
    numbers = pd.to_numeric(values, errors="coerce").dropna()

    if numbers.empty:
        return None

    return {"mean": float(numbers.mean()), "min": float(numbers.min()), "max": float(numbers.max())}


def mode_profile(values: pd.Series) -> str | None:
    """Most frequent value of a categorical series (``None`` when empty)."""
    cleaned = values.dropna().astype(str)
    cleaned = cleaned[cleaned.str.strip() != ""]

    if cleaned.empty:
        return None

    return str(cleaned.value_counts().index[0])


def describe_differences(
    profiles: dict[int, dict[str, dict[str, float] | str | None]],
    names: dict[str, str],
    units: dict[str, str | None],
    texts: dict[str, str],
    tolerance: float = 0.05,
) -> list[str]:
    """Explain, per variable, which group has the highest and lowest mean.

    Args:
        profiles: ``{group: {variable: {"mean", "min", "max"} | mode | None}}``.
        names: Full name of every variable.
        units: Unit of every variable (``None`` when none).
        texts: Report strings of the language.
        tolerance: Relative difference below which groups are called similar.

    Returns:
        One sentence per variable.
    """
    sentences: list[str] = []

    for variable, name in names.items():
        means = {group: profile[variable]["mean"] for group, profile in profiles.items() if isinstance(profile.get(variable), dict)}

        # Categorical variables: list the dominant value of every group.
        if not means:
            modes = {group: profile.get(variable) for group, profile in profiles.items() if isinstance(profile.get(variable), str)}
            if modes:
                sentences.append(f"{name}: " + "; ".join(f"{texts['group'].format(code=group)} {texts['mostly'].format(value=value)}" for group, value in modes.items()))
            continue

        high = max(means, key=means.get)
        low = min(means, key=means.get)
        spread = abs(means[high] - means[low])
        scale = max(abs(means[high]), abs(means[low]), 1e-9)

        # Nearly equal means are reported as similar instead of ranking them.
        if spread / scale < tolerance:
            sentences.append(texts["same"].format(name=name, value=fmt_number(means[high], units.get(variable))))
        else:
            sentences.append(
                texts["highest"].format(
                    name=name, high=high, high_value=fmt_number(means[high], units.get(variable)),
                    low=low, low_value=fmt_number(means[low], units.get(variable)),
                )
            )

    return sentences
