"""Builds the human-readable report of a climate/soil clustering.

The report is composed from the real analysis results (indicator catalogue
entries, per-cell values returned by the Subsetting API and the cluster of
every accession) in the user's language. Indicators are always named with
their full catalogue name; the interpretation sentences come from
``INTERPRETATIONS`` (by indicator prefix) or ``CATEGORY_INTERPRETATIONS``
(by catalogue category) with a generic fallback.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.reporting import describe_differences, fmt_number, mode_profile, numeric_profile, report_strings
from sdks.subsetting.models import Indicator

# What each indicator reveals, by prefix (lower case). Keys: "en", "es".
INTERPRETATIONS: dict[str, dict[str, str]] = {
    "cdd": {"en": "the longest run of consecutive days without rain; higher values mean longer dry spells and a higher drought risk",
            "es": "la racha más larga de días consecutivos sin lluvia; valores más altos indican sequías más prolongadas y mayor riesgo de déficit hídrico"},
    "t_rain": {"en": "the accumulated rainfall of the period; lower values indicate drier environments",
               "es": "la lluvia acumulada del período; valores más bajos indican ambientes más secos"},
    "ndws": {"en": "the number of days with water stress for the crop; higher values mean the soil water was insufficient more often",
             "es": "el número de días con estrés hídrico para el cultivo; valores más altos indican que el agua del suelo fue insuficiente con más frecuencia"},
    "ndwl": {"en": "the number of days with excess water (waterlogging); higher values indicate flooding-prone sites",
             "es": "el número de días con exceso de agua (anegamiento); valores más altos indican sitios propensos a inundación"},
    "tx": {"en": "the average maximum daily temperature; higher values mean hotter days",
           "es": "la temperatura máxima diaria promedio; valores más altos indican días más calurosos"},
    "tm": {"en": "the average daily temperature; it separates warm from cool environments",
           "es": "la temperatura diaria promedio; separa los ambientes cálidos de los frescos"},
    "tn": {"en": "the average minimum daily temperature; lower values mean colder nights",
           "es": "la temperatura mínima diaria promedio; valores más bajos indican noches más frías"},
    "nvpd4": {"en": "the number of days with high atmospheric demand (vapour pressure deficit); higher values mean more days of hot, dry air",
              "es": "el número de días con alta demanda atmosférica (déficit de presión de vapor); valores más altos indican más días de aire caliente y seco"},
    "days_heat": {"en": "the number of days above the heat threshold of the crop; higher values mean more heat stress",
                  "es": "el número de días por encima del umbral de calor del cultivo; valores más altos indican más estrés por calor"},
    "days_cold": {"en": "the number of days below the cold threshold of the crop; higher values mean more cold stress",
                  "es": "el número de días por debajo del umbral de frío del cultivo; valores más altos indican más estrés por frío"},
    "days_optm": {"en": "the number of days within the optimum temperature range of the crop; higher values mean more favourable conditions",
                  "es": "el número de días dentro del rango óptimo de temperatura del cultivo; valores más altos indican condiciones más favorables"},
    "phihox": {"en": "the soil pH; values below 5.5 indicate acid soils and above 7.5 alkaline soils",
               "es": "el pH del suelo; valores por debajo de 5.5 indican suelos ácidos y por encima de 7.5 suelos alcalinos"},
    "orcdrc": {"en": "the soil organic carbon content; higher values indicate more fertile, better structured soils",
               "es": "el contenido de carbono orgánico del suelo; valores más altos indican suelos más fértiles y mejor estructurados"},
    "clyppt": {"en": "the clay fraction of the soil; higher values mean heavier soils that hold more water",
               "es": "la fracción de arcilla del suelo; valores más altos indican suelos más pesados que retienen más agua"},
    "sndppt": {"en": "the sand fraction of the soil; higher values mean lighter soils that drain faster",
               "es": "la fracción de arena del suelo; valores más altos indican suelos más livianos que drenan más rápido"},
    "sltppt": {"en": "the silt fraction of the soil", "es": "la fracción de limo del suelo"},
    "cecsol": {"en": "the cation exchange capacity of the soil; higher values indicate a better capacity to retain nutrients",
               "es": "la capacidad de intercambio catiónico del suelo; valores más altos indican mejor capacidad para retener nutrientes"},
    "bldfie": {"en": "the soil bulk density; higher values indicate more compact soils",
               "es": "la densidad aparente del suelo; valores más altos indican suelos más compactos"},
}

# Fallback by catalogue category keyword (matched as substring, lower case).
CATEGORY_INTERPRETATIONS: dict[str, dict[str, str]] = {
    "drought": {"en": "the exposure of the site to water deficit", "es": "la exposición del sitio al déficit hídrico"},
    "heat": {"en": "the exposure of the site to high temperatures", "es": "la exposición del sitio a altas temperaturas"},
    "flood": {"en": "the exposure of the site to excess water", "es": "la exposición del sitio al exceso de agua"},
    "photoperiod": {"en": "the day length and radiation regime of the site", "es": "el régimen de duración del día y radiación del sitio"},
    "soil": {"en": "a physical or chemical property of the soil", "es": "una propiedad física o química del suelo"},
    "cold": {"en": "the exposure of the site to low temperatures", "es": "la exposición del sitio a bajas temperaturas"},
}


def interpret_indicator(indicator: Indicator, language: str) -> str:
    """Return the sentence explaining what an indicator reveals.

    Args:
        indicator: Catalogue entry.
        language: ``"es"`` or ``"en"``.

    Returns:
        Explanation text (without the indicator name).
    """
    lang = language if language in ("es", "en") else "en"
    known = INTERPRETATIONS.get(indicator.pref.lower())

    if known:
        return known[lang]

    # Fall back to the catalogue category, then to a generic sentence.
    category = (indicator.category or "").lower()
    for keyword, texts in CATEGORY_INTERPRETATIONS.items():
        if keyword in category:
            return texts[lang]

    return report_strings(lang)["no_interpretation"].format(name=indicator.name.lower())


def cell_value(values: dict[str, Any], indicator: Indicator) -> Any:
    """Reduce the per-cell analysis columns of one indicator to a single value.

    Monthly columns (``pref_month1``...) are averaged; extracted indicators use
    ``pref_value``; categorical ones use ``pref_category``.

    Args:
        values: ``{column: value}`` of the cell as returned by ``/cluster``.
        indicator: Indicator to read.

    Returns:
        A float, a category text or ``None``.
    """
    prefix = indicator.pref.lower() + "_"
    numbers = [v for k, v in values.items() if str(k).lower().startswith(prefix) and isinstance(v, (int, float)) and not pd.isna(v)]
    categories = [v for k, v in values.items() if str(k).lower() == f"{prefix}category"]

    if numbers:
        return sum(numbers) / len(numbers)
    if categories:
        return categories[0]

    return None


def build_climate_report(
    *,
    language: str | None,
    crop: str | None,
    indicators: list[Indicator],
    candidate: pd.DataFrame,
    cluster_column: str,
    cellid_column: str,
    indicator_values: dict[int, dict[str, Any]],
    without_cellid: int,
    without_data: int,
    cells_without_data: int,
    criteria_column: str,
) -> str:
    """Compose the markdown report of a climate clustering.

    Args:
        language: User language.
        crop: Crop used for crop-specific indicators (``None`` when unknown).
        indicators: Catalogue entries used in the analysis.
        candidate: Candidate list after the clustering.
        cluster_column: Column holding the cluster of every accession.
        cellid_column: Column holding the cell id.
        indicator_values: Per-cell analysis values returned by the API.
        without_cellid: Accessions without coordinates.
        without_data: Accessions whose cells have no climate data.
        cells_without_data: Distinct cells without climate data.
        criteria_column: Column where the reasons were written.

    Returns:
        Markdown text with the five sections of the report.
    """
    lang = language if language in ("es", "en") else "en"
    texts = report_strings(lang)
    names = {indicator.pref: indicator.name for indicator in indicators}
    units = {indicator.pref: indicator.unit for indicator in indicators}

    assigned = candidate[candidate[cluster_column].notna()]
    groups = sorted(int(code) for code in assigned[cluster_column].unique())

    # Per-cell value of every indicator, then per-group statistics weighted by accession.
    per_row = pd.DataFrame({
        indicator.pref: assigned[cellid_column].map(lambda cell, ind=indicator: cell_value(indicator_values.get(int(cell), {}), ind) if pd.notna(cell) else None)
        for indicator in indicators
    }, index=assigned.index)
    per_row[cluster_column] = assigned[cluster_column].astype(int)

    profiles: dict[int, dict[str, Any]] = {}
    for group in groups:
        rows = per_row[per_row[cluster_column] == group]
        profile: dict[str, Any] = {}
        for indicator in indicators:
            stats = numeric_profile(rows[indicator.pref])
            profile[indicator.pref] = stats if stats is not None else mode_profile(rows[indicator.pref])
        profiles[group] = profile

    lines = [texts["title_climate"], "", texts["overview"], ""]
    lines.append(f"- {texts['crop']}: {crop or texts['crop_unknown']}.")
    lines.append(f"- {texts['indicators']}: " + "; ".join(f"{indicator.name}" + (f" ({indicator.unit})" if indicator.unit else "") for indicator in indicators) + ".")
    lines.append(f"- {texts['purpose_climate']}")
    lines.append(f"- {texts['groups_found'].format(n=len(groups), assigned=len(assigned))}.")

    lines += ["", texts["interpretation"], ""]
    for indicator in indicators:
        lines.append(f"- {indicator.name}: {interpret_indicator(indicator, lang)}.")

    lines += ["", texts["profiles"], ""]
    for group in groups:
        size = int((per_row[cluster_column] == group).sum())
        parts = []
        for indicator in indicators:
            stats = profiles[group][indicator.pref]
            if isinstance(stats, dict):
                parts.append(f"{indicator.name} {texts['mean_min_max'].format(mean=fmt_number(stats['mean'], indicator.unit), min=fmt_number(stats['min']), max=fmt_number(stats['max']))}")
            elif stats is not None:
                parts.append(f"{indicator.name} {texts['mostly'].format(value=stats)}")
        lines.append(f"- {texts['group'].format(code=group)} ({size} {texts['accessions']}): " + "; ".join(parts) + ".")

    lines += ["", texts["differences"], ""]
    for sentence in describe_differences(profiles, names, units, texts):
        lines.append(f"- {sentence}.")

    lines += ["", texts["left_out"], ""]
    if without_cellid == 0 and without_data == 0:
        lines.append(f"- {texts['none_left_out']}")
    else:
        if without_cellid:
            lines.append(f"- {texts['no_coordinates'].format(n=without_cellid)}.")
        if without_data:
            lines.append(f"- {texts['no_climate_data'].format(n=without_data, cells=cells_without_data)}.")
        lines.append(f"- {texts['reason_column'].format(column=criteria_column)}")

    return "\n".join(lines)
