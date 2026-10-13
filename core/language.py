"""Lightweight language detection and the user-facing strings of the fixed response blocks.

The LLM writes its own answer in the user's language, but the blocks added by
the system (status line, activity summary, preview header) are rendered in
Python, so they need translations of their own. Only Spanish and English are
supported; anything else falls back to English.
"""

import re

# Languages supported by the fixed blocks.
SUPPORTED_LANGUAGES = ("es", "en")
DEFAULT_LANGUAGE = "en"
# Full names used in the system prompt instruction.
LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}

# Frequent function words that rarely appear in the other language.
_SPANISH_WORDS = frozenset(
    "el la los las de del que y en un una por para con sin es son está están como más pero "
    "sus al lo le me dame quiero necesito busca buscar lista listado accesiones cultivo "
    "filtra filtrar agrupa agrupar crea crear exporta exportar hola gracias cuales cuáles donde dónde".split()
)
_ENGLISH_WORDS = frozenset(
    "the of and to in is are with for from that this which what where how give me want need "
    "find search list accessions crop filter group create export hello thanks please show".split()
)

# Strings of the fixed response blocks.
STRINGS: dict[str, dict[str, str]] = {
    "en": {
        "status_no_data": "**Session:** no accession list loaded yet.",
        "status": "**Session:** {original} accessions in the Original list → **{candidate}** in the Candidate list ({mode} mode).",
        "mode_local": "local file",
        "mode_genesys": "Genesys",
        "activities": "**Activities performed so far:**",
        "accessions": "accessions",
        "preview": "**Candidate list preview** (showing {shown} of {total} accessions):",
        "preview_empty": "**Candidate list preview:** the Candidate list is currently empty.",
        "hidden_columns": "_{hidden} additional columns not shown in the preview; export the list to see them all._",
    },
    "es": {
        "status_no_data": "**Sesión:** todavía no se ha cargado ninguna lista de accesiones.",
        "status": "**Sesión:** {original} accesiones en la lista original → **{candidate}** en la lista candidata (modo {mode}).",
        "mode_local": "archivo local",
        "mode_genesys": "Genesys",
        "activities": "**Actividades realizadas hasta ahora:**",
        "accessions": "accesiones",
        "preview": "**Vista previa de la lista candidata** (se muestran {shown} de {total} accesiones):",
        "preview_empty": "**Vista previa de la lista candidata:** la lista candidata está vacía.",
        "hidden_columns": "_{hidden} columnas adicionales no se muestran en la vista previa; exporta la lista para verlas todas._",
    },
}


def detect_language(text: str, default: str | None = None) -> str | None:
    """Guess whether a message is written in Spanish or English.

    Args:
        text: Message typed by the user.
        default: Value returned when the text gives no evidence (e.g. a bare file upload).

    Returns:
        ``"es"``, ``"en"`` or ``default``.
    """
    words = re.findall(r"[a-záéíóúñü]+", (text or "").lower())

    # Count the function words of each language; accents are strong Spanish evidence.
    spanish = sum(1 for word in words if word in _SPANISH_WORDS) + sum(1 for word in words if re.search(r"[áéíóúñ]", word))
    english = sum(1 for word in words if word in _ENGLISH_WORDS)

    if spanish == 0 and english == 0:
        return default
    return "es" if spanish >= english else "en"


def strings_for(language: str | None) -> dict[str, str]:
    """Return the fixed strings of a language (English when unsupported)."""
    return STRINGS.get(language or DEFAULT_LANGUAGE, STRINGS[DEFAULT_LANGUAGE])
