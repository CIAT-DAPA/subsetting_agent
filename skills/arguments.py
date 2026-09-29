"""Tolerant parsing of tool arguments produced by language models.

Small models often serialise arrays as JSON text (``'["COL"]'``), nest lists
(``[[300]]``) or send numbers and booleans as strings. Every skill should
normalise its arguments through these helpers instead of trusting the schema.
"""

import json
import re
from typing import Any


def flatten(items: list[Any]) -> list[Any]:
    """Flatten nested lists/tuples of any depth into a single list.

    Args:
        items: Possibly nested list.

    Returns:
        Flat list preserving order.
    """
    flat: list[Any] = []

    # Recurse into nested containers, append scalars as they are.
    for item in items:
        if isinstance(item, (list, tuple)):
            flat.extend(flatten(list(item)))
        else:
            flat.append(item)

    return flat


def parse_text_list(text: str) -> list[Any]:
    """Interpret a string argument that should have been a list.

    Small models often serialise arrays as text: ``'["COL","PER"]'``, ``'[300]'``
    or ``'300, 100'``. JSON is tried first, then common separators.

    Args:
        text: Raw string sent by the model.

    Returns:
        The items found in the text (possibly a single item).
    """
    stripped = text.strip()

    # Looks like a JSON array/scalar: decode it when possible.
    if stripped.startswith("[") or stripped.startswith('"'):
        try:
            decoded = json.loads(stripped)
            return decoded if isinstance(decoded, list) else [decoded]
        except json.JSONDecodeError:
            # Fall through to the separator-based split.
            stripped = stripped.strip("[]")

    # Split on the usual separators; a plain value yields a single item.
    return [part.strip().strip("\"'") for part in re.split(r"[,;|]", stripped) if part.strip()]


def as_list(value: Any) -> list[Any] | None:
    """Normalise a scalar, list or text argument coming from the model into a list.

    Handles nested lists (``[[300]]``), JSON encoded arrays (``'["COL"]'``) and
    separator-delimited strings (``'300, 100'``).

    Args:
        value: ``None``, a scalar, a list or a string.

    Returns:
        ``None`` when the value is empty, otherwise a flat list without blanks.
    """
    # Nothing given: keep the criterion out of the filter.
    if value is None:
        return None

    # Strings may hide a serialised list; decode them before flattening.
    if isinstance(value, str):
        items: list[Any] = parse_text_list(value)
    elif isinstance(value, (list, tuple)):
        items = flatten(list(value))
    else:
        items = [value]

    # Strings inside lists may themselves hide several values (e.g. ['["COL"]']
    # or ['100, 500']); decode/split them too.
    expanded: list[Any] = []
    for item in items:
        if isinstance(item, str) and (item.strip().startswith("[") or re.search(r"[,;|]", item)):
            expanded.extend(parse_text_list(item))
        else:
            expanded.append(item)

    cleaned = [item for item in expanded if item is not None and str(item).strip() != ""]

    return cleaned or None


def as_upper_list(value: Any) -> list[str] | None:
    """Like :func:`as_list` but upper-cases every item (ISO3 / WIEWS codes)."""
    items = as_list(value)
    return None if items is None else [str(item).strip().upper() for item in items]


def as_int_list(value: Any, name: str = "samp_stat") -> list[int] | None:
    """Like :func:`as_list` but converts every item to ``int``.

    Accepts integers, integral floats (``300.0``) and numeric strings (``"300"``).

    Args:
        value: Raw argument.
        name: Argument name used in the error message.

    Returns:
        List of integer codes, or ``None`` when nothing was given.

    Raises:
        ValueError: If any item is not an integer code; the message names the
            offending values so the model can explain them to the user.
    """
    items = as_list(value)

    if items is None:
        return None

    codes: list[int] = []
    invalid: list[Any] = []

    # Convert item by item, collecting the ones that are not integer codes.
    for item in items:
        try:
            number = float(str(item).strip())
        except ValueError:
            invalid.append(item)
            continue

        if number != int(number):
            invalid.append(item)
            continue

        codes.append(int(number))

    if invalid:
        raise ValueError(
            f"'{name}' must contain MCPD integer codes such as 100, 300 or 500; "
            f"received invalid values: {invalid}"
        )

    return codes


def as_bool(value: Any) -> bool | None:
    """Interpret a boolean argument that may arrive as text.

    Args:
        value: ``None``, a bool or a string such as ``"true"``/``"false"``.

    Returns:
        The boolean, or ``None`` when empty or unrecognised.
    """
    if value is None or isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    # Only explicit truthy/falsy words are accepted; anything else is ignored.
    if text in ("true", "yes", "1"):
        return True
    if text in ("false", "no", "0"):
        return False

    return None


def as_int(value: Any) -> int | None:
    """Interpret an integer argument that may arrive as text.

    Args:
        value: ``None``, a number or a numeric string.

    Returns:
        The integer, or ``None`` when empty or not numeric.
    """
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def as_float(value: Any) -> float | None:
    """Interpret a numeric argument that may arrive as text.

    Args:
        value: ``None``, a number or a numeric string.

    Returns:
        The float, or ``None`` when empty or not numeric.
    """
    if value is None or isinstance(value, bool):
        return None

    try:
        return float(str(value).strip().replace(",", "."))
    except ValueError:
        return None


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Interpret an argument that should be a list of JSON objects.

    Accepts a list of dicts, a single dict, or JSON text encoding either.

    Args:
        value: Raw argument.

    Returns:
        List of dictionaries (possibly empty).

    Raises:
        ValueError: If the text is not valid JSON or items are not objects.
    """
    # Nothing given: an empty list lets the caller decide what to do.
    if value is None:
        return []

    # JSON text: decode it first.
    if isinstance(value, str):
        stripped = value.strip()

        if not stripped:
            return []

        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Expected a JSON list of objects, received: {stripped[:200]}") from exc

    items = value if isinstance(value, list) else [value]
    result: list[dict[str, Any]] = []

    # Every item must be an object; nested JSON strings are decoded too.
    for item in items:
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Expected a JSON object, received: {item[:200]}") from exc

        if not isinstance(item, dict):
            raise ValueError(f"Expected a JSON object, received: {item!r}")

        result.append(item)

    return result
