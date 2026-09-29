"""Skill that loads the first list of accessions of a session.

Two source modes exist:

* ``local``   - the user uploaded an Excel/CSV file; every column is passport data.
* ``genesys`` - accessions are searched in Genesys PGR through its API using
  structured criteria (crop, genus, species, country, institute, sampStat...).

Loading a list creates the Original list and the Candidate list in the session
state (see :meth:`core.state.SessionState.set_original_list`).
"""

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState, SourceMode
from sdks.genesys import (
    AccessionFilter,
    CountryFilter,
    GenesysAuthError,
    GenesysClient,
    GenesysConnectionError,
    GenesysError,
    GenesysRequestError,
    InstituteFilter,
    TaxonomyFilter,
)
from skills.base import Skill
from skills.list_accessions.loaders import (
    SUPPORTED_EXTENSIONS,
    UnsupportedFileError,
    detect_coordinate_columns,
    read_accession_file,
)

logger = get_logger(__name__)

# Maximum number of column names echoed back to the model.
_MAX_COLUMNS_IN_RESULT = 60

# Factory that builds a Genesys client on demand (injectable for tests).
ClientFactory = Callable[[], GenesysClient]


def _default_client_factory() -> GenesysClient:
    """Build a Genesys client from the application settings.

    Returns:
        A configured ``GenesysClient``.
    """
    return GenesysClient.from_settings(get_settings())


def _flatten(items: list[Any]) -> list[Any]:
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
            flat.extend(_flatten(list(item)))
        else:
            flat.append(item)

    return flat


def _parse_text_list(text: str) -> list[Any]:
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


def _as_list(value: Any) -> list[Any] | None:
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
        items: list[Any] = _parse_text_list(value)
    elif isinstance(value, (list, tuple)):
        items = _flatten(list(value))
    else:
        items = [value]

    # Strings inside lists may themselves hide several values (e.g. ['["COL"]']
    # or ['100, 500']); decode/split them too.
    expanded: list[Any] = []
    for item in items:
        if isinstance(item, str) and (item.strip().startswith("[") or re.search(r"[,;|]", item)):
            expanded.extend(_parse_text_list(item))
        else:
            expanded.append(item)

    cleaned = [item for item in expanded if item is not None and str(item).strip() != ""]

    return cleaned or None


def _as_upper_list(value: Any) -> list[str] | None:
    """Like :func:`_as_list` but upper-cases every item (ISO3 / WIEWS codes)."""
    items = _as_list(value)
    return None if items is None else [str(item).strip().upper() for item in items]


def _as_int_list(value: Any, name: str = "samp_stat") -> list[int] | None:
    """Like :func:`_as_list` but converts every item to ``int``.

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
    items = _as_list(value)

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


def _as_bool(value: Any) -> bool | None:
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


def _as_int(value: Any) -> int | None:
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


class ListAccessionsSkill(Skill):
    """Load accessions from a local file or from Genesys and start the session lists."""

    name = "list_accessions"
    description = (
        "Load the first list of accessions of the session and create the Original list "
        "and the Candidate list. Use source='local' with the path of a file the user "
        "attached (Excel or CSV; every column becomes passport data). Use "
        "source='genesys' with structured criteria (crop, genus, species, "
        "country_of_origin, institute_code, samp_stat, accession_numbers, text) when "
        "the user gives no file and wants accessions from Genesys PGR; at least one "
        "criterion is required. Calling it again replaces the current lists."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["local", "genesys"],
                "description": "Where the accessions come from: 'local' (attached file) or 'genesys'.",
            },
            "file_path": {
                "type": "string",
                "description": "Path of the attached Excel/CSV file (required when source='local').",
            },
            "sheet_name": {
                "type": "string",
                "description": "Excel sheet to read. Defaults to the first sheet.",
            },
            "crop": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Crop names as the user said them, e.g. ['beans'], ['frijol'], ['maize']; "
                    "they are resolved against the Genesys crop catalogue. Do not use when "
                    "genus/species are given (genesys mode)."
                ),
            },
            "genus": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Genera, e.g. ['Phaseolus'] (genesys mode).",
            },
            "species": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific epithets, e.g. ['vulgaris'] (genesys mode).",
            },
            "country_of_origin": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ISO-3166 alpha-3 codes of the country of origin, e.g. ['COL','PER'].",
            },
            "institute_code": {
                "type": "array",
                "items": {"type": "string"},
                "description": "FAO WIEWS codes of the holding institute, e.g. ['COL003'].",
            },
            "samp_stat": {
                "type": "array",
                "items": {"type": "integer"},
                "description": (
                    "MCPD biological status codes: 100 wild, 200 weedy, 300 traditional "
                    "cultivar/landrace, 400 breeding material, 500 advanced cultivar, 999 other."
                ),
            },
            "accession_numbers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exact accession numbers to retrieve, e.g. ['G50001'].",
            },
            "text": {
                "type": "string",
                "description": "Free-text keywords searched across the accession database.",
            },
            "query": {
                "type": "string",
                "description": "Deprecated alias of 'text'.",
            },
            "historic": {
                "type": "boolean",
                "description": "true to include only historical records. Omit for active accessions.",
            },
            "max_records": {
                "type": "integer",
                "description": "Maximum accessions to download from Genesys (default from configuration).",
            },
        },
        "required": ["source"],
    }

    def __init__(self, client_factory: ClientFactory | None = None) -> None:
        """Create the skill.

        Args:
            client_factory: Callable returning a ``GenesysClient``. Defaults to a
                factory that reads the application settings (lazy, first use).
        """
        self._client_factory = client_factory or _default_client_factory
        self._client: GenesysClient | None = None

    # ---------------------------------------------------------------- run
    def run(
        self,
        state: SessionState,
        paths: SessionPaths,
        source: str = "",
        file_path: str | None = None,
        sheet_name: str | None = None,
        crop: list[str] | str | None = None,
        genus: list[str] | str | None = None,
        species: list[str] | str | None = None,
        country_of_origin: list[str] | str | None = None,
        institute_code: list[str] | str | None = None,
        samp_stat: list[int] | int | None = None,
        accession_numbers: list[str] | str | None = None,
        text: str | None = None,
        query: str | None = None,
        historic: bool | str | None = None,
        max_records: int | str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Dispatch to the local or Genesys loader.

        Args:
            state: Session state to populate.
            paths: Session folders (uploads must live in ``paths.inputs``).
            source: ``"local"`` or ``"genesys"``.
            file_path: Attached file path (local mode).
            sheet_name: Excel sheet (local mode, optional).
            crop: Genesys crop codes (genesys mode).
            genus: Genera (genesys mode).
            species: Specific epithets (genesys mode).
            country_of_origin: ISO3 country codes (genesys mode).
            institute_code: WIEWS institute codes (genesys mode).
            samp_stat: MCPD biological status codes (genesys mode).
            accession_numbers: Exact accession numbers (genesys mode).
            text: Free-text keywords (genesys mode).
            query: Deprecated alias of ``text``.
            historic: Restrict to historical records (genesys mode).
            max_records: Download cap (genesys mode).
            **_: Extra arguments sent by the model are ignored.

        Returns:
            Tool result dictionary.
        """
        mode = (source or "").strip().lower()

        # Route by mode; anything else is a model mistake we report explicitly.
        if mode == SourceMode.LOCAL.value:
            return self._run_local(state, paths, file_path, sheet_name)

        if mode == SourceMode.GENESYS.value:
            return self._run_genesys(
                state,
                crop=crop,
                genus=genus,
                species=species,
                country_of_origin=country_of_origin,
                institute_code=institute_code,
                samp_stat=samp_stat,
                accession_numbers=accession_numbers,
                text=text or query,
                historic=_as_bool(historic),
                max_records=_as_int(max_records),
            )

        return self.error(
            f"Unknown source '{source}'. Use 'local' for an attached file or 'genesys' "
            "to search accessions in Genesys PGR."
        )

    # ------------------------------------------------------------- local
    def _run_local(
        self,
        state: SessionState,
        paths: SessionPaths,
        file_path: str | None,
        sheet_name: str | None,
    ) -> dict[str, Any]:
        """Load accessions from an uploaded Excel/CSV file.

        Args:
            state: Session state to populate.
            paths: Session folders.
            file_path: Path of the uploaded file.
            sheet_name: Excel sheet to read (optional).

        Returns:
            Tool result with the shape of the loaded table.
        """
        # Without a path the model must ask the user for the file.
        if not file_path:
            return self.error(
                "Local mode needs the path of an attached Excel or CSV file. "
                "Ask the user to upload the file."
            )

        resolved = self._resolve_input_file(paths, Path(file_path))

        # Only files inside the session inputs folder may be read.
        if resolved is None:
            uploaded = [path.name for path in state.uploaded_files]
            return self.error(
                f"File '{file_path}' is not available in this session. "
                f"Files uploaded so far: {uploaded or 'none'}."
            )

        try:
            dataframe = read_accession_file(resolved, sheet_name)
        except UnsupportedFileError as exc:
            return self.error(str(exc), supported_extensions=list(SUPPORTED_EXTENSIONS))
        except (FileNotFoundError, ValueError) as exc:
            return self.error(str(exc))

        replaced = state.has_data
        state.set_original_list(dataframe, SourceMode.LOCAL)
        state.extras["source_file"] = str(resolved)

        description = f"Loaded {len(dataframe)} accessions from file '{resolved.name}' (local mode)"

        # Tell the user the previous lists were discarded when re-loading.
        if replaced:
            description += ", replacing the previous Original and Candidate lists"

        state.log_activity(self.name, description)

        columns = [str(column) for column in dataframe.columns]
        coordinates = detect_coordinate_columns(columns)

        return self.ok(
            f"{description}. All {len(columns)} columns are treated as passport data.",
            mode=SourceMode.LOCAL.value,
            accessions=len(dataframe),
            columns=columns[:_MAX_COLUMNS_IN_RESULT],
            total_columns=len(columns),
            coordinate_columns=coordinates,
            replaced_previous_lists=replaced,
        )

    @staticmethod
    def _resolve_input_file(paths: SessionPaths, candidate: Path) -> Path | None:
        """Locate an uploaded file inside the session ``inputs`` folder.

        The model may pass the full path or just the file name; both are accepted
        as long as the final file lives under ``paths.inputs``.

        Args:
            paths: Session folders.
            candidate: Path or name given by the model.

        Returns:
            The resolved path, or ``None`` when the file is outside ``inputs`` or missing.
        """
        inputs_root = paths.inputs.resolve()
        options = [candidate, paths.inputs / candidate.name]

        # Try the path as given first, then the same file name inside inputs.
        for option in options:
            resolved = option.resolve()

            try:
                resolved.relative_to(inputs_root)
            except ValueError:
                # Outside the session folder: never read it, even if it exists.
                continue

            if resolved.is_file():
                return resolved

        return None

    # ----------------------------------------------------------- genesys
    def _get_client(self) -> GenesysClient:
        """Return the Genesys client, creating it on first use."""
        # Lazy creation so importing the skill never requires credentials.
        if self._client is None:
            self._client = self._client_factory()

        return self._client

    @staticmethod
    def build_genesys_filter(
        *,
        crop: Any = None,
        genus: Any = None,
        species: Any = None,
        country_of_origin: Any = None,
        institute_code: Any = None,
        samp_stat: Any = None,
        accession_numbers: Any = None,
        text: str | None = None,
        historic: bool | None = None,
    ) -> AccessionFilter:
        """Translate the tool arguments into an ``AccessionFilter``.

        Args:
            crop: Genesys crop codes.
            genus: Genera.
            species: Specific epithets.
            country_of_origin: ISO3 codes.
            institute_code: WIEWS codes.
            samp_stat: MCPD biological status codes.
            accession_numbers: Exact accession numbers.
            text: Free-text keywords.
            historic: Restrict to historical records.

        Returns:
            The filter; empty when no criterion was given.

        Raises:
            ValueError: If ``samp_stat`` contains non-integer values.
        """
        return AccessionFilter(
            crop=[str(item).strip().lower() for item in _as_list(crop)] if _as_list(crop) else None,
            taxonomy=TaxonomyFilter(
                genus=[str(item).strip().capitalize() for item in _as_list(genus)] if _as_list(genus) else None,
                species=[str(item).strip().lower() for item in _as_list(species)] if _as_list(species) else None,
            ),
            country_of_origin=CountryFilter(code3=_as_upper_list(country_of_origin)),
            institute=InstituteFilter(code=_as_upper_list(institute_code)),
            samp_stat=_as_int_list(samp_stat),
            accession_numbers=[str(item).strip() for item in _as_list(accession_numbers)]
            if _as_list(accession_numbers)
            else None,
            text=" ".join(str(item) for item in _as_list(text)) if _as_list(text) else None,
            historic=historic,
        )

    def _run_genesys(
        self,
        state: SessionState,
        *,
        crop: Any,
        genus: Any,
        species: Any,
        country_of_origin: Any,
        institute_code: Any,
        samp_stat: Any,
        accession_numbers: Any,
        text: str | None,
        historic: bool | None,
        max_records: int | None,
    ) -> dict[str, Any]:
        """Load accessions from Genesys PGR using structured criteria.

        Args:
            state: Session state to populate.
            crop: Genesys crop codes.
            genus: Genera.
            species: Specific epithets.
            country_of_origin: ISO3 codes.
            institute_code: WIEWS codes.
            samp_stat: MCPD biological status codes.
            accession_numbers: Exact accession numbers.
            text: Free-text keywords.
            historic: Restrict to historical records.
            max_records: Download cap.

        Returns:
            Tool result with the shape of the loaded table.
        """
        notes: list[str] = []  # explanations appended to the result for the model
        unresolved_crops: list[str] = []

        try:
            # Crop names typed by users ("bean", "frijol") are translated into the
            # real Genesys crop codes; unknown names are dropped and reported.
            crop_names = [str(item) for item in (_as_list(crop) or [])]
            crop_codes: list[str] | None = None

            if crop_names:
                crop_codes, unresolved_crops = self._get_client().resolve_crop_codes(crop_names)

                if unresolved_crops:
                    notes.append(
                        f"Crop names not recognised by Genesys and ignored: {unresolved_crops}."
                    )

            accession_filter = self.build_genesys_filter(
                crop=crop_codes or None,
                genus=genus,
                species=species,
                country_of_origin=country_of_origin,
                institute_code=institute_code,
                samp_stat=samp_stat,
                accession_numbers=accession_numbers,
                text=text,
                historic=historic,
            )

            # Searching the whole of Genesys is never what the user means.
            if accession_filter.is_empty():
                return self.error(
                    "Genesys mode needs at least one search criterion: crop, genus, species, "
                    "country of origin, institute code, biological status (samp_stat), accession "
                    "numbers or free text. Ask the user what accessions they are looking for.",
                    unresolved_crops=unresolved_crops,
                )

            body = accession_filter.to_body()
            logger.info("Session %s: Genesys search with %s", state.session_id, body)
            client = self._get_client()
            total = client.count_accessions(accession_filter)

            # A crop code combined with a taxonomy can be contradictory in Genesys
            # (the taxon may be catalogued under another crop): retry on taxonomy only.
            if total == 0 and accession_filter.crop and accession_filter.taxonomy is not None:
                retry_filter = accession_filter.model_copy(update={"crop": None})

                if not retry_filter.is_empty():
                    retry_total = client.count_accessions(retry_filter)

                    if retry_total > 0:
                        notes.append(
                            f"Crop {accession_filter.crop} combined with the taxonomy returned no "
                            "accessions; the search was repeated using the taxonomy without the crop."
                        )
                        accession_filter = retry_filter
                        body = accession_filter.to_body()
                        total = retry_total

            # Nothing matched: report it without touching the current lists.
            if total == 0:
                state.log_activity(self.name, f"Genesys search returned no accessions for {body}")
                return self.error(
                    "No accessions in Genesys match the given criteria. Suggest the user to "
                    "relax the filters (other country, genus only, fewer codes). " + " ".join(notes),
                    mode=SourceMode.GENESYS.value,
                    filter=body,
                    total_matching=0,
                    unresolved_crops=unresolved_crops,
                )

            cap = max_records or client.max_records
            dataframe = client.list_accessions_dataframe(accession_filter, max_records=cap)
        except ValueError as exc:
            return self.error(f"Invalid Genesys criteria: {exc}")
        except GenesysAuthError as exc:
            logger.error("Genesys authentication failed: %s", exc)
            return self.error(
                "Genesys rejected the configured credentials. The API token must be checked "
                "by the system administrator; meanwhile the user can upload a file (local mode)."
            )
        except GenesysConnectionError as exc:
            logger.error("Genesys unreachable: %s", exc)
            return self.error(
                "Genesys could not be reached right now. Suggest retrying later or uploading a file."
            )
        except (GenesysRequestError, GenesysError) as exc:
            logger.error("Genesys request failed: %s", exc)
            return self.error(f"Genesys returned an error for this search: {exc}")

        replaced = state.has_data
        state.set_original_list(dataframe, SourceMode.GENESYS)
        state.extras["genesys_filter"] = body
        state.extras["genesys_total_matching"] = total

        loaded = len(dataframe)
        truncated = loaded < total
        description = f"Loaded {loaded} accessions from Genesys PGR (genesys mode) matching {body}"

        # Be explicit when the download cap hid part of the results.
        if truncated:
            description += f"; {total} accessions match but only the first {loaded} were loaded"

        # Tell the user the previous lists were discarded when re-loading.
        if replaced:
            description += ", replacing the previous Original and Candidate lists"

        state.log_activity(self.name, description)

        columns = [str(column) for column in dataframe.columns]

        return self.ok(
            f"{description}. Passport data columns follow the MCPD standard. " + " ".join(notes),
            mode=SourceMode.GENESYS.value,
            accessions=loaded,
            total_matching=total,
            truncated=truncated,
            filter=body,
            notes=notes,
            unresolved_crops=unresolved_crops,
            columns=columns[:_MAX_COLUMNS_IN_RESULT],
            total_columns=len(columns),
            coordinate_columns=detect_coordinate_columns(columns),
            replaced_previous_lists=replaced,
        )
