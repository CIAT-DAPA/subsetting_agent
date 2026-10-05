"""Skill that loads the first list of accessions of a session.

Two source modes exist:

* ``local``   - the user uploaded an Excel/CSV file; every column is passport data.
* ``genesys`` - accessions are searched in Genesys PGR through its API using
  structured criteria (crop, genus, species, country, institute, sampStat...).

Loading a list creates the Original list and the Candidate list in the session
state (see :meth:`core.state.SessionState.set_original_list`).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import get_settings
from core.logger import get_logger
from core.session import SessionPaths
from core.state import SessionState, SourceMode
from sdks.subsetting import add_cellid_column
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
from skills.arguments import as_bool, as_int, as_int_list, as_list, as_upper_list
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
# Name of the column that stores the base-raster cell id of each accession.
CELLID_COLUMN = "cellid"

# Factory that builds a Genesys client on demand (injectable for tests).
ClientFactory = Callable[[], GenesysClient]


def _default_client_factory() -> GenesysClient:
    """Build a Genesys client from the application settings.

    Returns:
        A configured ``GenesysClient``.
    """
    return GenesysClient.from_settings(get_settings())


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
            "latitude_column": {
                "type": "string",
                "description": "Column with the latitude, only when the tool could not detect it.",
            },
            "longitude_column": {
                "type": "string",
                "description": "Column with the longitude, only when the tool could not detect it.",
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
        latitude_column: str | None = None,
        longitude_column: str | None = None,
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
            latitude_column: Column holding the latitude when auto-detection fails.
            longitude_column: Column holding the longitude when auto-detection fails.
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
        coordinate_hint = (latitude_column, longitude_column)

        # Route by mode; anything else is a model mistake we report explicitly.
        if mode == SourceMode.LOCAL.value:
            return self._run_local(state, paths, file_path, sheet_name, coordinate_hint)

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
                historic=as_bool(historic),
                max_records=as_int(max_records),
                coordinate_hint=coordinate_hint,
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
        coordinate_hint: tuple[str | None, str | None] = (None, None),
    ) -> dict[str, Any]:
        """Load accessions from an uploaded Excel/CSV file.

        Args:
            state: Session state to populate.
            paths: Session folders.
            file_path: Path of the uploaded file.
            sheet_name: Excel sheet to read (optional).
            coordinate_hint: ``(latitude_column, longitude_column)`` given by the model.

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

        dataframe, geo = self._attach_cellid(dataframe, coordinate_hint)

        replaced = state.has_data
        state.set_original_list(dataframe, SourceMode.LOCAL)
        state.extras["source_file"] = str(resolved)
        state.extras["coordinate_columns"] = geo["coordinate_columns"]

        description = f"Loaded {len(dataframe)} accessions from file '{resolved.name}' (local mode)"

        # Tell the user the previous lists were discarded when re-loading.
        if replaced:
            description += ", replacing the previous Original and Candidate lists"

        state.log_activity(self.name, description)
        self._log_cellid(state, geo)

        columns = [str(column) for column in dataframe.columns]

        return self.ok(
            f"{description}. All {len(columns)} columns are treated as passport data. {geo['message']}",
            mode=SourceMode.LOCAL.value,
            accessions=len(dataframe),
            columns=columns[:_MAX_COLUMNS_IN_RESULT],
            total_columns=len(columns),
            replaced_previous_lists=replaced,
            **{key: value for key, value in geo.items() if key != "message"},
        )

    # ------------------------------------------------------------ cellid
    @staticmethod
    def _attach_cellid(
        dataframe: pd.DataFrame, coordinate_hint: tuple[str | None, str | None] = (None, None)
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Compute the base-raster cell id for every georeferenced accession.

        The cell id is a property of the accession, so it is computed once at
        load time and stored in both the Original and the Candidate lists.

        Args:
            dataframe: Loaded accessions.
            coordinate_hint: Latitude/longitude columns given explicitly by the
                model; auto-detection is used when they are missing.

        Returns:
            ``(dataframe, info)`` where ``info`` holds ``coordinate_columns``,
            ``georeferenced``, ``without_cellid``, ``cellid_computed`` and a
            human readable ``message``.
        """
        columns = [str(column) for column in dataframe.columns]
        detected = detect_coordinate_columns(columns)
        latitude = coordinate_hint[0] if coordinate_hint[0] in columns else detected["latitude"]
        longitude = coordinate_hint[1] if coordinate_hint[1] in columns else detected["longitude"]
        coordinates = {"latitude": latitude, "longitude": longitude}

        # Without both coordinate columns the cell id cannot be computed.
        if not latitude or not longitude:
            return dataframe, {
                "coordinate_columns": coordinates,
                "cellid_computed": False,
                "georeferenced": 0,
                "without_cellid": len(dataframe),
                "message": (
                    "Latitude/longitude columns were not detected, so the 'cellid' needed by the "
                    "climate tools was not computed. If the file has coordinates, ask the user which "
                    "columns hold latitude and longitude and reload with latitude_column/longitude_column."
                ),
            }

        with_cellid = add_cellid_column(dataframe, latitude, longitude, get_settings().grid, CELLID_COLUMN)
        georeferenced = int(with_cellid[CELLID_COLUMN].notna().sum())
        missing = len(with_cellid) - georeferenced

        return with_cellid, {
            "coordinate_columns": coordinates,
            "cellid_computed": True,
            "georeferenced": georeferenced,
            "without_cellid": missing,
            "message": (
                f"Computed 'cellid' for {georeferenced} accessions with valid coordinates "
                f"({latitude}/{longitude}); {missing} accessions have no usable coordinates."
            ),
        }

    def _log_cellid(self, state: SessionState, geo: dict[str, Any]) -> None:
        """Record the cell id computation in the session activity log.

        Args:
            state: Session state.
            geo: Information returned by :meth:`_attach_cellid`.
        """
        # Log both outcomes so the user sees why climate tools may be unavailable.
        if geo["cellid_computed"]:
            columns = geo["coordinate_columns"]
            state.log_activity(
                self.name,
                f"Computed cellid for {geo['georeferenced']} of "
                f"{geo['georeferenced'] + geo['without_cellid']} accessions "
                f"(columns {columns['latitude']}/{columns['longitude']})",
            )
        else:
            state.log_activity(self.name, "No coordinate columns detected; cellid not computed")

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
            crop=[str(item).strip().lower() for item in as_list(crop)] if as_list(crop) else None,
            taxonomy=TaxonomyFilter(
                genus=[str(item).strip().capitalize() for item in as_list(genus)] if as_list(genus) else None,
                species=[str(item).strip().lower() for item in as_list(species)] if as_list(species) else None,
            ),
            country_of_origin=CountryFilter(code3=as_upper_list(country_of_origin)),
            institute=InstituteFilter(code=as_upper_list(institute_code)),
            samp_stat=as_int_list(samp_stat),
            accession_numbers=[str(item).strip() for item in as_list(accession_numbers)]
            if as_list(accession_numbers)
            else None,
            text=" ".join(str(item) for item in as_list(text)) if as_list(text) else None,
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
        coordinate_hint: tuple[str | None, str | None] = (None, None),
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
            coordinate_hint: Latitude/longitude columns given by the model (rarely needed).

        Returns:
            Tool result with the shape of the loaded table.
        """
        notes: list[str] = []  # explanations appended to the result for the model
        unresolved_crops: list[str] = []

        try:
            # Crop names typed by users ("bean", "frijol") are translated into the
            # real Genesys crop codes; unknown names are dropped and reported.
            crop_names = [str(item) for item in (as_list(crop) or [])]
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

        dataframe, geo = self._attach_cellid(dataframe, coordinate_hint)

        replaced = state.has_data
        state.set_original_list(dataframe, SourceMode.GENESYS)
        state.extras["genesys_filter"] = body
        state.extras["genesys_total_matching"] = total
        state.extras["coordinate_columns"] = geo["coordinate_columns"]

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
        self._log_cellid(state, geo)

        columns = [str(column) for column in dataframe.columns]

        return self.ok(
            f"{description}. Passport data columns follow the MCPD standard. {geo['message']} " + " ".join(notes),
            mode=SourceMode.GENESYS.value,
            accessions=loaded,
            total_matching=total,
            truncated=truncated,
            filter=body,
            notes=notes,
            unresolved_crops=unresolved_crops,
            columns=columns[:_MAX_COLUMNS_IN_RESULT],
            total_columns=len(columns),
            replaced_previous_lists=replaced,
            **{key: value for key, value in geo.items() if key != "message"},
        )
