# list_accessions

Loads the **first list of accessions** of a session and creates the two working
lists defined by the business rules:

* **Original list** – exact copy of the loaded data; never modified afterwards.
* **Candidate list** – copy of the data plus the empty columns `criteria_passport`,
  `criteria_traits`, `criteria_research`, `criteria_climate`, `cluster_traits`,
  `cluster_research`, `cluster_climate`. Every later skill narrows or annotates it.

## When to call it

| Situation | Arguments |
|-----------|-----------|
| The user attached an Excel/CSV file (its path appears in the *Attached files* block) | `source="local"`, `file_path=<attached path>`, optional `sheet_name`, `latitude_column`, `longitude_column` |
| The user gives no file and describes accessions (crop, genus/species, country, institute, biological status, accession numbers, keywords) | `source="genesys"` + the matching criteria (at least one) |
| The user attaches a new file or asks for a new Genesys search after a list is already loaded | Call it again: the previous lists are **replaced** and the activity log says so |

Do **not** call it for greetings, questions about capabilities or when the user
only wants to filter a list that is already loaded.

## Cell id (computed at load time, both modes)

Right after loading, the tool detects the latitude/longitude columns
(`DECLATITUDE`/`DECLONGITUDE`, `latitude`/`longitude`, `lat`/`lon`...) and adds a
`cellid` column to **both** the Original and the Candidate lists: the 1-based
cell of the base raster (7198 x 2000 cells of 0.05°, from `.env`) used by the
climate indicators. Accessions without valid coordinates or outside the raster
get an empty `cellid`. The result reports `cellid_computed`, `georeferenced`
and `without_cellid`; mention those numbers to the user.

If the coordinate columns are not detected (`cellid_computed=false`), ask the
user which columns hold latitude and longitude and call the tool again with
`latitude_column` / `longitude_column`. Climate tools need the `cellid`.

## Local mode

* Supported formats: `.xlsx`, `.xlsm`, `.xls`, `.csv`, `.tsv`, `.txt` (delimiter auto-detected).
* Every column of the file becomes passport data; nothing is dropped or renamed
  beyond trimming header spaces and naming empty headers `column_N`.
* Only files stored in the session `inputs` folder can be read.
* The result reports the number of accessions, the column names and the columns
  detected as latitude/longitude (useful later for the climate skill).

## Genesys mode

Criteria are translated into a Genesys `AccessionFilter`; only the criteria with
a value are sent. Passport data are the MCPD columns returned by the SDK
(`INSTCODE, ACCENUMB, GENUS, SPECIES, ORIGCTY, SAMPSTAT, DECLATITUDE,
DECLONGITUDE, ...`).

| Argument | Type | Maps to | Example |
|----------|------|---------|---------|
| `crop` | list[str] | `crop` – common names resolved against the Genesys crop catalogue (`GET /api/v2/crop`); unknown names are ignored and reported in `unresolved_crops` | `["beans"]`, `["frijol"]`, `["maize"]` |
| `genus` | list[str] | `taxonomy.genus` | `["Phaseolus"]` |
| `species` | list[str] | `taxonomy.species` | `["vulgaris"]` |
| `country_of_origin` | list[str] | `countryOfOrigin.code3` (ISO3) | `["COL", "PER"]` |
| `institute_code` | list[str] | `institute.code` (FAO WIEWS) | `["COL003"]` (CIAT), `["USA022"]` |
| `samp_stat` | list[int] | `sampStat` (MCPD) | `[300]` |
| `accession_numbers` | list[str] | `accessionNumbers` | `["G50001", "G50002"]` |
| `text` | str | `_text` full-text search | `"drought tolerant"` |
| `historic` | bool | `historic` | omit unless the user asks for historical records |
| `max_records` | int | download cap | defaults to `GENESYS_MAX_RECORDS` |

MCPD `SAMPSTAT` codes: 100 wild, 110 natural, 120 semi-natural/wild, 200 weedy,
300 traditional cultivar / landrace, 400 breeding/research material,
410 breeder's line, 500 advanced/improved cultivar, 600 GMO, 999 other.

Mapping hints for the model:

* A taxon in the request ("Phaseolus vulgaris") → `genus`/`species`, **no** `crop`.
* A crop without taxon ("frijoles", "beans", "maíz") → `crop=["frijoles"]` as said by the user
  (or `genus=["Phaseolus"]`, `["Zea"]`, `["Oryza"]`, `["Manihot"]` when the genus is obvious).
* When `crop` + taxonomy return 0 accessions, the tool retries with the taxonomy only and
  explains it in `notes`.
* Country names → ISO3 (`Colombia`→`COL`, `Perú`→`PER`, `México`→`MEX`, `Brasil`→`BRA`, `Guatemala`→`GTM`).
* "landraces / variedades tradicionales / criollas" → `samp_stat=[300]`; "silvestres / wild" → `samp_stat=[100]`;
  "cultivares mejorados / improved" → `samp_stat=[500]`.
* "del CIAT" → `institute_code=["COL003"]`; "del CIMMYT" → `["MEX002"]`; "del IRRI" → `["PHL001"]`.

The tool first counts the matching accessions, then downloads up to the cap.
When more accessions match than were downloaded, the result has
`truncated=true` and the message says how many match versus how many were
loaded: tell the user and offer to narrow the criteria.

## Result

```json
{
  "status": "ok",
  "message": "Loaded 5000 accessions from Genesys PGR (genesys mode) matching {...}; 5649 accessions match but only the first 5000 were loaded. Passport data columns follow the MCPD standard.",
  "mode": "genesys",
  "accessions": 5000,
  "total_matching": 5649,
  "truncated": true,
  "filter": {"taxonomy": {"genus": ["Phaseolus"], "species": ["vulgaris"]}, "countryOfOrigin": {"code3": ["COL"]}},
  "columns": ["INSTCODE", "ACCENUMB", "GENUS", "SPECIES", "ORIGCTY", "..."],
  "total_columns": 49,
  "coordinate_columns": {"latitude": "DECLATITUDE", "longitude": "DECLONGITUDE"},
  "cellid_computed": true,
  "georeferenced": 4812,
  "without_cellid": 188,
  "notes": [],
  "unresolved_crops": [],
  "replaced_previous_lists": false
}
```

Local mode returns the same shape with `"mode": "local"` and without
`total_matching`/`truncated`/`filter`.

On failure `status` is `"error"` and `message` explains what the user should do
(upload a file, add a criterion, relax the filters, retry later...). Relay that
guidance to the user in their language; never invent data.
