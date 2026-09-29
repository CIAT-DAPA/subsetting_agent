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
| The user attached an Excel/CSV file (its path appears in the *Attached files* block) | `source="local"`, `file_path=<attached path>`, optional `sheet_name` |
| The user gives no file and asks for accessions by name, crop, taxon, country... | `source="genesys"`, `query=<search text>` |
| The user attaches a new file after a list is already loaded | Call it again with the new file: the previous lists are **replaced** and the activity log says so |

Do **not** call it for greetings, questions about capabilities or when the user
only wants to filter a list that is already loaded.

## Local mode

* Supported formats: `.xlsx`, `.xlsm`, `.xls`, `.csv`, `.tsv`, `.txt` (delimiter auto-detected).
* Every column of the file becomes passport data; nothing is dropped or renamed
  beyond trimming header spaces and naming empty headers `column_N`.
* Only files stored in the session `inputs` folder can be read.
* The result reports the number of accessions, the column names and the columns
  detected as latitude/longitude (useful later for the climate skill).

## Genesys mode

Not implemented yet: the tool returns a formal error asking the user to upload a
file. The interface (`source="genesys"`, `query`) is final and will be backed by
the Genesys SDK.

## Result

```json
{
  "status": "ok",
  "message": "Loaded 120 accessions from file 'beans.xlsx' (local mode). All 14 columns are treated as passport data.",
  "mode": "local",
  "accessions": 120,
  "columns": ["ACCENUMB", "ORIGCTY", "DECLATITUDE", "DECLONGITUDE", "..."],
  "total_columns": 14,
  "coordinate_columns": {"latitude": "DECLATITUDE", "longitude": "DECLONGITUDE"},
  "replaced_previous_lists": false
}
```

On failure `status` is `"error"` and `message` explains what the user should do
(upload a file, choose an existing sheet, use a supported format...). Relay that
guidance to the user in their language; never invent data.
