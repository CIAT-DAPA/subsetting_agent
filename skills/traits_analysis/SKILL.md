# traits_analysis

Annotates and groups the Candidate list by **trait data** (characterization /
evaluation / phenotypic data). It never removes accessions.

## Actions

| action | What it does | Writes |
|--------|--------------|--------|
| `detect` | Lists the columns of an uploaded list that look like traits (anything that is not passport data), with type and range/categories. Creates nothing. | – |
| `fetch` | Downloads trait data from Genesys datasets for the accessions of the Candidate list (matched by `UUID`, or by `ACCENUMB` + `INSTCODE`). Descriptors can be chosen with `query` ("iron zinc", "seed color") or `traits=["Fe.Mean"]`; without a selection the first 10 descriptors are taken. Values are aggregated per accession: **mean** for numeric descriptors, **mode** for categorical ones. Creates nothing. | `trait_<columnName>` columns |
| `group` | Creates the subsets. **One trait with a condition** (`conditions=[{"trait": "Fe.Mean", "operator": "gte", "value": 60}]`) → `cluster_traits` 1 (meets) / 0 (does not meet). **Several traits** (`traits=[...]` and/or several conditions) → groups by combination: numeric traits in terciles `low/medium/high`, categorical traits by category, conditions as `met/not met`. Accessions without trait values get no group (`cluster_traits` empty). | `cluster_traits`, `criteria_traits` (criterion + values) |

Operators: `equals, not_equals, in, not_in, contains, not_contains, starts_with, gt, gte, lt, lte, between, is_null, not_null`.

## Genesys workflow used by `fetch`

1. `POST /api/v2/dataset/accessions-datasets` → datasets holding data for the accessions.
2. `GET /api/v2/dataset/{uuid}` and `/descriptors` → descriptors (title, columnName, dataType, uom).
3. `POST /api/v2/dataset/data?datasetUuids=&fields=` with `{"filters": {"accession": {"uuid": [...]}}}`
   in chunks of 200 accessions → rows `{"accession": "<uuid>", "<descriptor uuid>": [values]}`.
4. `observations_to_dataframe` + `aggregate_observations` → one value per accession.

## Result keys

`subsets_created` (only `group` sets it to `true`), `next_step`, `trait_columns`,
`coverage`, `accessions_with_data` / `accessions_without_data`, `groups`
(`cluster_traits`, `label`, `accessions`), `assigned`, `without_data`.
