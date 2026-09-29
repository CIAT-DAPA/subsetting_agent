# passport_filter

Works **always on the Candidate list**. Filters accessions by any passport data
column, describes columns, or resets the Candidate list to the Original list.

## Actions

| `action` | Purpose | Arguments |
|----------|---------|-----------|
| `filter` (default) | Keep only the accessions satisfying the conditions; record them in `criteria_passport` | `conditions`, optional `logic` |
| `describe` | Show columns with type, nulls, distinct count, min/max or most frequent values | optional `columns`, `top` |
| `reset` | Rebuild the Candidate list from the Original list (clears all filters and annotations) | – |

## Conditions

Each condition is an object `{"column": ..., "operator": ..., "value": ...}`.

| Operator | Value | Example |
|----------|-------|---------|
| `equals`, `not_equals` | scalar | `{"column": "SAMPSTAT", "operator": "equals", "value": 300}` |
| `in`, `not_in` | list | `{"column": "ORIGCTY", "operator": "in", "value": ["COL", "PER"]}` |
| `contains`, `not_contains`, `starts_with` | text | `{"column": "COLLSITE", "operator": "contains", "value": "Cauca"}` |
| `gt`, `gte`, `lt`, `lte` | number | `{"column": "ELEVATION", "operator": "gte", "value": 1500}` |
| `between` | `[min, max]` | `{"column": "DECLATITUDE", "operator": "between", "value": [-5, 12]}` |
| `is_null`, `not_null` | – | `{"column": "DECLATITUDE", "operator": "not_null"}` |

* Text comparisons ignore case and surrounding spaces; numeric operators convert the
  column to numbers (Excel data is loaded as text).
* `logic` combines the conditions of one call: `"and"` (default) or `"or"`.
* Successive calls **chain**: each call filters the current Candidate list. Use
  `reset` to start over.
* Column names are resolved flexibly: exact name, case/separator-insensitive
  match, MCPD synonyms (`country`→`ORIGCTY`, `latitude`→`DECLATITUDE`,
  `institute`→`INSTCODE`, `status`→`SAMPSTAT`, `genus`, `species`, `elevation`…)
  and unique substring. Unknown columns return an error with suggestions and the
  available columns.
* If no accession satisfies the conditions, the list is **not** changed and the
  result includes `column_hints` (frequent values) to help the user adjust.

## criteria_passport

Every surviving row gets the readable expression of the filter, e.g.
`ORIGCTY in [COL, PER] AND SAMPSTAT = 300`; later filters are appended with
` | `, so the column tells the full history of passport filters applied.

## Typical flow

1. User: "solo las de Colombia y Perú que sean landraces" →
   `filter` with `[{"column":"ORIGCTY","operator":"in","value":["COL","PER"]},
   {"column":"SAMPSTAT","operator":"equals","value":300}]`.
2. User: "¿qué países hay?" → `describe` with `columns=["ORIGCTY"]`.
3. User: "quita todos los filtros" → `reset`.

## Result (filter)

```json
{
  "status": "ok",
  "message": "Filtered Candidate list by passport data 'ORIGCTY in [COL, PER] AND SAMPSTAT = 300': 5649 -> 812 accessions. The criteria were recorded in the column 'criteria_passport'.",
  "criteria": "ORIGCTY in [COL, PER] AND SAMPSTAT = 300",
  "accessions_before": 5649,
  "accessions_after": 812,
  "removed": 4837,
  "conditions": ["ORIGCTY in [COL, PER]", "SAMPSTAT = 300"]
}
```
