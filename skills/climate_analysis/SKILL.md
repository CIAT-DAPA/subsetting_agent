# climate_analysis

Creates subsets of the **Candidate list** with **climate and soil indicators**
of the collecting sites (Subsetting API). Needs the `cellid` column computed
when the accessions were loaded (accessions without coordinates cannot be
analysed).

## Workflow (business rules)

1. **Identify the indicators.** Two ways, both accepted by `cluster` and `filter`:
   * the user names them (`CDD`, `t_rain`, `TX`, `PHIHOX`, `days_heat`...) → `indicators=[...]`;
   * the user describes a need ("sequía", "drought tolerance", "calor",
     "suelos ácidos", "inundación") → `query="..."`; the tool picks up to five
     generic indicators of the matching categories (Spanish and English
     vocabulary) and reports them. Prefixes written inside the query are honoured.
   `action="list_indicators"` is only for exploring the catalogue (it answers
   "which indicators exist?") and **creates nothing**: its result says so and
   points to `cluster`/`filter`.
2. **Cluster** (`action="cluster"`): "crea subconjuntos / agrupa" → groups the
   accessions into 2–10 clusters chosen by the API. No accession is removed;
   `cluster_climate` gets the cluster of each accession and `criteria_climate`
   the indicators and values used. Result has `subsets_created=true`.
3. **Filter** (`action="filter"`): thresholds → keeps the accessions whose
   indicator values satisfy the conditions. `criteria_climate` records the
   criteria and the values of each accession; `cluster_climate` is set to `0`.
   Result has `subsets_created=true`.

## Catalogue (sandbox, 88 indicators)

| Category | Prefixes | Type |
|----------|----------|------|
| Drought stress | `CDD` consecutive dry days, `ndws` water stress days, `t_rain` total precipitation | generic (monthly) |
| Flooding stress | `ndwl` waterlogging days, `p_95` extreme daily precipitation | generic |
| Heat stress | `TX` avg max temperature, `TN` avg min temperature, `VPD`, `NVPD4` | generic |
| Photoperiod | `ind_time` daylength, `m_srad` solar radiation | generic |
| Crop-specific | `days_heat`, `days_cold`, `days_optm` for Beans, Maize, Rice, Cassava, Potato, Wheat, Sorghum, Soybean, Banana, Barley, Cowpea, Pearl millet, Sweetpotato, Yam, Andean roots... | specific (monthly, per crop) |
| Soil | `PHIHOX` pH, `ORCDRC` organic carbon, `CECSOL` CEC, `BLDFIE` bulk density | extracted (single value) |
| Soil | `TEXMHT` texture class, `salinity_num_cat` salinity class | categorical |

Periods: `mean` (default), `min`, `max`, `1983-2016`, `1990-2016`, `1995-2016`,
`2000-2016`, `2005-2016`, `2010-2016`, `2021-2040`, `2041-2060` (future scenarios).

## Arguments

| Argument | Used by | Meaning |
|----------|---------|---------|
| `query` | cluster, filter, list_indicators | Need in words: `"drought"`, `"sequía"`, `"heat"`, `"soil"`, `"suelos"`, `"CDD t_rain"`. Selects the indicators for cluster/filter when `indicators` is empty; filters the catalogue in list_indicators |
| `indicators` | cluster, list_indicators | Prefixes/names/ids. `cluster` uses them (else `query`, else the remembered ones); `list_indicators` remembers them |
| `conditions` | filter | `[{"indicator", "operator", "value", "statistic"?, "months"?}]`. Operators: `equals`, `not_equals`, `gt`, `gte`, `lt`, `lte`, `between` (`[min,max]`), `in`, `not_in` |
| `statistic` | filter (per condition) | Aggregation of the monthly values: `mean`, `sum`, `min`, `max`. Default: `sum` for `t_rain`, `CDD`, `ndws`, `ndwl`, `days_*`, `NVPD4`; `mean` otherwise. Ignored for soil indicators |
| `months` | filter, cluster | Inclusive range `[first, last]` (default `[1, 12]`; wraps, e.g. `[11, 3]`) |
| `period` | filter, cluster | Period label (default `mean`) |
| `crop` | all | Crop for crop-specific indicators. Inferred from `CROPNAME` when possible; the tool asks for it when ambiguous |
| `logic` | filter | `and` (default) / `or` |
| `min_clusters`, `max_clusters` | cluster | Range explored by the API (2–10) |

## Behaviour details

* Accessions without `cellid`, or whose cell has no data for the indicators,
  are **excluded** by `filter` and left **unassigned** (`cluster_climate` empty)
  by `cluster`. `cluster` writes the reason in `criteria_climate`
  (`no coordinates: cellid not available...` / `no climate data in the Subsetting
  database for cell N (indicators ...)`) and reports `without_cellid`,
  `without_climate_data` and `cells_without_climate_data`. Tell the user: it is
  a coverage gap of the climate database (e.g. cassava accessions located at
  CIAT headquarters had no indicator values in the sandbox), not an error.
* Monthly values may have missing months; aggregations ignore them.
* `criteria_climate` texts: filter → `CDD sum(m1-12) > 150 AND t_rain sum(m1-12) < 800 [CDD=182.30, t_rain=655.10]`;
  cluster → `cluster by CDD, t_rain (period mean, months 1-12) [CDD=15.20, t_rain=98.40]`.
  Successive calls append with ` | `.
* When no accession passes a filter the list is unchanged and `value_ranges`
  shows the observed min/max to help adjust the thresholds.
* The API only holds data for cells that contain accessions of the loaded
  crops; a `SubsettingNoDataError` is turned into a formal message.

## Examples

* "Crea subconjuntos usando indicadores para sequía" → `cluster`, `query="sequía"`
  (tool picks `CDD`, `ndws`, `t_rain`).
* "Crea subconjuntos usando CDD, ndws, t_rain" → `cluster`, `indicators=["CDD","ndws","t_rain"]`.
* "Agrupa por tipo de suelo" → `cluster`, `query="suelo"` (picks `PHIHOX`, `ORCDRC`, `CECSOL`, `BLDFIE`, `TEXMHT`).
* "Quiero las accesiones de zonas con más de 20 días secos consecutivos en promedio" →
  `filter`, `conditions=[{"indicator":"CDD","operator":"gt","value":20,"statistic":"mean"}]`.
* "Agrúpalas por clima de sequía" → `list_indicators` with `query="drought"`,
  then `cluster` with `indicators=["CDD","ndws","t_rain"]`.
* "Solo suelos con pH entre 5.5 y 7" → `filter`,
  `conditions=[{"indicator":"PHIHOX","operator":"between","value":[5.5,7]}]`.
* "Días de calor para frijol mayores a 60 en el primer semestre" →
  `filter`, `conditions=[{"indicator":"days_heat","operator":"gt","value":60,"months":[1,6]}]`, `crop="Beans"`.
