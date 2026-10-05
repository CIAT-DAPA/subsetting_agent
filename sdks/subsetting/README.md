# Subsetting SDK

Typed client for the CIAT **Subsetting API** (climate and agro-climatic
indicators). Request/response shapes come from the API source code
(`subsets_api/api.py`) and its Angular front-end.

## Configuration (`.env`)

| Variable | Meaning |
|----------|---------|
| `SUBSETTING_API_URL` | Full API URL including the version segment, e.g. `https://sandbox.genesys-pgr.org/api/subsetting/v1`. Endpoints are appended to it. |
| `SUBSETTING_API_TOKEN` | Tried first: `Authorization: API-Token <token>` |
| `SUBSETTING_ACCESS_TOKEN` | JWT of the same environment. On 401/403 the SDK tries `Authorization: Bearer <jwt>` and then the cookie `access_token=<jwt>` (the scheme the Genesys web client uses). The first accepted scheme is kept for the session. |
| `SUBSETTING_TIMEOUT` | Seconds (default 120; clustering is slow) |
| `SUBSETTING_DEFAULT_PERIOD` | Indicator period label (`mean` by default) |
| `SUBSETTING_GRID_*` | Base raster used to compute cell ids |

## Endpoints used

| Method & path | Purpose | Body |
|---------------|---------|------|
| `GET /indicators` | Catalogue grouped by category → flattened to `Indicator{id, name, pref, indicator_type, crop, category, unit}` | – |
| `GET /indicator-period` | Periods per indicator (`IndicatorPeriod{id, indicator, period, ssp}`); needed because the data endpoints take **period ids** | – |
| `POST /indicators-data` | Indicator values per cell | `{"cellid": [...], "indicators": [period ids]}` |
| `POST /cluster` | Multivariate clustering | `{"cellid_list": [{"crop", "cellids"}], "data": [{"name", "indicator": [period ids], "type", "crop"}], "analysis": {"algorithm": ["agglomerative"], "hyperparameter": {"n_clusters": max, "min_cluster": min}, "summary": false}, "months": [1, 12]}` |

The agglomerative algorithm picks the optimal number of clusters (silhouette)
between `min_clusters` and `max_clusters`; the business rule uses 2–10.

## Usage

```python
from core.config import get_settings
from sdks.subsetting import SubsettingClient, add_cellid_column

settings = get_settings()
frame = add_cellid_column(candidate_df, "DECLATITUDE", "DECLONGITUDE", settings.grid)  # adds "cellid"

with SubsettingClient.from_settings(settings) as client:
    drought = client.find_indicators("drought")                 # search by name/pref/category
    ids = [i.id for i in drought]
    data = client.get_indicators_data(cells, ids)                # long DataFrame: cellid, pref_indicator, month1..12, value, category
    result = client.generate_clusters(cells, ids, crop="bean")   # ClusterResult.assignments = {cellid: cluster}
```

## Cell id

`compute_cellid(lat, lon, grid)` reproduces `raster::cellFromXY` (the function
used by the data pipeline): ids are 1-based, numbered row by row from the
north-west corner; row 1 touches `ymax = ymin + nrows * cellsize`.
Coordinates outside the extent, missing or non-numeric return `None`.

The indicator database is indexed by the **global** 0.05° raster
(`raster_base_complete`: 7200 × 3600 cells, −180..180 / −90..90), which is the
default of `SUBSETTING_GRID_*`. It was confirmed against the sandbox with
`scripts/subsetting_cellid_probe.py` (the 7198 × 2000 / −50 raster shipped in
the source repository does not index the data). Example: (11.39, −72.22) →
`11320556`.

## Errors

`SubsettingNoDataError` (no indicator data for the cells: indicator values only
exist for cells that hold accessions; arbitrary coordinates return nothing),
`SubsettingAuthError` (every credential rejected), `SubsettingRequestError`
(other HTTP errors with `status_code`/`body`), `SubsettingConnectionError`
(network/timeouts), `SubsettingError` (invalid arguments, unknown indicator or
period, empty analysis). Transient errors (429/5xx/network) are retried.

## Smoke test

```bash
uv run python scripts/subsetting_smoke.py --list-indicators
# coordinates of real accessions (indicator data only exists where accessions are)
uv run python scripts/subsetting_smoke.py --data --indicators CDD t_rain --from-genesys --genus Phaseolus --species vulgaris --country COL --institute COL003 --limit 30
uv run python scripts/subsetting_smoke.py --cluster --indicators CDD t_rain --from-genesys --genus Phaseolus --species vulgaris --country COL --institute COL003 --limit 30
# explicit pairs (quote the string; negatives are fine)
uv run python scripts/subsetting_smoke.py --data --indicators CDD --coords "3.5,-76.35 -12.0,-77.0"
```
