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
north-west corner; row 1 touches `ymax = ymin + nrows * cellsize` (50°).
Coordinates outside the extent, missing or non-numeric return `None`.

## Errors

`SubsettingAuthError` (every credential rejected), `SubsettingRequestError`
(other HTTP errors with `status_code`/`body`), `SubsettingConnectionError`
(network/timeouts), `SubsettingError` (invalid arguments, unknown indicator or
period, empty analysis). Transient errors (429/5xx/network) are retried.

## Smoke test

```bash
uv run python scripts/subsetting_smoke.py --list-indicators
uv run python scripts/subsetting_smoke.py --data --indicators cdd t_rain --coords 3.5,-76.35 4.7,-74.1
uv run python scripts/subsetting_smoke.py --cluster --indicators cdd t_rain --coords 3.5,-76.35 4.7,-74.1 -12.0,-77.0 19.4,-99.1
```
