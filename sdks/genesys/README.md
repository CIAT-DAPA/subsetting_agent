# Genesys SDK

Typed Python client for the [Genesys PGR](https://www.genesys-pgr.org) API v2.
Only the accession listing endpoint is covered for now:

```
POST /api/v2/acn/list?p=<page>&l=<size>&s=<sort>&d=<direction>
```

## Authentication

| Method | `.env` variables | Header sent |
|--------|------------------|-------------|
| Personal API token (preferred) | `GENESYS_API_TOKEN` | `Authorization: API-Token <token>` |
| OAuth2 client credentials | `GENESYS_CLIENT_ID`, `GENESYS_CLIENT_SECRET` | `Authorization: Bearer <jwt>` (fetched from `/oauth/token`, cached and refreshed) |

Other settings: `GENESYS_API_URL` (production `https://api.genesys-pgr.org`,
sandbox `https://api.sandbox.genesys-pgr.org`), `GENESYS_PAGE_SIZE` (max 1000),
`GENESYS_MAX_RECORDS` (safety cap per search), `GENESYS_TIMEOUT`.

## Usage

```python
from core.config import get_settings
from sdks.genesys import AccessionFilter, CountryFilter, GenesysClient, TaxonomyFilter

accession_filter = AccessionFilter(
    taxonomy=TaxonomyFilter(genus=["Phaseolus"], species=["vulgaris"]),
    country_of_origin=CountryFilter(code3=["COL", "PER"]),
    samp_stat=[300],          # landraces
    historic=False,
)

with GenesysClient.from_settings(get_settings()) as client:
    total = client.count_accessions(accession_filter)
    page = client.list_accessions(accession_filter, page=0, size=100)
    frame = client.list_accessions_dataframe(accession_filter, max_records=2000)
```

`AccessionFilter.to_body()` serialises **only the criteria with a value**:
`None`, blank strings, empty lists and empty nested filters are dropped, while
`False`/`0` are kept because they are real criteria. The example above sends:

```json
{
  "taxonomy": {"genus": ["Phaseolus"], "species": ["vulgaris"]},
  "countryOfOrigin": {"code3": ["COL", "PER"]},
  "sampStat": [300],
  "historic": false
}
```

Python attribute names use snake_case and are serialised with the exact JSON
names of the API (`samp_stat` → `sampStat`, `text` → `_text`, `not_` → `NOT`).
Filters can also be built from JSON names: `AccessionFilter.model_validate({"sampStat": [300]})`.

## Modelled filter properties

`_text`, `crop`, `cropName`, `institute{code, country{code3, region}, fullName, accessions}`,
`accessionNumber`, `accessionNumbers`, `doi`, `uuid`,
`taxonomy{genus, species, genusSpecies, subtaxa, taxonName, family, synonyms}`,
`sampStat`, `countryOfOrigin{code3, region}`, `geo{latitude, longitude, elevation, referenced}`,
`coll{collDate, collNumb, collMissId, collSite}`, `storage`, `historic`, `available`,
`mlsStatus`, `sgsv`, `inTrust`, `hasDoi`, `hasDataset`, `hasSubset`, `images`, `genotyped`,
`duplSite`, `donorCode`, `donorName`, `breederCode`, `curationType`, `acquisitionDate`,
`collection`, `lists`, `subsets`, `datasets`, `seqNo`, `pdci`, `lastModifiedDate`,
and the logical operators `NOT`, `AND`, `OR` (nested `AccessionFilter`).

## Flattened MCPD columns

`list_accessions_dataframe()` converts each nested `AccessionDTO` into one row
with MCPD-style columns (same convention as the official `genesysr` client):

`INSTCODE, INSTNAME, INSTCTY, ACCENUMB, DOI, UUID, GENUS, SPECIES, SPAUTHOR, SUBTAXA,
SUBTAUTHOR, TAXONNAME, CROPNAME, CROPCODE, SAMPSTAT, ACQDATE, ACCENAME, ORIGCTY,
ORIGCTYNAME, COLLSITE, DECLATITUDE, DECLONGITUDE, COORDUNCERT, COORDDATUM, GEOREFMETH,
ELEVATION, COLLDATE, COLLSRC, COLLNUMB, COLLCODE, COLLNAME, COLLMISSID, DONORCODE,
DONORNAME, DONORNUMB, BREDCODE, BREDNAME, ANCEST, DUPLSITE, STORAGE, MLSSTAT, AVAILABLE,
HISTORIC, INSVALBARD, INTRUST, ACCEURL, CURATION, PDCI, LASTMODIFIED`

Multi-valued properties (`STORAGE`, `DUPLSITE`, `COLLCODE`...) are joined with `;`.

## Crop catalogue

`client.list_crops()` calls `GET /api/v2/crop` (cached per client) and
`client.resolve_crop_codes(["frijol", "maize"])` returns
`(["<shortName>", ...], [unresolved names])`, matching case-insensitively
against `shortName`, `name` and `otherNames`. Use it instead of guessing crop
codes: `uv run python scripts/genesys_smoke.py --list-crops` prints the catalogue.

## Errors

`GenesysAuthError` (401/403 or token problems), `GenesysRequestError`
(other HTTP errors, with `status_code` and `body`), `GenesysConnectionError`
(network/timeouts). Transient errors (429, 5xx, network) are retried up to
`max_retries` times with exponential back-off.

## Smoke test against the real API

```bash
uv run python scripts/genesys_smoke.py --genus Phaseolus --species vulgaris --country COL --limit 20
```
