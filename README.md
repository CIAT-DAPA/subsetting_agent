# Subsetting Agent

An AI agent that helps genebank users (curators, breeders, researchers) build
**subsets of accessions** that match their needs by combining four sources of
information, always in this order:

1. **Passport data** — identification and collecting-site data of accessions,
   from the [Genesys PGR](https://www.genesys-pgr.org) REST API **or from a
   spreadsheet the user uploads** (Excel/CSV with accession ids and coordinates).
2. **Traits** — characterization and evaluation data (descriptors and
   observations), also from Genesys.
3. **Scientific documents** — PDFs uploaded by the user, converted to Markdown
   and searched by the agent.
4. **Climate indicators** — drought, flood, heat, photoperiod, soil and
   crop-specific indicators computed for the collecting site of each accession,
   from the Genesys **Subsetting API**, which also clusters accessions by their
   environmental profile.

The agent is served through a Gradio chat interface and follows the same
tool-calling loop as the AClimate "Melisa" agent (litellm + local LLM).

> **Status:** all blocks are implemented and unit-tested (214 tests, no
> network): SDKs, document processing, accession spreadsheets, tools, system
> prompt, agent and Gradio app. Next: validation against the real Genesys
> sandbox with an API token.

---

## Architecture

```
subsetting_agent/
├── app.py                      # Gradio ChatInterface, multimodal (PDF uploads)
├── subsetting_agent.py         # SubsettingAgent: LLM tool-calling loop (litellm)
├── prompts/
│   └── system_prompt.py        # System prompt enforcing the business order
├── genesys_sdk/                # Genesys PGR REST API (accessions, crops, traits)
│   ├── client.py               #   GenesysClient (httpx, async, token auth, paging)
│   ├── models.py               #   AccessionFilter builder, Accession, pages, descriptors
│   └── exceptions.py
├── subsetting_sdk/             # Subsetting API (climate indicators, clustering)
│   ├── client.py               #   SubsettingClient (8 endpoints, typed)
│   ├── models.py               #   IndicatorFilter, ClusterRequest/Result, ...
│   ├── catalog.py              #   IndicatorCatalog: names -> indicator period ids
│   ├── grid.py                 #   GridSpec: lat/lon -> cellid (raster_base grid)
│   └── exceptions.py
├── accession_files/            # User spreadsheets with accession ids + coordinates
│   └── reader.py               #   read_accession_file: column detection, validation, cellid
├── document_processing/        # PDF -> Markdown -> searchable sections
│   ├── pdf_converter.py        #   convert_pdf_to_markdown (cached on disk)
│   ├── document_store.py       #   DocumentStore: sections, outline, lexical search
│   └── models.py
├── tools/                      # Tools exposed to the LLM + session state
│   ├── accession_context.py    #   AccessionContext: selection, stage, steps, clusters (JSON)
│   ├── services.py             #   ToolServices: clients, document store, context, catalog
│   ├── genesys_tools.py        #   passport + trait tools (Genesys mode)
│   ├── file_tools.py           #   spreadsheet loading tools (file mode)
│   ├── document_tools.py       #   document tools
│   ├── subsetting_tools.py     #   climate tools + describe_selection
│   └── registry.py             #   ToolRegistry: OpenAI schemas, validation, dispatch
├── tests/                      # pytest suite (no network access required)
├── pyproject.toml              # uv project, Python 3.10
├── uv.lock
└── .env.example
```

### Two accession sources

The source is decided **at the start of every turn, deterministically**, from
the files attached in the conversation:

| Situation | Mode | Stage 1 | Traits | Documents | Climate |
|---|---|---|---|---|---|
| No spreadsheet attached | **Genesys mode** | passport filters on the Genesys API | yes | yes | yes |
| An Excel/CSV with accession ids + coordinates attached | **File mode** | load the spreadsheet; `cellid` computed locally with the Subsetting grid | no (the file has no trait data) | yes | yes |

In file mode no Genesys client is created and the Genesys tools are not even
registered, so the model cannot call them. The spreadsheet reader detects the
identifier, latitude and longitude columns (and an optional crop column) from
common header names; the user can name the columns explicitly when detection
fails. Rows with missing/invalid coordinates, outside the indicator grid, or
duplicated ids are reported and skipped.

### How the pieces fit together

```
 user request ──► passport filter ──► Genesys /acn/list ──► accessions (+ cellid)
        or    ──► spreadsheet (id, lat, lon) ──► grid.cellid() ──► accessions (+ cellid)
                                                                   │
                        traits? ──► Genesys /acn/{uuid}/observations ──► reduced selection
                                                                   │
                     documents? ──► DocumentStore.search(...) ──► evidence / reduced selection
                                                                   │
                     climate ──► Subsetting /subset | /cluster ──► cellids per cluster
                                                                   │
                                            cellid -> accessions map ──► final subset
```

The Subsetting API works with **grid cells (`cellid`)**, not with accession
identifiers. Each accession returned by Genesys carries the cell of its
collecting site; the agent keeps a `cellid -> accessions` map so cluster results
can be translated back into concrete accessions.

---

## Requirements

- Python **3.10** (pinned in `.python-version`; `requires-python = ">=3.10,<3.11"`)
- [uv](https://docs.astral.sh/uv/) for dependency management and running commands
- For the agent (next blocks): an LLM reachable through litellm, by default a
  local [Ollama](https://ollama.com) serving `llama3.1:8b`
- A Genesys API token (see configuration)

---

## Installation

```bash
git clone <this repository>
cd subsetting_agent
uv sync            # creates .venv with Python 3.10 and installs all dependencies
cp .env.example .env
```

Edit `.env` and fill in at least `GENESYS_API_TOKEN`.

Never use `pip` directly in this project; every command goes through `uv run`.

## Running the agent

```bash
ollama pull llama3.1:8b        # or point SUBSETTING_AGENT_MODEL/API_BASE to another litellm model
uv run python app.py           # http://localhost:7860
```

The chat accepts text, PDF attachments and one accession spreadsheet
(`.xlsx`, `.xls`, `.csv`, `.tsv`). Attaching a spreadsheet switches the whole
conversation to file mode (see *Two accession sources*). Every PDF attached during the
conversation stays available to the agent (converted once and cached). The
accession selection persists across turns inside the browser session; opening
a new tab starts a fresh conversation. Nothing is shared between users.

Example requests:

- "Find bean landraces from Colombia and Peru with coordinates."
- "Keep only the ones with drought tolerance score above 3."
- "Group them by total precipitation and maximum temperature between May and
  September, and keep the driest cluster."
- (attach a paper) "Which of the selected accessions does this paper report as
  heat tolerant?"
- (attach `my_accessions.xlsx`) "Cluster my accessions by drought indicators and
  keep the two driest groups."

---

## Configuration

All settings are read from environment variables (loaded from `.env` by the
app). Defaults are shown in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `GENESYS_API_URL` | `https://api.sandbox.genesys-pgr.org` | Root URL of the Genesys REST API |
| `GENESYS_API_TOKEN` | *(empty)* | Sent as `Authorization: API-Token <token>` |
| `GENESYS_API_TIMEOUT` | `60` | Per-request timeout in seconds |
| `GENESYS_CELLID_FIELD` | `geo.tileIndex` | Dotted path of the accession field used as Subsetting `cellid` (alternative: `tileIndex3min`) |
| `GENESYS_MAX_ACCESSIONS` | `2000` | Maximum accessions fetched for one passport filter |
| `SUBSETTING_GRID_NCOLS` / `_NROWS` / `_XMIN` / `_YMIN` / `_CELLSIZE` | `7198` / `2000` / `-180` / `-50` / `0.05` | Grid used to compute `cellid` from spreadsheet coordinates |
| `SUBSETTING_API_URL` | `https://sandbox.genesys-pgr.org/api/subsetting` | Root URL of the Subsetting API |
| `SUBSETTING_API_PREFIX` | `/api/v1` | Route prefix; set empty if the proxy strips it |
| `SUBSETTING_API_TIMEOUT` | `120` | Per-request timeout (clustering can be slow) |
| `DOCUMENT_CACHE_DIR` | `<system temp>/subsetting_agent_documents` | Where PDF conversions are cached |
| `SUBSETTING_AGENT_MODEL` | `ollama_chat/llama3.1:8b` | litellm model name |
| `SUBSETTING_AGENT_API_BASE` | `http://localhost:11434` | LLM endpoint |
| `SUBSETTING_AGENT_HOST` / `_PORT` | `localhost` / `7860` | Gradio server |
| `SUBSETTING_AGENT_LOG_LEVEL` | `INFO` | Logging level of the app |

---

## Using the SDKs

All clients are asynchronous and independent from the agent, so they can be
used from scripts, notebooks or tests.

### Genesys: passport data and traits

```python
import asyncio
from genesys_sdk import AccessionFilter, GenesysClient


async def main() -> None:
    async with GenesysClient() as genesys:
        # Domain-level builder; omitted criteria are not applied.
        passport = AccessionFilter.passport(
            crop_codes=["bean"],
            origin_countries=["COL", "PER"],
            sample_status=[300],  # MCPD SAMPSTAT: landraces
            with_coordinates=True,  # required for any climate analysis
        )

        overview = await genesys.accession_overview(passport)
        print(overview.accession_count, "matching accessions")

        accessions, total = await genesys.collect_accessions(passport, max_records=500)
        by_cell = {a.cellid(): a for a in accessions if a.cellid() is not None}

        observations = await genesys.get_observations(accessions[0].uuid)
        print(observations.all_records)


asyncio.run(main())
```

`Accession.cellid()` reads the field configured in `GENESYS_CELLID_FIELD`, so
switching to another field is an environment change only.

### Subsetting: climate indicators and clustering

```python
import asyncio
from subsetting_sdk import (
    ClusteringAlgorithm,
    ClusterRequest,
    CropCellIds,
    IndicatorCatalog,
    SubsettingClient,
)


async def main() -> None:
    async with SubsettingClient() as subsetting:
        catalog = await IndicatorCatalog.from_client(subsetting)
        print(catalog.category_names())  # e.g. Drought stress, Heat stress, ...

        # Resolve human names into validated filters with the right period ids.
        precipitation = catalog.build_filter(
            "total precipitation", months=(5, 9), value_range=(200, 800)
        )
        tmax = catalog.build_filter("maximum temperature", months=(5, 9))

        cells = [CropCellIds(crop="bean", cellids=[101, 102, 103, 201])]

        # Univariate filter: keep cells inside every range.
        subset = await subsetting.filter_subset(cells, [precipitation])
        print(subset.all_cellids())

        # Multivariate clustering of the remaining cells.
        result = await subsetting.cluster(
            ClusterRequest(
                cellid_list=cells,
                filters=[precipitation, tmax],
                algorithms=[ClusteringAlgorithm.AGGLOMERATIVE],
            )
        )
        print(result.clusters(crop="bean"))  # {0: [101, 102], 1: [103], ...}


asyncio.run(main())
```

Business errors are typed: `NoMatchingDataError` (no cell satisfies the
filters) and `CoreCollectionError` (requested core collection larger than the
available cells). Network and gateway errors are retried; 4xx never are.

### Documents: PDF to Markdown and search

```python
from document_processing import DocumentStore

store = DocumentStore()
added, errors = store.add_pdfs(["paper_drought_beans.pdf"])

for hit in store.search("drought tolerance landraces Mexico", top_k=3):
    print(hit.section.heading, hit.score, hit.snippet)

section = store.read_section(added[0].document_id, index=0)
store.cleanup()  # deletes the cache directory
```

Each PDF is converted once and cached as
`<DOCUMENT_CACHE_DIR>/YYYYMMDD_HHmmss_<hash>.md`, where `<hash>` is the first 12
characters of the SHA-256 of the PDF content. The same PDF uploaded again is
served from the cache. Documents are split into heading-based sections bounded
to ~1500 characters so that a small local model can consume search hits.

### Tools exposed to the LLM

The registry (`tools.build_registry(source)`) defines 16 tools, grouped by stage;
`source` is `"genesys"` or `"file"` and selects which stage-1 tools are exposed.
Every tool receives a `ToolServices` object (clients, document store and the
session `AccessionContext`) and returns a compact JSON-serializable result;
failures become `{"error": ...}` so the model can recover.

| Stage | Tool | Purpose |
|---|---|---|
| Passport | `search_crops` | Resolve crop names to Genesys crop codes |
| Passport | `preview_accessions` | Count matches and show a breakdown, without loading |
| Passport | `select_accessions` | Load georeferenced accessions and start the selection |
| Passport (file mode) | `list_accession_files`, `load_accessions_from_file` | Read the user's spreadsheet, compute cellids, start the selection |
| Traits | `search_trait_descriptors` | Find descriptors by keyword and crop |
| Traits | `filter_selection_by_trait` | Keep accessions whose observations satisfy a condition |
| Documents | `list_documents`, `search_documents`, `read_document_section` | Explore uploaded PDFs |
| Documents | `keep_accessions_from_documents` | Reduce to accession numbers a paper identifies |
| Climate | `list_climate_indicators` | Indicators by stress category |
| Climate | `filter_selection_by_climate` | Keep sites with an indicator inside a range |
| Climate | `cluster_selection_by_climate` | Cluster sites; then `pick_cluster` |
| Any | `describe_selection` | Stage, counts, applied steps and examples |

```python
from tools import AccessionContext, ToolServices, build_registry

registry = build_registry()
services = ToolServices(
    genesys=genesys_client,
    subsetting=subsetting_client,
    documents=store,
    context=AccessionContext.from_json(saved_state),
)

result = await registry.execute(services, "select_accessions", {"crop_codes": ["bean"]})
saved_state = services.context.to_json()  # travels with the chat history
openai_tools = registry.openai_tools()  # passed to litellm
```

The context only moves forward through the stages (passport → traits →
documents → climate) and climate tools operate solely on the cellids of the
accessions currently selected, which is how the business order is enforced by
data and not only by the prompt.

### The agent

`SubsettingAgent` keeps the tool-calling loop of the AClimate agent: the
conversation memory, a per-turn cache that serves repeated calls without
re-executing them, rescue of tool calls that small models emit as plain-text
JSON, stall detection and an iteration cap. Tools run locally through the
registry; there is no MCP session.

```python
from subsetting_agent import SubsettingAgent

agent = SubsettingAgent()  # model and endpoint from .env
agent.memory = previous_memory  # rebuilt from the chat history

turn = await agent.chat(
    "Find bean landraces from Colombia in dry areas",
    document_paths=uploaded_pdfs,  # every PDF attached so far
    context_json=previous_context_json,  # selection from the previous turn
)
turn.answer  # text for the user
turn.memory  # updated memory to store
turn.context_json  # updated selection to store
turn.document_errors  # PDFs that could not be converted
```

The system prompt (`prompts/system_prompt.py`) is written in English and asks
the model to answer in the user's language. It spells out the four stages in
order, when each one applies, how to recover from tool errors and how to close
a request (`describe_selection` before the final answer).

---

## Development

```bash
uv run pytest                 # full test suite (mocked HTTP, generated PDFs)
uv run ruff check .           # lint (pycodestyle, pyflakes, isort, bugbear, pyupgrade, pydocstyle)
uv run ruff format .          # formatter
```

Conventions used across the code base:

- **English only** for code, comments, docstrings and messages.
- **Every function, loop and conditional is documented**: functions describe
  purpose and parameters (Google style docstrings), loops explain what each
  iteration does, conditionals explain what they validate.
- Public request/response contracts are **pydantic models**; hand-built
  dictionaries are not sent to any API.
- Tests never hit the network (`pytest-httpx` mocks) and generate their own
  fixtures (small PDFs are produced with `pymupdf`).
- Dependencies are declared in `pyproject.toml` and locked in `uv.lock`; add
  them with `uv add <package>` (or `uv add --dev <package>`).

---

## Design notes

- **Passport → traits → documents → climate.** The order is a business rule.
  It is enforced by the agent's system prompt and by the data flow: climate
  tools only receive the cellids of the accessions that survived the previous
  stages.
- **File mode computes `cellid` locally.** `subsetting_sdk/grid.py` reproduces
  R `raster::cellFromXY` on the `raster_base.asc` grid (0.05°, 7198 × 2000,
  origin −180/−50, 1-based row-major from the north-west corner). If the
  deployed raster differs, override it with the `SUBSETTING_GRID_*` variables.
- **Traits are per accession.** Genesys exposes observations only through
  `/acn/{uuid}/observations`, not as a list filter. The trait stage therefore
  runs over a bounded selection (`GENESYS_MAX_ACCESSIONS`); the agent may need
  to ask the user to narrow the passport filter first.
- **Subsetting API quirks handled by the SDK.** Two GET endpoints return JSON
  with a `text/html` content type; `/subset` reports "no match" as HTTP 400;
  `/core-collection` reports "too few cells" as HTTP 422; `/cluster` answers
  `{}` when the analysis fails internally. Month windows may wrap the year
  (e.g. November to February). "Total precipitation" is summed over the window,
  every other monthly indicator is averaged.
- **Indicator ids are MongoDB ObjectIds** of *indicator periods*
  (indicator × period × SSP scenario). `IndicatorCatalog.build_filter` hides
  this from the agent and from the LLM.
- **Statelessness.** Following the AClimate pattern, a new agent is created per
  chat call; the conversation memory and the accession selection are rebuilt
  from the Gradio per-browser history.

---

## References

- Subsetting tool user guide (indicator descriptions):
  <https://www.genesys-pgr.org/content/help/subsetting-tool-user-guide>
- Subsetting API source: <https://github.com/CIAT-DAPA/subsets_genebank_accessions>
  (`src/subsets_api`, branch `dev`)
- Genesys API for MCP (Swagger):
  <https://api.sandbox.genesys-pgr.org/swagger-ui/index.html?urls.primaryName=Genesys%20API%20for%20MCP>
- Genesys PGR: <https://www.genesys-pgr.org>
