# SubsettingAgent

AI agent that helps genebank users build subsets of accessions (seed samples)
combining **passport data**, **traits**, **climate indicators** and **research
papers**. Accessions come from an uploaded Excel/CSV file (local mode) or from
[Genesys PGR](https://www.genesys-pgr.org/) (Genesys mode).

## Architecture

```
app.py                  Gradio chat interface (multimodal: text + files)
agent/
  subsetting_agent.py   LLM tool-calling loop (LiteLLM) that dispatches skills
  prompts.py            System prompt template
  skill_registry.py     Auto-discovery of skills under skills/
skills/
  base.py               Skill contract (name, description, parameters, run)
  <skill_name>/         One folder per skill: SKILL.md + skill.py
  list_accessions/      Load the first list (local Excel/CSV, or Genesys PGR with structured criteria)
  passport_filter/      Filter / describe / reset the Candidate list by passport columns
  climate_analysis/     list_indicators / filter / cluster by climate indicators (Subsetting API)
  traits_analysis/      detect / fetch (Genesys datasets) / group by trait data -> cluster_traits
  export_list/          Export Candidate or Original list as CSV (attached to the chat)
  arguments.py          Tolerant parsing of tool arguments (lists, ints, bools, JSON text)
core/
  config.py             Settings loaded from .env (pydantic-settings)
  session.py            tmp/<session>/inputs|outputs folders
  state.py              SessionState: Original list, Candidate list, mode, log
  formatter.py          Activity summary + 15-row preview + CSV exports
  logger.py             Logging setup
sdks/
  genesys/              Genesys PGR client (accession listing, crops, datasets/descriptors/observations) - see sdks/genesys/README.md
  subsetting/           Subsetting API client (indicators, indicator data, clusters, cellid) - see sdks/subsetting/README.md
scripts/
  genesys_smoke.py      Manual check of the Genesys SDK against the real API
  genesys_traits_smoke.py  Manual check of the Genesys trait/dataset endpoints (prints raw rows)
  subsetting_smoke.py   Manual check of the Subsetting SDK against the real API
tests/                  pytest suite
```

## Requirements

* Python 3.10
* [uv](https://docs.astral.sh/uv/)
* An LLM reachable through LiteLLM (Ollama with `llama3.1:8b` by default)

## Setup

```bash
uv sync                      # creates .venv and installs dependencies
cp .env.example .env         # then edit the values you need
```

## Run

```bash
uv run app.py
```

Open `http://localhost:7860` (or the host/port set in `.env`).

## Test

```bash
uv run --group dev pytest
```

## Business rules in short

* **Original list**: first list of accessions loaded in the session; never changes.
* **Candidate list**: working list returned to the user; every skill narrows or
  annotates it (not cumulative). It carries the columns `criteria_passport`,
  `criteria_traits`, `criteria_research`, `criteria_climate`, `cluster_traits`,
  `cluster_research` and `cluster_climate`.
* Every answer contains a summary of the activities performed and a preview of
  the first 15 rows of the Candidate list. Full CSV exports are generated on request.
