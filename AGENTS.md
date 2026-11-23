# PyPSA-GB: guide for AI coding assistants

This file orients an AI assistant (or a new contributor) working in this repository. The documentation
(`docs/source/`, published at https://pypsa-gb.readthedocs.io) is the full reference; this page says where things are
and which conventions to keep.

## At a glance

| | |
|---|---|
| Language | Python 3.11+ |
| Framework | PyPSA (networks and optimisation), linopy, Snakemake (workflow) |
| Solver | Gurobi by default (`solve.solver.name`), HiGHS as the open-source alternative |
| Environment | `envs/pypsa-gb-stable.yaml` (pinned; `conda activate pypsa-gb-stable`) |
| What to run | `config/config.yaml` (`run_scenarios`, `studies`) |
| Default run | `Test_Market`: six hours of 2035 on the Reduced network, wholesale market then balancing mechanism |

## Repository layout

```
Snakefile                 main workflow (includes rules/*.smk); Snakefile_cutouts downloads weather cutouts
rules/                    Snakemake rules by area: network_build, demand, renewables, generators, storage,
                          hydrogen, interconnectors, solve, market, CP30, FES, fes_long_range,
                          consumer_bill, public_ownership, neso_documents, ...
scripts/                  the Python each rule runs, grouped by area (network_build/, demand/, generators/,
                          renewables/, storage/, hydrogen/, interconnectors/, solve/, market/, CP30/, FES/,
                          sources/, consumer_bill/, public_ownership/, utilities/, analysis/)
config/                   configuration (see below)
data/                     input data, tracked; data/neso_documents.yaml lists NESO documents the workflow
                          downloads instead of shipping
resources/                everything the workflow builds (networks, results); not tracked
docs/source/              Sphinx + MyST documentation; figures drawn by scripts/analysis/docs_figures/
tests/                    pytest suite (tests/README.md)
```

## Configuration

- **Defaults** live in `config/defaults/`: `base.yaml` plus `base/**`, where the file path is the key path
  (`base/market/balancing.yaml` holds `market.balancing`). Overlays attach by a scenario's declared axes:
  `method/<method>.yaml`, then `source/<targets.source>.yaml`, then `dispatch/<market.dispatch>.yaml`.
  Merge order: base, `config.yaml` globals, method, source, dispatch, the scenario.
- **Scenarios**: `config/scenarios.yaml` (the canonical set) and study folders `config/studies/<study>/`
  (`study.yaml`, `templates.yaml`, `families.yaml`, `scenarios.yaml`), loaded only when listed under `studies` in
  `config.yaml`. A family expands to twelve monthly scenarios; `inherits: [a, b]` merges templates left to right.
- **Every merged scenario carries** `targets: {source, edition, pathway}` (source `CP30`, `FES` or `historical`) and
  `method` (`brownfield`, `gsp_block` or `historical`). A new scenario id must write both; only the frozen ids in
  `data/scenario_sources/legacy_scenario_ids.txt` may have them derived from the old keys. The canonical scenarios in
  `config/scenarios.yaml` write them explicitly. Supported pairs: CP30 + brownfield, FES + gsp_block,
  historical + historical.
- **Keys are nested by domain** (`solve.*`, `network.*`, `market.*`, `demand.*`, `support.*`, ...). An old key path
  is refused with the new one named (`config_loader.RENAMED_KEYS`).
- **Read configuration only through `config/config_loader.py`** (`load_config`, `load_defaults`, `load_layer`,
  `read_config`, `load_scenario_entries`); never `yaml.safe_load` a scenario or defaults file directly.
  `python config/config_loader.py --explain <scenario> --path <key>` shows which file sets a value.
- **Code defaults equal the base defaults.** When a script falls back to a default, it must be the value in
  `config/defaults/base/`.
- Comments in the configuration say what a value is, its unit and its source; rationale belongs in the docs.

## Running the workflow

```bash
snakemake -n                                  # dry run of config.yaml's run_scenarios
snakemake --cores 4                           # build and solve
snakemake --cores 4 --config scenario=HT35    # one scenario instead of run_scenarios
```

Before a run that builds or solves anything, do a dry run (`-n`) and tell the user what it will run: on the full ETYS
network a build or solve can take hours. When trying something out, use a one-week `solve_period` on the Reduced or
Zonal network. When reporting a result, say which file it came from.

Build and solve rules receive the scenario through `model_scenario_param(cfg)` and report rules through
`scenario_param(cfg)`. Editing an input file a rule reads makes Snakemake rebuild everything downstream of it, so
change shared data and scripts deliberately and check which outputs move.

## The model, in one paragraph per part

- **Network** (`scripts/network_build/`, `rules/network_build.smk`): ETYS full network from the ETYS appendices,
  Reduced (32 buses) or Zonal. Branch impedances are carried as pu on 100 MVA and converted to PyPSA's units once
  (`branch_units.py`); change a transformer rating only through `set_transformer_s_nom_preserving_impedance`.
  Bus coordinates are OSGB36 (EPSG:27700) metres; site data are WGS84, so convert before nearest-bus matching
  (`scripts/utilities/spatial_utils.py`).
- **Constraints** (`scripts/solve/`): seasonal line ratings, a voltage floor below which branches are relaxed,
  secured boundary limits with outage factors (`secured_boundaries.py`), network-delivery cases in the balancing
  stage.
- **Supply and demand** (`scripts/generators/`, `scripts/demand/`, `scripts/storage/`, `scripts/hydrogen/`,
  `scripts/interconnectors/`): historical years from DUKES, REPD and ESPENI; Clean Power 2030 (brownfield
  placement) and FES (GSP blocks) for future years.
- **Dispatch** (`scripts/solve/solve_network.py`, `scripts/market/`): a single network-constrained solve, or Stage 1
  wholesale (copperplate) followed by Stage 2 balancing anchored to Stage 1 positions, with bid and offer prices
  from `market_utils.py` and support schemes (`cfd_bm_bids.py`, `support_bids.py`); the `central` dispatch overlay
  (`config/defaults/dispatch/central.yaml`) is the alternative.
- **After the solve** (`scripts/consumer_bill/`, `scripts/public_ownership/`): household bill and public-ownership
  accounting layers on solved market years.

## Conventions

- Scripts run by a rule's `script:` directive must not use `from __future__` imports (a test enforces it).
- Scripts run as `python scripts/...` put the repository root on `sys.path` before importing `scripts.*`.
- Use the logging set up by `scripts/utilities/logging_config.py`; type hints and docstrings with Args/Returns.
- Ruff runs on errors only (`pyproject.toml`); a pre-commit hook refuses files over 10 MB
  (`.pre-commit-config.yaml`).
- Units: MW, MWh, GBP; costs in the price year the configuration states.

## Tests and docs

```bash
python -m pytest                                                     # what CI runs
python -m sphinx -b html docs/source docs/build/html                  # the docs (docs/requirements.txt)
```

Tests that need workflow outputs skip on a fresh clone (`tests/helpers`: `requires_outputs`). Docs figures are
redrawn by `scripts/analysis/docs_figures/<section>.py` and `scripts/analysis/build_landing_figures.py` from
`resources/`; the shared style is `scripts/analysis/docs_figures/common.py`.

## Debugging pointers

- All generators at one corner bus: WGS84 coordinates compared with OSGB36 metres.
- Infeasible or load shedding: compare `n.generators.p_nom.sum()` with peak demand; check for isolated buses and for
  branches with very small ratings.
- High curtailment: check line loading (`n.lines_t.p0.abs() / n.lines.s_nom`) and boundary limits for the hours
  concerned.
- Logs are under `logs/<area>/`.
