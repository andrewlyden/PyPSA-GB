# Architecture

PyPSA-GB is a Snakemake workflow around PyPSA. Snakemake decides what to build from the requested outputs; each
rule runs one Python script; the scripts share helper modules under `scripts/`. Nothing is installed as a package:
everything runs from the repository root.

## Folder layout

```
PyPSA-GB/
├── Snakefile              main workflow (assumes weather cutouts exist)
├── Snakefile_cutouts      weather cutouts: Zenodo download or ERA5 build (run first, separately)
├── Snakefile_consumer_bill  rebuilds historical bill comparisons from saved model outputs
├── config/                what to run and the settings (see below)
├── rules/                 Snakemake rules, one file per stage (*.smk)
├── scripts/               the Python code, one folder per stage (module index: reference/api)
├── data/                  inputs: published datasets and registries (versioned)
├── resources/             everything the workflow builds (not versioned)
├── logs/                  rule logs (not versioned)
├── tests/                 pytest suite: smoke, unit and workflow tests
├── envs/                  conda environments (pypsa-gb.yaml)
└── docs/                  this documentation (Sphinx, MyST Markdown)
```

The `Snakefile` loads the configuration, works out which scenarios to run, and includes the rule files in stage
order: renewables, network build, clustering, demand, FES, NESO documents, CP30, FES long-range, generator data
preparation, generators, storage, hydrogen, interconnectors, solve, analysis, market, turn-down allocation, consumer
bill and public ownership.

## The rule chain

A scenario's network is built in steps, each adding one part of the system and writing the network under a longer
file name (see {doc}`/reference/outputs`):

```{mermaid}
flowchart TD
    subgraph r1["Network and demand"]
        direction LR
        A["build_ETYS_base_network<br/>build_reduced_base_network<br/>build_zonal_base_network"] --> B["build_base_demand"]
    end
    subgraph r2["Generation"]
        direction LR
        C["finalize_demand"] --> D["add_renewables_to_network"]
    end
    subgraph r3[" "]
        direction LR
        E["add_thermal_generators_to_network"] --> F["finalize_generator_integration"]
    end
    subgraph r4["Costs and storage"]
        direction LR
        G["apply_marginal_costs_to_network"] --> H["add_storage_to_network"]
    end
    subgraph r5["Hydrogen and interconnectors"]
        direction LR
        I["add_hydrogen_system"] --> J["add_interconnectors_to_network"]
    end
    r1 --> r2 --> r3 --> r4 --> r5
    r5 --> K["cluster_network (optional)"]
    r5 --> L["finalize_network"]
    K --> L
    L --> M["solve_network"]
    L --> N["solve_wholesale_market"]
    N --> O["solve_balancing_mechanism"]
    O --> P["analyze_market_results"]
```

- **Inputs to the chain** come from data rules: renewable profiles from weather cutouts
  (`generate_renewable_profiles`), FES tables (`FES_data`), CP30 tables, interconnector availability and foreign
  prices, and NESO documents (`retrieve_neso_document`).
- **Without the market**, `solve_network` solves the finished network once.
- **With the market** (`market.enabled: true`), Stage 1 (`solve_wholesale_market`) solves a copperplate wholesale
  dispatch and Stage 2 (`solve_balancing_mechanism`) redispatches it on the constrained network at bid and offer
  prices. See {doc}`/model/dispatch/index`.
- **Studies** solved month by month share one network build: each month's market rules read the build's finished
  network, and a family's months can share a Stage 1 (`wholesale_from`). Twelve months are summed by study-level
  rules (for example `aggregate_CP30_months`).
- **After the solve**, validation rules compare the results with published data, and the consumer-bill and
  public-ownership layers (`rules/consumer_bill.smk`, `rules/public_ownership.smk`) read completed market outputs.
  They never trigger a solve.

Historical years (up to 2024) and future years take different data at several steps (DUKES and REPD against FES or
CP30); the routing is decided by the scenario's `targets` and `method`, through
`scripts/utilities/scenario_detection.py`.

## The configuration system

`config/config.yaml` says which studies to load and which scenarios to run (`run_scenarios`). Each scenario's settings
are merged in layers: the complete defaults in `config/defaults/base.yaml` and `config/defaults/base/**` (file path =
key path), the overlays chosen by the scenario's declared axes (`method/`, `source/`, `dispatch/`), then the
scenario's own entry in `config/scenarios.yaml` or a study under `config/studies/`. `config/config_loader.py` does the
merge, refuses renamed keys and unknown values, and derives `targets` and `method` for older scenarios. See
{doc}`/model/scenarios_configuration`; every key is listed on {doc}`/reference/configuration_keys`.

Rules pass the merged scenario to their script as a parameter. Build and solve rules exclude the reporting-only blocks
from that parameter, so editing a report setting does not re-run a solve.

## Code conventions

- **Scripts run by rules** read inputs, outputs, parameters and the log path from the `snakemake` object. Many also
  run from the command line for checks and rebuilds.
- **Shared helpers** live beside the scripts that use them, or in `scripts/utilities/` when several stages share them
  (spatial mapping, logging, calendars, scenario detection).
- **Configuration** is read only through `config_loader` (`load_config`, `load_defaults`, `load_scenario_entries`),
  never by loading a YAML file directly. Code defaults equal the base defaults.
- **Coordinates**: network buses are in OSGB36 (British National Grid, metres); site data is usually WGS84
  (degrees). `scripts/utilities/spatial_utils.py` converts between them; never compare the two directly.
- **Branch impedances** are carried through the build in per unit on 100 MVA and converted to PyPSA's units once,
  at the end of the ETYS build (`scripts/network_build/branch_units.py`); the solve refuses a network without that
  conversion.
- **Logging** goes through `scripts/utilities/logging_config.py`; each rule writes its own log under `logs/`.
- **Fail loudly.** A missing input, an unknown configuration value or a failed check stops the rule with a message
  naming the cause; the workflow does not fall back silently to other data.

## Performance

| Network | Buses | Typical use |
|---|---|---|
| Zonal | 17 | quick tests, teaching |
| Reduced | 32 | fast runs, `Test_Market` |
| ETYS clustered | 100-300 | studies of a week to a year on a workstation |
| ETYS | about 2,000 | the CP30 and FES studies, solved month by month |

Full ETYS networks need 8 GB of memory or more. Solve time grows with buses and hours: use `solve_period` to solve a
short window, a rolling mode to solve in windows, or clustering to reduce the network.
