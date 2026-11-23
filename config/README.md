# PyPSA-GB configuration

```
config/
├── config.yaml           what to run: the studies to load and the scenarios to run (edit this)
├── scenarios.yaml        the canonical scenarios: tutorials, examples, quick tests (add yours here)
├── studies/              larger studies, solved month by month (README.md there)
│   ├── cp30_2030/        NESO Clean Power 2030, Further Flex and Renewables
│   ├── fes2050/          FES 2025 Holistic Transition 2050
│   ├── validation_2020/  the 2020 validation year
│   └── archive/          retired scenarios, kept loadable
├── defaults/             the defaults every scenario starts from (README.md there)
│   ├── base.yaml, base/  the neutral defaults (file path = key path)
│   ├── method/           how a fleet is built: brownfield, gsp_block, historical
│   ├── source/           what a publication gives: CP30, FES, historical
│   └── dispatch/         central dispatch (central.yaml)
├── clustering.yaml       network clustering presets
├── cutouts_config.yaml   weather cutouts (Snakefile_cutouts)
└── config_loader.py      loads and merges all of it
```

## Running

List scenario ids, or a study's family names, under `run_scenarios` in `config.yaml`, then:

```bash
snakemake -n            # dry run: what would be built
snakemake -j 4          # run
```

A family name (for example `no_acceleration` in `cp30_2030`) runs its twelve months; their shared network build and
Stage 1 family are built as inputs.

## How a scenario's settings are merged

Each layer is deep-merged over the one before (details in `defaults/README.md`):

1. the base (`defaults/base.yaml` and `defaults/base/**`);
2. global overrides in `config.yaml` (any key other than `studies` and `run_scenarios`; normally none);
3. the method overlay, `defaults/method/<method>.yaml`;
4. the source overlay, `defaults/source/<targets.source>.yaml`;
5. the dispatch overlay, `defaults/dispatch/<market.dispatch>.yaml` (`central` only);
6. the scenario's templates (`inherits`), then its own entry;
7. what the loader derives (old-style keys from `targets`, clustering presets, year parameters).

So a scenario lists only what differs from the defaults its axes select.

## Writing a scenario

Add it to `scenarios.yaml` (or to a study). Every new scenario states where its numbers come from (`targets`) and how
its fleet is built (`method`):

```yaml
My_Historical_2020:
  description: 2020 on the Reduced network, one week
  targets: {source: historical}
  method: historical
  modelled_year: 2020
  renewables_year: 2020          # weather year
  demand_year: 2020              # demand profile year
  network_model: Reduced         # Reduced (32 buses) | Zonal (17 zones) | ETYS (about 2,000 buses)
  solve_period: {enabled: true, start: '2020-01-01 00:00', end: '2020-01-07 23:00'}

My_Future_2035:
  description: FES 2024 Holistic Transition 2035 on ETYS clustered to GSP regions
  targets: {source: FES, edition: 2024, pathway: HT}
  method: gsp_block
  modelled_year: 2035
  renewables_year: 2020
  demand_year: 2020
  network_model: ETYS
  network:
    clustering: gsp_spatial      # a preset from clustering.yaml, or an inline block
```

- `targets.source`: `CP30`, `FES` or `historical`. `edition` is the publication year (the FES edition; 2024 for
  CP30), `pathway` the pathway code (`HT`, `EE`, `HE`, `FFR`, `ND`, ...); historical takes neither.
- `method`: `brownfield` (named REPD/TEC projects on the ETYS network; CP30), `gsp_block` (FES GSP building blocks)
  or `historical` (DUKES, REPD and ESPENI; `modelled_year` up to 2024). Supported pairs: CP30 + brownfield,
  FES + gsp_block, historical + historical.
- Only the scenario ids that existed before 27 October 2026 (`data/scenario_sources/legacy_scenario_ids.txt`) may
  leave `targets` and `method` out: they may write `scenario_source`/`CP30_pathway` or `FES_year`/`FES_scenario`, or
  only a historical `modelled_year`, from which the loader derives both. Any other id must write both, a historical
  year included (`targets: {source: historical}`, `method: historical`). The canonical scenarios in `scenarios.yaml`
  write them explicitly, so any of them can be copied under a new id.
- Settings are nested by domain: `solve.*`, `network.*`, `renewables.*`, `thermal.*`, `storage.*`, `demand.*`,
  `fleet.*`, `market.*`, `support.*`, `interconnectors.*`, `marginal_costs.*`, `consumer_bill.*`,
  `public_ownership.*`. An old key path is refused with its new path named.

FES pathways by edition:

| FES edition | Pathways (code) |
|-------------|-----------------|
| 2025 | Holistic Transition (`HT`), Electric Engagement (`EE`), Hydrogen Evolution (`HE`), Falling Behind (`FB`) |
| 2024 | Holistic Transition (`HT`), Electric Engagement (`EE`), Hydrogen Evolution (`HE`), Counterfactual (`CF`) |
| 2023, 2022 | Leading the Way (`LW`), Consumer Transformation (`CT`), System Transformation (`ST`), Falling Short (`FS`), Falling Short (No Shale) (`FSNS`) |
| 2021 | Leading the Way (`LW`), Consumer Transformation (`CT`), System Transformation (`ST`), Steady Progression (`SP`) |

FES 2020 is not supported: NESO published no GSP regional breakdown for it, so its capacity cannot be placed on the
network, and a scenario that names it is refused.

Each edition also has its forecast (Five Year Forecast `FYF`; Ten Year Forecast `TYF` in 2025). The list, with
aliases and successors, is `data/scenario_sources/pathways.csv`.

For a future year, `demand_year` sets the shape of the demand profile; the annual total is FES's consumer demand
(ES1/ED1), spread over GSPs by its building-block shares.

## Clustering presets (`clustering.yaml`)

| Preset | Method | |
|--------|--------|--|
| `gsp_spatial` | spatial | to GSP regions |
| `admin_regions` | spatial | to administrative regions |
| `kmeans_10`, `kmeans_5` | kmeans | geographical clusters |
| `hierarchical_15` | hierarchical | 15 clusters |

## Seeing what a scenario gets

```bash
python config/config_loader.py --list                       # every scenario
python config/config_loader.py --active                     # what run_scenarios selects
python config/config_loader.py --scenario HT35 [--json]     # one scenario, fully merged
python config/config_loader.py --explain HT35 [--path market.wholesale]   # which file set each value
python config/config_loader.py --validate                   # check every scenario
python config/config_loader.py --dump-merged FILE           # every merged config as JSON lines
```

In Python:

```python
from config.config_loader import load_config, get_scenario

config = load_config()                     # include_archive=True also loads config/studies/archive/
scenario = get_scenario("HT35")
print(scenario["targets"], scenario["solve"]["solver"]["name"])
```

Snakemake compares a rule's settings to decide what to rebuild: changing a value a build or solve reads re-runs it;
reordering keys or editing the report-only blocks (`config_loader.NOT_READ_BY_MODEL`) does not.
