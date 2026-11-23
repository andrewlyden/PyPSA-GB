# Scenarios and configuration

## What it does

Every run of PyPSA-GB is driven by a scenario: a named set of settings that says which year to model, where its
numbers come from, which network to use and how to dispatch it. A scenario writes only what differs from the
defaults. The configuration loader (`config/config_loader.py`) merges it over layers of defaults chosen by what the
scenario declares, checks it, and hands each Snakemake rule the merged result.

```
config/
├── config.yaml           which studies to load and which scenarios to run
├── scenarios.yaml        the canonical scenarios: tutorials, examples, quick tests
├── studies/<study>/      larger studies, solved month by month
│   └── archive/          retired scenarios, kept loadable
├── defaults/
│   ├── base.yaml, base/  the complete, neutral defaults (file path = key path)
│   ├── method/           how a fleet is built: brownfield, gsp_block, historical
│   ├── source/           what a publication gives: CP30, FES, historical
│   └── dispatch/         central dispatch (central.yaml)
├── clustering.yaml       network clustering presets
├── cutouts_config.yaml   weather cutouts (Snakefile_cutouts)
└── config_loader.py      loads and merges all of it
```

## Method and assumptions

### The layers and their merge order

A scenario's configuration is built by deep-merging these layers, each over the one before (a mapping merges key
by key; any other value replaces what was there):

| Order | Layer | Chosen by |
|---|---|---|
| 1 | The base: `config/defaults/base.yaml` and every file under `config/defaults/base/` | always |
| 2 | Global overrides in `config/config.yaml` (any key other than `studies` and `run_scenarios`) | always |
| 3 | The method overlay, `config/defaults/method/<method>.yaml` | the scenario's `method` |
| 4 | The source overlay, `config/defaults/source/<targets.source>.yaml` | the scenario's `targets.source` |
| 5 | The dispatch overlay, `config/defaults/dispatch/<market.dispatch>.yaml` | the scenario's `market.dispatch` |
| 6 | The scenario: its `inherits` parents, then its own entry | the scenario id |
| 7 | The loader's derivations: `targets` and `method` themselves, year parameters, clustering presets | the loader |

The overlays attach by the axes a scenario declares, never by its name, so two scenarios with the same `targets`,
`method` and `market.dispatch` start from the same defaults.

**The base holds the neutral defaults.** Every setting that applies to every scenario is in the base, at the value a
scenario gets unless a later layer changes it. The method, source and dispatch overlays also introduce keys of their
own, which only their scenarios carry (the CP30 build and data keys, `hydrogen`, `source_data`, `validation`); those
are listed in the overlay files themselves. A file's path is its key path:
`config/defaults/base/market/balancing.yaml` holds the contents of
`market.balancing`, `base/market.yaml` holds `market`'s other keys, and `base.yaml` holds the top-level keys
(`timestep_minutes`, `solve_period`, `logging`). A key defined in two base files is an error. The code's own
defaults equal the base, so a base setting missing from a scenario never silently takes a different value in a
script.

**Overlays hold only what every scenario on that axis needs.** Today only Clean Power 2030 differs from the base;
the other overlay files exist and are empty on purpose:

| Axis | Value | File | Holds |
|---|---|---|---|
| `method` | `brownfield` | `method/brownfield.yaml` | build choices: named-fleet placement, the component demand build, electrolyser siting, regional profiles for proxy sites |
| `method` | `gsp_block`, `historical` | `method/gsp_block.yaml`, `method/historical.yaml` | nothing |
| `targets.source` | `CP30` | `source/CP30.yaml` | the publication's numbers and pinned data: source files and editions, prices, hydrogen demand, interconnector set, validation thresholds |
| `targets.source` | `FES`, `historical` | `source/FES.yaml`, `source/historical.yaml` | nothing |
| `market.dispatch` | `central` | `dispatch/central.yaml` | central dispatch at cost (below) |
| `market.dispatch` | `self` | (no file) | the market, as the other layers set it |

The method overlay holds no source data, so a brownfield build on another source would not inherit CP30's files.
The method and source overlays never set the same key. The dispatch overlay is a mode switch merged after them and
touches only `market`, `solve` and `support`.

### Keys are nested by domain

Apart from a few identity keys, every setting sits under its domain:

| Domain | Holds (examples) |
|---|---|
| `solve` | `solver` (name, threads, tolerances), `mode` (`LP` or `MILP`), `voll` (value of lost load, GBP/MWh) |
| `network` | `etys` (publication year), `upgrades`, `transmission`, `spatial_mapping`, `clustering`, `component_aggregation` |
| `demand` | `timeseries`, `disaggregation`, `flexibility` |
| `renewables` | `aggregation`, `performance_factors`, `embedded`, `hydro` |
| `thermal` | `availability_factors`, `outages`, `ramping`, `dispatch_anchors` |
| `storage` | `round_trip_efficiencies`, `dispatch_costs`, `standing_losses`, `placement` |
| `fleet` | `brownfield`, `station_registry` and, in CP30 scenarios, `spatial_proxy` |
| `interconnectors` | `availability`, `pricing`, `historical_flow_mode` |
| `marginal_costs` | fuel and carbon prices, correction factors, empirical calibration |
| `market` | `enabled`, `dispatch`, `wholesale`, `balancing` |
| `support` | subsidy labels (`labels`), dispatch costs, balancing-mechanism bids (`bm_bids`), revenue |
| `hydrogen` | electrolysis capacity and running hours, exogenous hydrogen demand, hydrogen turbine fuel and costs, electrolyser siting (the CP30 overlays; the FES long-range studies set `hydrogen.fes`) |
| `source_data` | the publication's workbook, version and derived source files (`workbook`, `files`; the CP30 overlay) |
| `validation` | the checks of the build and the solve against the publication (`stage1`, `balancing`, `fleet_location`, `locational`) and their thresholds (the CP30 overlay) |
| `consumer_bill`, `public_ownership` | the research layers after the solve |

The identity keys stay at the top level: `description`, `targets`, `method`, `modelled_year`, `renewables_year`,
`demand_year`, `network_model`, `solve_period`, `timestep_minutes`, `inherits`, `network_from`, `wholesale_from`.
Every key and its default is listed in {doc}`/reference/configuration_keys`.

**Renamed keys are refused, not aliased.** The table `RENAMED_KEYS` in `config/config_loader.py` maps each
pre-restructure key path to its new one (for example `solver` to `solve.solver`, `clustering` to
`network.clustering`, `etys_upgrades` to `network.upgrades`, `subsidy_tracking` to `support.labels`). A scenario,
template, overlay or `config.yaml` that writes an old path stops the loader with the new path named.

### Targets and method

Two keys on every merged scenario separate where the numbers come from from how the system is built:

```yaml
targets: {source: FES, edition: 2024, pathway: HT}   # where capacity, energy and demand targets come from
method: gsp_block                                    # how the fleet and network are built
```

- `targets.source` is `CP30`, `FES` or `historical`. `edition` is the publication year (the FES edition; 2024 for
  CP30, whose data workbook revision is a separate setting); `pathway` is a code (`HT`, `EE`, `HE`, `FFR`, `ND`,
  ...) or a pathway name, looked up in the edition's own list. Historical scenarios take neither.
- `method` is `brownfield` (named REPD and TEC projects placed on the ETYS network until the publication's
  capacity and regional targets are met), `gsp_block` (FES building blocks by Grid Supply Point) or `historical`
  (DUKES, REPD and ESPENI for the year itself).

The chain can build three pairs; any other is refused when the configuration is loaded rather than failing part
way through a build:

| `targets.source` | `method` | Years |
|---|---|---|
| `historical` | `historical` | `modelled_year` up to 2024 |
| `CP30` | `brownfield` | 2030 |
| `FES` | `gsp_block` | any future year |

A scenario that writes `targets` must also write `method`. A `CP30` or `FES` source for a year up to 2024 is
refused: a year up to 2024 is always historical. A new id still writes it out
(`targets: {source: historical}`, `method: historical`); only the legacy ids below may leave both keys out. The
canonical scenarios in `config/scenarios.yaml` write both explicitly, so any of them can be copied under a new id.

**Legacy scenario ids.** Scenarios that existed before these keys were introduced (the frozen list in
`data/scenario_sources/legacy_scenario_ids.txt`) may instead write the old keys, from which the loader derives both:

| Old keys | Derived |
|---|---|
| `modelled_year` up to 2024 | `historical` / `historical` |
| `scenario_source: CP30`, or a `CP30_pathway` key | `CP30`, edition 2024, the pathway's code / `brownfield` |
| `FES_year` and `FES_scenario` | `FES`, edition `FES_year`, the pathway's code / `gsp_block` |
| none of these, future year | error: write `targets` and `method` |

Any other id must write `targets` and `method`. The reverse also holds: a scenario that writes only `targets` gets
the old keys (`FES_year` and `FES_scenario`, or `scenario_source` and `CP30_pathway`) filled in, because some
scripts still read them; old keys written beside `targets` must agree with it.

Scripts ask the question they mean through `scripts/utilities/scenario_detection.py`: `target_source()` where
numbers or source files come from, `uses_brownfield()` and `uses_gsp_blocks()` where the build method matters.

### Where scenarios are defined

- **`config/scenarios.yaml`** holds the canonical scenarios: one entry each, written in full. It is always loaded.
- **Study folders**, `config/studies/<study>/`, hold the larger studies. A study is loaded only when
  `config/config.yaml` lists it under `studies` (or the environment variable `PYPSA_GB_STUDIES` names it, or is
  `all`); until then its scenario ids do not exist. A study folder has four
  files, read in this order:

| File | Holds |
|---|---|
| `study.yaml` | The study's own settings: `label`, `year`, `family_id` (the id pattern of its months), and `monthly_aggregates` (families whose twelve months make one calendar year). |
| `templates.yaml` | Named building blocks, composed with `inherits`. A template is never a scenario. |
| `families.yaml` | Month families (below). |
| `scenarios.yaml` | Single-period scenarios: the shared network build, one-off runs. |

- **The archive**, `config/studies/archive/<study>/`, keeps retired families and scenarios with their ids
  unchanged, so earlier results stay reproducible; `archive/canonical/` holds those retired from
  `config/scenarios.yaml`. It is not loaded by default; `load_config(include_archive=True)` loads it.

A name may be defined once across everything loaded; a template and a scenario may not share a name.

### `inherits`

A scenario or template may name one parent or a list:

```yaml
inherits: [cp30_ffr_2030_month, market_balancing, network_no_acceleration]
```

The parents are resolved first and merged left to right, then the entry's own settings on top. Two parents may
not set the same setting to different values unless the entry sets it itself: the result never depends on the
order of the list. Cycles are refused. A parent may be a template or a scenario.

### Families and months

A family in `families.yaml` is one entry that becomes one scenario per calendar month. Its id comes from the
family's own `id` or the study's `family_id`, with `{Mon}`, `{year}` and `{family}` filled in. In the `cp30_2030`
study (`family_id: CP30_FFR_2030_{Mon}_{family}`), the family

```yaml
no_acceleration:
  description: balancing mechanism, CENTRAL network (CP30's accelerations not delivered)
  inherits: [cp30_ffr_2030_month, market_balancing, network_no_acceleration]
  stage1: stage1
```

becomes `CP30_FFR_2030_Jan_no_acceleration` to `CP30_FFR_2030_Dec_no_acceleration`. Everything in the entry is a
setting every month gets, except these keys of the family itself:

| Key | Effect |
|---|---|
| `year` | The year (default: the study's `year`). |
| `months` | A subset of the months (default: all twelve). |
| `stage1: <family>` | Take the wholesale (Stage 1) result from that family's same month: sets each month's `wholesale_from`. |
| `network_from: <id>` | Solve on another scenario's network instead of building one. |
| `month_overrides: {Mar: {...}}` | One month's exceptions, merged over that month only. |
| `description` | Appended to each month's generated description. |

Each month gets its own `solve_period`, from the first hour of the month to the last, keeping any other
`solve_period` settings the family has.

Two sharing keys save repeated work, and the workflow checks both when it parses the configuration:

- **`network_from`**: the months solve on one full-year network built once (in `cp30_2030`, the scenario
  `CP30_FFR_2030`). A child may differ from its parent only in settings the build never reads: the solve period,
  `market`, `solve.solver`, the description and a few support and bill settings read only at solve time.
- **`wholesale_from`**: a balancing family takes its Stage 1 dispatch, prices and storage from another family's
  month instead of solving an identical wholesale market again. Both must solve the same period on the same
  network, and only the balancing settings may differ.

A family name in `run_scenarios` runs its twelve months; where two loaded studies share a family name, write
`<study>:<family>`.

### Central dispatch

`market.dispatch` is `self` (the market: Stage 1 clears, the balancing mechanism pays bids and offers) or
`central`, the system operator dispatching every unit at cost. `central` attaches `dispatch/central.yaml`, which
sets several things that only make sense together: Stage 1 on the real network (no copperplate), no scarcity
pricing, the balancing stage priced at cost, no support-based balancing bids, small positive marginal-cost floors
for renewables, and a solver profile for the harder problem.

Setting only some of these would give a wrong answer silently, so the loader refuses a central-dispatch scenario
whose own entry changes any of the keys the overlay fixes (`CENTRAL_DISPATCH_FIXED`):

- `market.wholesale.copperplate_mode`
- `market.wholesale.scarcity_pricing.enabled`
- `market.balancing.pricing_basis`
- `support.bm_bids.enabled`
- `support.bm_bids.contract_value_bids.enabled`

The renewable floors and the solver settings remain the scenario's choice. A scenario that wants an operator
paying premiums stays on `market.dispatch: self` and sets the keys itself. The method is described in
{doc}`dispatch/central_dispatch`.

### How the merged configuration reaches the rules

The `Snakefile` loads the configuration once. Build and solve rules receive the scenario's merged configuration
through `model_scenario_param`, report rules through `scenario_param`. Snakemake compares these parameters with
their keys sorted, so reordering changes nothing, and the build and solve rules ignore the blocks no build or solve
script reads (`NOT_READ_BY_MODEL`: `consumer_bill`, `public_ownership`, `validation`, `support.audit`,
`source_data.version`, `description`, `logging`). Any other change re-runs the rules that take the configuration.
See {doc}`/getting_started/running_the_workflow`.

## Data

| File | Used for |
|---|---|
| `data/scenario_sources/pathways.csv` | Every pathway of FES 2020 to 2025 and of CP30: name, code, role, successor and aliases. The loader resolves `targets.pathway` against it. |
| `data/scenario_sources/legacy_scenario_ids.txt` | The ids allowed to derive `targets` and `method` from the old keys. |
| `data/FES/fes_year_parameters.csv` | Calibrated values by year and pathway, each with its source. A setting written `year_parameter:<key>` takes the value for the scenario's `modelled_year` and pathway. |
| `config/clustering.yaml` | Clustering presets (`gsp_spatial`, `admin_regions`, `kmeans_10`, `kmeans_5`, `hierarchical_15`, ...) that `network.clustering` may name. |

## Configuration

Tools for seeing what a scenario gets:

```bash
python config/config_loader.py --list                       # every scenario id loaded
python config/config_loader.py --active                     # what run_scenarios selects (families expanded)
python config/config_loader.py --scenario <id> [--json]     # one scenario, fully merged
python config/config_loader.py --explain <id>               # its layers, and which file set each value
python config/config_loader.py --explain <id> --path market.wholesale   # only values under a dotted path
python config/config_loader.py --validate                   # check every scenario
python config/config_loader.py --dump-merged FILE           # every scenario's own entry and merged config, JSON lines
```

`--explain` lists the layers in merge order (each base file, `config.yaml`, the overlays, every template in the
`inherits` chain, the scenario's entry) and then each value with the last layer that set it; `loader` marks a value
the loader derived. `--dump-merged` writes one line per scenario with sorted keys, which makes it easy to check
with `diff` that a change to the defaults or templates moved only what you meant.

In Python:

```python
from config.config_loader import load_config, get_scenario, load_scenario_entries

config = load_config()                      # include_archive=True also loads config/studies/archive/
scenario = get_scenario("HT35", config)     # the merged scenario
print(scenario["targets"], scenario["method"], scenario["solve"]["solver"]["name"])

own = load_scenario_entries()["HT35"]       # what the scenario itself wrote, inherits resolved
```

Read the defaults in code only through `load_defaults()`, `load_layer(layer, name)` and `read_config(path)`, and a
scenario's own entry only through `load_scenario_entries()`; reading the YAML files directly skips the layering,
`inherits` and the checks. Comments in the configuration files say what a value is, its unit and its source.

## Validation

The loader refuses, with a message naming the scenario and the fix:

- an old key path (`RENAMED_KEYS`);
- a new future scenario without `targets` and `method`, `targets` without `method`, an unknown `targets` key,
  source, method or pathway, or an unsupported source and method pair;
- a `CP30` or `FES` source for a year up to 2024;
- old keys that disagree with `targets`;
- a central-dispatch scenario that changes a key the overlay fixes;
- two parents in `inherits` that disagree, a cycle, or a parent that does not exist;
- a name defined twice, a template and a scenario with the same name, or a family month that does not exist;
- study settings written in `config.yaml` instead of the study's `study.yaml`.

The `Snakefile` adds the `network_from` and `wholesale_from` checks above, and refuses an id in `run_scenarios`
that no loaded file defines. `--validate` checks each merged scenario's required fields and, for CP30 scenarios,
that the source files they name exist. The unit tests hold the layering rules (`tests/unit/test_config_layers.py`)
and keep every build and solve gate equal across the old and new keys (`tests/unit/test_targets_and_method.py`).

## Limitations

- Only three source and method pairs can be built. Brownfield placement (named REPD and TEC projects, each at its own
  connection) is built only for CP30, so a FES pathway cannot yet be built from named projects (`targets.source: FES`
  with `method: brownfield`). This concerns where plant is placed, not contracts: the FES chain tracks every CfD
  round, the RO and DPAs by build-year cohort ({doc}`dispatch/bids_offers_support`).
- A change to most of a scenario's configuration re-runs every rule that takes the whole configuration, even
  where that rule does not read the changed value.
