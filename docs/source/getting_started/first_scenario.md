# Your first scenario

A scenario is a named entry in `config/scenarios.yaml` (or in a study folder). It lists only what differs from the
defaults; everything else comes from `config/defaults/`. This page writes one, checks it and runs it. The full
rules are in {doc}`../model/scenarios_configuration`.

## 1. Start from a canonical scenario

`config/scenarios.yaml` holds the canonical scenarios: tutorials, examples and quick tests. A good starting point
is `HT35_clustered`, FES 2024 Holistic Transition in 2035 on the ETYS network clustered to GSP regions, solved for
one winter week. Copied under a new id, it reads:

```yaml
My_HT35_clustered:
  description: FES 2024 Holistic Transition 2035, ETYS clustered to GSP regions, one winter week
  targets: {source: FES, edition: 2024, pathway: HT}
  method: gsp_block
  modelled_year: 2035
  renewables_year: 2019          # weather year
  demand_year: 2020              # demand profile shape year
  network_model: ETYS
  network:
    clustering: gsp_spatial      # a preset from config/clustering.yaml
  solve_period: {enabled: true, start: '2035-01-15 00:00', end: '2035-01-21 23:00'}
```

Add it to the end of `config/scenarios.yaml`. The scenario id (`My_HT35_clustered`) becomes part of every output
file name, so use letters, digits and underscores. Every new scenario writes `targets` and `method`, as this one
does; the loader refuses a future scenario without them.

## 2. What each setting does

| Key | Meaning |
|---|---|
| `description` | Free text, shown by the loader and in reports. |
| `targets` | Where the capacity, energy and demand numbers come from: `source` (`FES`, `CP30` or `historical`), `edition` (the publication year: the FES edition, 2024 for CP30) and `pathway` (a code such as `HT`, `EE`, `HE`, or a pathway name). Historical scenarios write `targets: {source: historical}`. |
| `method` | How the fleet is built: `gsp_block` for FES, `brownfield` for CP30, `historical` for historical years. |
| `modelled_year` | The year modelled. Up to 2024 the scenario is historical; after 2024 it is future. |
| `renewables_year` | The weather year for wind, solar and hydro. It needs a cutout; the published ones cover 2010 to 2024. |
| `demand_year` | The year whose half-hourly demand shape is used. For a future year the annual total comes from FES; this sets only the shape. |
| `network_model` | `ETYS` (about 2,000 buses), `Reduced` (32 buses) or `Zonal` (17 zones). |
| `network.clustering` | Optional: a preset name from `config/clustering.yaml` (`gsp_spatial`, `admin_regions`, `kmeans_10`, `kmeans_5`, `hierarchical_15`) or an inline block such as `{method: kmeans, n_clusters: 100}`. |
| `solve_period` | The hours to solve. Without it the full modelled year is solved; with `enabled: true` only `start` to `end` (inclusive). The network is always built for the full year. |
| `timestep_minutes` | Optional: 60 (the default) or 30. |

The FES pathways by edition are listed in `data/scenario_sources/pathways.csv`: for 2024 and 2025 they are
Holistic Transition (`HT`), Electric Engagement (`EE`) and Hydrogen Evolution (`HE`), plus each edition's
counterfactual (Counterfactual `CF` in 2024, Falling Behind `FB` in 2025).

## 3. Check it

Show the scenario as the workflow will see it, merged over the defaults:

```bash
python config/config_loader.py --scenario My_HT35_clustered
```

and see which file set a value, for example the clustering settings:

```bash
python config/config_loader.py --explain My_HT35_clustered --path network.clustering
```

The loader stops with a clear message when something is wrong, for example:

- a future scenario with no `targets`: "a new scenario must write targets: {source, edition, pathway} and method";
- an old key path: "writes solver, which was renamed in the config restructure: write solve.solver";
- an unknown pathway, or a `targets.source` that does not fit the year (a 2020 scenario cannot take `FES` targets).

`python scripts/validate_scenarios.py` runs further checks on the scenarios selected in `config/config.yaml`,
including whether their data files and weather cutouts are present (`--no-files` skips the file checks).

## 4. Run it

Run it without editing `config/config.yaml`, dry run first:

```bash
snakemake -n --config scenario=My_HT35_clustered
snakemake -j 4 --config scenario=My_HT35_clustered
```

or add it to `run_scenarios` in `config/config.yaml` and run `snakemake -j 4`.

The first run downloads the 2019 cutout if it is missing and builds the FES 2024 inputs; later scenarios reuse
both. The solved network is `resources/network/My_HT35_clustered_solved.nc`; see {doc}`results`.

## 5. Variations

**The same system through the market.** `inherits` copies another entry and lets you change only what differs.
Setting `market.enabled` replaces the single solve with the wholesale market and the balancing mechanism:

```yaml
My_HT35_clustered_market:
  inherits: My_HT35_clustered
  description: as My_HT35_clustered, through the wholesale market and the balancing mechanism
  market:
    enabled: true
```

Both scenarios build the same network. To build it only once, give the child `network_from: My_HT35_clustered`;
the workflow refuses this unless the two differ only in settings the build never reads (the market, the solver,
the solve period and the description).

**A historical year.**

```yaml
My_Historical_2022:
  description: 2022 on the Reduced network, one week in January
  targets: {source: historical}
  method: historical
  modelled_year: 2022
  renewables_year: 2022
  demand_year: 2022
  network_model: Reduced
  solve_period: {enabled: true, start: '2022-01-10 00:00', end: '2022-01-16 23:00'}
```

**Another solver.** `solve: {solver: {name: highs}}` in the scenario (see {doc}`installation`).

**Another FES pathway or edition.** Change `targets`, for example `{source: FES, edition: 2025, pathway: EE}`.

## Next

- {doc}`running_the_workflow`: targets, re-runs, studies and logs.
- {doc}`../model/scenarios_configuration`: the configuration system in full.
- {doc}`/reference/configuration_keys`: every setting and its default.
