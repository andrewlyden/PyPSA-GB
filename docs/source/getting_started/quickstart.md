# Quickstart

This page runs `Test_Market`, the scenario `config/config.yaml` selects out of the box. It is the smallest run
that goes through the whole model: six hours of 2035 on the 32-bus Reduced network, cleared first as a wholesale
market and then redispatched through the balancing mechanism. Once the inputs are downloaded it takes minutes.

You need the environment and a solver from {doc}`installation`.

## The scenario

`Test_Market` is defined in `config/scenarios.yaml`:

```yaml
Test_Market:
  modelled_year: 2035
  FES_year: 2024
  FES_scenario: Holistic Transition
  renewables_year: 2020
  demand_year: 2020
  network_model: Reduced
  renewables:
    aggregation:
      enabled: true
  network:
    component_aggregation:
      enabled: true
      include_storage_units: true
      include_loads: true
  solve_period:
    enabled: true
    start: 2035-01-01 00:00
    end: 2035-01-01 05:00
  description: Market test - Reduced network, 6 hours (fast smoke test)
  market:
    enabled: true
```

| Setting | Meaning |
|---|---|
| `modelled_year: 2035` with `FES_year`, `FES_scenario` | A future system: NESO's FES 2024 Holistic Transition pathway in 2035. (This scenario predates the `targets` and `method` keys; the loader derives `targets: {source: FES, edition: 2024, pathway: HT}` and `method: gsp_block` from these. New scenarios write `targets` and `method`, see {doc}`first_scenario`.) |
| `renewables_year: 2020` | Wind and solar output from 2020 weather. |
| `demand_year: 2020` | The shape of demand through the day and year from 2020; the annual total is FES's. |
| `network_model: Reduced` | The 32-bus network. |
| `renewables.aggregation`, `network.component_aggregation` | Merge generators, storage and loads of the same type at the same bus, which makes the problem smaller without changing its answer much. |
| `solve_period` | The network is built for all of 2035, but only 00:00 to 05:00 on 1 January is solved. |
| `market.enabled: true` | Solve as a two-stage market (wholesale, then balancing mechanism) instead of a single network-constrained solve. |

Everything else comes from the defaults in `config/defaults/`. To see the scenario fully merged:

```bash
python config/config_loader.py --scenario Test_Market
```

## Run it

Look first. A dry run lists every job and why it would run, without running anything:

```bash
snakemake -n
```

Then run it with up to four jobs at once:

```bash
snakemake -j 4
```

`config/config.yaml` selects the scenario:

```yaml
run_scenarios:
  - Test_Market
```

To run it whatever `config/config.yaml` says, add `--config scenario=Test_Market`. To build only Test_Market's own
outputs, name the file you want; Snakemake then runs exactly the chain that file needs:

```bash
snakemake -j 4 resources/analysis/Test_Market_market_summary.json
```

```{note}
If the dry run lists jobs for other scenarios or study-level checks (for example `validate_CP30_balancing`),
`config/config.yaml` loads a study under `studies`. A study's checks join the default target whatever
`run_scenarios` says. Ask for the file above instead, or remove the study from `studies`. See
{doc}`running_the_workflow`.
```

## What runs

On a fresh clone the run first fetches what it needs: the FES 2024 tables from the NESO data portal and the 2020
weather cutout from Zenodo (about 700 MB). Then, for Test_Market:

1. **Inputs.** Renewable sites and their hourly profiles from the 2020 cutout (wind, solar, hydro, marine);
   FES 2024 capacities and demand; fuel and carbon prices.
2. **Build.** The Reduced network (`build_reduced_base_network`), then demand, renewables, thermal plant and
   their marginal costs, storage, hydrogen and interconnectors, each step adding to the network
   (`resources/network/Test_Market_network_*.pkl`, `.nc`). `finalize_network` writes
   `resources/network/Test_Market.nc`.
3. **Stage 1: wholesale market** (`solve_wholesale_market`). The GB network is treated as a single node (a
   "copperplate") and the six hours are cleared at least cost, giving each unit's schedule and one GB wholesale
   price per hour.
4. **Stage 2: balancing mechanism** (`solve_balancing_mechanism`). The network limits are restored. Units start
   from their Stage 1 schedules and are moved up (paid their offer price) or down (paying their bid price) until
   the flows fit the network, at least total cost.
5. **Analysis** (`analyze_market_results`, `generate_market_analysis_notebook`). Accounting of volumes, costs and
   prices, a dashboard and a notebook.

The methods are described in {doc}`../model/dispatch/wholesale` and {doc}`../model/dispatch/balancing`.

## Where the outputs are

| File | Contents |
|---|---|
| `resources/network/Test_Market.nc` | The finalised network, before any solve. |
| `resources/market/Test_Market_wholesale.nc` | The solved Stage 1 network. |
| `resources/market/Test_Market_wholesale_price.csv` | The wholesale price per hour (GBP/MWh). |
| `resources/market/Test_Market_wholesale_dispatch.csv`, `_wholesale_storage.csv`, `_wholesale_links.csv` | Stage 1 schedules of generators, storage and links (MW). |
| `resources/market/Test_Market_balancing.nc` | The solved Stage 2 network: physical dispatch and flows. |
| `resources/market/Test_Market_redispatch_summary.csv` | Per unit: energy moved up and down (MWh) and its offer and bid cost (GBP). |
| `resources/market/Test_Market_constraint_costs.csv` | The same by technology. |
| `resources/market/Test_Market_congestion.csv` | Lines and transformers that reached their limit, with hours congested. |
| `resources/market/Test_Market_price_comparison.csv` | Per hour: the wholesale price beside the balancing stage's nodal prices and their spread. |
| `resources/analysis/Test_Market_market_summary.json` | Headline figures: demand, redispatch volume and cost, prices, interconnector flows. |
| `resources/analysis/Test_Market_market_dashboard.html` | An interactive dashboard; open it in a browser. |
| `resources/analysis/Test_Market_market_notebook.ipynb` | A notebook that loads and plots the results. |

Logs are in `logs/` (the market stages in `logs/market/`). How to read these files is in {doc}`results`.

```{note}
Six hours is a smoke test, not a study. Its costs and prices say little about 2035; use it to check that the
installation works, then write a scenario of your own.
```

## Next

- {doc}`results`: open the networks and read the summaries.
- {doc}`first_scenario`: write and run your own scenario.
