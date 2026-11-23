# Results

Every run leaves solved PyPSA networks (`.nc`, netCDF) and a set of CSV and JSON summaries under `resources/`.
This page lists the ones a new user needs and shows how to open them. The full list of outputs and their columns
is in {doc}`/reference/outputs`.

## What a run writes

Which files you get depends on how the scenario is dispatched (see {doc}`running_the_workflow`).

### Single solve (`market.enabled: false`)

| File | Contents |
|---|---|
| `resources/network/<id>.nc` | The finalised network, before the solve. |
| `resources/network/<id>_solved.nc` | The solved network: dispatch, flows, storage and prices. |
| `resources/network/<id>_optimization_summary.txt` | Solver, status, objective and a short summary of the solution. |
| `resources/results/<id>_generation.csv` | Output of every generator per snapshot (MW); columns are `<generator>_<carrier>`. |
| `resources/results/<id>_storage.csv` | Per snapshot and storage unit: carrier, dispatch (`p_dispatch`, MW) and state of charge (MWh). |
| `resources/results/<id>_flows.csv` | Active power flow on every line per snapshot (MW, positive from `bus0` to `bus1`). |
| `resources/results/<id>_costs.csv` | Per generator: energy (MWh), marginal cost (GBP/MWh) and total cost (GBP). |
| `resources/results/<id>_emissions.csv` | Per generator: CO2 intensity (kg/MWh) and emissions (kg), where generators carry an intensity; the standard builds keep emission factors on the carriers, so this is usually a placeholder line. |
| `resources/analysis/<id>_summary.json` | Headline figures: cost, generation and demand, load shed, capacity and generation by carrier. |
| `resources/analysis/<id>_spatial.html`, `<id>_dashboard.html` | An interactive map and a results dashboard; open them in a browser. |
| `resources/analysis/<id>_notebook.ipynb` | A notebook that loads and plots the solved network. |

### Two-stage market (`market.enabled: true`)

| File | Contents |
|---|---|
| `resources/market/<id>_wholesale.nc` | Stage 1, the wholesale market solved on a copperplate. Its schedules and prices are meaningful; its line flows are not physical. |
| `resources/market/<id>_wholesale_price.csv` | The GB wholesale price per snapshot (GBP/MWh). |
| `resources/market/<id>_wholesale_dispatch.csv`, `_wholesale_storage.csv`, `_wholesale_links.csv` | Stage 1 schedules (MW). |
| `resources/market/<id>_balancing.nc` | Stage 2, the balancing mechanism on the real network: the physical dispatch and flows. |
| `resources/market/<id>_balancing_dispatch.csv`, `_balancing_storage.csv` | Stage 2 dispatch (MW). |
| `resources/market/<id>_redispatch_summary.csv` | Per unit: energy moved up and down from the Stage 1 schedule (MWh) and its offer and bid cost (GBP). |
| `resources/market/<id>_constraint_costs.csv` | The same by carrier. |
| `resources/market/<id>_congestion.csv` | Lines and transformers that reached their limit: rating (MVA), maximum and mean loading, hours congested. |
| `resources/market/<id>_price_comparison.csv` | Per snapshot: wholesale price, mean, minimum and maximum nodal price in Stage 2, and their spread. |
| `resources/analysis/<id>_market_summary.json` | Headline figures (see below). |
| `resources/analysis/<id>_market_accounting.csv` | The accounting metrics behind the summary, one row each, with units. |
| `resources/analysis/<id>_interconnector_flows.csv` | Per interconnector: imports, exports, losses, utilisation and congestion rent, by stage. |
| `resources/analysis/<id>_market_dashboard.html`, `<id>_market_notebook.ipynb` | Dashboard and notebook. |

A scenario that takes its Stage 1 from another (`wholesale_from`) has no wholesale files of its own: read its
parent's.

## Opening a network

A solved network opens with PyPSA. Static data (one row per component) is in `n.generators`, `n.lines`,
`n.storage_units` and so on; time series are in `n.generators_t.p`, `n.lines_t.p0`, `n.buses_t.marginal_price` and
so on.

```python
import pypsa

n = pypsa.Network("resources/network/HT35_clustered_solved.nc")

# Installed capacity by technology (GW)
print((n.generators.groupby("carrier").p_nom.sum() / 1e3).sort_values(ascending=False).round(1))

# Energy by technology over the solve period (TWh); snapshot weightings turn MW into MWh
weights = n.snapshot_weightings.generators
energy = n.generators_t.p.mul(weights, axis=0).sum()
print((energy.groupby(n.generators.carrier).sum() / 1e6).sort_values(ascending=False).round(2))

# Curtailment of wind and solar (TWh): available minus dispatched
vre = n.generators.index[n.generators.carrier.isin(["wind_onshore", "wind_offshore", "solar_pv"])]
p_max_pu = n.get_switchable_as_dense("Generator", "p_max_pu")[vre]   # time series or static value
available = p_max_pu.mul(n.generators.p_nom[vre]).mul(weights, axis=0).sum().sum()
print(f"curtailment {(available - energy[vre].sum()) / 1e6:.2f} TWh")

# The most heavily loaded lines (share of the static rating; ETYS lines may also carry seasonal
# ratings in n.lines_t.s_max_pu)
loading = n.lines_t.p0.abs().max() / (n.lines.s_nom * n.lines.s_max_pu)
print(loading.sort_values(ascending=False).head(10))
```

For a market run, open both stages and compare them:

```python
wholesale = pypsa.Network("resources/market/Test_Market_wholesale.nc")
balancing = pypsa.Network("resources/market/Test_Market_balancing.nc")

moved = balancing.generators_t.p - wholesale.generators_t.p     # MW, positive = turned up in the BM
print(moved.T.groupby(balancing.generators.carrier).sum().T.sum().round(0))
```

```{note}
In the balancing network, `buses_t.marginal_price` is a dual of the redispatch problem at each bus, not an
energy price or an imbalance price. Use `<id>_wholesale_price.csv` for the market price and
`<id>_price_comparison.csv` for the spread between buses.
```

## The market summary

`<id>_market_summary.json` collects the figures most studies start from:

| Section | Holds |
|---|---|
| `network_size`, `snapshots` | What was solved. |
| `accounting` | Demand (MWh); accepted offer and bid costs; the modelled redispatch cost (GBP and GBP/MWh of demand); unserved energy, reported apart as feasibility slack; gross redispatch volume; interconnector re-trading; `warnings` and `definitions` of every term. |
| `redispatch` | Total energy turned up and down (MWh), offer and bid cost, and the breakdown by carrier. |
| `congestion` | Number of congested lines and transformers, and the most congested. |
| `prices` | Mean and demand-weighted wholesale price, mean nodal price and spread. |
| `interconnectors` | Gross and net imports, losses and estimated congestion rent. |

Read `accounting.warnings` first. A short run is flagged as a diagnostic: its costs and volumes must not be scaled
up to a year, because the hours it covers are not a representative sample.

## Logs

If an output is missing or looks wrong, the job's log is in `logs/` (the market stages in `logs/market/`) and its
run time and memory in `benchmarks/`.

## Next

- {doc}`/model/index`: what the model does at each step.
- {doc}`/tutorials/index`: notebooks that work through runs and their results.
