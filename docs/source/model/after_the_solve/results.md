# Results and accounting

What the workflow writes once a network is solved: tables and summaries a script or a person can read, interactive
dashboards and notebooks, the accounts of a two-stage market, and, for a study solved month by month, the year
summed from its twelve months.

## What it does

| Run | Rule (script) | Main outputs |
|---|---|---|
| Single solve | `solve_network` (`scripts/solve/solve_network.py`) | `resources/network/{scenario}_solved.nc`, `_optimization_summary.txt`; `resources/results/{scenario}_generation.csv`, `_storage.csv`, `_flows.csv`, `_costs.csv`, `_emissions.csv` |
| Single solve | `analyze_and_visualize_solved_network` (`scripts/analysis/analyze_solved_network.py`) | `resources/analysis/{scenario}_summary.json`, `_dashboard.html`, `_spatial.html` |
| Single solve | `generate_analysis_notebook` (`scripts/analysis/generate_analysis_notebook.py`) | `resources/analysis/{scenario}_notebook.ipynb` |
| Two-stage market | `analyze_market_results` (`scripts/market/analyze_market.py`, `market_accounting.py`) | `resources/analysis/{scenario}_market_summary.json`, `_market_accounting.csv`, `_interconnector_flows.csv`, `_market_dashboard.html` |
| Two-stage market | `generate_market_analysis_notebook` | `resources/analysis/{scenario}_market_notebook.ipynb` |
| Twelve-month study | `aggregate_CP30_months` (`scripts/market/aggregate_neso_comparator.py`) | `resources/analysis/{study}_annual_summary.json`, `_monthly_metrics.csv` |

A scenario with `market.enabled: true` is solved as a two-stage market and has no single-solve outputs; a scenario
without it has no market outputs. The balancing stage's own tables (redispatch by unit, constraint costs, congestion,
prices) are listed on {doc}`/model/dispatch/balancing`.

## Method and assumptions

### Single solve

`solve_network` writes the solved network and, from it, five tables:

| File | Content |
|---|---|
| `_generation.csv` | Output of every generator per snapshot (MW), columns `<generator>_<carrier>` |
| `_storage.csv` | Per snapshot and storage unit: carrier, dispatch `p_dispatch` (MW) and state of charge (MWh) |
| `_flows.csv` | Active power on every AC line per snapshot (MW, positive from `bus0` to `bus1`); links are not included |
| `_costs.csv` | Per generator: energy (MWh), marginal cost (GBP/MWh) and their product (GBP); storage and links are not included |
| `_emissions.csv` | Per generator: emission intensity (kg/MWh) and emissions (kg), where generators carry one |

`_optimization_summary.txt` gives the solver, its status and time, the objective (total system cost, and per hour),
generation by carrier, and a warning wherever load shedding exceeds 1 MWh.

`analyze_and_visualize_solved_network` adds:

- **A summary** (`_summary.json`): network size; total cost (the objective); generation, storage net output and demand
  (MWh); load shed (MWh and per cent of demand); peak and mean demand; generation, storage output and installed
  capacity by carrier. Hydrogen turbines count as generation.
- **A dashboard** (`_dashboard.html`): hourly generation mix, peak output by carrier, the state of charge of the five
  largest storage units, load shedding over time, the distribution of line loading, and the cost split.
- **A map** (`_spatial.html`): buses by voltage, sized by connected capacity; lines, transformers and links
  (interconnectors, internal HVDC, electrolysers and hydrogen turbines told apart); generators by technology; storage.

`generate_analysis_notebook` writes an unexecuted notebook that loads the solved network and walks through the mix,
topology, storage, hydrogen, transmission, curtailment, demand flexibility and congestion hotspots.

### Two-stage market

`analyze_market_results` reads both stages and writes the market summary. Its sections:

| Section | Content |
|---|---|
| `accounting` | The cost measures of {doc}`/model/dispatch/balancing` (accepted offers and bids, ordinary redispatch, feasibility slack, non-participant turn-down, interconnector re-trading, the NESO-comparable cost), with their definitions; demand and consumer flexibility (MWh); gross redispatch (MWh and per cent of demand, GB only and including the foreign side); storage burn and the held churn sink; warnings |
| `redispatch` | Energy turned up and down (MWh), offer, bid and net cost, the number of units moved, and the same by carrier |
| `prices` | The Stage 1 price as a simple mean and weighted by demand (consumer flexibility's draw included); the mean balancing nodal dual at demand buses and the nodal spread, which are marginal values of the balancing LP, not imbalance prices |
| `interconnectors` | Imports, exports, net imports and losses (MWh); congestion rent; hours of simultaneous counter-flow; the re-trading volume and cost, with the counter-trade premium; the same by cable |
| `congestion` | How many branches reached their limit, and the five most congested |
| `revenue` | CfD payments and RO income, where support labels and revenue tracking are on (the default) |
| `wholesale`, `balancing`, `internal_dc_tie_break` | Each stage's objective, and the internal HVDC tie-break term, which is in no cost measure |

**Congestion rent** per cable and hour is the GB price times the energy delivered into GB, less the GB-side export,
less the trade cost in the Stage 1 LP.

The accounting warns where a result needs care: a run shorter than a year (its totals must not be annualised), any
feasibility slack above 0.1 MWh, storage burn above 1 MWh, or non-participant turn-down above 1 MWh.

`_market_accounting.csv` holds the same accounting measures, one per row, with units; the definitions are only in the
summary. `_interconnector_flows.csv` gives per-cable totals over the period.

### A year from twelve months

A study solved month by month declares each complete year it wants summed under `monthly_aggregates` in its
`study.yaml` (a family name, or a list of twelve scenario ids). `aggregate_CP30_months`, used by every such study
(CP30 2030 and the FES long-range years alike), requires twelve distinct months that together cover exactly 8,760 or
8,784 hours, and every accounting field it sums. Then:

- Costs and volumes are summed.
- The wholesale price is weighted by demand, hour by hour where every month carries hourly values, and otherwise from
  the monthly means weighted by monthly demand; the method is recorded beside the value (`_basis`).
- The nodal dual is weighted by demand; congestion counts take the largest month.
- Costs per MWh use the year's total demand. The NESO-comparable cost and the cost including re-trading are derived
  from the summed parts.
- A diagnostic that any month lacks is left empty rather than summed over fewer months.

`_monthly_metrics.csv` keeps one row per month. The year is built when a later step needs it (the turn-down
re-pricing, the CP30 checks, the bill), not by `snakemake all` on its own.

## Data

Every step here reads only the solved networks and the tables the solve wrote, plus the scenario's configuration.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `market.enabled` | `false` | Two-stage market instead of a single solve |
| `market.wholesale_only` | `false` | Stop after Stage 1: a wholesale notebook (merit order, dispatch, price against ELEXON's MID) instead of the market summary |
| `support.revenue.enabled` | `true` | CfD and RO revenue in the market summary |
| `monthly_aggregates` (study `study.yaml`) | none | The complete years a study sums |

## Validation

Every check of a solved run is on {doc}`validation`.

## Limitations

- **Single-solve sums assume an hourly step.** The single-solve tables and summary add MW over snapshots without
  their weightings, so they equal MWh only at a 60-minute step.
- **The single-solve energy balance is a report, not a check.** The summary's `energy_balance_error_pct` compares
  generation and storage with demand but leaves out interconnector flows, so net imports appear as an error.
- **The emissions table is usually empty.** It is filled only where generators carry an emission intensity; the
  standard builds keep emission factors on the carriers (`n.carriers.co2_emissions`, tonnes per MWh of fuel), so the
  file holds a placeholder line. Emissions follow from each generator's output, its carrier's factor and its
  efficiency.
- **The single-solve cost table covers generators only.** The objective in the summary includes storage and links.
