# FES long-range years: 2040 and 2050

## Question

NESO's Future Energy Scenarios (FES) set out how much generation, storage, demand and hydrogen GB could have in 2040 and
2050, but FES does not publish how that system would be dispatched on the transmission network or what balancing it
would need. This study takes the FES 2025 **Holistic Transition** (HT) pathway in 2040 and 2050 and asks:

- How does the FES 2025 HT system dispatch hour by hour, as a wholesale market followed by a balancing mechanism?
- How much renewable output is turned down for network reasons, and what do constraint and balancing actions cost?
- What wholesale prices and cross-border flows does it give?

## Setup

### The year-generic FES chain

Both years are built by the same chain (`rules/fes_long_range.smk`), which runs for any FES year:

| Part | How | Model page |
|---|---|---|
| Targets | FES 2025, pathway HT (`targets: {source: FES, edition: 2025, pathway: HT}`) | {doc}`/model/scenarios_configuration` |
| Fleet method | `gsp_block`: FES capacity by Grid Supply Point (GSP) and technology, each GSP row on its GSP's demand node; transmission-connected ("Direct") capacity split by licence area (`scripts/generators/fes_supply.py`) | {doc}`/model/generation/renewables`, {doc}`/model/generation/thermal` |
| Demand | FES demand by GSP and component, with consumer flexibility (`scripts/demand/fes_demand.py`) | {doc}`/model/demand/fes_long_range` |
| Hydrogen | FES electrolysis, hydrogen storage and turbines, with a seasonal hydrogen pass (`scripts/hydrogen/fes_hydrogen.py`) | {doc}`/model/flexibility/hydrogen` |
| Network | The ETYS network with upgrades, clustered to NESO's 17 CP.08 economic zones (`scripts/network_clustering/neso_zone_busmap.py`); zones are joined by the secured boundaries | {doc}`/model/network/clustering`, {doc}`/model/network/constraints` |
| Foreign prices | TYNDP 2026 draft National Trends+ prices for the model year (`scripts/interconnectors/tyndp_prices.py`) | {doc}`/model/interconnectors` |
| Weather and demand shape | 2013 | {doc}`/model/demand/fes_long_range` |
| Wind and solar profiles | Each FES block takes the average profile of the existing sites in its own region (offshore wind: its sea basin), with each technology's national energy unchanged | {doc}`/model/generation/renewables` |

Every generator, store and load is placed on the full ETYS network first and then moves to its bus's zone, so
placement is the same as on an unclustered build. The solved network has 29
buses<!-- resources/analysis/FES25_HT_2050_Jan_market_market_summary.json: network_size.buses; the same in every 2040 and 2050 month summary -->:
the 17 GB zones plus the foreign market buses and the hydrogen and gas buses.

### Dispatch

Each month is a two-stage market ({doc}`/model/dispatch/wholesale`, {doc}`/model/dispatch/balancing`), on the same
methods as the Clean Power 2030 study ({doc}`cp30_2030`):

- **Stage 1**, a rolling day-ahead copperplate dispatch.
- **Balancing mechanism**, rolling 24-hour windows on the zonal network with secured boundary limits and NESO's
  2022-25 generic boundary outages replayed. Gas offers at 1.6 times short-run marginal cost (SRMC); CfD renewables bid their contract value
  (floored at zero); interconnectors are re-traded at the foreign trade value plus GBP 20/MWh
  ({doc}`/model/dispatch/bids_offers_support`, {doc}`/model/dispatch/interconnector_trading`).
- **Turn-down** is as solved (cheapest first, `market.balancing.turn_down_allocation.mode: cheapest_first`): merchant
  renewables first, then CfD units by the support they lose. The pro rata basis is computed alongside for comparison.
- **Central dispatch** at resource cost (family `central_dispatch`) is solved beside the market year
  ({doc}`/model/dispatch/central_dispatch`).

### Families and ids

The settings are in `config/studies/fes2040/` and `config/studies/fes2050/`. Scenario ids are
`FES25_HT_<year>_<Mon>_<family>`, for example `FES25_HT_2050_Jan_market`. The families are:

| Family | What it is |
|---|---|
| `market` | The market year: Stage 1 and the balancing mechanism |
| `central_dispatch` | Central dispatch at resource cost (solved; not reported on this page) |
| `tyndp2026_prices` | Sensitivity with its own hydrogen trajectory on the TYNDP 2026 draft prices; configured for both years, not reported here |

The twelve months of the `market` family are summed into the annual aggregate `FES25_HT_<year>`.

`config/studies/fes2045/` prepares 2045 on the same chain (ids `FES25_HT_2045_<Mon>_<family>`); it has not been
solved, so there are no 2045 results.

## How to reproduce

In `config/config.yaml`, load the study and name the families with their study prefix (a family name used in two
studies is ambiguous otherwise):

```yaml
studies:
  - fes2040
  - fes2050
run_scenarios:
  - fes2040:market
  - fes2050:market
  - fes2050:central_dispatch
```

Then `snakemake -n` to check the plan and `snakemake -j 4` to run it ({doc}`/getting_started/running_the_workflow`).
You need the 2013 weather cutout and a Gurobi licence; FES tables are fetched from NESO's data portal and cached under
`resources/sources/raw/`. Each year is a full-year build and twelve monthly two-stage solves on the zonal network;
expect a day or more of compute per family.

Outputs:

| Output | File |
|---|---|
| Annual aggregate | `resources/analysis/FES25_HT_<year>_annual_summary.json` (and `_monthly_metrics.csv`) |
| Turn-down on both bases | `resources/analysis/FES25_HT_<year>_pro_rata_turn_down_summary.json` |
| Constraint cost and network curtailment | `resources/analysis/FES25_HT_<year>_future_balancing_costs_summary.json` |
| Monthly market summary | `resources/analysis/FES25_HT_<year>_<Mon>_market_market_summary.json` |
| Dispatch against FES 2025 ES1 | `resources/analysis/FES25_HT_<year>_fes_energy_validation.json` (rule `validate_fes_dispatch`, run on request through target `fes_long_range_validation`) |

## Results

Results are for the `market` family of each year. Costs are in 2024 prices.

```{figure} /_static/figures/studies/fes_capacity_2040_2050.png
:alt: Grouped bar chart of installed capacity by technology in 2040 and 2050; offshore wind, solar and hydrogen turbines grow most, and gas falls.
:width: 100%

Installed capacity in the two built networks, by technology. Gas includes gas with carbon capture; other renewables
are bioenergy (with and without capture), energy from waste, hydro and marine; hydrogen turbines
are the hydrogen-to-power links; storage is batteries, pumped hydro, liquid air and compressed air, without consumer
flexibility; interconnectors are the GB cables' ratings. Final results, FES 2025 Holistic Transition builds.
```

### Annual summary

| | 2040 | 2050 |
|---|---|---|
| Modelled demand, TWh (incl. hydrogen offtake and consumer flexibility) | 576<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.demand_MWh --> | 725<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.demand_MWh --> |
| GB electricity demand, TWh | 569<!-- sum over months of accounting.gb_electricity_demand_MWh in resources/analysis/FES25_HT_2040_<Mon>_market_market_summary.json --> | 721<!-- sum over months of accounting.gb_electricity_demand_MWh in resources/analysis/FES25_HT_2050_<Mon>_market_market_summary.json --> |
| Stage 1 wholesale price, demand-weighted, GBP/MWh | 37.21<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.mean_wholesale_price_GBP_per_MWh_demand_weighted --> | 42.69<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.mean_wholesale_price_GBP_per_MWh_demand_weighted --> |
| Balancing nodal dual at demand buses, demand-weighted, GBP/MWh | 46.19<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.mean_nodal_dual_GBP_per_MWh_demand_weighted --> | 52.60<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.mean_nodal_dual_GBP_per_MWh_demand_weighted --> |
| Interconnector imports, Stage 1, TWh | 43.6<!-- sum over months of interconnectors.gross_import_MWh in resources/analysis/FES25_HT_2040_<Mon>_market_market_summary.json --> | 58.1<!-- sum over months of interconnectors.gross_import_MWh in resources/analysis/FES25_HT_2050_<Mon>_market_market_summary.json --> |
| Interconnector exports, Stage 1, TWh | 82.5<!-- sum over months of interconnectors.gross_export_MWh in resources/analysis/FES25_HT_2040_<Mon>_market_market_summary.json --> | 84.6<!-- sum over months of interconnectors.gross_export_MWh in resources/analysis/FES25_HT_2050_<Mon>_market_market_summary.json --> |
| Unserved energy (feasibility slack), MWh | 1.4<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.feasibility_slack_energy_MWh --> | 1.3<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.feasibility_slack_energy_MWh --> |

GB is a net exporter in both years: 38.9 TWh in 2040 and 26.5 TWh in 2050 in the Stage 1 market<!-- exports minus
imports from the two rows above -->, and 29.9 and 16.2 TWh after balancing, when interconnectors are
re-traded<!-- exports minus imports in the dispatch table below -->. The balancing nodal dual is a marginal value of the balancing LP, not an imbalance price.

### Constraint and balancing cost

| GBP m | 2040 | 2050 |
|---|---|---|
| Accepted offers | 770<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.accepted_offer_cost_GBP --> | 1083<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.accepted_offer_cost_GBP --> |
| Accepted bids | 699<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.accepted_bid_cost_GBP --> | 853<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.accepted_bid_cost_GBP --> |
| Interconnector re-trading | 858<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.interconnector_retrading_cost_GBP --> | 954<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.interconnector_retrading_cost_GBP --> |
| **NESO-comparable redispatch cost** (as solved) | **2,324**<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.neso_comparable_redispatch_cost_GBP --> | **2,888**<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.neso_comparable_redispatch_cost_GBP --> |
| Same, pro rata zonal turn-down | 2,393<!-- resources/analysis/FES25_HT_2040_pro_rata_turn_down_summary.json: neso_comparable_redispatch_cost_by_basis_GBP.pro_rata_zonal --> | 2,926<!-- resources/analysis/FES25_HT_2050_pro_rata_turn_down_summary.json: neso_comparable_redispatch_cost_by_basis_GBP.pro_rata_zonal --> |
| Gross redispatch, % of demand | 7.0%<!-- resources/analysis/FES25_HT_2040_annual_summary.json: aggregate.gross_redispatch_pct_demand --> | 7.5%<!-- resources/analysis/FES25_HT_2050_annual_summary.json: aggregate.gross_redispatch_pct_demand --> |

The NESO-comparable measure is defined as in the CP30 study: accepted GB offers and bids plus interconnector
re-trading, without unserved energy and forced non-participant turn-down. Both long-range years cost less to balance
than CP30 2030 on its recommended network (GBP 3,375 m, {doc}`cp30_2030`). The comparison is not like for like: the
long-range years are solved on the zonal network, which sees no constraints inside a zone (see Limitations), while
2030 is solved on the full ETYS network. The two turn-down bases differ by
under GBP 70 m in either year, much less than in 2030.

### Network curtailment

Turn-down of CfD-contracted units in the balancing stage (as solved), by carrier, and of all wind:

| TWh | 2040 | 2050 |
|---|---|---|
| Offshore wind | 4.8<!-- resources/analysis/FES25_HT_2040_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.wind_offshore --> | 7.9<!-- resources/analysis/FES25_HT_2050_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.wind_offshore --> |
| Onshore wind | 4.9<!-- resources/analysis/FES25_HT_2040_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.wind_onshore --> | 3.7<!-- resources/analysis/FES25_HT_2050_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.wind_onshore --> |
| Solar PV | 0.8<!-- resources/analysis/FES25_HT_2040_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.solar_pv --> | 1.2<!-- resources/analysis/FES25_HT_2050_future_balancing_costs_summary.json: annual.network_curtailment_twh_by_carrier.solar_pv --> |
| All wind, whatever its support | 12.6<!-- resources/analysis/FES25_HT_2040_pro_rata_turn_down_summary.json: turn_down_MWh --> | 16.5<!-- resources/analysis/FES25_HT_2050_pro_rata_turn_down_summary.json: turn_down_MWh --> |

The carrier rows are the CfD-contracted units' turn-down (the volume behind the bill's CfD compensation line); the last
row counts every wind unit's turn-down below its Stage 1 output, whatever its support, so it is larger. Economic curtailment in Stage 1
(surplus renewable output with nowhere to go) is not in these figures.

```{figure} /_static/figures/studies/fes_cost_and_curtailment.png
:alt: Two bar charts for 2040 and 2050; left, constraint and balancing cost stacked into accepted offers, accepted bids and interconnector re-trading, rising from 2,324 to 2,888 GBP m; right, turn-down of CfD-contracted offshore wind, onshore wind and solar, with offshore wind rising from 4.8 to 7.9 TWh.
:width: 100%

Left: the NESO-comparable redispatch cost (as solved) and its parts. Right: turn-down of CfD-contracted units
in the balancing stage, by carrier. Final results, FES 2025 Holistic Transition, twelve monthly solves per year.
```

```{figure} /_static/figures/studies/fes_wind_turn_down_map.png
:alt: Two maps of Great Britain's 17 zones in 2040 and 2050 with circles sized by wind turned down; the largest are offshore wind in East Anglia, growing from 2040 to 2050, and onshore wind in northern Scotland.
:width: 100%
:align: center

Wind turned down in the balancing stage over the year, by zone, as solved, on the zonal network (zones drawn at their
centres, joined by the AC boundaries and the HVDC links). Final results, FES 2025 Holistic Transition.
```

### Dispatch against FES 2025 ES1

The final dispatch (after balancing) of each year is compared with FES 2025 ES1 by `validate_fes_dispatch`
(`scripts/FES/fes_dispatch_validation.py`), run on the final solves. Energy in TWh; PASS within the check's
tolerance, WARN outside it.

| TWh | 2040 FES | 2040 model | 2040 check | 2050 FES | 2050 model | 2050 check |
|---|---:|---:|---|---:|---:|---|
| Offshore wind | 366.7 | 323.8<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / wind_offshore --> | -12% WARN | 395.5 | 383.8<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / wind_offshore --> | -3% PASS |
| Onshore wind | 84.5 | 76.9<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / wind_onshore --> | -9% PASS | 95.4 | 97.8<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / wind_onshore --> | +2% PASS |
| Solar | 61.3 | 70.4<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / solar_pv --> | +15% WARN | 83.3 | 90.2<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / solar_pv --> | +8% PASS |
| Nuclear | 66.7 | 63.4<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / nuclear --> | -5% PASS | 93.5 | 95.2<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / nuclear --> | +2% PASS |
| Gas with CCS | 38.3 | 25.2<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / CCS Gas --> | -34% WARN | 42.4 | 38.1<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / CCS Gas --> | -10% WARN |
| Unabated gas | 1.3 | 3.6<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / gas (CCGT + OCGT + gas_engine) --> | +173% WARN | - | - | no line |
| Biomass with CCS | 20.3 | 21.2<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / CCS Biomass --> | +5% PASS | 20.7 | 21.1<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / CCS Biomass --> | +2% PASS |
| Hydrogen turbines | 3.9 | 0.8<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: dispatched_energy / Hydrogen --> | -80% WARN | 10.4 | 1.5<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: dispatched_energy / Hydrogen --> | -86% WARN |
| Imports | 38.6 | 44.9<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: interconnectors / Imports --> | +16% WARN | 51.1 | 59.0<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: interconnectors / Imports --> | +16% WARN |
| Exports | 87.6 | 74.7<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: interconnectors / Exports --> | -15% WARN | 68.9 | 75.3<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: interconnectors / Exports --> | +9% PASS |
| Demand excluding electrolysis | 519.0 | 519.3<!-- resources/analysis/FES25_HT_2040_fes_energy_validation.csv: demand / Electricity demand excluding electrolysis --> | +0% PASS | 615.3 | 615.6<!-- resources/analysis/FES25_HT_2050_fes_energy_validation.csv: demand / Electricity demand excluding electrolysis --> | +0% PASS |

Demand matches ES1 to within 0.1% in both years, and in 2050 wind, solar and nuclear all pass. In 2040 offshore wind
delivers 12% less than ES1. Gas with CCS runs less than in ES1 in both years, hydrogen turbines run far less (by 80%
in 2040 and 86% in 2050), and imports are higher. Rows more than 10% from ES1 are marked WARN.

## Limitations

- **Zonal network.** The network is clustered to NESO's 17 CP.08 zones. Constraints inside a zone are not seen, so
  network curtailment and constraint costs are lower bounds relative to a full ETYS solve.
- **The network is as FES plans it.** The 2040 and 2050 networks include every planned upgrade on time, with only
  NESO's historic boundary outage pattern applied. There is no delivery-delay case as in the CP30 study.
- **Boundary capabilities beyond the published horizon.** Secured boundary capabilities are held at their last
  published year where ETYS does not reach the model year.
- **Foreign prices are external.** Continental prices come from TYNDP 2026 draft National Trends+ and do not respond to
  GB's dispatch; interconnector flows and re-trading costs follow from them.
- **Perfect foresight within each window, one weather year (2013).** As in the CP30 study.
- **Bids and offers are derived from costs and support**, not observed: the subsidy-dependent turn-down order and the
  re-trading premium carry over from 2030 assumptions.
- **Differences from ES1 dispatch.** Gas with CCS and hydrogen turbines run less than in ES1: the model dispatches
  them on cost against renewables, storage and imports and does not reproduce ES1's running hours for them (see
  Dispatch against FES 2025 ES1).
