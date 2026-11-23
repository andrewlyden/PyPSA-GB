# Outputs and file names

Everything the workflow writes goes under `resources/` (or the folder given by `--config resources_path=...`) and
`logs/`. Neither is in the repository: both are rebuilt by Snakemake. File names start with the scenario id, written
`{scenario}` below. A monthly study shares one network build between its months, so a month's market files carry the
month's id while its network is the build's.

## Network build chain

Each rule adds one part of the system and writes the network under a longer name. The intermediate networks are
pickles (`.pkl`); the base network, the network after interconnectors and the finished network are NetCDF (`.nc`),
readable with `pypsa.Network(path)`.

| Step | Rule | File in `resources/network/` |
|---|---|---|
| Topology | `build_ETYS_base_network`, `build_reduced_base_network`, `build_zonal_base_network` | `{scenario}_network.nc` |
| Base demand | `build_base_demand` | `{scenario}_network_demand_base.pkl` |
| Demand components and flexibility | `finalize_demand` | `{scenario}_network_demand.pkl` |
| Renewables | `add_renewables_to_network` | `{scenario}_network_demand_renewables.pkl` |
| Thermal plant | `add_thermal_generators_to_network` | `{scenario}_network_demand_renewables_thermal.pkl` |
| Generator checks | `finalize_generator_integration` | `{scenario}_network_demand_renewables_thermal_generators.pkl` |
| Marginal costs | `apply_marginal_costs_to_network` | `{scenario}_network_demand_renewables_thermal_generators_costs.pkl` |
| Storage | `add_storage_to_network` | `{scenario}_network_demand_renewables_thermal_generators_storage.pkl` |
| Hydrogen | `add_hydrogen_system` | `{scenario}_network_demand_renewables_thermal_generators_storage_hydrogen.pkl` |
| Interconnectors | `add_interconnectors_to_network` | `{scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc` |
| Clustering (optional) | `cluster_network` | `{scenario}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc` and `{scenario}_clustering_busmap.csv` |
| Finished network | `finalize_network` | `{scenario}.nc` and `{scenario}_network_summary.txt` |

Build side files:

| File | Contents |
|---|---|
| `resources/network/{scenario}_etys_upgrade_audit.csv` | ETYS Appendix B rows applied, and rows that could not be represented |
| `resources/network/{scenario}_cp30_reinforcement_audit.csv`, `..._hnd_hybrid_audit.csv` | CP30 reinforcements and offshore hybrid links applied |
| `resources/generators/{scenario}_renewables_summary.csv`, `..._thermal_summary.csv` | capacity added by each integration step |
| `resources/generators/{scenario}_generators_full.csv`, `..._generators_summary_by_carrier.csv` | every generator, and capacity by carrier |
| `resources/generators/{scenario}_marginal_costs_breakdown.csv` | each generator's marginal cost and the layers that set it |
| `resources/demand/{scenario}_demand_integration_summary.csv` | demand added, by component |

## Single network solve

Scenarios without the market solve the finished network once (rule `solve_network`):

| File | Contents |
|---|---|
| `resources/network/{scenario}_solved.nc` | the solved network |
| `resources/network/{scenario}_optimization_summary.txt` | solver status, objective and headline figures |
| `resources/results/{scenario}_generation.csv` | generation by generator and hour |
| `resources/results/{scenario}_storage.csv` | storage dispatch and state of charge |
| `resources/results/{scenario}_flows.csv` | AC line flows (links are not included) |
| `resources/results/{scenario}_costs.csv` | energy and cost by generator |
| `resources/results/{scenario}_emissions.csv` | emissions by generator, where generators carry an intensity (usually a placeholder line) |
| `resources/analysis/{scenario}_summary.json`, `_dashboard.html`, `_spatial.html`, `_notebook.ipynb` | summary, dashboard, map and notebook (rules `analyze_and_visualize_solved_network`, `generate_analysis_notebook`) |

## Two-stage market

Market scenarios (`market.enabled: true`) solve a wholesale stage and a balancing stage instead.

**Wholesale (Stage 1, rule `solve_wholesale_market`)**, in `resources/market/`:

| File | Contents |
|---|---|
| `{scenario}_wholesale.nc` | the solved copperplate network |
| `{scenario}_wholesale_dispatch.csv` | generator positions (MW by hour) |
| `{scenario}_wholesale_storage.csv` | storage positions |
| `{scenario}_wholesale_links.csv` | link and interconnector flows |
| `{scenario}_wholesale_price.csv` | the wholesale price (GBP/MWh), over demand buses |

**Balancing mechanism (Stage 2, rule `solve_balancing_mechanism`)**, in `resources/market/` unless stated:

| File | Contents |
|---|---|
| `{scenario}_balancing.nc` | the solved network-constrained redispatch |
| `{scenario}_balancing_dispatch.csv`, `{scenario}_balancing_storage.csv` | physical positions after redispatch |
| `{scenario}_redispatch_summary.csv` | per-unit increases and decreases (MWh) and their cost |
| `{scenario}_constraint_costs.csv` | redispatch cost and volume by carrier (offer, bid and net cost in GBP; increase and decrease in MWh), with a TOTAL row; the interconnector counter-trade premium is reported in the market summary, not here |
| `{scenario}_congestion.csv` | branches at their limit |
| `{scenario}_price_comparison.csv` | wholesale against balancing nodal prices |
| `{scenario}_contract_value_bids.csv` | the bid each supported unit was given (CfD contract value, RO, merchant) |
| `resources/analysis/{scenario}_adequacy_relaxation.csv` | any unserved energy and where |

**Turn-down allocation** (`rules/turn_down_allocation.smk`): `{scenario}_pro_rata_turn_down.csv/.json` re-prices
the month's wind turn-down pro rata by zone, and `{scenario}_turn_down_basis.json` records which basis the month's
reports use.

**Analysis (rule `analyze_market_results`)**, in `resources/analysis/`:

| File | Contents |
|---|---|
| `{scenario}_market_summary.json` | the month's headline figures: prices, volumes, costs |
| `{scenario}_market_accounting.csv` | the accounting measures, one per row, with units |
| `{scenario}_interconnector_flows.csv` | flows by cable |
| `{scenario}_market_dashboard.html` | an interactive dashboard |

**Studies of twelve months** combine their months with rule `aggregate_CP30_months` into
`resources/analysis/{study}_monthly_metrics.csv` and `{study}_annual_summary.json`: costs and volumes summed, prices
weighted by demand ({doc}`/model/after_the_solve/results`).

## Validation outputs

The `validate_CP30_network`, `_locational` and `_fleet_location` rows check a build; the rest check a solved run
({doc}`/model/after_the_solve/validation`).

| File | Rule | Compares |
|---|---|---|
| `resources/market/{scenario}_bm_validation.csv`, `resources/analysis/{scenario}_bm_validation.html` (and `_boav_by_carrier.csv`, `_boalf_by_flag.csv`, `_disbsad_summary.csv`, `_redispatch_mapped_unmapped.csv` in `resources/market/`) | `validate_bm_results` | historical balancing volumes, costs, prices and dispatch against ELEXON |
| `resources/market/{scenario}_bm_calibration.csv`, `resources/analysis/{scenario}_bm_calibration.html` | `validate_bm_calibration` | bid and offer prices and volumes by technology against ELEXON |
| `resources/market/{scenario}_neso_validation.csv`, `resources/analysis/{scenario}_neso_validation.html` | `validate_neso_constraints` | constraint costs and boundary flows against NESO data |
| `resources/CP30/validation/{scenario}_component_totals.csv`, `..._energy_totals.csv` | `validate_CP30_network` | a CP30 build's capacity and available energy against CP30 |
| `resources/CP30/validation/{scenario}_locational_checks.csv` | `validate_CP30_locational` | plant location against CP30's regional split |
| `resources/CP30/validation/{scenario}_fleet_location.csv/.json` | `validate_CP30_fleet_location` | capacity by region, zone and boundary side |
| `resources/analysis/{study}_cp30_stage1_validation.csv/.json` | `validate_CP30_stage1` | Stage 1 energy and prices against CP30 |
| `resources/analysis/{study}_cp30_energy_validation.csv/.json` | `validate_CP30_dispatch` | dispatched energy and curtailment over the year against CP30 |
| `resources/analysis/{study}_cp30_balancing_validation.csv/.json` | `validate_CP30_balancing` | a CP30 year's balancing volumes, cost, dispatch and turn-down by zone against NESO |
| `resources/analysis/{study}_fes_energy_validation.csv/.json` | `validate_fes_dispatch` | a FES long-range year's energy against FES |
| `resources/analysis/{study}_ic1_diagnostics_summary.json` (and its tables), `{study}_bm_storage_churn.csv/.json` | `diagnose_fes_ic1`, `measure_fes_bm_storage_churn` | a FES long-range year's interconnector flows and spreads, and storage churn in balancing |
| `resources/analysis/secured_capability_by_family.csv` | `audit_secured_capability` | the boundary capability each family's balancing stage enforces |

## After the solve

The consumer-bill layer (preliminary) reads completed market outputs and never triggers a solve. Its main output is
`resources/analysis/{bill_case}_consumer_bill_future_summary.json` (with `..._components.csv` and an HTML report).
See
{doc}`/model/after_the_solve/index`.

## Logs

Each rule writes a log under `logs/` (for example `logs/solve_network_{scenario}.log`, `logs/market/...`). When a rule
fails, Snakemake prints the log path; read it first.
