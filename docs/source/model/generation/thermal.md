# Thermal generation

Dispatchable plant that burns a fuel or splits atoms: gas (CCGT, OCGT, engines, CHP, gas with CCS), nuclear, coal,
oil, biomass (with and without CCS), energy from waste, the smaller biofuels, geothermal, and hydrogen turbines (see
{doc}`../flexibility/hydrogen`). This page covers which plant is in a scenario, where it is connected, how available it is and what
it costs to run.

## What it does

Rule `add_thermal_generators_to_network` (`rules/generators.smk`, script
`scripts/generators/integrate_thermal_generators.py`) adds the thermal fleet as PyPSA `Generator` components. Rule
`apply_marginal_costs_to_network` (script `scripts/generators/apply_marginal_costs.py`) then sets every generator's
marginal cost and writes the breakdown to `resources/generators/{scenario}_marginal_costs_breakdown.csv`.

| Scenario type | Capacity | Location |
|---|---|---|
| Historical | DUKES 5.11 power stations, plus REPD dispatchable renewables (biomass, waste, biogas, landfill and sewage gas, advanced biofuels, geothermal) | each station's own coordinates |
| FES GSP blocks | FES generation building blocks by GSP and technology | GSP rows at their GSP; `Direct` rows spread within their licence area |
| CP30 brownfield | CP30 ES1 national totals by technology | named stations from declared schedules; FES proxy rows for any residual |
| FES long range | FES building blocks, with a declared station schedule for `Direct` rows | named or declared sites |

## Method and assumptions

### Historical fleet

Historical runs use only published historical sources; there is no FES fallback. DUKES table 5.11 gives conventional
capacity (coal, gas, oil, nuclear) by station. REPD adds the dispatchable renewables. Stations that closed before
1 January of the modelled year are removed using `data/generators/coal_phase_out_dates.csv` and
`nuclear_phase_out_dates.csv`; a station closing during the year is kept, at reduced capacity where it ran so.

Each station is placed at its coordinates (`data/generators/dukes_power_station_coordinates.csv` and the other location
tables) on the nearest bus. On ETYS networks the BMU-to-node table of `GB_network.xlsx` then moves large units from
nearby 132 kV buses to the transmission node they connect to.

Nuclear availability is historical too: the hourly ESPENI nuclear output divided by installed nuclear capacity becomes
the nuclear `p_max_pu`. `thermal.nuclear_smoothing` keeps it hourly (individual outages) or averages it over days or
months.

For back-tests, `thermal.dispatch_anchors` can tie listed carriers to observed commitment decisions the linear model
cannot represent. It is never used in future scenarios.

### Future fleet: FES GSP blocks

FES gives capacity by technology and GSP. Rows at a named GSP go to that GSP's node. `Direct(NGET)`, `Direct(SPTL)`
and `Direct(SHETL)` rows (transmission-connected plant with no site) are spread over buses in their licence area, in
proportion to where that carrier is today (`distribute_fes_generators_spatially`).

### Future fleet: CP30 brownfield

NESO Clean Power 2030 (CP30) publishes thermal capacity as national totals per pathway (ES1). The build keeps those
totals exact and replaces anonymous proxies with named plant wherever there is evidence. The station schedules are data
files in `data/market/CP30/`, named by keys under `fleet.spatial_proxy`:

| Schedule | What it does |
|---|---|
| `gas_station_schedule` | CP30's unabated gas total is split into CCGT, OCGT and gas engines in the FES proxy's proportions. Named stations fill each carrier's target at their DUKES capacity and declared bus. If they exceed the target, they are kept in order of evidence that they run in the year: Capacity Market agreements for both delivery years the calendar year spans, then for one, then none; then FES retention at the station's own GSP or licence area; then newest first. The station at the margin takes the residual, so the total is exact. |
| `nuclear_station_schedule` | Places CP30's nuclear total at named stations with their own connection bus and support label; one station may take the residual. |
| `biomass_station_schedule`, `ccs_biomass_station_schedule` | Named biomass and BECCS units at their buses; biomass retires by age. |
| `cfd_station_schedule` | Carves named CfD units (energy from waste, advanced conversion, geothermal) out of the anonymous waste and "other renewables" proxies, so their contracts can be settled. |
| `committed_overlay` | Adds committed plant that CP30 omits. It is refused where CP30 already has capacity for that carrier. `committed_overlay_carve_out` can take the overlay's capacity out of a member carrier of the same CP30 line instead of adding it on top, so the line total stays CP30's. |

Each named CCGT carries a heat-rate index from its empirical marginal cost. The class efficiency sets the fleet level;
the index sets the spread around it, with the kept fleet's capacity-weighted harmonic mean efficiency equal to the class
value. The fleet's average cost does not move; plants differ.

### Future fleet: FES long range

With `fleet.fes_supply.enabled` (the FES long-range studies, `scripts/generators/fes_supply.py`):

- FES's "CCGTs (non CHP)" block holds gas CCS and unabated CCGT together. `fleet.fes_supply.thermal.gas_ccs_split`
  splits every row in that year's ES1 proportion.
- The `Direct` rows of nuclear, gas CCS, transmission hydrogen power and BECCS are replaced by a declared station
  schedule (rule `build_fes_station_schedule`, from `data/generators/fes/station_list.csv`). Each row names a site
  and bus with its evidence: a named project from the TEC register, a declared site (for example a licensed nuclear
  site for small modular reactors), or a share of the FES residual over declared sites. The rows of every
  (carrier, licence area) sum to FES's total exactly.
- Declared buses are restored after the generic placement corrections, so they survive.

### Availability and operating limits

The model has no unit commitment. Thermal operation is limited by linear constraints:

| Limit | Key | How it acts |
|---|---|---|
| Availability factor | `thermal.availability_factors.factors` | caps `p_max_pu` per carrier (fuel supply, maintenance, heat-led CHP) |
| Must-run floor | `thermal.availability_factors.min_output` | sets `p_min_pu` (for example biomass, energy from waste) |
| Energy-window floor | `thermal.availability_factors.min_output_energy_window_hours` | turns a carrier's hourly floor into an energy budget per block of hours in Stage 1; the balancing stage drops the hourly floor |
| Group overrides | `thermal.availability_factors.group_overrides` | factor and floor for one (carrier, data source) group |
| Outages | `thermal.outages` | reproducible planned and forced outages applied to `p_max_pu` (seeded) |
| Ramp limits | `thermal.ramping` | `ramp_limit_up` / `ramp_limit_down` as a share of `p_nom` per hour |

**Example: nuclear with limited flexibility.** Nuclear has no ramp limit by default. A scenario can hold it to a
minimum output and a ramp rate with the same two keys:

```yaml
thermal:
  availability_factors:
    enabled: true
    min_output:
      nuclear: 0.20        # floor, share of p_nom
  ramping:
    enabled: true
    carriers:
      nuclear: {ramp_limit_up: 0.025, ramp_limit_down: 0.025}   # share of p_nom per snapshot
```

Ramp limits act between consecutive snapshots, so at 30-minute resolution the same value allows twice the change per
hour. With no unit commitment there are no start-up decisions, minimum up and down times or cycling costs.

Outages follow, per carrier, a number of planned outages a year of a given length in given months, and a forced outage
rate with a mean duration. Units are drawn independently with a fixed seed, so a rebuild gives the same profile. Gas CHP
can carry a seasonal must-run floor that follows heat demand (see the heat credit below).

```{figure} /_static/figures/supply/gas_availability_2030.png
:alt: Line of available gas capacity in GW through 2030, below a dashed line of installed capacity.
:width: 100%

Available CCGT, OCGT and gas-engine capacity in the 2030 build, daily mean, after the seeded planned and forced outages.
Planned outages start between April and September; forced outages give the day-to-day variation.
```

### Marginal costs

The short-run marginal cost of a fuel-burning generator, in GBP/MWh of electricity, is

$$
c = \frac{p_\text{fuel}}{\eta} + \frac{e_\text{fuel}\, p_\text{CO2}}{1000\,\eta} + c_\text{VOM},
$$

with fuel price $p_\text{fuel}$ (GBP/MWh thermal), electrical efficiency $\eta$, emission factor $e_\text{fuel}$
(kgCO2/MWh thermal), carbon price $p_\text{CO2}$ (GBP/tCO2) and variable O&M $c_\text{VOM}$ (GBP/MWh).

**Prices.** Historical runs use built-in tables of each year's fuel and carbon prices. Future runs use the case in
`marginal_costs.future_costs.case`:

| Case | Source |
|---|---|
| `low`, `central`, `high` | fuel, carbon (UK ETS plus Carbon Price Support) and VOM in the base config |
| `neso_cp2030` | gas and carbon from NESO CP30 Annex 4 (rule `fetch_neso_cp2030_annex4_costs`); other fuels from a fallback case |
| a study's own case | for example the FES 2025 commodity prices of the modelled year (rule `build_fes2025_future_costs_source`) |

Further settings refine the case: `gas_price_override_pence_per_therm` replaces its gas price;
`carbon_price_support_end` drops Carbon Price Support from a date; `gas_emission_factor_basis` chooses the gross-CV
(0.183 t/MWh) or net-CV (0.2016 t/MWh) gas factor to match the efficiency basis; `efficiency_defaults` sets class
efficiencies and `unit_efficiency_overrides` single units. Gas with CCS emits 10% of the unabated factor.

**Order of precedence for each generator:**

1. load shedding and other protected carriers keep their price;
2. nuclear takes `marginal_costs.nuclear_mc` (its bids in the balancing mechanism reflect strategy, not short-run cost);
3. a thermal unit with an empirical marginal cost (historical calibration from Elexon data,
   `marginal_costs.empirical_calibration`) takes it;
4. other thermal units take the formula, optionally times a carrier correction factor
   (`marginal_costs.carrier_correction_factors`, off for future runs by default);
5. renewables take a cost from their support label (see {doc}`/model/dispatch/bids_offers_support`);
6. everything else takes the formula.

**By-product adjustments** (`marginal_costs.byproduct_adjustments`). Energy from waste is paid a gate fee to take its
fuel, so its marginal cost is VOM only. Gas CHP is credited for the heat it sells, at the cost of an 85%-efficient gas
boiler; the credit follows a seasonal heat-demand shape from monthly heating degree days, which can also scale the CHP
must-run floor.

**Support.** Biofuels holding ROCs can have their ROC income deducted (`marginal_costs.biofuel_subsidy`), floored at a
minimum. Gas CCS under a Dispatchable Power Agreement is priced to run ahead of unabated gas: its cost is the lower of
its own and the cheapest CCGT's less a margin. These rules belong to the support chapter
({doc}`/model/dispatch/bids_offers_support`).

```{figure} /_static/figures/supply/merit_order_2030.png
:alt: Merit-order chart: marginal cost in GBP/MWh against cumulative installed capacity in GW, coloured by technology group.
:width: 100%

Merit order of the 2030 build: each generator's marginal cost against cumulative installed capacity, coloured by
technology group (gas CHP at its annual mean after the heat credit; interconnectors and load shedding left out). Support
labels set the renewables' costs: ROC-accredited plant below zero, CfD and merchant plant at zero. The thermal fleet
rises from nuclear and biomass through CCGTs to OCGTs, gas engines and oil.
```

## Data

| Input | File |
|---|---|
| DUKES 5.11 power stations | `data/generators/DUKES_5.11_2025.xlsx` |
| Station locations | `data/generators/dukes_power_station_coordinates.csv`, `power_stations_locations.csv` |
| Closures | `data/generators/coal_phase_out_dates.csv`, `nuclear_phase_out_dates.csv` |
| REPD dispatchable sites | `data/renewables/repd-q2-jul-2025.csv` (rule `extract_repd_dispatchable_sites`) |
| FES building blocks | `resources/FES/FES_{edition}_data.csv` |
| CP30 totals | `resources/CP30/CP30_ES1.csv` |
| CP30 station schedules | `data/market/CP30/*_station_schedule_2030.csv`, `committed_overlay_2030.csv` |
| FES long-range stations | `data/generators/fes/station_list.csv` |
| Capacity Market register (gas retention) | `data/market/capacity_market/` |
| ESPENI (historical nuclear) | `data/demand/espeni.csv` |
| TEC register | `data/generators/tec-register-02-september-2025.csv` |

The gas schedule's Capacity Market and FES columns are rebuilt by `scripts/CP30/build_gas_station_schedule.py`, and
rule `verify_CP30_gas_station_schedule` checks them against the register. See {doc}`/reference/data/data_sources`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `thermal.availability_factors` | off in the base; on in CP30 | carrier caps, floors, group overrides |
| `thermal.outages` | on | planned and forced outages per carrier |
| `thermal.ramping` | on | ramp limits per carrier |
| `thermal.nuclear_smoothing` | `hourly` | historical nuclear availability: `hourly`, `daily`, `monthly` |
| `thermal.dispatch_anchors` | off | historical back-tests only |
| `fleet.spatial_proxy.*_station_schedule` | CP30 study | named-station schedules |
| `fleet.spatial_proxy.committed_overlay`, `committed_overlay_carve_out` | CP30 study | committed plant CP30 omits |
| `fleet.fes_supply` | FES long-range studies | gas CCS split, station schedule |
| `marginal_costs.future_costs.case` | `central` (CP30: `neso_cp2030`) | future commodity case |
| `marginal_costs.future_costs.gas_price_override_pence_per_therm` | `null` (CP30: 100.68) | gas price, p/therm |
| `marginal_costs.future_costs.gas_emission_factor_basis` | `gross_cv` (CP30: `net_cv`) | gas emission factor basis |
| `marginal_costs.nuclear_mc` | 10.0 | nuclear marginal cost, GBP/MWh |
| `marginal_costs.byproduct_adjustments` | EfW gate fee, CHP heat credit | by-product treatment |
| `marginal_costs.biofuel_subsidy.enabled` | `false` | deduct ROC income from biofuels |

Defaults are in `config/defaults/base/thermal.yaml` and `config/defaults/base/marginal_costs.yaml`; the CP30 values in
`config/defaults/source/CP30.yaml` and the CP30 study's templates.

## Validation

- CP30 builds compare thermal capacity by ES1 line with CP30 (rule `validate_CP30_network`), counting overlay
  capacity inside the line it was carved from. After Stage 1, the energy of each main generation line is checked
  against ES1 (rule `validate_CP30_stage1`, failing beyond 15%), and the full year's dispatch again after balancing
  (rule `validate_CP30_dispatch`).
- The fleet-location check (rule `validate_CP30_fleet_location`) compares capacity by region and on each side of the
  secured boundaries with CP30.
- FES long-range years are checked against FES ES1 (rule `validate_fes_dispatch`).
- Historical dispatch is compared with ESPENI and Elexon in {doc}`/studies/validation_2020`.

## Limitations

- No unit commitment: start-up costs, minimum up and down times and part-load efficiency are absent. Must-run floors,
  energy windows and ramp limits stand in for them.
- A carrier's efficiency is one class value (with a per-unit spread for named CCGTs); there is no part-load curve.
- Outages are a seeded random draw, not a forecast of real maintenance plans.
- Future fuel and carbon prices are flat for the year in a case; they do not vary by month.
