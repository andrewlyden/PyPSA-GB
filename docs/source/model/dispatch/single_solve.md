# Single network-constrained solve

## What it does

The single solve finds the least-cost dispatch of every generator, storage unit, link and
interconnector over the solve period, with every transmission limit in force. It is one linear
optimal power flow (LOPF) in PyPSA. Each bus gets its own marginal price (a nodal price), and the
solve reports how much load, if any, could not be served.

It is what a scenario runs when `market.enabled` is `false` (the default). A scenario with
`market.enabled: true` skips it and runs the two-stage market instead ({doc}`wholesale`,
{doc}`balancing`); both start from the same finalised network.

Two Snakemake rules do the work (`rules/solve.smk`):

| Rule | Script | Output |
|------|--------|--------|
| `finalize_network` | `scripts/solve/finalize_network.py` | `resources/network/{scenario}.nc` and `{scenario}_network_summary.txt` |
| `solve_network` | `scripts/solve/solve_network.py` | `resources/network/{scenario}_solved.nc`, `{scenario}_optimization_summary.txt` and the results tables |

## Method and assumptions

### Finalising the network

`finalize_network` takes the complete network (demand, renewables, thermal plant, storage, hydrogen
and interconnectors added; the clustered version when clustering is on) and writes it under the
scenario's own name. On the way it:

1. **Aggregates identical components** when `network.component_aggregation.enabled` is true:
   generators (and, if asked, storage units, stores and loads) with identical parameters at the same
   bus are merged, which shortens solves without changing the optimum.
2. **Clips tiny availabilities**: hourly `p_max_pu` values between 0 and 0.001 are set to zero. Very
   small non-zero bounds hurt the solver's numerics and carry no energy.
3. **Prepares large hydro** (power ceiling and reservoir path, `renewables.hydro`) on the finished
   network, so every large-hydro unit is covered whichever step added it.
4. **Validates** the network component by component (`scripts/solve/deep_network_validation.py`).
   A critical error stops the workflow before any solve; warnings go into the summary report.

A scenario that declares `network_from: <parent>` (for example one month of a year built once)
solves on its parent's finalised network.

### Preparing the solve

`solve_network` loads the finalised network and applies, in this order:

1. **Cost checks.** Load-shedding units must carry a positive cost and thermal units a non-zero
   one; a network that would make the LP unbounded is refused.
2. **Transmission settings** from `network.transmission`: capacity floors and scaling, the voltage
   floor below which branches are left unconstrained (`min_bm_constraint_voltage_kv` with its
   Scottish exception `bm_voltage_floor`), explicit `line_rating_overrides`, and the outage schedule
   or secured boundary limits (`outage_schedule`). These are described in
   {doc}`/model/network/constraints`.
3. **Numerical conditioning** (below).
4. **The solve period** (`solve_period`): the network is built for a full year and solved on a
   window of it, either `start` and `end` or `auto_select` (`peak_demand_week`, `peak_wind_week`,
   `low_wind_week`).
5. **Seasonal line ratings** (`network.transmission.seasonal_ratings`), applied as a monthly factor
   on each line's rating once the period is known.
6. **Load-shedding caps** (below), and **feasible operating bounds**: wherever a generator's hourly
   minimum output exceeds its available maximum (a must-run floor in an hour the unit is out), the
   minimum is cut to the maximum, so an outage cannot make the LP infeasible.

The solve is a linear programme. Every generator's `committable` flag is cleared before the solve,
so there is no unit commitment: plant can run anywhere between its hourly minimum and maximum.
PyPSA's LOPF uses the linearised (DC) power flow: Kirchhoff's laws on line reactances, no losses,
no reactive power.

Three sets of extra constraints are added to PyPSA's own:

- **Hydro constraints** (`scripts/solve/hydro_constraints.py`): reservoir and pumped-storage limits
  configured under `renewables.hydro`.
- **Boundary constraints**: where the outage schedule supplies limits on whole boundaries (sums of
  branch flows), they are imposed as aggregate constraints.
- **Interconnector cable limits** (`scripts/interconnectors/market_coupling.py`): a cable modelled as
  two one-way links shares one rating between the two directions.

### Load shedding at the value of lost load

Every bus with demand carries a load-shedding generator, added at generator integration with a
marginal cost equal to `solve.voll` (GBP 6,000/MWh). It keeps the LP feasible when supply or the
network cannot meet demand, and its output is unserved energy. Before the solve each unit is capped,
hour by hour, at the demand on its own bus (`apply_load_shedding_limits`), so it behaves as demand
interruption and cannot export power through the network as an emergency generator. A bus with no
load gets a cap of zero.

### Numerical conditioning

Large GB networks have coefficients that span many orders of magnitude, which is what makes a
barrier solver report numerical trouble. `improve_numerical_conditioning` therefore:

- removes generators (other than load shedding), storage units and links below 0.1 MW;
- solves **consumer-flexibility units smaller than 2 MW** (`MIN_CONSUMER_FLEX_PNOM_MW`) as fixed
  loads: each unit's drain becomes a `Load` named `<unit> fixed drain` on its bus, at its reference
  draw, so its energy stays in the demand and only its flexibility is lost. Many very small units
  make the LP slow and poorly conditioned for a negligible share of flexible capacity;
- clamps transformer reactances above 10 pu on 100 MVA.

The same step runs in both market stages, so a market run and a single solve see the same fleet.

### Solver

The solver and its options come from `solve.solver`. Gurobi is the default; HiGHS is the open-source
alternative (`solve.solver.name: highs`, which passes only the thread count). The default Gurobi
profile is a barrier solve without crossover (`method: 2`, `crossover: 0`) at relaxed tolerances
(`1e-4`) and two threads.

Barrier without crossover returns an interior solution. Where the price is exactly zero a storage
unit is then indifferent between charging and discharging, and can do both at once, wasting energy
through its round-trip losses. The storage dispatch costs set at build time
(`storage.dispatch_costs`) break that tie; crossover (`crossover: -1`) is the slower alternative.

The solve must end with status `ok`, or the rule fails. The fallback ladder of the market stages
({doc}`wholesale`) is not used here. Every solver setting, and what to do when a solve is slow or fails, is in
{doc}`solvers`.

## Data

The single solve reads only the finalised network and the configuration. Everything it optimises
(costs, availabilities, demand, ratings) was set by the build steps listed on {doc}`/model/index`:
network, demand, generation, storage, hydrogen and interconnectors.

## Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `market.enabled` | `false` | `false` runs this solve; `true` runs the two-stage market instead |
| `solve_period.enabled` | `false` | Restrict the solve to part of the year |
| `solve_period.start`, `.end` | | Explicit period (timestamps) |
| `solve_period.auto_select` | | `peak_demand_week`, `peak_wind_week` or `low_wind_week` |
| `solve.voll` | `6000.0` | Value of lost load (GBP/MWh), the load-shedding cost |
| `solve.solver.name` | `gurobi` | `gurobi` or `highs` (also `glpk`, `cplex`) |
| `solve.solver.threads` | `2` | Solver threads |
| `solve.solver.method` | `2` | Gurobi method (2 = barrier) |
| `solve.solver.crossover` | `0` | 0 = off (faster); -1 = automatic |
| `solve.solver.BarConvTol`, `FeasibilityTol`, `OptimalityTol` | `1e-4` | Gurobi tolerances |
| `solve.solver.BarIterLimit` | `200` | Barrier iteration cap |
| `network.component_aggregation.enabled` | `false` | Merge identical components at finalisation |
| `network.transmission.*` | | Ratings, voltage floor, outages and boundaries ({doc}`/model/network/constraints`) |

## Validation

- `finalize_network` refuses a network that fails its component checks; the summary report lists
  every warning.
- `{scenario}_optimization_summary.txt` reports the solver status, the objective, the generation mix
  and the total load shed. Any load shedding means the network or the fleet cannot meet demand;
  {doc}`/development/troubleshooting` lists the usual causes.

## Outputs

| File | Content |
|------|---------|
| `resources/network/{scenario}_solved.nc` | Solved network: dispatch, flows, storage state, nodal prices (`buses_t.marginal_price`) |
| `resources/network/{scenario}_optimization_summary.txt` | Status, objective, generation mix, load shed |
| `resources/results/{scenario}_generation.csv` | Generator output (snapshots x generators) |
| `resources/results/{scenario}_storage.csv` | Storage dispatch and state of charge |
| `resources/results/{scenario}_flows.csv` | Line flows |
| `resources/results/{scenario}_costs.csv` | Costs by generator |
| `resources/results/{scenario}_emissions.csv` | Emissions by generator |

## Limitations

- **One optimiser, perfect foresight.** The whole solve period is one LP, so storage plans against
  every hour in it. Real operation schedules a day ahead and corrects in real time; the two-stage
  market represents that.
- **Nodal prices are not GB market prices.** GB has one national wholesale price; the single
  solve's prices differ by bus wherever the network binds. Use the two-stage market for a GB price
  and a constraint cost.
- **No unit commitment, linear power flow.** Minimum stable generation, start costs and losses are
  not modelled.
