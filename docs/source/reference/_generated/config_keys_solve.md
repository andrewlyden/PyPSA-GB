## `solve`

File: `config/defaults/base/solve.yaml`. Keys below are relative to `solve`.

| Key | Default | Description |
|---|---|---|
| `voll` | `6000.0` | Value of lost load (GBP/MWh) |
| `solver` | (mapping) |  |
| `solver.name` | `gurobi` | Options: gurobi, highs, glpk, cplex |
| `solver.threads` | `2` | Barrier scales poorly past two threads on these LPs; the market rules declare threads: 2 so four months can solve at once on a 4-core machine. |
| `solver.method` | `2` | 0=primal, 1=dual, 2=barrier |
| `solver.crossover` | `0` | Barrier without crossover returns interior (non-vertex) solutions. Where the price is exactly zero a storage unit is indifferent between charging and discharging, so an interior point can do both at once and dissipate energy through round-trip losses. The market stages' storage_marginal_cost_floor and storage.dispatch_costs break that tie; crossover (-1) is the slower alternative. 0=disabled (faster), -1=auto (enables crossover) |
| `solver.BarHomogeneous` | `0` | Fast barrier profile. The market solves run these options through a fallback ladder (scripts/market/market_utils.py solve_with_fallback): configured -&gt; conservative barrier (NumericFocus 3, BarHomogeneous 1, ScaleFlag 2, DualReductions 0) -&gt; dual simplex with crossover; a window that needs a later rung is logged as a WARNING. 0=standard barrier; 1 is the conservative fallback rung |
| `solver.BarConvTol` | `0.0001` | Relaxed for numerical stability with large storage |
| `solver.FeasibilityTol` | `0.0001` | Relaxed for numerical stability with large storage |
| `solver.OptimalityTol` | `0.0001` | Relaxed for numerical stability with large storage |
| `solver.NumericFocus` | `0` | 0=Gurobi default; 3 is the conservative fallback rung |
| `solver.ScaleFlag` | `-1` | -1=Gurobi auto scaling; 2 is the conservative fallback rung |
| `solver.DualReductions` | `1` | 1=Gurobi default; 0 is the conservative fallback rung |
| `solver.BarIterLimit` | `200` | Barrier iteration cap (Gurobi default 1000) |
| `solver.io_api` | `direct` | How linopy hands the model to the solver in the market stages: "direct" (through gurobipy/highspy, fastest) \| "lp" \| "mps" (write a file and read it back). Not a Gurobi parameter: read by market_utils.solver_io_kwargs. |
| `mode` | `LP` | LP: fast, no unit commitment. MILP: slow, full unit commitment constraints. Options: "LP" or "MILP" |
