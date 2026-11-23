# Solvers and solver settings

Every PyPSA-GB solve is a linear programme: least-cost dispatch of every generator, storage unit and link over the
solve period, subject to the network's limits. This page covers which solver to use, every setting the workflow
passes to it, what to choose for runs of different sizes, and what to do when a solve is slow or fails. Installing
a solver is in {doc}`/getting_started/installation`.

## Which solver

| Solver | Licence | Use it for |
|---|---|---|
| **Gurobi** (default) | commercial; free named licences for academics | everything, and in practice the full ETYS network and the two-stage market studies |
| **HiGHS** | open source, installed with the environment | the Reduced and Zonal networks, clustered networks, and tutorials 1 to 4 |

Select a solver with `solve.solver.name` (`gurobi` or `highs`). The workflow also accepts `glpk` and `cplex`, but
passes them only the thread count.

HiGHS solves the small and clustered networks well. On the full ETYS network, and in the month-long market solves of
the studies, it is much slower than Gurobi, and the market stages' fallback ladder (below) exists only for Gurobi.

## Where the settings are

The settings are under `solve.solver`. The defaults are in `config/defaults/base/solve.yaml`. A scenario can override
any of them:

```yaml
My_Scenario:
  inherits: Tutorial_2020_reduced
  description: the tutorial week solved with HiGHS on four threads
  solve:
    solver:
      name: highs
      threads: 4
```

The same `solve:` block in `config/config.yaml` changes every scenario. To see the settings a scenario ends up with,
and which file set each one:

```bash
python config/config_loader.py --explain <scenario_id> --path solve
```

The study templates set a few of their own: the CP30, FES 2040 and FES 2050 studies keep two threads, and the CP30
study accepts a `suboptimal` Stage 1 window and uses `NumericFocus: 1` for central dispatch.

## The settings

| Key | Default | Meaning |
|---|---|---|
| `name` | `gurobi` | `gurobi`, `highs`, `glpk` or `cplex` |
| `threads` | `2` | Threads for the solver. The barrier method gains little beyond two on these problems |
| `method` | `2` | Gurobi algorithm: 0 primal simplex, 1 dual simplex, 2 barrier |
| `crossover` | `0` | Gurobi crossover after barrier: 0 off (faster), -1 automatic (a vertex solution, slower) |
| `BarConvTol`, `FeasibilityTol`, `OptimalityTol` | `1e-4` | Gurobi tolerances, relaxed from Gurobi's defaults for numerical stability with large storage fleets |
| `BarIterLimit` | `200` | Barrier iteration cap |
| `BarHomogeneous`, `NumericFocus`, `ScaleFlag`, `DualReductions` | `0`, `0`, `-1`, `1` | Gurobi's numerical settings at their standard values; the market fallback ladder moves them to the conservative values |
| `io_api` | `direct` | How the model reaches the solver in the market stages: `direct` (in memory, fastest), `lp` or `mps` (through a file) |
| `accepted_conditions` | `[optimal]` | Solver outcomes Stage 1 of the market accepts (add `suboptimal` to accept a window Gurobi stops short of the optimum) |

Only `threads` is passed to HiGHS; the Gurobi settings are ignored.

**Barrier without crossover.** The default returns an interior solution, much faster than a vertex. One side effect:
where the price is exactly zero, a storage unit is indifferent between charging and discharging and can do both at
once, wasting energy through its round-trip losses. The storage dispatch costs set at build time
(`storage.dispatch_costs`) and the market stages' storage cost floor break that tie. Set `crossover: -1` if you need
vertex solutions, for example for exact dual prices, and accept the longer run time.

## Choosing settings by run

| Run | Solver | Settings | Typical time and memory |
|---|---|---|---|
| Reduced or Zonal network, a week | HiGHS or Gurobi | defaults | minutes, 4 GB |
| ETYS clustered (about 300 buses), a week | HiGHS or Gurobi | defaults | 10 to 20 minutes, 8 GB |
| Full ETYS network, a week, single solve | Gurobi | defaults | 30 to 60 minutes, 16 GB |
| Full ETYS network, two-stage market, a month (the studies) | Gurobi | defaults (two threads) | about 30 minutes per stage per month, 16 GB or more |

**Threads and parallel jobs.** Snakemake's `-j` sets how many cores it may use; each rule declares its threads (the
single solve 4, the market stages 2). With `-j 8`, four market months solve side by side. Memory is usually the limit
on the full network, not cores: give Snakemake a budget, for example
`snakemake -j 8 --resources mem_mb=20000`, or lower `-j` if jobs are killed or the machine starts swapping
({doc}`/getting_started/running_the_workflow`).

**Long periods.** A full year in one solve is possible on the small networks. On the full network the studies solve
month by month (twelve scenarios sharing one build), which keeps each LP to a size Gurobi solves in about half an
hour.

## When a solve is slow or fails

**What the workflow does first.** Before every solve, `improve_numerical_conditioning` removes units below 0.1 MW,
solves consumer-flexibility units below 2 MW as fixed loads, and caps extreme transformer reactances, because
coefficients spanning many orders of magnitude are what make a barrier solver struggle
({doc}`single_solve`).

**The market stages' fallback ladder.** Each window of the wholesale and balancing stages is tried in turn on:

1. the configured settings;
2. a conservative barrier (`NumericFocus: 3`, `BarHomogeneous: 1`, `ScaleFlag: 2`, `DualReductions: 0`);
3. dual simplex with crossover, which returns a vertex solution.

A window solved on a later rung is logged as a WARNING naming the rung. A solution that breaks a unit's bounds by more
than 1 MW counts as failed, and the next rung is tried. The single solve has no ladder: it must end with status `ok`,
or the rule fails.

**If it is slow:**

- Solve a shorter period first (`solve_period`), or a clustered network, to check the scenario.
- Keep the barrier defaults (`method: 2`, `crossover: 0`). Crossover can multiply the run time.
- On the full network, use Gurobi rather than HiGHS.
- More threads rarely help a single barrier solve; more parallel jobs do, if memory allows.

**If it fails:**

- **Infeasible.** Load shedding at the value of lost load (`solve.voll`, GBP 6,000/MWh) is available at every demand
  bus, so a network that cannot meet demand normally reports load shed, not infeasibility. A true infeasibility
  usually points to an inconsistent constraint (a fixed schedule that cannot be met, a storage level that cannot be
  reached). In the balancing stage, `market.balancing.diagnostics.compute_iis: true` makes Gurobi log the conflicting
  constraints of a failed window ({doc}`/development/troubleshooting`).
- **Numerical trouble or a suboptimal stop.** Try `NumericFocus: 1` to `3`, or `crossover: -1`, in the scenario.
- **Out of memory.** Lower `-j`, solve a shorter period, or cluster the network.

The solver's log is in the rule's log file (`logs/solve_network_{scenario}.log`, and `logs/market/` for the market
stages); a failed rule prints the path.
