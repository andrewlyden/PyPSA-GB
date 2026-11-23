# Running the workflow

PyPSA-GB is a [Snakemake](https://snakemake.github.io/) workflow. You ask for output files (targets); Snakemake
works out which rules make them, runs those whose outputs are missing or out of date, and skips the rest. This
page covers what you need to drive it day to day. Run every command from the repository root, in the activated
environment.

## Files

| File | Role |
|---|---|
| `Snakefile` | The main workflow. Loads the configuration, decides which scenarios to run, includes the rule files and defines `rule all`. |
| `rules/*.smk` | The rules, one file per part of the model: `network_build`, `network_clustering`, `demand`, `renewables`, `generators`, `storage`, `hydrogen`, `interconnectors`, `solve`, `market`, `analysis`, and the source and research layers (`FES`, `CP30`, `fes_long_range`, `consumer_bill`, `public_ownership`, ...). |
| `scripts/` | The Python each rule runs. Module docstrings describe the method. |
| `Snakefile_cutouts` | A separate workflow that only fetches weather cutouts (see {doc}`installation`). |
| `config/config.yaml` | Which studies to load and which scenarios to run. |

## The build chain

For each scenario the main workflow builds one network in steps, each writing an intermediate file under
`resources/network/`:

```{mermaid}
flowchart TD
    subgraph b1["Build"]
        direction LR
        A["{id}_network.nc<br/>topology"] --> B["+ demand"] --> C["+ renewables"]
    end
    subgraph b2[" "]
        direction LR
        D["+ thermal,<br/>marginal costs"] --> E["+ storage"] --> F["+ hydrogen"]
    end
    subgraph b3[" "]
        direction LR
        G["+ interconnectors"] --> H["clustering<br/>(optional)"]
    end
    b1 --> b2 --> b3
    b3 --> I["{id}.nc<br/>finalised"]
    I --> J["{id}_solved.nc<br/>single solve"]
    I --> K["{id}_wholesale.nc<br/>Stage 1"]
    K --> L["{id}_balancing.nc<br/>Stage 2"]
```

A scenario with `market.enabled: true` goes down the market branch; any other goes to the single solve. The
intermediate files are named by what they contain, for example
`{id}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc`.

## Choosing what to run

`snakemake` with no target builds `rule all`: the outputs of every scenario in `run_scenarios` in
`config/config.yaml`.

```yaml
run_scenarios:
  - Test_Market
  - My_HT35_clustered
```

Three other ways to choose:

```bash
# one scenario, without editing config.yaml
snakemake -j 4 --config scenario=HT35_clustered

# one output file (and only what it needs)
snakemake -j 4 resources/network/HT35_clustered_solved.nc

# everything up to one rule, for the active scenarios
snakemake -j 4 --until finalize_network
```

Asking for a file is the most precise: Snakemake builds exactly the chain that file needs. Rule names are listed
by `snakemake --list-rules`.

`python config/config_loader.py --active` prints what `run_scenarios` selects, with family names expanded to
their months; `--list` prints every scenario id the loaded configuration defines.

## Dry runs

Always look before you run:

```bash
snakemake -n                 # the jobs, and why each would run
snakemake -n --quiet rules   # just the count per rule
```

Each job comes with its reason: output missing, an input newer than the output, code or parameters changed
since the last run. Read the reasons before a long run; they tell you whether you are about to re-solve something
you did not mean to.

## Cores and threads

```bash
snakemake -j 4        # run up to 4 cores' worth of jobs at once (same as --cores 4)
```

Rules declare how many threads they use. The single solve declares 4 and the market stages 2, because the barrier
method gains little beyond two threads on these problems; with `-j 4`, two market months solve side by side. The
solver's own thread count is `solve.solver.threads` (2 by default).

Memory, not cores, is usually the limit on the full ETYS network. The market rules declare their peak memory
(`mem_mb`), measured on the full-year Clean Power 2030 network; Snakemake enforces it only when you give it a
budget, for example `snakemake -j 8 --resources mem_mb=20000`. Otherwise lower `-j` if jobs are killed or the
machine starts swapping.

## When things re-run

By default Snakemake re-runs a job when an input file is newer than its output, or when the rule's code, its
parameters, its set of input files or its software environment have changed since the output was made.

Most model rules receive the scenario's whole merged configuration as one parameter, so a configuration change is
a parameter change. Two things keep this from re-solving more than it should (`config/config_loader.py`):

- **Key order does not matter.** The configuration is compared with its keys sorted, so moving or reformatting
  entries changes nothing.
- **Report-only blocks are ignored by build and solve rules.** Those rules take the configuration through
  `model_scenario_param`, which leaves out the blocks no build or solve script reads: `consumer_bill`,
  `public_ownership`, `validation`, `support.audit`, `source_data.version`, `description` and `logging`
  (`NOT_READ_BY_MODEL`). Editing them re-runs only the report rules that read them.

Any other change to a scenario's merged configuration re-runs the rules that take it (renewable and thermal
integration, storage, the solve and the market stages) and everything downstream, even if the rule does not use
the value you changed. Rules that take only a few settings re-run only when those change.

To see what has changed before deciding:

```bash
snakemake --list-changes params    # outputs whose rule parameters changed
snakemake --list-changes code      # outputs whose rule code changed
```

**Running on file times only.** `--rerun-triggers mtime` tells Snakemake to look only at file modification
times and ignore code and parameter changes:

```bash
snakemake -j 4 --rerun-triggers mtime
```

This is useful after a change you know cannot alter results (a comment or log message in a script, a
configuration edit for a scenario you are not running), when the default triggers would re-solve months of
dispatch. The cost is that a real change is not picked up either: after a change that should alter results,
force the affected rule with `-R <rule>` (or `--forcerun <rule>`), which also re-runs everything downstream.

```bash
snakemake -j 4 -R solve_network --config scenario=HT35_clustered
```

## Studies and month families

The larger studies live in `config/studies/<study>/` and are loaded only when `config/config.yaml` lists them:

```yaml
studies:
  - cp30_2030
  - validation_2020
```

The environment variable `PYPSA_GB_STUDIES` replaces that list without editing the file: a comma list of study
names, or `all` for every study folder (a study whose `study.yaml` says `draft: true` is left out unless it is
listed). The test suite sets `all`, so its checks of the studies run whatever `config.yaml` loads.

A study defines month families: one entry that becomes twelve scenarios, one per calendar month (for example the
family `no_acceleration` of `cp30_2030` becomes `CP30_FFR_2030_Jan_no_acceleration` to
`CP30_FFR_2030_Dec_no_acceleration`). A family name in `run_scenarios` runs its twelve months. Where two loaded
studies use the same family name, write `<study>:<family>` (for example `fes2050:market`).

The months of a family share work in two ways, and the workflow checks both when it parses the configuration:

- `network_from: <id>`: the months solve on one full-year network built once. The months may differ from it only
  in settings the build never reads (the solve period, the market and solver settings, the description).
- `wholesale_from: <id>`, set by a family's `stage1:` key: a balancing family takes its Stage 1 (wholesale) result
  from another family's same month instead of solving an identical wholesale market again. Only the balancing
  settings may differ.

```{warning}
Loading a study also adds its study-level checks and reports to `rule all` (for example the Clean Power 2030
balancing check, which reads the twelve solved months of several families), whatever `run_scenarios` says. List
under `studies` only the studies you mean to run, or ask for your scenario's output files by name. A study year is
a full-year ETYS model solved month by month: expect hours of solver time per family.
```

See {doc}`../model/scenarios_configuration` for how families, templates and `inherits` are written, and
{doc}`/studies/index` for the studies themselves.

## Logs and benchmarks

- **Rule logs** are in `logs/`, one file per job, named by the rule's `log:` entry: for example
  `logs/solve_network_<id>.log`, `logs/market/solve_wholesale_<id>.log`,
  `logs/market/solve_balancing_<id>.log`, `logs/FES/FES_data_<year>.log`. When a job fails, Snakemake prints its
  log path; start there.
- **Benchmarks** (run time and peak memory per job) are in `benchmarks/`, for example
  `benchmarks/solve/solve_<id>.txt`.

## Useful commands

```bash
snakemake -n                                   # dry run of rule all
snakemake -j 4 --keep-going                    # carry on with independent jobs after a failure
snakemake -j 4 --rerun-incomplete              # redo jobs left incomplete by an interrupted run
snakemake --unlock                             # clear the lock after a crash or a killed run
snakemake --rulegraph | dot -Tpng > rules.png  # draw the rule graph (needs graphviz)
```

## Next

{doc}`results`: what the outputs contain and how to open them.
