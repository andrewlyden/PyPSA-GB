# Getting Started

From a fresh clone to a solved scenario and its results. Work through the pages in order the first time.

```{toctree}
:maxdepth: 2

installation
quickstart
first_scenario
running_the_workflow
results
ai_assistants
```

## What you need

- Git, and conda or mamba ([Miniforge](https://github.com/conda-forge/miniforge) is the simplest way to get both).
- An internet connection on the first run: the workflow downloads the weather cutout for each weather year
  (about 700 MB each) and the NESO data a scenario reads.
- A linear programming solver: Gurobi (the default; free academic licences) or HiGHS (open source, installed with
  the environment).
- Memory to suit the network: the Reduced (32-bus) and Zonal networks run on a laptop; the full ETYS network
  (about 2,000 buses) needs 8 GB of RAM or more.

## The pages

| Page | What you do |
|---|---|
| {doc}`installation` | Clone the repository, create the environment, set up a solver, and learn what the workflow downloads. |
| {doc}`quickstart` | Run `Test_Market`, six hours of 2035 on the Reduced network through the wholesale market and the balancing mechanism. |
| {doc}`first_scenario` | Write your own scenario in `config/scenarios.yaml`, check it, and run it. |
| {doc}`running_the_workflow` | How the Snakemake workflow is organised: targets, dry runs, cores, re-runs, studies and logs. |
| {doc}`results` | Open the solved networks and read the main CSV and JSON summaries. |
| {doc}`ai_assistants` | Use the site's Ask AI button, or a coding assistant in your clone, to write scenarios, debug runs and read results. |

If something goes wrong, see {doc}`../development/troubleshooting` or open a
[GitHub issue](https://github.com/andrewlyden/PyPSA-GB/issues).
