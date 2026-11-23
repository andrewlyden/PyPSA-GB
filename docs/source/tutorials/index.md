# Tutorials

Five Jupyter notebooks that take you from a first run to the two-stage market. They follow one week, 10 to 16 February,
through three years: 2020 as it happened, 2030 on NESO's Clean Power 2030 pathway, and 2040 on FES 2025's Holistic
Transition. Each notebook says how to run its scenario, reads the result, explains what it shows, and ends with
changes to try. The pages below keep the notebooks' outputs, so they can be read without running anything.

```{nbgallery}
1-first-run-2020
2-network-models
3-demand
4-future-2040-holistic-transition
5-two-stage-market-2030
```

## Learning path

| # | Notebook | What you learn | Scenario | Run time |
|---|----------|----------------|----------|----------|
| 1 | {doc}`1-first-run-2020` | Run a scenario; what a solved network holds; how demand was met; the model against the real week; prices and full lines | `Tutorial_2020_reduced` | about 5 minutes |
| 2 | {doc}`2-network-models` | The four network models side by side; what detail changes; coordinates; clustering | `Tutorial_2020_zonal`, `_reduced`, `_clustered`, `_etys` | minutes; full ETYS 30 to 60 minutes |
| 3 | {doc}`3-demand` | How demand is built in each year; flexible demand in 2030; where demand is | reads tutorials 1, 4 and 5 | none |
| 4 | {doc}`4-future-2040-holistic-transition` | A FES scenario; the 2040 fleet and week against 2020; storage and hydrogen; curtailment and prices | `Tutorial_2040_HT_clustered` | 10 to 20 minutes |
| 5 | {doc}`5-two-stage-market-2030` | The wholesale market and the balancing mechanism; constraint costs; where wind is turned down; interconnector re-trading | `CP30_FFR_2030_tutorial_week` | about 40 minutes the first time |

The scenarios are in `config/studies/tutorials/scenarios.yaml`, except tutorial 5's, which is part of the CP30 study
(`config/studies/cp30_2030/scenarios.yaml`) because it is solved on that study's network and market settings.

## Before you start

1. Install the model and a solver ({doc}`../getting_started/installation`). The notebooks run in the `pypsa-gb`
   environment, which includes Jupyter.
2. Check that `config/config.yaml` loads the tutorial scenarios, and the CP30 study for tutorial 5:

   ```yaml
   studies:
     - tutorials
     - cp30_2030
   ```

3. Run a tutorial's scenario from the repository root before opening its notebook, as each notebook's "Run it"
   section says, for example:

   ```bash
   snakemake -j 4 --config scenario=Tutorial_2020_reduced
   ```

   The first run of a weather year fetches its weather data (2020 for tutorials 1 and 2, 2013 for 4 and 5), which
   adds time. Tutorial 5 and the full ETYS run in tutorial 2 need about 16 GB of memory and are slow without Gurobi;
   the others run with HiGHS on a laptop.
4. Start Jupyter and open the notebooks in `docs/source/tutorials/`:

   ```bash
   conda activate pypsa-gb
   jupyter lab
   ```

   Each notebook finds the repository root itself, so it runs from any folder.

## Going further

The notebooks show how to run the model and read its results. For complete applications, with their setup, how to
reproduce them and their results, see the {doc}`../studies/index` pages: the 2020 validation, Clean Power 2030 and the
FES long-range years.
