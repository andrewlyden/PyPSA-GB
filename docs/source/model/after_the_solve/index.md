# After the solve

What the workflow computes from a solved network: the results tables and dashboards of every run, the market
accounts of a two-stage run, the sum of a twelve-month study, the checks against outturn and published targets, and the
household electricity bill.

| Page | What it covers | Main rules |
|---|---|---|
| {doc}`results` | Results tables, summaries, dashboards and notebooks; the market accounts; a study's year from its twelve months | `solve_network`, `analyze_and_visualize_solved_network`, `analyze_market_results`, `aggregate_CP30_months` |
| {doc}`validation` | Every check of a solved run: against ELEXON and NESO for historical years, against CP30 and FES for future ones | `validate_bm_results`, `validate_neso_constraints`, `validate_CP30_stage1`, `validate_CP30_dispatch`, `validate_CP30_balancing`, `validate_fes_dispatch` |
| {doc}`consumer_bill/index` | The household electricity bill built from a solved market year | `project_future_consumer_bill` and the bill rules |

None of these steps changes a solved network, and none triggers a solve. The checks that run while a network is being
built (capacities, placement, available energy) are on the pages of the build steps ({doc}`/model/index`).

For where the files are and how to open them, see {doc}`/getting_started/results`; for every output, see
{doc}`/reference/outputs`.

```{toctree}
:maxdepth: 2

results
validation
consumer_bill/index
```
