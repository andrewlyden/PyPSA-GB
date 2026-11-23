# Validation

Every check of a solved run, in one place: what it compares, against which data, how it passes, and whether a failure
stops the workflow. Historical years are checked against what happened (ELEXON and NESO); future years against the
publication they are built from (CP30 or FES). The results of the checks are on the study pages
({doc}`/studies/validation_2020`, {doc}`/studies/cp30_2030`, {doc}`/studies/fes_long_range`).

## What it does

| Rule (script) | Runs on | Compares | Fails the run? |
|---|---|---|---|
| `validate_bm_results` (`scripts/market/validate_bm.py`) | historical market scenarios | balancing volumes, costs, prices and dispatch against ELEXON | no: ratios only |
| `validate_bm_calibration` (`scripts/market/bm_calibration_scorecard.py`) | historical market scenarios priced from ELEXON | bid and offer prices and volumes by technology against ELEXON | only if `fail_on_regression` |
| `validate_neso_constraints` (`scripts/market/validate_neso_constraints.py`) | historical market scenarios | constraint cost, volume and boundary flows against NESO | no: ratios only |
| `validate_CP30_stage1` (`scripts/CP30/stage1_validation.py`) | a CP30 family's twelve Stage 1 months | Stage 1 energy against CP30 ES1, and price against CP.32 | yes, on FAIL |
| `validate_CP30_dispatch` (`scripts/CP30/validate_cp30_dispatch.py`) | a CP30 study year | energy over the year against ES1; clean power turned down against CP.08 | yes, on FAIL |
| `validate_CP30_balancing` (`scripts/CP30/balancing_validation.py`) | a CP30 study year | balancing volumes, cost, final dispatch and turn-down by zone against NESO's published figures | never |
| `validate_fes_dispatch` (`scripts/FES/fes_dispatch_validation.py`) | a FES long-range study year | energy over the year against FES ES1, hydrogen against FES | yes, on FAIL or an unexplained WARN |

Rules `diagnose_fes_ic1` and `measure_fes_bm_storage_churn` add two reports to a FES long-range year (interconnector
flows and spreads in January and July; storage charging and discharging at once in the balancing stage); rule
`fes_long_range_validation` runs all three. Rule `audit_secured_capability` reads only the configuration and reports
the boundary limits each CP30 family's balancing stage enforces ({doc}`/model/network/constraints`).

## Method and assumptions

### Historical years: against outturn

`validate_bm_results` sets the model's balancing stage beside ELEXON's settlement data for the same period: accepted
volumes (BOAV), acceptances by flag (BOALF), system buy and sell prices, generation by unit (B1610) and disaggregated
balancing services (DISBSAD), with the Market Index Data price for the wholesale stage. Each row of
`resources/market/{scenario}_bm_validation.csv` gives the model value, ELEXON's, the unit and their ratio; the page
`resources/analysis/{scenario}_bm_validation.html` shows the same. The total balancing cost is also set against NESO's
published annual balancing cost for 2019-2024.

`validate_bm_calibration` checks the prices and volumes behind that result, technology by technology: each
technology's effective offer and bid price against the median ELEXON bid-offer price of its mapped units, and its
volumes against ELEXON's accepted volumes. A priority technology with at least `min_volume_mwh` of redispatch is
flagged when its ratio is more than `tolerance` from 1.

`validate_neso_constraints` compares with NESO's thermal constraint data: the cost against NESO's six-boundary
constraint cost and its total thermal constraint cost, the volume against NESO's thermal volume, and the flow on each
boundary against NESO's day-ahead flows and limits. Boundaries are mapped from the ETYS network by
`data/network/neso_boundary_mapping.yaml`. A clustered network that keeps fewer than three quarters of the mapped
boundaries' circuits gets a `not_applicable` report instead.

### CP30 2030: against the publication

CP30's energy table, ES1, comes from NESO's market model without network limits, so the wholesale stage (Stage 1) is
the stage judged against it.

`validate_CP30_stage1` runs once a family's twelve Stage 1 months are solved:

| Status | When |
|---|---|
| FAIL | a main generation line more than 15% from ES1, interconnectors more than 20%, demand more than 0.5%, or an ES1 line the model cannot map |
| WARN | any generation line more than 10% off; curtailment more than 25%; interconnectors or electrolysis more than 10%; the price mean or a point of the price duration curve more than 25% from CP.32 (a gap under GBP 5/MWh never warns); the share of zero-price hours more than 0.10 off |
| INFO | storage, whose cycling in NESO's model a least-cost dispatch does not reproduce |

A line under 1 TWh in ES1 is judged on its absolute gap. On a FAIL the rule writes its outputs and a copy marked
`_failed_run`, then stops.

`validate_CP30_dispatch` sums the twelve months of a study year. It judges energy, curtailment, storage and
interconnectors on Stage 1 (dispatch WARN beyond 10%, curtailment beyond 25%, demand FAIL beyond 0.5%) and reports the
final, post-balancing values for information, with clean power turned down in the balancing stage set against CP.08's
bid volume.

`validate_CP30_balancing` compares a study year with NESO's own balancing figures for CP30
(`data/CP30/validation_references/neso_balancing_benchmarks.csv`): volumes against CP.18 (within 15%), the
NESO-comparable cost against NESO's constraint cost for that network (15%; 20% for the delay cases), final dispatch
against N2 (5%) and the unabated gas share (1 percentage point), and turn-down by NESO's 17 CP.08 zones (5 percentage
points; Scotland's share 15%). It reports PASS, WARN or INFO and never fails.

### FES long-range years: against FES

`validate_fes_dispatch` compares a year's final dispatch, with the Stage 1 value beside each row, against FES 2025's
ES1 energy table, and hydrogen against FES's hydrogen tables. It uses the CP30 dispatch tolerances, and it fails on an
unmapped line, a demand gap, or any WARN without a coded explanation, so every known difference is named.

## Data

| Input | Used by |
|---|---|
| ELEXON BOAV, BOALF, system prices, B1610, DISBSAD, MID (fetched and cached under `resources/market/elexon/`) | `validate_bm_results`, `validate_bm_calibration` |
| NESO thermal constraint costs, constraint breakdown, day-ahead constraint flows and limits (cached in `data/validation/`) | `validate_neso_constraints` |
| CP30 ES1 and the CP.32 price curve (`data/CP30/cp32_price_duration_curve_2030.csv`) | `validate_CP30_stage1`, `validate_CP30_dispatch` |
| NESO's CP30 balancing figures and zones (`data/CP30/validation_references/`) | `validate_CP30_balancing`, `validate_CP30_dispatch` |
| FES 2025 data workbook (`data/FES/`) | `validate_fes_dispatch` |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `market.balancing.calibration.tolerance` | `0.20` | Calibration flag threshold |
| `market.balancing.calibration.min_volume_mwh` | `1000` | Minimum redispatch for a technology to be judged |
| `market.balancing.calibration.fail_on_regression` | `false` | Fail the rule on a flagged technology |
| `validation.stage1` (`config/defaults/source/CP30.yaml`) | see above | Stage 1 tolerances (equal to `DEFAULT_CONFIG` in `stage1_validation.py`) |
| `validation.balancing` (`config/defaults/source/CP30.yaml`) | see above | Balancing-check tolerances |
| `balancing_validation_families` (`config/studies/cp30_2030/study.yaml`) | four network cases | Families whose balancing check joins `snakemake all` |

## Limitations

- **Historical checks report; they do not judge.** The ELEXON and NESO comparisons give ratios without thresholds;
  {doc}`/studies/validation_2020` reads them.
- **Future checks are against another model.** CP30 and FES are NESO's own model results, not outturn; agreement shows
  that PyPSA-GB reproduces the publication's system, not that either is right.
- **The 2020 study's summary tables** (cost and volume by month, the scorecard) are built by
  `scripts/analysis/validation_volume_cost.py` and `scripts/analysis/review_completed_validation_months_2020.py`, run
  by hand on the rules' outputs.
