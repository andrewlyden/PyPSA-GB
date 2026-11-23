# Historical validation: 2020

## Question

Before PyPSA-GB is used to estimate constraint costs in 2030 and beyond, it should reproduce a year that has happened.
This study solves calendar 2020 month by month as a two-stage market on the full ETYS network and compares it with
outturn:

- **Constraint cost and volume** against NESO's thermal constraint costs and volumes.
- **Boundary flows** against NESO's day-ahead flows on the six boundaries NESO reports.
- **Dispatch** by technology against ESPENI (half-hourly GB generation by fuel).
- **Wholesale price** against ELEXON's Market Index Data (MID) price.

2020 is a demanding test year. Demand fell sharply during the COVID-19 lockdowns, wind output was high, two storms
crossed GB in August, and the Western HVDC link was out of service for most of January.

## Setup

The study is `config/studies/validation_2020/`. Its `validation` family expands to one scenario per month,
`Validation_<Mon>2020` (for example `Validation_Jan2020`), each solved over its own calendar month at a 30-minute time
step.

| Part | Choice | Model page |
|---|---|---|
| Fleet | DUKES thermal plant, REPD renewables and storage, CfD and ROC units kept apart. Stations and offshore wind farms sit on their declared connection buses (`data/generators/historical_station_buses.csv`, `data/network/ETYS/repd_offshore_landing_crosswalk.csv`). | {doc}`/model/generation/thermal`, {doc}`/model/generation/renewables` |
| Availability | Nuclear stations from their ELEXON notified output by month; Peterhead held at its notified output; other thermal units with random outages | {doc}`/model/generation/thermal` |
| Hydro | Large-hydro inflow is ESPENI's metered hydro output, day by day | {doc}`/model/generation/renewables` |
| Demand | ESPENI 2020 | {doc}`/model/demand/historical` |
| Network | Full ETYS network (about 2,100 buses) with NESO's day-ahead limits on SCOTEX, SSE-SP, SSHARN, ESTEX, SEIMP and SWALEX. SSHARN is the ETYS B7a cut, the AC circuits and the Western link. The Western link is out from 10 January to 8 February (`network.transmission.declared_link_outages`). | {doc}`/model/network/topology`, {doc}`/model/network/constraints` |
| Weather | 2020, with monthly wind and solar performance factors fitted to outturn | {doc}`/model/generation/renewables` |
| Market | Stage 1 rolling day ahead, without scarcity pricing. The balancing mechanism runs in rolling 24-hour windows with ELEXON price ladders for bids and offers. Embedded wind and solar are not balancing-mechanism units: they stay at their Stage 1 output (`participation.generators.hold_carriers`). | {doc}`/model/dispatch/wholesale`, {doc}`/model/dispatch/balancing` |

The benchmark is NESO's **total thermal constraint cost** (constraint breakdown, GBP 596 m in 2020). It covers every
thermal constraint, local ones included, the same scope as the full network the model enforces. NESO's cost by
constraint group covers only the six named boundaries (GBP 450 m), and is shown as a secondary series.

## How to reproduce

```yaml
# config/config.yaml
studies:
  - validation_2020
run_scenarios:
  - validation          # Validation_Jan2020 ... Validation_Dec2020
```

Run `snakemake -n`, then `snakemake -j 6` ({doc}`/getting_started/running_the_workflow`). You need the 2020 weather
cutout and a Gurobi licence. ELEXON data come from the ELEXON API, or from the compressed calibration cache
(`python scripts/utilities/elexon_cache.py restore`). Solving six months at a time, the year takes about three hours on
a 12-core laptop with 32 GB of memory; more months at once can run out of memory. Then:

```bash
python scripts/analysis/validation_volume_cost.py
python scripts/analysis/review_completed_validation_months_2020.py
```

| Output | What it holds |
|---|---|
| `resources/analysis/Validation_<Mon>2020_market_summary.json` | Each month's costs, volumes and redispatch by technology |
| `resources/market/Validation_<Mon>2020_neso_validation.csv`, `resources/analysis/Validation_<Mon>2020_neso_validation.html` | The month against NESO: cost, volume and boundary flows |
| `resources/analysis/validation_2020_volume_cost.csv` | Cost, volume and cost per MWh by month against NESO, and turn-down by technology against ELEXON |
| `resources/analysis/validation_2020_completed_months_scorecard.csv` | The monthly scorecard, with prices against ELEXON MID |
| `resources/analysis/validation_2020_completed_months_dispatch_vs_espeni.csv` | Dispatch by technology against ESPENI |

## Results

| Month | Model cost, GBP m | NESO, GBP m | Ratio | Model turn-down, GWh | NESO volume, GWh | Ratio | Model price, GBP/MWh | ELEXON MID, GBP/MWh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jan | 104.2 | 72.6 | 1.43 | 1,193 | 736 | 1.62 | 35.30 | 34.68 |
| Feb | 108.3 | 76.2 | 1.42 | 1,284 | 779 | 1.65 | 32.43 | 29.49 |
| Mar | 43.9 | 40.5 | 1.08 | 696 | 461 | 1.51 | 28.86 | 28.50 |
| Apr | 24.8 | 10.7 | 2.32 | 336 | 123 | 2.74 | 24.00 | 23.05 |
| May | 33.8 | 34.8 | 0.97 | 389 | 328 | 1.19 | 21.32 | 21.97 |
| Jun | 45.9 | 33.4 | 1.37 | 506 | 473 | 1.07 | 28.55 | 25.51 |
| Jul | 43.3 | 40.3 | 1.07 | 666 | 453 | 1.47 | 28.08 | 27.77 |
| Aug | 50.1 | 17.4 | 2.88 | 500 | 179 | 2.80 | 33.46 | 35.24 |
| Sep | 45.6 | 40.1 | 1.14 | 544 | 320 | 1.70 | 39.09 | 43.15 |
| Oct | 57.2 | 54.2 | 1.05 | 544 | 453 | 1.20 | 40.54 | 41.48 |
| Nov | 115.3 | 111.1 | 1.04 | 1,131 | 887 | 1.28 | 36.98 | 41.48 |
| Dec | 96.3 | 64.8 | 1.49 | 942 | 418 | 2.25 | 45.14 | 54.05 |
| **Year** | **768.5** | **596.1** | **1.29** | **8,729** | **5,609** | **1.56** | 32.81 | 33.86 |

<!-- resources/analysis/validation_2020_volume_cost.csv: model_cost_gbp_m, neso_cost_gbp_m (NESO
constraint breakdown, thermal), cost_ratio, model_volume_gwh (one-sided redispatch), neso_volume_gwh (NESO thermal
volume, daily absolute), volume_ratio. Prices: resources/analysis/validation_2020_completed_months_scorecard.csv
mean_smp_gbp_mwh and mean_mid_gbp_mwh; the year row is the mean of the months. Model cost includes GBP 4.0 m of
penalty-priced turn-down of held embedded units (market summaries, non_participant_turn_down_cost_at_penalty_GBP). -->

```{figure} /_static/figures/studies/validation_2020_monthly.png
:alt: Three panels by month of 2020. Top, the model's constraint cost beside NESO's total thermal constraint cost, with ticks for NESO's six-boundary cost; the model is close in March, May, July, September, October and November and above in January, February, April, August and December. Middle, the model's turn-down beside NESO's thermal constraint volume, the model higher in every month. Bottom, the model's wholesale price beside ELEXON's MID price, the two lines close all year.
:width: 100%

Constraint cost (top) and volume (middle) against NESO, and the wholesale price against ELEXON's MID price (bottom),
by month of 2020.
```

What the comparison shows:

- **Over the year the model's constraint cost is 1.29 times NESO's** (GBP 768 m against 596 m). Six months are within
  15%: March, May, July, September, October and November, which was the costliest month for both (GBP 115 m against
  111 m).
- **The flows across the Scottish boundaries match.** On SSHARN, the costliest boundary of 2020, the model's mean flow
  is 0.93 to 1.08 of NESO's day-ahead flow in the costly months, and it is near its limit for a similar number of
  hours. In November, SSHARN is above 90% of its limit for 762 hours in the model and 597 in NESO's data.
- **The model turns down more than NESO did, 1.56 times the volume over the year.** It pays less per MWh (GBP 88
  against 106). The largest gaps are April, August and December. August's excess is mostly on four storm days: Storm
  Ellen (19 to 21 August) and Storm Francis (25 August) account for most of its cost.
- **Wholesale prices track outturn.** The monthly model price is within 17% of ELEXON's MID price in every month; over
  the year the mean of the months is GBP 32.81/MWh against 33.86. The largest gap is December (-16%).

<!-- Boundaries: resources/analysis/validation_2020_completed_months_neso_boundaries.csv, SSHARN rows for Jun, Jul,
Oct, Nov, Dec (flow_ratio 0.95, 1.00, 1.08, 0.93, 0.95; Nov model_hours_above_90 762, neso_hours_above_90 597).
Storm days: per-window BM costs in logs/market/solve_balancing_Validation_Aug2020.log. -->

### Dispatch against ESPENI

Model energy over ESPENI energy, by month:

| Technology | Range over the twelve months | Year |
|---|---|---|
| Demand | 1.01 to 1.04 | 1.02 |
| Nuclear | 0.95 to 1.00 | 0.98 |
| Wind | 0.98 to 1.06 | 1.03 |
| Solar | 1.00 to 1.12 | 1.02 |
| CCGT | 0.87 to 0.96 | 0.91 |
| Biomass | 0.90 to 1.11 | 0.97 |
| Hydro (non-pumped) | 0.58 to 1.23 | 0.83 |
| Pumped storage, discharge | 0.09 to 0.74 | 0.45 |
| Coal | 1.00 | 1.00 |

<!-- resources/analysis/validation_2020_completed_months_dispatch_vs_espeni.csv: ratio by comparison,
minimum and maximum over the months; year = sum of model_mwh over sum of espeni_mwh. -->

Demand, nuclear and coal match outturn because they are inputs or held to their 2020 notified output; wind and solar
are calibrated by the monthly performance factors. Gas runs 9% less than it did over the year. Hydro receives
ESPENI's metered output as inflow, and runs below it because the balancing stage turns it down (below). Pumped storage
cycles less than in reality: a cost-minimising dispatch with perfect foresight has little reason to cycle storage that
the real market used for reserve, inertia and hedging.

### Turn-down by technology

The model turns down 4.8 TWh of wind against 3.6 TWh in ELEXON's accepted bids, and 1.4 TWh of large hydro against
0.3 TWh. ELEXON's figures include energy-balancing actions as well as constraints. Hydro is the cheapest turn-down in
the model, as it is in the real bids: most stations bid GBP 2.5 to 5/MWh in January 2020. NESO took far less of it,
probably because of hydro stations' operating limits in the balancing mechanism, which the model does not represent.

<!-- resources/analysis/validation_2020_volume_cost.csv year row: model_wind_down_gwh 4,777, elexon_wind_down_gwh
3,621, model_hydro_down_gwh 1,355, elexon_hydro_down_gwh 323. Hydro bids: resources/market/Validation_Jan2020/elexon/
elexon_bids.csv, median of nonzero hours by BMU (T_ERRO 2.5, T_SLOY 5.0, E_CLAC 2.5). -->

## Limitations

- **Perfect foresight.** Each balancing window knows its day in advance, with no forecast error. NESO's day-ahead
  limits are themselves forecasts, and real-time limits were often looser, which the model cannot see. Both push the
  model's volume up.
- **Only bids and offers.** NESO also relieved B6 and B7a with intertrips, constraint-management contracts and other
  services outside the balancing mechanism, especially in storms. The model has no equivalent, which is the main reason
  for August's excess.
- **Hydro operating limits.** Minimum run times and stable export limits of hydro stations in the balancing mechanism
  are not modelled, so hydro is turned down more than it was.
- **The Western link's availability.** Its January outage is modelled. Any other restriction in 2020 is not, because
  no hourly record of its availability is published (it is not a balancing-mechanism unit).
- **Thermal limits only.** Voltage, stability and inertia actions, a large part of NESO's 2020 costs during the low
  demand of the lockdowns, are not modelled; the comparison is with NESO's thermal constraint costs only.
- **Monthly performance factors are fitted to outturn**, so wind and solar energy agree by construction; the test is
  of where and when they run, not how much.
- **No scarcity pricing.** The Stage 1 scarcity curve is calibrated on 2024 and reads 2020's fleet as tight; it is
  switched off for this study.
- **A validation, not a calibration.** Every change made to reach these results corrects an input to match a
  documented fact (connection buses, metered hydro, the link outage, which units are in the balancing mechanism). None
  of them is fitted to NESO's costs.
