# Wholesale market (stage 1)

## What it does

Stage 1 represents the GB wholesale (day-ahead) market. GB has one national wholesale price: a
trade between two parties anywhere in GB clears at the same price whatever the grid between them can
carry. Stage 1 therefore meets demand at least cost **without internal transmission limits** (a
copperplate) and produces:

- a **position** (schedule) for every generator, storage unit and link, hour by hour; the balancing
  stage starts from these ({doc}`balancing`);
- an hourly **GB wholesale price**;
- an hourly **hydrogen value**, the price on the hydrogen bus, which the balancing stage uses to
  price electrolysers and hydrogen turbines.

Cross-border interconnectors are not relaxed: they trade against foreign prices up to their ratings
({doc}`/model/interconnectors`).

```{figure} /_static/figures/dispatch/stage1_week.png
:alt: Two stacked panels over one January week: Stage 1 generation by technology as stacked areas with demand as a line, and below it the GB wholesale price, which sits between 130 and 250 GBP/MWh while gas is marginal and falls to about zero once offshore wind rises.
:width: 100%

Stage 1 schedule and GB price over an example winter week (14-20 January) from the 2030 central case of the CP30 2030 study. Above zero: generation, storage discharge and imports; below zero: storage charging (with the churn sink) and exports; the line is GB demand including flexible demand and electrolysis. The price falls from gas-set levels to about zero as offshore wind rises on Friday.
```

| Rule | Script | Input |
|------|--------|-------|
| `solve_wholesale_market` | `scripts/market/solve_wholesale.py` | `resources/network/{scenario}.nc` from {doc}`single_solve` (`finalize_network`) |

A central-dispatch Stage 1 (`copperplate_mode: none`) runs through the twin rule
`solve_wholesale_market_central`, which is identical except that Snakemake schedules it first because
it is the slowest job in a run.

## Method and assumptions

### Steps before the solve

In order, `solve_wholesale.py`:

1. Applies any scenario-level re-pricing of the interconnectors and hydrogen system (off unless a
   study sets it), and checks the cost data as the single solve does.
2. Sets the **marginal-cost basis** (`marginal_cost_basis.mode`). `as_built`, the default, keeps the
   market-shaped costs the build wrote: ROC plant at minus its certificate value, CfD plant at its
   dispatch cost. `resource_cost` is for central dispatch ({doc}`central_dispatch`).
3. Sets the **curtailment order of zero-cost renewables** (`renewable_curtailment_order`, below).
4. Applies the optional **day-ahead forecast error** (`forecast_error`, below).
5. Makes the **copperplate** (`copperplate_mode`, below), adds the storage cost floor and the
   optional churn sink.
6. Runs the numerical conditioning of {doc}`single_solve`, restricts the snapshots to the solve
   period, caps load shedding at local demand, fixes must-take generation, sets the starting storage
   and hydrogen-store levels, and lifts island HVDC limits.
7. Adds the optional unit commitment overlay (`unit_commitment`), collapses the mainland, moves to
   coarser time blocks if `temporal_resolution_hours` > 1, and solves in the chosen `mode`.

Every generator-level step runs on the real buses before the mainland is collapsed, so for example
each load-shedding unit is capped at the demand of its own bus.

### The copperplate

`copperplate_mode` sets how the single GB price is produced:

| Mode | What it does |
|------|--------------|
| `collapse` (default) | The mainland synchronous network, which carries every load, is collapsed onto one bus just before the solve (`scripts/market/copperplate.py`). Islands, the external (foreign) buses and every link keep their identity and limits. Results are expanded back onto the full network for the saved file. |
| `relax` | Every line and transformer keeps its place with `s_nom` set to `transmission_relaxation` (1e6 MVA). The original ratings are kept in an `s_nom_original` column; transformer impedances are rescaled with the rating so the physics is unchanged. |
| `none` | No copperplate: Stage 1 solves on the real network with the balancing stage's network settings. This is central dispatch, not a market ({doc}`central_dispatch`). |

`collapse` and `relax` give the same optimum: with unlimited internal ratings every mainland bus
already shares one price, so the mainland is one node either way. `collapse` is exact, and much
faster and better conditioned, because it removes thousands of power-flow equations and bounds that
span nine orders of magnitude.

**Island HVDC.** The collapse keeps every link's rating, so without a further step the day-ahead
market would see the limits of the HVDC links to island systems (Shetland, for example) and curtail
island wind in Stage 1. With `relax_island_hvdc: true` every GB-internal link of
`island_hvdc_carriers` with an end off the mainland is lifted for Stage 1; the saved Stage 1 network
and the balancing stage keep the real ratings.

### Dispatch modes

`mode` sets how the solve period is split:

| Mode | What it does |
|------|--------------|
| `single` | One LP over the whole solve period: full foresight across it. |
| `rolling_day_ahead` (base default) | The period is solved in consecutive windows of `window_hours` (24). Each window sees only its own hours plus any look-ahead. With `carry_soc: true` storage starts each window where the last one ended. |

**Look-ahead.** Without it, a rolling window gives stored energy no value at its last hour, so
storage sells everything it holds by midnight at any price above its cost floor and starts the next
day by recharging. With `look_ahead_hours` $L$ > 0, window $k$ is optimised over

$$[t_k,\ t_k + W + L)$$

(cut at the end of the solve period) and only its first $W$ = `window_hours` are kept: their dispatch
and prices, and the storage state of charge, hydrogen-store level and reservoir level at the last kept
hour, which the next window starts from. End-of-day energy is then valued at the next day's prices
from the same LP. This is PyPSA's rolling-horizon overlap.

`look_ahead_resolution_hours` (6) solves the look-ahead hours as blocks of that many hours: every
time-varying input is the block mean and the block's snapshot weighting is its length, so energy
balances and storage levels carry the block's true duration. The kept hours stay hourly. A window of
24 kept hours with a 24-hour look-ahead in 6-hour blocks is 28 snapshots instead of 48, and keeps
nearly all of the hourly look-ahead's value to storage. Ramp limits act per snapshot, so inside a
look-ahead block a unit may move by one hour's ramp.

**Temporal resolution.** `temporal_resolution_hours` R > 1 solves the whole of Stage 1 on R-hour
blocks (`scripts/market/temporal_resolution.py`) and writes the results back hourly: power held
within each block, storage levels interpolated linearly inside it. The balancing stage then runs
hourly on the held positions. Windows and look-ahead must be whole blocks, and it cannot be combined
with Stage 1 unit commitment. The default is 1 (hourly).

### Storage and hydrogen across windows

- **Storage cost floor** (`storage_marginal_cost_floor`, GBP 1/MWh discharged): every storage unit
  with a lower dispatch cost is raised to it. Without it a barrier solve without crossover can charge
  and discharge the same unit at once in zero-price hours.
- **Month-boundary state of charge** (`market.storage_soc`): both stages start storage at the same
  level (`initial_fraction`, or the network's own). With `month_end: at_least_initial`, Stage 1's last
  window must end each unit at or above its starting level (capped at what it can recharge in that
  window), so a month cannot sell energy it was given for free.
- **Hydrogen store** (`market.hydrogen_store`): the hydrogen store level is carried across windows
  like state of charge, from `initial_fraction` of its capacity, with the same `month_end` rule.
  `end_of_window_value_gbp_per_mwh_h2` (off in the base) credits hydrogen left in the store at the
  end of each rolling window, so a day-ahead window values hydrogen it leaves for the next and
  electrolysers run whenever electricity is cheaper than efficiency times that value. The credit is
  objective-only and is removed from the reported objective.

### Which renewables are curtailed first

When supply exceeds demand, many renewables sit at the same cost and the barrier solver splits the
curtailment among them arbitrarily. `renewable_curtailment_order.mode` sets the order
(`scripts/market/curtailment_order.py`):

| Mode | Order |
|------|-------|
| `as_built` (base default) | Every CfD and merchant unit at its build cost (zero): an arbitrary tie. |
| `contract_aware` | Each unit backs off where its support terms make it: merchant renewables (`merchant_carriers`) at their cost plus `merchant_variable_cost_gbp_per_mwh` (0.5), so they go first; AR4 and later CfDs at `ar4_plus_bid_gbp_per_mwh` (0, because their contract pays whenever the price is not negative); AR1-AR3, Investment Contracts and Hinkley Point C at minus their strike, so only they and ROC plant take the price below zero. |

`contract_aware` needs the CfD strike table ({doc}`bids_offers_support`) and applies to the market
only: not under `copperplate_mode: none` or `marginal_cost_basis.mode: resource_cost`.

### Must-take generation

`market.must_take_generation.data_sources` lists generators, by the data source the build stamped on
them, that run at their availability in both stages (`scripts/market/must_take.py`): in Stage 1 their
minimum output equals their maximum hour by hour, so the market cannot curtail them, and the balancing
stage holds them at their Stage 1 output. It is used for rooftop solar, which no market curtails in
practice. The key sits at `market` level because both stages read it. The base list is empty.

### Churn sink (optional)

`churn_sink` adds a negative-only generator at each storage bus, sized at `share_of_storage_power`
of that bus's storage power and priced just above the AR4+ curtailment bid. It absorbs surplus energy
only where the market would otherwise curtail, so storage can carry its own cycling cost without
changing curtailment. It is a calibration device for studies that match a published market model
in which storage charges and discharges at the same time. The balancing stage holds it at its
Stage 1 output, and it is in no redispatch volume or cost. Off in the base.

### Scarcity pricing

A copperplate LP clears at the marginal unit's cost, so with ample capacity the price can never rise
above one plant's cost however tight the system is. Real GB prices rise as the de-rated margin
falls. `scarcity_pricing` adds a stepped reserve shortage curve to the LP
(`scripts/market/scarcity_pricing.py`):

- **Margin.** De-rated GB capacity less the demand it serves: conventional plant at its available
  capacity times a Capacity Market de-rating factor, wind and other variable renewables at their
  available output, storage and interconnectors by their flows or de-rated capacity (set in the
  curve file). Dispatch uses up margin one for one. Load shedding and the foreign markets
  (`excluded_carriers`) are left out.
- **Curve.** Below each threshold the LP buys shortfall at a rising price per MWh. Because every MW of
  demand uses up a MW of margin, the price of the deepest step in use becomes part of the energy
  price: the scarcity mark-up.
- **Calibration.** Thresholds and prices (`curve_file`) are fitted to the 2024 gap between N2EX
  day-ahead prices and the model, binned on NESO's published de-rated margin
  (`scripts/market/calibrate_scarcity_curve.py`), with the BSC Reserve Scarcity Price (loss of load
  probability times VoLL) below 1 GW. `max_drm_threshold_mw` drops the steps above a margin.

On by default; always off under central dispatch. Off as well in the 2020 validation study and the canonical
2020 market scenarios: the curve is fitted on 2024, and on the 2020 fleet with its monthly availability it read the
margin as tight often enough to clear about a sixth of August 2020's half-hours near GBP 1,400/MWh, against an ELEXON
market index price of about GBP 35/MWh. The curve is a 2024 calibration, not a general one.

### Internal HVDC tie-break

GB-internal HVDC links (the Eastern Green Links, the Western HVDC, the links to island hubs) are
lossless and free in both directions in the model. Beside the AC network their flow is a degenerate
tie: the LP can place any flow on them at the same cost. Both stages therefore add a small cost
$c$ = `market.internal_dc_tie_break_gbp_per_mwh` (GBP 0.75/MWh, a proxy for converter losses) on the
absolute flow of every internal DC link $l$ (`scripts/market/internal_dc_tie_break.py`):

$$p_{l,t} = f_{l,t} - r_{l,t}, \quad f_{l,t},\ r_{l,t} \ge 0, \qquad
\text{objective} \mathrel{+}= c \sum_{l,t} w_t \,(f_{l,t} + r_{l,t})$$

A plain marginal cost cannot do this on a two-way link, because reverse flow would then earn money.
Cross-border interconnectors and hydrogen links are not touched. In the collapsed copperplate the
mainland-internal links disappear, so in Stage 1 the term reaches only the links to islands. The cost
is **objective only**: it is stripped from the reported Stage 1 objective, recorded in the network's
metadata and reported on its own line (`internal_dc_tie_break_cost_GBP` in the market summary).

### Optional: forecast error

With `forecast_error.enabled`, Stage 1 schedules on a **forecast** rather than the outturn
(`scripts/market/forecast_error.py`): one national AR(1) error per technology
(`technologies`, standard deviation as a share of capacity) with time constant
`autocorrelation_hours`, shaped by $4\,\mathrm{CF}(1-\mathrm{CF})$ so there is no error where there is
no resource, plus a demand error (`demand_rmse`). The series is generated for the whole year from
`seed`, so a week and the month containing it see the same forecast. The balancing stage reloads the
network and sees the outturn, so its redispatch includes the cost of correcting the error. Off by
default.

### Optional: unit commitment

`unit_commitment.enabled` makes Stage 1 a mixed-integer schedule (minimum stable output, minimum up
and down times, start costs, ramp limits by carrier in `carrier_parameters`) while the balancing
stage stays an anchored LP (`scripts/market/wholesale_commitment.py`). Units below `min_p_nom_mw` and
the carriers in `exclude_carriers` are not committed. Off by default because it is much slower.

### Solving

The market stages solve through a **fallback ladder** (`market_utils.solve_with_fallback`): the
configured options, then a conservative barrier (NumericFocus 3, BarHomogeneous 1, ScaleFlag 2,
DualReductions 0), then dual simplex with crossover. A window that needs a later rung is logged as a
warning. `solve.solver.accepted_conditions` widens the termination conditions accepted from a rung
(the central-dispatch overlay accepts `suboptimal`), and `solve.solver.io_api` sets how the model is
handed to the solver (`direct` by default).

### The reported price

The GB price written to `{scenario}_wholesale_price.csv` is, for each hour, the mean of the marginal
prices at **GB demand buses** (buses with a load and `country` GB). `price_spread` is the highest
minus the lowest of those prices, which is about zero on a copperplate.

Only demand buses are used because other buses keep price differences even in a copperplate. The
buses of DC links (hydrogen turbines, electrolysers, internal HVDC) and of islands sit behind links
that keep their limits and directional costs, and their duals can be very large in either direction;
foreign buses carry foreign prices. Averaging over every bus would report a spread of tens of
thousands of GBP/MWh that no GB consumer pays. Under central dispatch the demand-bus prices are nodal
and do differ; the column is then their mean.

The same file carries `hydrogen_price`, the dual of the hydrogen bus, where the network has a
hydrogen system.

## Data

| Input | Use |
|-------|-----|
| `resources/network/{scenario}.nc` | The finalised network (shared by a year's monthly scenarios through `network_from`) |
| `resources/generators/{build}_marginal_costs_breakdown.csv` | The build's marginal-cost audit; strips subsidy layers under `resource_cost` |
| `data/market/scarcity/scarcity_curve_2024.json` | The scarcity curve |
| `data/market/lccc/cfd_strike_prices_by_round.csv` (through `support.labels.cfd_strike_table`, set in the CP30 2030 study's templates) | CfD strikes for `contract_aware` ordering |

Foreign prices and interconnector trading costs are set at build time
({doc}`/model/interconnectors`).

## Configuration

Stage 1 settings are under `market.wholesale` (`config/defaults/base/market/wholesale.yaml`); the
shared ones are under `market` (`config/defaults/base/market.yaml`).

| Key | Default | Meaning |
|-----|---------|---------|
| `market.enabled` | `false` | Run the two-stage market |
| `market.wholesale_only` | `false` | Run Stage 1 only (no balancing stage) |
| `market.dispatch` | `self` | `self` (market) or `central` ({doc}`central_dispatch`) |
| `market.wholesale.mode` | `rolling_day_ahead` | `single` or `rolling_day_ahead` |
| `market.wholesale.window_hours` | `24` | Hours kept per rolling window |
| `market.wholesale.carry_soc` | `true` | Carry storage state of charge between windows |
| `market.wholesale.look_ahead_hours` | `24` | Extra hours optimised and discarded (0 or null = none) |
| `market.wholesale.look_ahead_resolution_hours` | `6` | Block length of the look-ahead (null = hourly) |
| `market.wholesale.temporal_resolution_hours` | `1` | Stage 1 block length |
| `market.wholesale.copperplate_mode` | `collapse` | `collapse`, `relax` or `none` |
| `market.wholesale.transmission_relaxation` | `1.0e6` | Rating (MVA) in `relax` mode |
| `market.wholesale.relax_island_hvdc` | `true` | Lift island HVDC limits in Stage 1 |
| `market.wholesale.storage_marginal_cost_floor` | `1.0` | Minimum storage dispatch cost (GBP/MWh) |
| `market.wholesale.scarcity_pricing.enabled` | `true` | Scarcity curve in the LP |
| `market.wholesale.marginal_cost_basis.mode` | `as_built` | `as_built` or `resource_cost` |
| `market.wholesale.renewable_curtailment_order.mode` | `as_built` | `as_built` or `contract_aware` |
| `market.wholesale.forecast_error.enabled` | `false` | Schedule on a forecast |
| `market.wholesale.unit_commitment.enabled` | `false` | Mixed-integer Stage 1 |
| `market.wholesale.churn_sink.enabled` | `false` | Storage churn sink |
| `market.wholesale.central_constraint_screening.enabled` | `false` | Screened branch limits for a central Stage 1 |
| `market.storage_soc` | | Starting state of charge and month-end rule (both stages) |
| `market.hydrogen_store` | | Carried hydrogen-store level (both stages) |
| `market.must_take_generation.data_sources` | `[]` | Generators never curtailed or redispatched |
| `market.internal_dc_tie_break_gbp_per_mwh` | `0.75` | Cost on internal HVDC flow (both stages) |

A scenario that differs from another only in its balancing settings can reuse that scenario's Stage 1
with `wholesale_from: <scenario>`; the workflow refuses the declaration unless everything Stage 1
reads is identical.

## Validation

- **Historical years.** The wholesale notebook (`{scenario}_wholesale_notebook.ipynb`, for
  `wholesale_only` runs) compares the model price with the ELEXON market index price where it is
  available; the scarcity curve is fitted on 2024 prices.
- **CP30 2030.** `validate_CP30_stage1` (`scripts/CP30/stage1_validation.py`) runs as soon as a
  family's twelve Stage 1 months exist. It checks Stage 1 energy by technology against NESO's ES1
  table and the price distribution against NESO's CP.32 price curve, and fails the run if a main line
  is out of tolerance. Results are in {doc}`/studies/cp30_2030`.

## Outputs

| File | Content |
|------|---------|
| `resources/market/{scenario}_wholesale.nc` | Solved Stage 1 network on the full bus set, with real ratings |
| `resources/market/{scenario}_wholesale_dispatch.csv` | Generator positions (MW, snapshots x generators) |
| `resources/market/{scenario}_wholesale_storage.csv` | Storage positions (net MW, positive = discharge) |
| `resources/market/{scenario}_wholesale_links.csv` | Link and interconnector positions (MW at bus0) |
| `resources/market/{scenario}_wholesale_price.csv` | `wholesale_price`, `price_spread` and, where present, `hydrogen_price` (GBP/MWh) |

## Limitations

- **The copperplate is deliberate.** It describes how the GB market prices energy, not where power
  can flow. Stage 1 alone is not a feasible dispatch; the balancing stage makes it one.
- **No strategic bidding in Stage 1.** Units offer at their marginal cost (plus the scarcity mark-up
  and the curtailment order). Market power and contract positions other than support schemes are not
  represented.
- **Limited foresight, not real uncertainty.** Rolling windows with look-ahead see the outturn of the
  next day unless the forecast error is on.
- **Exogenous foreign prices.** Continental and Irish prices do not respond to GB.
