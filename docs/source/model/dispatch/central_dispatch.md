# Central dispatch

## What it does

Central dispatch replaces the market with a system operator that schedules the fleet itself, at
cost, on the real network. It answers a counterfactual question: on the same build, with the same
fleet, demand and weather, what would it cost to run the system if no one were paid a bid or an
offer, and scheduling saw the grid from the start?

A scenario chooses it with `market.dispatch: central`. The two-stage machinery stays
({doc}`wholesale`, {doc}`balancing`), but each stage means something else:

| Stage | Market (`market.dispatch: self`) | Central dispatch (`market.dispatch: central`) |
|-------|----------------------------------|-----------------------------------------------|
| Stage 1 | A copperplate market: one GB price, units scheduled on what they bid | The operator meets demand at least cost **on the real network**: nodal prices |
| Stage 2 | NESO buys bids and offers to make the schedule fit the network | The operator corrects its own schedule **at cost** |

Because Stage 1 already respects the network, Stage 2 finds little to redispatch unless something
differs between the two stages (a forecast error, a coarser look-ahead, the rolling windows). Its
constraint cost then comes out near zero by construction, rather than being assumed.

## Method and assumptions

### One switch, not five settings

Central dispatch needs several settings to move together: the copperplate off, the balancing stage
priced at cost, support-based bids off, renewable costs floored so that spilling subsidised wind
cannot count as income, and a solver profile that copes with the harder LP. Setting only some of them
silently gives a wrong answer. So `market.dispatch: central` attaches one overlay,
`config/defaults/dispatch/central.yaml`, merged after the method and source overlays and before the
scenario's own entry (base, `config.yaml` globals, method, source, dispatch, scenario). It sets:

| Key | Value under `central` | Why |
|-----|-----------------------|-----|
| `market.wholesale.copperplate_mode` | `none` | Stage 1 schedules on the real network |
| `market.wholesale.scarcity_pricing.enabled` | `false` | The scarcity mark-up is a rent generators earn when margins are tight; an operator instructing its own plant at cost pays none |
| `market.balancing.pricing_basis` | `at_cost` | Every unit offers and bids at its marginal cost: no default markup or discount, every `carrier_overrides` entry read as markup at zero (`scripts/market/pricing_basis.py`) |
| `market.balancing.mc_floors` | 0.01 for wind, solar, marine and hydro carriers | Supported renewables carry negative marginal costs that stand for subsidy-backed bidding; at cost, turning wind down would otherwise look like income |
| `support.bm_bids.enabled` | `false` | No support-based bids |
| `support.bm_bids.contract_value_bids.enabled` | `false` | Read by the balancing stage on its own, so switched off separately |
| `solve.solver.*` | NumericFocus 1, BarHomogeneous 1, ScaleFlag 2, BarIterLimit 400, `accepted_conditions: [optimal, suboptimal]` | A barrier profile for an LP that frees every unit against the full power flow; a solve stopped short of full convergence is accepted and logged |

### What the loader refuses

Five of these are fixed (`config_loader.CENTRAL_DISPATCH_FIXED`): the copperplate setting, scarcity
pricing, the balancing pricing basis, and both support-bid switches. `check_dispatch_mode` refuses a
central scenario whose own entry sets any of them to something else, and names the keys. A scenario
that wants, say, an operator paying a premium on top of cost should stay on `market.dispatch: self`
and set the keys itself. The renewable floors and the solver settings stay the scenario's choice.

Interconnector counter-trading is left alone ({doc}`interconnector_trading`): moving a cable off its
day-ahead schedule pays a foreign counterparty whoever scheduled GB's side, so both institutions
counter-trade on the same terms. Its foreign-plant multipliers borrow the CCGT markups, which the
at-cost basis sets to zero.

### Stage 1 on the real network

With `copperplate_mode: none`, `solve_wholesale.py` applies to Stage 1 the network preparations the
balancing stage applies, in the balancing stage's order: the network delivery case
(`market.balancing.network_delivery`), rating overrides and the voltage floor, the outage schedule or
secured boundary limits, and seasonal ratings ({doc}`/model/network/constraints`). The schedule is
made on the network the balancing stage will enforce. Island HVDC limits are not lifted.

The rolling windows, look-ahead, storage and hydrogen-store rules of {doc}`wholesale` are unchanged.
The reported Stage 1 price is still the mean over GB demand buses, but those prices are now nodal, so
`price_spread` is no longer near zero.

A network-constrained window LP over thousands of branches is slow and memory-hungry; Snakemake runs
these jobs through the twin rule `solve_wholesale_market_central`, first. Optionally,
`market.wholesale.central_constraint_screening` (`scripts/market/central_constraint_screening.py`)
solves each window with thermal limits on a monitored subset of branches, checks every branch, and
re-solves with the violated ones added until none is exceeded. The screened LP is the full LP with
some constraints left out, so when its optimum satisfies them all it is an optimum of the full LP:
same objective, and nodal prices up to degeneracy.

```{figure} /_static/figures/dispatch/central_price_map.png
:alt: Map of Great Britain with each demand bus coloured by its mean January Stage 1 price under central dispatch: blue (about 55 to 70 GBP/MWh) in Scotland, grey across northern England and orange (about 95 to 110 GBP/MWh) in the Midlands, the South and London.
:width: 70%
:align: center

Mean Stage 1 nodal price at each GB demand bus under central dispatch in January 2030 (2030 central case of the CP30 2030 study, central-dispatch twin). Grey is the market's single GB price for the same month; behind the constrained boundaries in the north power is worth less, beyond them in the south more.
```

### The resource-cost basis

The network's marginal costs are market-shaped: a ROC generator carries minus its certificate value,
because that is what it forgoes when curtailed; a CfD generator carries zero. A market schedules on
those, correctly, because that is what the plant bids. An operator minimising the cost of the system
does not: certificates paid within the system are transfers, not costs. Left as built, a central
dispatch would rank ROC wind ahead of CfD wind for reasons that are not resource costs.

`market.wholesale.marginal_cost_basis.mode: resource_cost` (`scripts/market/marginal_cost_basis.py`)
therefore sets, before the forecast error and the copperplate step:

- every renewable carrier to `renewable_floor` (GBP 0.01/MWh): spilled wind costs nothing, and a
  cost of exactly zero is degenerate;
- every other generator whose cost carries a subsidy layer to its fuel plus carbon plus variable
  O&M, read from the build's marginal-cost audit
  (`resources/generators/{build}_marginal_costs_breakdown.csv`);
- nuclear, thermal plant on formula costs, storage and load shedding: unchanged.

The basis is a scenario setting, not part of the overlay; the central-dispatch families of the CP30
2030 and long-range FES studies set it. The base keeps `as_built`.

**Support order.** With every renewable at one floor, which unit gives way in a surplus is a tie the
solver breaks arbitrarily, and it can cut supported plant while running merchant plant, paying
support for energy that is then not produced. `support_order` (on by default) spreads the floor by
the support a unit loses when turned down, $v_i$: a CfD unit's strike, a ROC unit's certificate value,
zero for merchant:

$$c_i = c_{\mathrm{floor}} + s\left(1 - \frac{v_i}{\max_j v_j}\right)$$

with $s$ = `spread_gbp_per_mwh` (GBP 1/MWh). Merchant plant sits at the top of the band and is turned
down first; the plant with the most support sits at the floor and is turned down last. Resource cost
among renewables is zero either way, so the spread only resolves the tie. A long-range FES network
carries no support labels on its generators, so its studies point `support_order.labels_file` at the
scenario's support-label table.

### Stage 2 at cost

The balancing stage runs as described in {doc}`balancing`, with every unit priced at its marginal
cost. What it corrects depends on what differs between the stages:

- **Forecast error** (`market.wholesale.forecast_error`): the operator schedules on a forecast and
  corrects on the outturn. Without it, central dispatch has perfect foresight and is an optimistic
  bound on its cost.
- **Commitment** (`market.balancing.commitment_proxy`): makes late starts and deep turn-downs costly,
  which a linear model otherwise treats as free.
- **Window structure**: the look-ahead blocks and rolling windows of Stage 1 against the balancing
  windows.

### Comparing with market dispatch

A central-dispatch scenario is used as the twin of a market scenario on the same build
(`network_from`), with the same fleet, network case, calibration and interconnector rules; only the
institution differs. The comparison then separates:

- **Resource cost**: fuel, carbon, variable O&M and payments abroad for cross-border energy, from both
  years' solved networks;
- **Transfers**: bid and offer payments, support payments and scarcity rents, which exist under the
  market and not under central dispatch;
- **Volumes**: redispatch, curtailment and interconnector flows.

The results are in {doc}`/studies/cp30_2030` and {doc}`/studies/fes_long_range`.

```{figure} /_static/figures/dispatch/central_vs_market_redispatch.png
:alt: Paired monthly bars of redispatch volume: market dispatch between 4 and 12 TWh a month, 88 TWh in the year, and central dispatch below 1 TWh a month, 7 TWh in the year.
:width: 100%

Redispatch volume (turn-up plus turn-down) by month in the 2030 central case of the CP30 2030 study and its central-dispatch twin. Central dispatch schedules on the real network, so its balancing stage corrects only what differs between the stages, such as the window structure.
```

## Data

Nothing beyond the market stages' own inputs. The resource-cost basis reads the build's marginal-cost
audit; the support order reads the CfD strike table named in `support.labels.cfd_strike_table` (set in the CP30 2030
study's templates), or `support_order.labels_file` (set in the long-range studies' templates).

## Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `market.dispatch` | `self` | `self` or `central` (attaches `config/defaults/dispatch/central.yaml`) |
| `market.wholesale.marginal_cost_basis.mode` | `as_built` | `resource_cost` for central dispatch |
| `market.wholesale.marginal_cost_basis.renewable_floor` | `0.01` | Renewable cost under `resource_cost` (GBP/MWh) |
| `market.wholesale.marginal_cost_basis.support_order.enabled` | `true` | Spread the floor by lost support |
| `market.wholesale.marginal_cost_basis.support_order.spread_gbp_per_mwh` | `1.0` | Width of the spread (GBP/MWh) |
| `market.wholesale.central_constraint_screening.enabled` | `false` | Screened branch limits in a central Stage 1 |
| `market.wholesale.forecast_error.enabled` | `false` | Schedule on a forecast |
| `market.balancing.commitment_proxy.enabled` | `false` | Linear commitment proxy in Stage 2 |

`python config/config_loader.py --explain <scenario_id> --path market` shows which file set each
value of a central scenario.

## Validation

- The loader refuses an inconsistent central scenario (above).
- Each window logs the solver's termination condition; `suboptimal` windows are accepted only under
  `solve.solver.accepted_conditions`.
- A central Stage 2 should find little to redispatch when the forecast error is off; a large
  constraint cost points to a difference between the networks the two stages see.
- The CP30 checks of {doc}`wholesale` and {doc}`balancing` run on central families as on market ones.

## Limitations

- **An idealised operator.** It has the model's own cost data, no market power, no gaming and, without
  the forecast error, perfect foresight within each window.
- **Linear.** No unit commitment beyond the optional proxy, so the cost of correcting errors is
  understated.
- **Same fleet.** It compares how the same plant is run; it does not ask what a central planner would
  have built.
