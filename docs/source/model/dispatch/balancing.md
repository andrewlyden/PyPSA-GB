# Balancing mechanism (stage 2)

## What it does

Stage 2 represents NESO's balancing mechanism (BM). The Stage 1 schedule ignores the grid inside GB,
so it is usually not physically possible: too much generation behind a congested boundary, too
little beyond it. Stage 2 reloads the network with its real ratings and moves units off their
Stage 1 positions until the dispatch fits, at least cost to the system operator. A unit turned up is
paid its **offer**; a unit turned down pays or is paid its **bid**. The cost of those moves is the
model's constraint cost, and the volumes show where and why the network binds.

```{figure} /_static/figures/dispatch/balancing_week.png
:alt: Stacked areas above and below zero over one January week showing generation turned up and turned down by technology in the balancing stage; from Friday about 10 to 15 GW of offshore and onshore wind is turned down and replaced mainly by reduced exports and gas.
:width: 100%

The balancing stage's moves off the Stage 1 schedule shown in {doc}`wholesale`, over an example winter week (14-20 January) from the 2030 central case of the CP30 2030 study. When wind rises on Friday, Stage 1 schedules more wind and more exports than the network can carry; the balancing stage turns wind down behind the constraints and turns up gas and imports (largely by cutting exports) beyond them. Turn-up and turn-down balance hour by hour.
```

| Rule | Script | Inputs |
|------|--------|--------|
| `solve_balancing_mechanism` | `scripts/market/solve_balancing.py` | The finalised network and the Stage 1 positions and price ({doc}`wholesale`) |
| `analyze_market_results` | `scripts/market/analyze_market.py` | Both stages' networks and tables |

How bids and offers are priced is on {doc}`bids_offers_support`; how cross-border cables are treated
is on {doc}`interconnector_trading`.

## Method and assumptions

### Anchoring to Stage 1

Every participating generator $g$ is tied to its Stage 1 position $\hat p_{g,t}$ by two non-negative
variables, an increase $u_{g,t}$ (an accepted offer) and a decrease $d_{g,t}$ (an accepted bid):

$$p_{g,t} = \hat p_{g,t} + u_{g,t} - d_{g,t}, \qquad
0 \le u_{g,t} \le \bar p_{g,t} - \hat p_{g,t}, \qquad 0 \le d_{g,t} \le \hat p_{g,t} - \underline p_{g,t}$$

where $\bar p$ and $\underline p$ are the unit's available maximum and minimum output in that hour.

A storage unit is anchored on each leg separately, discharge and charge, each with its own increase
and decrease:

$$p^{\mathrm{dis}}_{s,t} = \hat p^{\mathrm{dis}}_{s,t} + u^{\mathrm{dis}}_{s,t} - d^{\mathrm{dis}}_{s,t}, \qquad
p^{\mathrm{sto}}_{s,t} = \hat p^{\mathrm{sto}}_{s,t} + u^{\mathrm{sto}}_{s,t} - d^{\mathrm{sto}}_{s,t}$$

More discharge or less charging is paid at the unit's offer; less discharge or more charging at its
bid. Anchoring the net position alone would leave the split behind it free, and a unit that charges
and discharges in the same hour destroys energy through its losses: the BM would use storage as a
cheap dump instead of paying wind to turn down. With both legs anchored, a simultaneous pair pays an
offer and a bid and is never worth it. Stage 1 saves only net storage output, so by default its legs
are recovered by sign (positive is discharge); a study can set `market.balancing.storage_anchor:
stage1_legs` to anchor on the gross legs Stage 1 saved on its network.

### The objective

PyPSA's cost objective is replaced by the cost of the moves:

$$\min \sum_t h_t \Big[ \sum_g \big(o_{g,t}\, u_{g,t} + b_{g,t}\, d_{g,t}\big)
+ \sum_s \big(o_{s,t}\,(u^{\mathrm{dis}}_{s,t} + d^{\mathrm{sto}}_{s,t}) + b_{s,t}\,(d^{\mathrm{dis}}_{s,t} + u^{\mathrm{sto}}_{s,t})\big) \Big]
+ C^{\mathrm{IC}} + C^{\mathrm{H_2}} + R$$

- $o$ and $b$ are offer and bid prices in GBP/MWh, in the **ESO-cost convention** used throughout:
  a positive bid costs NESO money to turn the unit down; a negative bid means NESO receives money
  (a gas plant turned down hands back the fuel it saves). They may vary by hour.
- $h_t$ is the snapshot length in hours.
- $C^{\mathrm{IC}}$ is the cost of re-trading interconnectors, when they are not fixed
  ({doc}`interconnector_trading`); $C^{\mathrm{H_2}}$ the cost of moving hydrogen links, priced at the
  Stage 1 hydrogen value ({doc}`bids_offers_support`).
- $R$ collects **objective-only terms**: guards that steer the solve but are not payments. Each is
  removed from the reported BM cost and logged on its own (list below).

All other PyPSA constraints stay: power balance at every bus, line and transformer limits, storage
energy balance, link limits, and the boundary constraints.

### The network in the balancing stage

The balancing stage reloads the finalised network, so it sees the outturn (actual availability and
demand) and the real ratings. It then applies, in the order of the single solve:

- the **network delivery case** (`market.balancing.network_delivery`): a delayed-network case
  applied to the built network in this stage only (`scripts/market/network_delivery.py`); null is
  the network as built;
- capacity floors, the **voltage floor** (`network.transmission.min_bm_constraint_voltage_kv`, by
  default 220 kV: branches with an end below it are left unconstrained, except Scottish 132 kV
  transmission under `bm_voltage_floor`) and `line_rating_overrides`;
- the **outage schedule** or **secured boundary limits** (`network.transmission.outage_schedule`);
- **seasonal line ratings**.

All of these are described in {doc}`/model/network/constraints`. The balancing stage starts storage
and the hydrogen store at the same levels as Stage 1 (`market.storage_soc`, `market.hydrogen_store`).
Marginal-cost floors (`market.balancing.mc_floors`) are applied to plant whose cost came out at zero,
so that dispatchable plant is never free to turn up.

### Full period or rolling windows

`market.balancing.mode` sets how the period is solved:

| Mode | What it does |
|------|--------------|
| `full_period` | One LP over the solve period. |
| `rolling` (base default) | Consecutive windows of `window_hours` (24 in the base; 1 gives a per-hour BM), each anchored to its own Stage 1 positions. Storage state of charge and the hydrogen-store level carry from one window to the next. |

A rolling window would otherwise be free to empty storage at its end, because the next window's
anchors never ask for the energy back. `storage_end_soc` therefore asks each participating unit to end
the window with at least the state of charge Stage 1 scheduled at that hour, softly: a shortfall is
allowed at `penalty_gbp_per_mwh` (GBP 200 per MWh of stored energy), objective only. In the last window
the target is Stage 1's month-end level.

### What is held at Stage 1

Some units are held at their Stage 1 position instead of being offered to the BM:

| Held | Why | Setting |
|------|-----|---------|
| Hydrogen turbines | The hydrogen store should move only with the electrolysers | `market.balancing.hydrogen_links.hold_carriers` (`[H2_turbine]`; a study may add `electrolysis`) |
| Electrolysers, when held | Demand stays at its wholesale schedule; otherwise they are anchored and priced at the Stage 1 hydrogen value | `hydrogen_links.hold_carriers`, `hydrogen_links.enabled` |
| Consumer flexibility (EV smart charging, V2G, heat, demand response) | It is demand: it sets its schedule in the market and does not bid in the BM | `market.balancing.participation.storage_units.hold_consumer_flexibility` (`true`) |
| Must-take generation | Never curtailed in either stage | `market.must_take_generation.data_sources` |
| The churn sink | A Stage 1 calibration device, not a balancing action | `market.wholesale.churn_sink` |
| Cross-border interconnectors, when fixed | Day-ahead schedules are not redispatched | `market.balancing.fix_interconnectors` ({doc}`interconnector_trading`) |

Held links and held storage are soft: if a window cannot be solved otherwise, they may move at a
large objective-only penalty, which is logged and stripped. When electrolysers are not held, they
may run harder to absorb constrained wind, but the month's hydrogen must match Stage 1: each window
ends with the store between its Stage 1 level and what the rest of the month can still take back,
and the period ends within `hydrogen_links.store_month_end_tolerance_pu` (share of the store's
capacity) of Stage 1's month-end level. The band is soft and priced like the storage floor.

### Who takes part

`market.balancing.participation` filters which generators and storage units can move, by carrier,
name and size (`min_p_nom_mw`, 1 MW). A non-participant is handled by its `behavior`:

| Behaviour | Effect |
|-----------|--------|
| `priced_out` (base default) | It can still move, but only at `penalty_offer_price` / `penalty_bid_price` (GBP 6,000/MWh for generators, 5,000 for storage), so it moves only when nothing else can keep the LP feasible |
| `fallback_priced` | It moves at the normal derived prices |
| `fixed` | It is held at its Stage 1 position (can make the LP infeasible) |

The base excludes load shedding, embedded solar and wind, foreign import units, CHP, energy from
waste and biogas. Most CHP and energy-from-waste plant is embedded and not BM-registered; with their
must-run floors held as energy budgets, the BM would otherwise use their fuel as cheap turn-down.

**Load shedding (VoLL slack).** Load-shedding units are non-participants priced at the penalty bid
and offer, which equals the value of lost load, and are capped at their own bus's demand (plus any
held consumer flexibility there). Energy they supply is unserved load. It keeps the LP feasible and is
reported as **feasibility slack**, never as an ordinary balancing cost.

### Optional: a linear proxy for unit commitment

Without commitment any thermal unit can go from nothing to full output within the hour at its
marginal cost, so correcting an error is symmetric and nothing makes being caught out expensive.
`commitment_proxy` (off by default; `scripts/market/commitment_proxy.py`) keys two edits on the Stage 1
schedule: a unit that Stage 1 left off can be started only with `start_premium_gbp_mwh` added to its
offer, and a unit that Stage 1 ran at or above `min_stable_fraction` of its capacity cannot be taken
below that level, so renewables are spilled instead.

### Objective-only terms

These steer the solve and are stripped from the reported BM cost: the storage dispatch-cost floor
(`storage_marginal_cost_floor`, on gross discharge), the end-of-window state-of-charge shortfall
(`storage_end_soc`), the reverse-leg guard on storage moving against its Stage 1 direction
(`storage_reverse_leg_guard_gbp_per_mwh`), interconnector movement beyond its flow limit
(`interconnector_counter_trade.beyond_limit_penalty_gbp_per_mwh`), the hydrogen-store band, the slack
on held links, rescheduled consumer flexibility, the internal HVDC tie-break, the turn-up floor on
renewables (`support.bm_bids.contract_value_bids.turn_up_floor_gbp_per_mwh`, where set), and the
Stage 1 value of the interconnector schedule (only the change in trade value is a balancing cost).

### Cost measures

`analyze_market_results` builds the accounting (`scripts/market/market_accounting.py`). The measures
differ in what they include, so name the one you quote:

| Measure (key in the market accounting) | Definition |
|------------------------------|------------|
| `modelled_redispatch_cost_GBP` | Accepted offer cost plus accepted bid cost of every GB generator and storage unit, relative to its Stage 1 position |
| `ordinary_redispatch_cost_excluding_feasibility_slack_GBP` | The above without feasibility slack (load shedding at VoLL) |
| `feasibility_slack_cost_GBP` | Unserved energy priced at VoLL, reported on its own |
| `non_participant_turn_down_cost_at_penalty_GBP` | Turn-down of non-participants the LP could only price at the penalty bid: forced, not accepted, reported on its own |
| `interconnector_retrading_cost_GBP` | Change in interconnector trade value plus counter-trade premium ({doc}`interconnector_trading`) |
| `neso_comparable_redispatch_cost_GBP` | Ordinary redispatch plus interconnector re-trading, without VoLL slack or non-participant turn-down: the measure comparable with NESO's published thermal constraint cost |
| `internal_dc_tie_break_cost_GBP` | The tie-break term, objective only |

`constraint_costs.csv` covers accepted bids and offers of GB units only, so interconnector re-trading
and the constraint table are additive and never double count. The nodal duals of the balancing LP are
diagnostic marginal values, not imbalance or cash-out prices.

### Turn-down allocation: cheapest first or pro rata by zone

The BM LP turns wind down **cheapest first**: inside a congested zone it takes the units with the
lowest bids and leaves the dearer ones running. NESO's constraint modelling for Clean Power 2030
(CP.08) shares a zone's turn-down across its wind fleet, and BM practice spreads acceptances over
many units. The constraint cost depends on which units are turned down, because their bids differ
(a ROC unit loses its certificates, a CfD unit its difference payment, a merchant unit very little).
So every month of a twelve-month study family is reported on **two bases**
(`scripts/market/turn_down_allocation.py`, rules in `rules/turn_down_allocation.smk`):

- **`cheapest_first`**: as solved.
- **`pro_rata_zonal`**: for each hour $t$ and each of NESO's 17 CP.08 zones $z$, the zone's wind
  turn-down $D_{z,t} = \sum_{j \in z} d_{j,t}$ (onshore and offshore wind,
  `turn_down_allocation.technologies`) is re-shared among the zone's units in proportion to their
  Stage 1 output:

  $$\tilde d_{i,t} = D_{z,t}\,\frac{\hat p_{i,t}}{\sum_{j \in z} \hat p_{j,t}}, \qquad i \in z$$

  and each unit's re-shared volume is priced at its own bid in that hour, rebuilt from the month's bid
  audit (`{scenario}_contract_value_bids.csv`) and the network (ROC banding times buy-out). The
  zone-hour volume is unchanged; only who takes it changes. A unit on a bus with no zone keeps its own
  volume.

The re-pricing never feeds back into a solve. Its check: re-pricing the model's own turn-down with
the rebuilt bids must reproduce the solve's wind bid cost to `check_tolerance_rel` (0.1%) of the gross
bid money (or `check_tolerance_abs_gbp`), or the month fails. The pro rata NESO-comparable cost is the
as-solved one plus the difference in wind bid cost:

$$C^{\mathrm{NESO}}_{\mathrm{pro\ rata}} = C^{\mathrm{NESO}}_{\mathrm{as\ solved}} + \big(B^{\mathrm{wind}}_{\mathrm{pro\ rata}} - B^{\mathrm{wind}}_{\mathrm{as\ solved}}\big)$$

`market.balancing.turn_down_allocation.mode` picks the **headline basis** of a family:
`cheapest_first`, `pro_rata_zonal`, or `closest_to_neso` (the base default): the basis whose
NESO-comparable cost is nearer NESO's published constraint cost for the family's network case
(`data/CP30/validation_references/neso_balancing_benchmarks.csv`). Each month records its basis in
`{scenario}_turn_down_basis.json`. The consumer bill layer reads per-unit BM
results only through the helpers in `turn_down_allocation.py`, on the basis of their network's market
family, so every group on one network uses the same basis. Fuel and carbon are not re-shared: wind
burns none, and zone-hour totals are conserved.

```{figure} /_static/figures/dispatch/turn_down_bases.png
:alt: Two bar charts comparing cheapest-first and pro-rata turn-down by support scheme over a year: volumes are similar (about 32 TWh of CfD wind either way) but pro rata moves turn-down from merchant to ROC wind and raises the bid cost from about 1.45 to 2.43 GBP bn.
:width: 100%

The year's wind turn-down on the two bases, by support scheme, for the 2030 central case of the CP30 2030 study (twelve monthly balancing solves). Re-sharing a zone's turn-down pro rata moves volume off merchant wind, which bids nothing, onto ROC wind, which loses its certificates, and onto dearer CfDs, so the same volume costs more.
```

| Rule | Output |
|------|--------|
| `reprice_turn_down_pro_rata` | `resources/market/{scenario}_pro_rata_turn_down.csv` (per unit: zone, support, Stage 1 MWh, as-solved and pro rata MWh and cost) and `.json` (totals, check) |
| `summarise_turn_down_allocation` | `resources/analysis/{study}_pro_rata_turn_down_summary.json` (both bases, `headline_basis`) |
| `select_turn_down_basis` | `resources/market/{scenario}_turn_down_basis.json` |

## Data

| Input | Use |
|-------|-----|
| `resources/network/{scenario}.nc` | Reloaded with real ratings and outturn |
| `resources/market/{source}_wholesale_*.csv` | Stage 1 positions and prices (`{source}` is the scenario's `wholesale_from`, else itself) |
| ELEXON bid-offer data, CSV prices or the future bid stack | Depending on `bid_offer_source` ({doc}`bids_offers_support`) |
| `data/network/neso_boundary_mapping.yaml` | Boundary definitions for the NESO outage replay |
| `data/CP30/validation_references/neso_cp08_zones.geojson` | NESO's 17 CP.08 zones for the turn-down re-pricing |

## Configuration

Balancing settings are under `market.balancing` (`config/defaults/base/market/balancing.yaml`).

| Key | Default | Meaning |
|-----|---------|---------|
| `mode` | `rolling` | `full_period` or `rolling` |
| `window_hours` | `24` | Hours per rolling window |
| `network_delivery` | `null` | Delayed-network case for this stage |
| `bid_offer_source` | `auto` | Where prices come from ({doc}`bids_offers_support`) |
| `pricing_basis` | `bid_offer` | `bid_offer` or `at_cost` |
| `fix_interconnectors` | `true` | Hold cross-border links at Stage 1 |
| `interconnector_retrading` | `null` | Explicit re-trading switch; null defers to `fix_interconnectors` |
| `hydrogen_links.hold_carriers` | `[H2_turbine]` | Hydrogen links held at Stage 1 |
| `participation.generators.behavior` | `priced_out` | Treatment of non-participants |
| `participation.storage_units.hold_consumer_flexibility` | `true` | Hold consumer flexibility at Stage 1 |
| `storage_end_soc.penalty_gbp_per_mwh` | `200.0` | Soft end-of-window storage floor |
| `storage_marginal_cost_floor` | `1.0` | Storage dispatch cost in the objective (GBP/MWh) |
| `storage_reverse_leg_guard_gbp_per_mwh` | `250.0` | Guard on storage moving against its Stage 1 direction |
| `mc_floors` | by carrier | Marginal-cost floors (GBP/MWh) for plant at zero cost |
| `commitment_proxy.enabled` | `false` | Linear commitment proxy |
| `turn_down_allocation.mode` | `closest_to_neso` | Headline turn-down basis |
| `temporal_resolution_hours` | `1` | Block-mean inputs to match a block Stage 1 |

## Validation

- **Historical years** (`modelled_year` 2024 or earlier): `validate_bm_results` compares BM volumes,
  costs, prices and dispatch with ELEXON actuals (`{scenario}_bm_validation.csv/.html`);
  `validate_neso_constraints` compares boundary flows and constraint costs with NESO data
  (`{scenario}_neso_validation.csv/.html`); `validate_bm_calibration` scores bid and offer prices and
  volumes against ELEXON by carrier ({doc}`bids_offers_support`). See {doc}`/studies/validation_2020`.
- **CP30 2030**: `validate_CP30_balancing` (`scripts/CP30/balancing_validation.py`) compares a family
  with NESO's published balancing volumes (CP.18), the NESO-comparable cost, final dispatch (N2) and
  turn-down by CP.08 zone. It reports and never fails a run. Results are in {doc}`/studies/cp30_2030`.

## Outputs

| File | Content |
|------|---------|
| `resources/market/{scenario}_balancing.nc` | Solved BM network: physical dispatch, flows, nodal duals |
| `resources/market/{scenario}_balancing_dispatch.csv` | Physical generator output (MW) |
| `resources/market/{scenario}_balancing_storage.csv` | Physical storage output (MW) |
| `resources/market/{scenario}_redispatch_summary.csv` | Per unit: `carrier`, `increase_MWh`, `decrease_MWh`, `offer_cost`, `bid_cost`, `net_cost` (hydrogen links included) |
| `resources/market/{scenario}_constraint_costs.csv` | The same by carrier, with a `TOTAL` row (GB units' accepted bids and offers only) |
| `resources/market/{scenario}_congestion.csv` | Lines and transformers loaded to 95% of their rating or more |
| `resources/market/{scenario}_price_comparison.csv` | Stage 1 price against the mean, minimum and maximum BM nodal duals at GB demand buses |
| `resources/market/{scenario}_contract_value_bids.csv` | Audit of the CfD and support bids used |
| `resources/analysis/{scenario}_adequacy_relaxation.csv` | Audit of any diagnostic transmission relaxation (empty unless configured) |
| `resources/analysis/{scenario}_market_summary.json` | Every cost measure above, volumes, prices, revenue tracking |
| `resources/analysis/{scenario}_market_accounting.csv` | The accounting measures, one per row, with units (their definitions are in the market summary) |
| `resources/analysis/{scenario}_interconnector_flows.csv` | Per cable: totals over the period in both stages |
| `resources/analysis/{scenario}_market_dashboard.html` | Interactive dashboard of both stages |

## Limitations

- **Thermal constraints only.** The real BM also buys reserve, response, voltage support and
  stability; the model redispatches for transmission thermal limits and boundary capability only.
- **Linear and lossless.** DC power flow and no unit commitment (unless the proxy is on).
- **Exogenous bids.** Bids and offers come from rules and calibration, not from a model of each
  participant's strategy.
- **Foresight within a window.** A rolling window sees the whole of its own outturn.
- **Pro rata is a reporting basis.** The pro rata re-pricing moves volume between units after the
  solve; it does not re-solve the network with those units turned down.
