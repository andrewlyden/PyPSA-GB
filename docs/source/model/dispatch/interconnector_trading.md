# Interconnector trading in balancing

## What it does

In Stage 1 every cross-border interconnector trades against an hourly foreign price, up to its
rating, as market coupling does day ahead ({doc}`wholesale`; the foreign prices and the cable model
are in {doc}`/model/interconnectors`). This page covers what the balancing stage
({doc}`balancing`) may do with those day-ahead schedules. There are two choices:

- **Fixed** (base default): each cable is held at its Stage 1 flow. Cross-border schedules are not
  redispatched by the GB balancing mechanism.
- **Re-traded**: the balancing stage may move a cable off its Stage 1 flow. In GB this is
  counter-trading: NESO pays a trader or a neighbouring system operator to change the flow, and
  someone abroad turns plant up or down. NESO's own constraint modelling replaces constrained wind
  most often by changing interconnector flows, usually by reducing exports, so for a constraint cost
  comparable with NESO's this matters.

```{figure} /_static/figures/dispatch/ic_week.png
:alt: Two lines over one January week, GB net imports in Stage 1 and after balancing, with the gap shaded: from Friday Stage 1 exports about 12 GW and the balancing stage cuts exports to between zero and 8 GW.
:width: 100%

GB net imports in Stage 1 and after balancing over an example winter week (14-20 January) from the 2030 central case of the CP30 2030 study. In the windy days the day-ahead market exports up to 12 GW that the network cannot carry to the coast; the balancing stage counter-trades most of it back.
```

```{figure} /_static/figures/dispatch/ic_retrading_by_country.png
:alt: Horizontal bars by counterparty country of the year's interconnector moves in the balancing stage: up moves (more import or less export) total about 33 TWh, led by France at 13 TWh, and down moves about 2 TWh.
:width: 100%

The year's interconnector re-trading by counterparty, 2030 central case of the CP30 2030 study (twelve monthly balancing solves against their Stage 1 flows, measured on the GB side). Almost all moves are up: the balancing stage cuts exports that Stage 1 scheduled from surplus wind.
```

The code is in `scripts/market/solve_balancing.py` (`resolve_fix_interconnectors`) and
`scripts/market/interconnector_counter_trade.py`, with the cable valuation in
`scripts/interconnectors/market_coupling.py`.

## Method and assumptions

### Fixed or re-traded

`market.balancing.interconnector_retrading` decides; when it is null (the base) the older switch
`fix_interconnectors` decides (`true` = fixed). A fixed cable's flow is constrained to its Stage 1
value in every hour. The CP30 2030 and long-range FES studies re-trade.

### What a re-traded move costs

A cable's position is signed: positive is import into GB. Each move off the Stage 1 flow
$\hat p_{l,t}$ is split into an **up** move $u_{l,t}$ (more import or less export: it adds GB supply,
like an offer) and a **down** move $d_{l,t}$ (less import or more export, like a bid). The balancing
objective pays three things for it.

**1. The change in trade value.** Stage 1 values each cable with a convex bid/ask curve: per hour,

$$V_{l,t}(p) = \max\big(a_{l,t}\,p,\ \eta_l^2\, b_{l,t}\,p\big)$$

where $a$ is the foreign ask (paid on imports), $b$ the foreign bid (earned on exports) and the export
pays the cable loss through $\eta_l^2$. The balancing stage values its own schedule the same way and
subtracts the constant value of the Stage 1 schedule, so it pays only the **change**,
$V_{l,t}(p) - V_{l,t}(\hat p)$. The Stage 1 trade was settled day ahead and is not a balancing cost.

**2. The counter-trade premium.** On top, every MWh moved pays a premium:
`premium_gbp_per_mwh` $\pi^{\uparrow}$ on up moves and `bid_premium_gbp_per_mwh` $\pi^{\downarrow}$ on
down moves (null = the same as the offer premium). FTI's assessment for Ofgem (2023) prices this at
about the short-run cost of a CCGT abroad, GBP 100/MWh from 2030 in the base, because changing a flow
usually means constraining on gas in a neighbouring country. Two rules set how the premium combines
with the trade value, separately for up moves (`offer_pricing`) and down moves (`bid_pricing`):

| Rule | Price of a move per MWh |
|------|-------------------------|
| `additive` (base default) | Change in trade value plus the premium |
| `all_in` | The premium is the whole price: $\max(\pi, m_{l,t})$, where $m$ is the move's marginal trade value at the Stage 1 schedule. FTI quotes the interconnector's BM price as a single "cost of reversing flow", which already pays for the foreign plant |

**3. The foreign-plant uplift (optional).** NESO's long-term constraint modelling prices a cable move
as the spread plus the foreign marginal plant's own offer or bid, with the same multipliers as GB
plant. With the move's marginal foreign value $m^{\uparrow}$ or $m^{\downarrow}$:

$$\text{up move: } (1 + s_o)\, m^{\uparrow}, \qquad \text{down move: } -(1 - s_b)\,\lvert m^{\downarrow}\rvert$$

The shares $s_o$, $s_b$ are `foreign_offer_uplift_share` and `foreign_bid_uplift_share`, or, with
`foreign_plant_multipliers: gb_gas`, the scenario's own CCGT `offer_markup` and `bid_discount`. Both
are scaled by `foreign_uplift_scale`. Both raise NESO's cost and are zero by default.

### What a re-traded move may do

**Flow limits.** LCP Delta (2026) reproduces present-day interconnector redispatch with a cap on how
far a cable may move from its day-ahead position, as a share of its capacity (2.0 being full export to
full import). `max_deviation_share_by_country` sets it by counterparty (Belgium 0.33, Denmark 0.22,
France 0.45, Germany 0.22, Netherlands 0.30, Ireland, Northern Ireland and Norway 0) and
`max_deviation_share` (0.45) applies to the rest. Each move is bounded in both directions by
$\text{share} \times p^{\mathrm{nom}}_l$.

**Minimum spread.** A trader needs a minimum margin before counter-trading
(`min_spread_gbp_per_mwh`, GBP 21/MWh). `min_spread_rule` applies it:

| Rule | Effect |
|------|--------|
| `stage1_gate` (base default) | In an hour whose Stage 1 spread $\lvert$GB price $-$ foreign price$\rvert$ is below the minimum, no move may **increase** the cable's absolute flow. A move back towards zero (the network shedding a flow it cannot carry) stays allowed inside the flow limit, up to the Stage 1 flow |
| `premium_floor` | The minimum spread is a floor on the premium |
| `none` | No spread rule |

**The band is soft.** When the network cannot carry a cable's Stage 1 flow to or from its landing
point, nothing else can stand in for it, and a hard band would make the window infeasible. Movement
beyond the band is therefore allowed at `beyond_limit_penalty_gbp_per_mwh` (GBP 300/MWh, above every
real bid and offer and far below VoLL), objective only and stripped from the reported cost, with the
volume logged; the premium is still paid on every MWh moved. Where the cable's own availability in
the balancing stage (an outage) no longer admits the Stage 1 flow, the band widens by exactly the
forced movement, so an outage costs no penalty.

**No wheeling (optional).** With the flow limits off, the BM could export more on a northern cable and
import more on a southern one in the same hour, a path round a GB boundary through Europe.
`no_wheeling: true` makes the counter-trades in each hour move GB's supply one way only
(`no_wheeling_formulation`: `binary`, exact, a mixed-integer constraint with one binary per hour; or
`hull`, its linear relaxation). Off by default.

### Under central dispatch

The same rules apply when a central operator corrects its own day-ahead schedule
({doc}`central_dispatch`). The foreign counterparty does not care who scheduled GB's side: the premium
is a resource cost abroad, and the flow limits describe trading arrangements, not GB's dispatch. The
spread gate then reads the central run's Stage 1 price, the demand-bus mean of its nodal prices. The
foreign-plant multipliers borrow the CCGT markups, which the at-cost basis sets to zero.

### How the cost is reported

The re-trading cost is a real payment: NESO's counter-trades are balancing services trades recovered
through BSUoS alongside bid-offer acceptances. So it sits in the balancing objective (it steers the
solve) and is reported as

$$C^{\mathrm{IC}} = \sum_{l,t} h_t\Big[V_{l,t}(p_{l,t}) - V_{l,t}(\hat p_{l,t})\Big]
+ \text{premiums} + \text{foreign-plant uplifts}$$

under `interconnector_retrading_cost_GBP` in the market summary and accounting. A positive value is a
net payment to foreign markets and traders; a negative value a net receipt. It is **not** in
`constraint_costs.csv`, which covers accepted bids and offers of GB units only, so the two are
additive and never double count. The NESO-comparable cost is ordinary redispatch plus this cost
({doc}`balancing`). The consumer bill adds it to its BSUoS constraint line
({doc}`/model/after_the_solve/consumer_bill/networks`).

The settings used are stamped on each re-traded link of the saved balancing network
(`market_counter_trade_premium`, `market_counter_trade_max_share`), so post-solve readers price it the
way the solve did. `{scenario}_interconnector_flows.csv` lists every cable's flow in both stages.

## Data

The foreign prices, cable ratings, losses and Stage 1 trading costs come from the network build
({doc}`/model/interconnectors`). The counter-trade settings are configuration only.

## Configuration

All under `market.balancing` (`config/defaults/base/market/balancing.yaml`).

| Key | Default | Meaning |
|-----|---------|---------|
| `fix_interconnectors` | `true` | Hold cables at Stage 1 (used when `interconnector_retrading` is null) |
| `interconnector_retrading` | `null` | `true` re-trades, `false` fixes, null defers |
| `interconnector_counter_trade.enabled` | `true` | Apply the premium and limits when re-trading |
| `interconnector_counter_trade.premium_gbp_per_mwh` | `100.0` | Premium on up moves (GBP/MWh) |
| `interconnector_counter_trade.bid_premium_gbp_per_mwh` | (code: null) | Premium on down moves; null = the up premium |
| `interconnector_counter_trade.offer_pricing`, `.bid_pricing` | (code: `additive`) | `additive` or `all_in` |
| `interconnector_counter_trade.max_deviation_share` | `0.45` | Flow limit for unlisted countries (share of rating) |
| `interconnector_counter_trade.max_deviation_share_by_country` | by country | Flow limits by counterparty |
| `interconnector_counter_trade.min_spread_gbp_per_mwh` | `21.0` | Minimum trader spread |
| `interconnector_counter_trade.min_spread_rule` | `stage1_gate` | `stage1_gate`, `premium_floor` or `none` |
| `interconnector_counter_trade.beyond_limit_penalty_gbp_per_mwh` | `300.0` | Objective-only price on movement beyond the band (null = hard band) |
| `interconnector_counter_trade.foreign_plant_multipliers` | `null` | `null` (use the shares) or `gb_gas` |
| `interconnector_counter_trade.foreign_offer_uplift_share`, `foreign_bid_uplift_share` | `0.0` | Foreign-plant uplift shares |
| `interconnector_counter_trade.foreign_uplift_scale` | `1.0` | Scales both shares |
| `interconnector_counter_trade.no_wheeling` | `false` | One direction of counter-trade per hour |

The CP30 source overlay sets an all-in offer price and a separate bid premium; the CP30 2030 study
lifts the country caps and the spread rule and sets its own premiums. The values in force for a
scenario are shown by `python config/config_loader.py --explain <scenario_id> --path
market.balancing.interconnector_counter_trade`.

## Validation

`validate_CP30_balancing` compares interconnector offers and bids (TWh) with NESO's published CP30
balancing volumes, and the final net export with NESO's constrained dispatch
({doc}`/studies/cp30_2030`). Each solve logs the counter-traded volume by cable and any movement
beyond the band.

## Limitations

- **The foreign side is a price, not a model.** Foreign prices are exogenous and do not respond to GB
  counter-trades; there is no foreign network.
- **One premium per direction.** The premium stands for the cost of foreign plant turned up or down;
  it does not vary by country or hour.
- **Flow limits are an empirical fit** to present-day behaviour, carried forward.
