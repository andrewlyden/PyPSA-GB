# Contracts for Difference

## What it does

A Contract for Difference pays a low-carbon generator the gap between a fixed strike price and a market reference
price for each MWh it generates. When the reference is above the strike, the generator pays back. Suppliers fund the
payments through the CfD supplier obligation, levied by the Low Carbon Contracts Company (LCCC) on eligible supply. In
Ofgem's cap the CfD sits in the wholesale allowance (Annex 2, sheet `7a`).

The line is computed in `scripts/consumer_bill/project_future_bill.py` (`read_cfd_payments`), under
`model_linked_rules.cfd` of the year's assumptions file.

## Method and assumptions

### Which plant holds a contract

Every CfD generator in the monthly networks carries an allocation-round label: AR1 to AR7a, Investment Contract,
Hinkley Point C, Drax bridging and Power BECCS ({doc}`/model/dispatch/bids_offers_support`).

- **A pathway's fleet is synthetic**, so rounds are assigned to its CfD capacity in proportion to the MW each round
  awarded for delivery by the modelled year. A plant's round says which tranche of capacity it represents, not which
  project it is.
- **Capacity with no contract yet** is a modelled round (`AR8_modelled` in 2030; `cfd_strike_table.MODELLED_ROUND`):
  plant generating in the modelled year that wins its contract in a future allocation round, at a modelled clearing
  price (below).
- **Hinkley Point C** in 2030 is unit 1 and unit 2 pro-rated for its connection date
  (`data/market/CP30/nuclear_station_schedule_2030.csv`). Sizewell B is merchant.

### Strike prices

Each round's published clearing price in 2012 money, indexed with CPI to the model's 2024 basis
(`data/market/lccc/cfd_strike_prices_by_round.csv`). Rounds with several delivery-year prices use the
capacity-weighted mean, because a model generator carries a round but no delivery year. Hinkley Point C is at
£89.50 (2012 money), £3 below its original strike, because the reduction took effect at Sizewell C's financial close.
The solves and the bill read the same strike table.

**The modelled round.** Its strike is modelled, not copied from the last round: the levelised cost at the private
hurdle rate, with each technology's capital cost calibrated so that its last auction's build year clears at that
auction's price (AR7 offshore wind 91.20, AR7a onshore 72.24, solar 65.23, tidal stream 265.00, 2024 money), taken at
the mean of the 2029 and 2030 build years (`scripts/market/build_ar8_modelled_strikes.py`). Two bill-only
sensitivities bracket it without a re-solve: every technology at its AR6 clearing price (low), and at AR8's
administrative strike prices (high).

### Settlement

For each contract and hour,

$$
\text{payment}_{g,t} = \left(S_g - R_{g,t}\right) \times Q_{g,t}
$$

where $S_g$ is the strike, $Q_{g,t}$ the generator's **physical** output in Stage 2 (after balancing actions, which is
what CfDs are paid on), and $R_{g,t}$ the reference price:

| Contracts | Reference price |
| --- | --- |
| Wind, solar, tidal | The hourly Stage 1 price, standing in for the day-ahead reference |
| Nuclear, biomass, Power BECCS (baseload contracts) | A seasonal baseload price: the demand-weighted mean Stage 1 price of the season (April to September, October to March), standing in for the season-ahead reference |

**Negative prices** (`negative_price_rule`). The bill and the balancing bids must use the same rule; the projector
refuses a bill whose rule differs from the bid audit's.

| Rule | Meaning |
| --- | --- |
| `none` (the CP30 and FES studies) | Every contract is paid in every hour. It matches how the balancing bids price CfD turn-down (strike minus the wholesale price, with no negative-price exemption), the convention NESO's balancing-cost studies use |
| `gb_contract_terms` | Each round's own term: AR1 none; AR2 and AR3 unpaid inside a run of six or more consecutive negative hours (SI 2015/1425); AR4 to AR7a and uncontracted capacity unpaid in any hour below zero; Investment Contracts and Hinkley Point C none |
| `ar4_zero_payment_when_price_le_0` | AR4 and later unpaid in any hour at or below zero |

A price counts as negative below -£0.01/MWh, so solver noise around zero is not read as a negative hour.

**Other terms:**

- **Curtailment order.** Stage 1 curtails plant in an order consistent with the contracts (each bids the support it
  loses), so no separate payment for "economic curtailment" is added.
- **Drax bridging collar.** The contract covers 4.5 TWh a year; output above that is merchant.
- **Power BECCS** settles with the other contracts but is booked on its own line, `allocated_ccs`, the electricity leg
  of the Power BECCS business model. The carbon-removal leg is paid by the Exchequer's carbon contract, not the bill.

### Recovery

$$
\text{CfD line} = \frac{\sum \text{payments} \times f}{\text{eligible GSP volume}} \times 3.1\ \text{MWh} \times L
$$

where $f$ = {{ rebasing_factor }} carries 2024 money to 2025, the eligible GSP volume is the GSP charging volume less
the energy-intensive-industry exemption ({{ eii_exemption_pct }}% of gross supplier sales on LCCC's 2024-25 reconciled
levy data), and $L$ = {{ loss_factor }} is the household's loss factor, because the levy is charged at the grid supply
point. See {doc}`recovery`.

### Assumptions to keep in view

- **The CfD and wholesale lines offset each other.** The top-up is large when the reference price is low. At a higher
  price level CfD payments fall while the wholesale lines rise.
- **Zero-price hours.** Under `gb_contract_terms`, whether surplus hours clear at zero or just below decides whether
  AR4+ plant is paid its full strike or nothing; under `none` it does not matter.
- **Volume.** A pathway may assume more contracted output than the official pipeline; the uncontracted part is the
  modelled round, which is partly a question about whether the pathway's fleet is deliverable.

## Data

- LCCC, allocation round results and reconciled daily levy rates (data portal).
- Explanatory Memorandum to the Contracts for Difference (Standard Terms) (Amendment) Regulations 2015
  (SI 2015/1425); Frontier Economics, *Impact of negative pricing on renewable support* (2019).
- OBR, Economic and Fiscal Outlook, March 2026, detailed forecast tables (receipts), 3.20 and 3.21.
- NAO, *Hinkley Point C* (HC 40, June 2017); EDF, Universal Registration Document 2025.
- DESNZ, Power BECCS business model update (December 2023).
- Ofgem, Annex 2 v1.30, sheets `7a` to `7d` (CfD allowance and its losses).

## Configuration

- `model_linked_rules.cfd` in the assumptions file: `negative_price_rule`, the reference-price basis, the recovery
  method, and the modelled-round sensitivities.
- `support.bm_bids.contract_value_bids.negative_price_rule` in the scenario: the rule the balancing bids use, which
  the bill must match.
- `consumer_bill.forecasts.cases.<case>.max_fallback_strike_share_of_payment`: the largest share of payment allowed to
  settle on a carrier-wide strike rather than an allocation-round strike (the case fails above it).

## Validation

**The settlement arithmetic reproduces LCCC.** `scripts/consumer_bill/reconcile_lccc_cfd_model_dispatch.py` replays a
historical year contract by contract against LCCC's published settlement data
(`data/market/lccc/actual_cfd_generation_and_payments.csv`). On LCCC's own generation the formula reproduces LCCC's
2020 payments within 0.01%, so the method (strike minus reference, times output) is right; the remaining gap on
modelled dispatch comes from the dispatch, which {doc}`/studies/validation_2020` tests.

**A future year cannot be reconciled with outturn**, so the CfD line is compared with the Office for Budget
Responsibility's forecast of CfD payments (EFO March 2026, tables 3.20 and 3.21): generation, capture price, strike and
total. The model's strikes and capture prices are output-weighted over every contract, so the two can be compared term
by term; differences in volume come from the pathway's fleet, differences in price from the modelled price level.

The worked example below is one run's settlement by technology and its largest contracts:

```{include} _generated/cfd_by_technology.md
```

```{include} _generated/cfd_largest_contracts.md
```
