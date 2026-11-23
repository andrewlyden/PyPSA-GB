# Wholesale energy and procurement

## What it does

These lines price the energy a household uses, as Ofgem's price cap does. Ofgem's direct fuel allowance is built in
three steps (Annex 2, sheet `1b`):

1. a **forward index**: 70% baseload and 30% peak contracts, averaged over a year of trading before the cap period;
2. the **allowances** for shaping, forecast error, imbalance, transaction costs and risk, added to the index;
3. **losses** from the transmission system to the meter.

The bill keeps that structure and replaces the market's forward index with the model's prices for the modelled year.
The code is `scripts/consumer_bill/project_future_bill.py` (the cap rule is `ofgem_cap_rule_price`); the rates are in
`model_linked_rules.direct_fuel` of the year's assumptions file.

## Method and assumptions

### The hourly price

The price is Stage 1's. For each of the twelve months, Stage 1 solves a copperplate market (no transmission limits)
in rolling day-ahead windows. The hourly GB price is the mean marginal price across GB demand buses
(`scripts/market/solve_wholesale.py`); on a copperplate these are equal, so it is the system marginal price. GB has one
national wholesale price, so a copperplate is the right basis; the cost of the grid's limits enters the bill
separately, through the BSUoS constraint line ({doc}`networks`).

The twelve monthly price series are joined into one calendar year, each hour weighted one hour
(`<case>_consumer_bill_future_hourly_inputs.csv`). The CP30 2030 study uses 2013 weather and demand shapes; 2013 and
2030 both start on a Tuesday and neither is a leap year, so weekdays line up. For other years the calendar is aligned
by month, day and hour (`scripts/utilities/calendar_year.py`).

Two things shape the Stage 1 price ({doc}`/model/dispatch/wholesale`):

- **The scarcity mark-up** (`market.wholesale.scarcity_pricing`): a BSC-style uplift in hours of tight margin. It is
  on by default and always off under central dispatch.
- **The curtailment order.** Renewables bid according to their support ({doc}`/model/dispatch/bids_offers_support`),
  so prices clear below zero only when contracted plant that loses support by being turned down is marginal.

The model works in 2024 money. The bill rebases to 2025 with ONS CPIH (132.9 to 138.0, a factor of
{{ rebasing_factor }}).

### Baseload and peak prices

For each calendar quarter, from the hourly series (`base_peak_prices_by_quarter` in the summary):

- **baseload price**: the average price over every hour of the quarter;
- **peak price**: the average price over the Elexon peak block, weekdays 07:00 to 19:00 UK local time
  ({{ peak_block_hours }} hours in a non-leap year, {{ peak_block_share_pct }}% of the year). The model's snapshots are
  UTC and are converted to local time before the block is applied.

Both are averages over hours, not weighted by demand, because baseload and peak contracts deliver a flat MW in every
hour of their block.

### Ofgem's cap rule

Each quarter's price is a 70/30 blend (Annex 2, `3b Demand`: 0.7 baseload, 0.3 peak). The four quarters are combined
with Ofgem's seasonal weights for single-rate households (Profile Class 1, PC1):

$$
P_{\text{blend}} = \sum_{q=1}^{4} w_q \left(0.7\,B_q + 0.3\,K_q\right)
$$

where $B_q$ and $K_q$ are the quarter's baseload and peak prices and $w_q$ its PC1 weight: 28.6%, 22.0%, 21.4% and
27.9% for Q1 to Q4 in the Oct-Dec 2026 cap (Annex 2 v1.30, `3b Demand`, column AD). Ofgem derives these from Elexon
data and updates them each cap period. They are declared in `direct_fuel.shaping.quarter_weights`.

### The lines

**Energy** (`direct_energy`):

$$
E = P_{\text{blend}} \times 1.0384 \times 3.1\ \text{MWh}
$$

The cap funds suppliers as if they bought the household's energy as 70/30 forward contracts. There is no forward
market for the modelled year, so the modelled price stands in for the forward index. That assumes the forward price
equals expected spot; see the forward premium below.

**The allowances.** Ofgem adds all its rates to the index together (sheet `1b`: index × (1 + sum of the rates)), and
applies losses after every allowance. The bill does the same (`procurement_bridge.rate_composition:
ofgem_annex2_1b`); each line is its own share of the total.

| Line | Rate | What it covers |
| --- | --- | --- |
| Shaping (`shaping_adjustment`) | {{ shaping_rate_pct }}% | Ofgem's two shaping allowances: seasonal to monthly (0.232%) and monthly peak/baseload to hourly (4.161%): turning flat blocks into the household's half-hourly profile, whose evening peak runs past the 19:00 end of the peak block |
| Imbalance (`procurement_imbalance`) | 1.310% | The supplier's cash-out: contracted volume never matches customers' use, and Elexon settles the difference at imbalance prices. Not BSUoS, which pays NESO's own balancing actions |
| Additional risk (`procurement_additional_risk_and_basis`) | 1.000% | Risks not covered elsewhere. For electricity there is no separate basis-risk allowance |
| Transaction costs (`procurement_transaction_cost`) | 0.394% | Exchange and broker fees, clearing, bid-offer spreads, collateral |
| Day-ahead re-hedging (`procurement_day_ahead_rehedging`) | 0.287% | Forecast error: hedges bought months ahead are trimmed day-ahead at different prices |
| Forward premium (`procurement_forward_premium`) | 0% centrally | See below |

All rates are Ofgem's (Annex 2, `3a Allowances`). Ofgem set the shaping allowances in 2018 from historical day-ahead
price ratios and seasonal-normal PC1 demand. The model's household profile is Ofgem's own two-block allocation, so the
model cannot measure shaping itself; the allowance stands in for it.

**Losses** (`procurement_losses`): index × (1 + sum of the rates) × $(L - 1)$. A household's meter records 3.1 MWh,
but its supplier buys more at the transmission system, because energy is lost in transmission and distribution. $L$
is the household's single-rate loss factor, {{ loss_factor }} (Ofgem Annex 2 v1.30, `3c Electricity losses`), the
same factor every line charged at a grid-supply-point rate carries ({doc}`recovery`).

**Forward premium**: zero centrally, with -3%, +5% and +10% sensitivities. The cap carries only its named allowances
and Ofgem treats backwardation as neutral in the long run. The historical gap between the cap's forward index and
outturn spot changes sign with the price trend; the CMA (2016) found no systematic bias in forward cost benchmarks;
German studies find ex-post premia of 3% to 10%. For scale, Ofgem's backwardation allowance was about £28 a year at
3,100 kWh in Oct-Dec 2026 (Annex 2, `1a`, rows 48 to 61).

### Assumptions to keep in view

- **The price level.** The modelled price level is a model output; the bill carries a CP30 price-level sensitivity
  (`consumer_bill.price_level_mapping`), which maps each Stage 1 hour onto CP30's published price duration curve
  (CP.32) and re-settles the same volumes. A higher price raises these lines and lowers the CfD line, so the effect on
  the total bill is smaller than on either.
- **Allowances scale with the price.** Every rate is a share of the price. That suits losses and imbalance;
  transaction costs are partly a fixed amount per MWh, so they are slightly understated at a low price.
- **Bank holidays** count as peak-block weekdays; Ofgem's calendar excludes them. A handful of hours.
- **A possible refinement.** With hourly model prices, shaping could be computed directly from Elexon's half-hourly
  PC1 profile rather than taken from Ofgem's 2018 allowance.

## Data

- Ofgem, Default tariff cap, Annex 2 wholesale cost allowance methodology v1.30 (26 August 2026): tabs `1a`, `1b`,
  `3a Allowances`, `3b Demand`, `3c Electricity losses`, `8a(i) Backwardation`.
- Ofgem, Default tariff cap level v1.31 (Oct-Dec 2026).
- ONS CPIH annual average, series L522.

## Configuration

`model_linked_rules.direct_fuel` in the assumptions file: `shaping` (rates, `quarter_weights`), the hedging rates,
`procurement_bridge.rate_composition` (`ofgem_annex2_1b`, or `on_shaped_energy`, which compounds the hedging rates on
the shaped price), the loss multiplier and the forward-premium sensitivities.

## Validation

The wholesale lines are compared with Ofgem's direct fuel allowance in {doc}`validation`. The worked example below
shows the build-up on one run (each rate as the run applied it):

```{include} _generated/wholesale_build_up.md
```

The quarters of that run:

```{include} _generated/wholesale_quarters.md
```

The sensitivities, reported in the bill summary (`diagnostics.wholesale_procurement`):

```{include} _generated/wholesale_sensitivities.md
```
