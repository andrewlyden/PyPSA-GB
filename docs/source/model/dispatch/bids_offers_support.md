# Bids, offers and support schemes

## What it does

The balancing stage ({doc}`balancing`) pays each unit it moves at that unit's price: an **offer** to
turn up, a **bid** to turn down. This page explains where those prices come from, and how the GB
support schemes (Contracts for Difference, the Renewables Obligation, and none for merchant plant)
enter both stages. Support matters to dispatch because it changes what a unit loses when it is
turned down: a merchant wind farm loses little, a ROC wind farm loses its certificates, a CfD wind
farm loses its difference payment.

The code is `market_utils.calculate_bid_offer_prices` (`scripts/market/market_utils.py`), with
`scripts/market/cfd_bm_bids.py` for contract-value CfD bids and `scripts/market/pricing_basis.py` for
the at-cost basis.

## Method and assumptions

### Sign convention

Every price is in the **ESO-cost convention**, GBP/MWh: positive means NESO pays.

- An offer of 120 means NESO pays 120 per MWh the unit turns up.
- A bid of 50 means NESO pays 50 per MWh the unit turns down (a supported wind farm compensated for
  lost support). A bid of -90 means NESO **receives** 90 per MWh (a gas plant turned down hands back
  most of the fuel it saves).

ELEXON publishes bids with the opposite sign (what the unit pays); they are negated when loaded.

### Where prices come from

`market.balancing.bid_offer_source` selects the source:

| Source | What it uses |
|--------|--------------|
| `auto` (base default) | `elexon` for a historical year (2024 or earlier) when the ELEXON files and BMU mapping exist, else `derived` |
| `derived` | Rules on each unit's marginal cost (below) |
| `elexon` | Observed ELEXON bid-offer data (BOD) for the solved period, mapped to model generators through BMU identifiers. Historical years only; refused for a future year |
| `csv` | User-supplied generator-by-snapshot price files (`csv.offer_file`, `csv.bid_file`) |
| `future_bid_stack` | Historical ELEXON prices turned into residuals against historical marginal cost, by carrier, and added back onto the future network's own marginal costs (`scripts/market/future_bid_stack.py`; settings in `future_bid_stack`) |

**ELEXON** (`market.balancing.elexon`). Rules `retrieve_elexon_market_data` and `build_bmu_mapping`
fetch the BOD data, aggregate it to hours and map BMUs to generators (with the Power Station
Dictionary crosswalk where available). Unmatched generators take `elexon.fallback`
(`carrier_average`, `derived` or `zero`), or a per-carrier linear fit against matched BMUs when
`fitted_fallback.enabled`. `elexon.price_ladders.enabled` uses BOD price and volume blocks instead of
one price per unit and hour.

Whatever the source, support pricing (below) and the participation filter ({doc}`balancing`) are
applied afterwards. ELEXON prices are observed evidence and are never replaced by support rules.

### Derived prices

Each carrier is priced in one of three modes (`market.balancing.carrier_overrides.<carrier>.mode`):

| Mode | Offer | Bid |
|------|-------|-----|
| `markup` (default) | $\mathrm{mc}\,(1 + m)$ | $-(\mathrm{mc} - \delta\,\lvert\mathrm{mc}\rvert)$ |
| `absolute` | `offer_price` | `bid_price` |
| `round_trip` (storage) | see below | see below |

$\mathrm{mc}$ is the unit's marginal cost, $m$ its `offer_markup` (else `default_offer_markup`, 0.10)
and $\delta$ its `bid_discount` (else `default_bid_discount`, 0.10). The markup bid follows
`derived_bid_convention: net_saving`: a unit turned down saves its marginal cost, and NESO receives
most of it, as a raw ELEXON bid of about 0.9 times marginal cost does. `legacy_cost` charges it
instead ($+\mathrm{mc}\,(1-\delta)$), so gas is never turned down; it is kept only for comparison.

The base `carrier_overrides` reflect ELEXON evidence: CCGT offers at 1.85 times marginal cost and OCGT
and gas engines at 2.3 times; nuclear at an absolute offer of 999 and bid of 150 (it almost never
offers up and is costly to restart); biomass, energy from waste and other must-run plant at absolute
prices that make them expensive to turn down; and renewables at absolute bids that stand for a blended
subsidy (onshore 25, offshore 90, solar 55). A unit that holds a CfD or a ROC is priced on its own
contract instead when support pricing is on (below). Source and study overlays change these: the CP30
source sets fossil offers at 1.29 times short-run marginal cost, and the CP30 2030 study at 1.6 times.

`mc_floors` sets a minimum marginal cost by carrier (gas, oil, coal, biogas and others) for plant that
came out of the cost pipeline at zero, so the BM never sees free dispatchable plant. It does not lower
a positive cost.

### Pricing basis

`market.balancing.pricing_basis` is `bid_offer` (the rules above) or `at_cost`. Under `at_cost` the
default markup and discount are zero and every `carrier_overrides` entry is read as `markup` at zero:
every unit offers and bids at its marginal cost. Every reader of the markups goes through
`pricing_basis.priced_balancing`, including the foreign-plant multipliers of interconnector
counter-trading. The central-dispatch overlay sets `at_cost` ({doc}`central_dispatch`).

### Storage

A storage unit's offer (discharge more) and bid (charge more or discharge less) are two halves of one
round trip. Priced independently, the BM could buy the cheap half and never pay for the dear one.
Storage carriers therefore use `mode: round_trip`. With $\eta$ the round-trip efficiency
(`efficiency_store` times `efficiency_dispatch`), $\pi$ the price the unit pays NESO to charge more,
and the `offer_premium` $P$:

$$\text{bid} = -\pi, \qquad \text{offer} = \frac{\pi}{\eta} + P$$

so a round trip costs NESO its losses plus the premium. With `storage_bid_reference: stage1_price`
(base default), $\pi$ follows the Stage 1 price $\lambda_t$ hour by hour: $\pi_t = \lambda_t -$
`bid_margin`, as GB storage bids track the market. `fixed` uses the carrier's `bid_price`. The base
levels come from 2024 ELEXON BOD: battery bids at the market index price less 44 and pumped storage
less 79, with premiums that put offers where 2024 offers sat.

Three guards keep storage from being used as a dump:

- **Churn guard** (`storage_churn_guard_gbp_per_mwh`, 250). A unit that charges and discharges at once
  destroys $(1-\eta)$ of the energy. Where needed the offer is raised to
  $\big(G(1-\eta) - \text{bid}\big)/\eta$, so churning costs at least $G$ per MWh destroyed, above
  any turn-down bid it could displace. The prices are refused if any storage unit could be charged for
  more than any storage unit's offer.
- **Offer floor** (`carrier_overrides.<carrier>.offer_floor`, optional): GB storage offers do not
  follow the market price down, so an indexed offer can be floored. The CP30 2030 study sets floors
  for batteries and long-duration storage.
- **Dispatch-cost floor** (`storage_marginal_cost_floor`, in both stages) and the **reverse-leg guard**
  ({doc}`balancing`).

### Hydrogen links

Electrolysers and hydrogen turbines, where not held at Stage 1, are anchored like generators and
priced at the hydrogen value Stage 1 implies (the dual of the hydrogen bus). For a link with marginal
cost $c_0$ that burns hydrogen ($w = 1$) or makes it ($w = -\eta_l$):

$$c_t = c_0 + w\,\lambda^{\mathrm{H_2}}_t, \qquad
\text{increase} = c_t + m\,\lvert c_t\rvert, \qquad \text{decrease} = -c_t + \delta\,\lvert c_t\rvert$$

with $m$ and $\delta$ from `hydrogen_links.offer_markup` and `bid_discount` (null takes the defaults).

### Support schemes

#### Labels, set at integration

When `support.labels.enabled`, every renewable and supported generator gets a `support_type`
(`CfD`, `ROC` or `merchant`) during generator integration
(`scripts/generators/integrate_renewable_generators.py`):

- **Named sites** (REPD): a CfD where the site has a positive CfD award; a ROC from its RO banding,
  or, where REPD leaves the banding blank, inferred for non-CfD renewables operational before
  `roc_closure_date` (1 April 2017), at the carrier's `default_roc_bandings`; ROC support ends
  `roc_eligibility_years` (20) after the operational date.
- **Scheme registers** (`support_registers`): the Ofgem RO register and certificates and the LCCC CfD
  portfolio override REPD for named sites. `checked_cfd_contracts` splits part-contracted sites into
  their rounds plus a merchant remainder, so each round settles at its own strike.
- **FES capacity without a site**: split between CfD and merchant by `fes_cfd_fractions`
  (offshore wind all CfD by default; onshore 0.3; solar 0.5).

CfD strikes come from the published strike table named in `support.labels.cfd_strike_table`
(`data/market/lccc/cfd_strike_prices_by_round.csv`, indexed to the model's price year; set in the CP30 2030 study's
templates): the same table
the consumer bill's CfD line settles against, so a bid and its compensation cannot drift apart.

The long-range FES chain (`method: gsp_block`) has no named sites: each clustered generator holds several build
years. It still tracks the contracts, by build-year cohort (`fleet_vintages.label_support_contracts`, rule
`build_support_labels`):

1. A stock-flow model splits each technology's FES capacity into build-year cohorts: the existing stock from REPD and
   DUKES with its asset lives, and new build from FES's year-on-year additions.
2. Every live phase of every known CfD round (`data/market/support/cfd_contracts.csv`), Hinkley Point C and Sizewell
   C's RAB goes on the cohorts of its technology and start year, to exactly its capacity.
3. Later build years take modelled contracts for their term (`PAR7_<year>` CfDs, `DPA_<year>` for gas with CCS),
   then turn merchant; the RO applies to the cohorts it accredited until it ends (31 March 2037).

Each generator bids the capacity-weighted blend of its cohorts' contracts (`scripts/market/support_bids.py`), and
the bill settles the same labels.

#### In Stage 1: dispatch costs

`support.dispatch_costs` sets the marginal costs the wholesale market sees:

| Support | Stage 1 marginal cost |
|---------|----------------------|
| CfD | `cfd_dispatch_cost` (0): the unit earns its strike through the difference payment, so it runs at any price that is not negative |
| CfD, fuelled (`cfd_dispatch_cost_exempt_carriers`: biomass, BECCS) | Its fuel and carbon cost |
| ROC | $-(\text{banding} \times \text{buy-out})$, at the buy-out price of the model year (`roc_buyout_prices`): it keeps generating below zero until the certificate value is lost |
| DPA (gas with CCS) | At most the cheapest unabated CCGT's cost less `dpa_margin_gbp_per_mwh` |
| Merchant | Its own cost |

The order among zero-cost renewables can be made contract-aware in Stage 1
(`market.wholesale.renewable_curtailment_order`, {doc}`wholesale`).

#### In the balancing stage: support bids

`support.bm_bids` (enabled in the base) prices supported units on their own contract rather than on
the blended carrier bid. In `mode: fixed`, a CfD unit offers and bids at `cfd.offer_price` and
`cfd.bid_price` (0, 0) and a ROC unit bids
$\text{banding} \times \text{buy-out} \times$ `roc.bid_multiplier`, what it forgoes. In
`mode: empirical`, carrier and support groups with at least `min_bmus` directly matched BMUs take
prices calibrated on historical ELEXON bids (rule `calibrate_support_bid_stack`). Carriers in
`exclude_carriers` (nuclear, BECCS, biomass) keep their carrier prices: a zero CfD offer would let the
BM turn Hinkley Point C up for free.

**Contract-value CfD bids** (`support.bm_bids.contract_value_bids`, configured in the CP30 source
overlay and switched on by the studies). One flat CfD bid makes the BM indifferent between turning
down a cheap contract and an expensive one, so the composition of curtailment, and so the constraint
cost, becomes arbitrary. With contract-value bids each CfD unit $i$ bids by its own strike $K_i$
against the Stage 1 price $\lambda_t$ (the reference price for intermittent contracts):

| `bid_basis` | Bid in an hour the contract pays |
|-------------|----------------------------------|
| `strike` | $K_i$, the whole strike |
| `contract_value` with `contract_value_floor: zero` | $\max(K_i - \lambda_t,\ 0)$, the difference payment it loses |
| `contract_value` with `contract_value_floor: none` | $K_i - \lambda_t$, negative when the price exceeds the strike: the unit then saves the payment it would hand back, and NESO receives it |

In an hour the contract pays nothing (`negative_price_rule: gb_contract_terms`: AR4 and later in any
negative-price hour, AR2 and AR3 inside a run of six or more negative hours) the bid is 0. Merchant
renewables bid `merchant.bid_price`. The CP30 2030 study sets it to zero (its balancing template). The CP30 source overlay's default of
GBP 20/MWh (what merchant wind was paid to turn down in the 2024 BM, for lost REGO and PPA volume) applies only to
scenarios that do not override it. ROC bids are unchanged. A unit with no strike entry
stops the solve (`missing_strike_policy: error`). `turn_up_floor_gbp_per_mwh` is a small objective-only
price on turning wind, solar or marine up, which resolves an otherwise arbitrary swap between identical
units. Every bid used is written to `{scenario}_contract_value_bids.csv`, and the basis and floor are
recorded there, so the bill's compensation and the asset settlement price turned-down energy the same
way.

```{figure} /_static/figures/dispatch/wind_bid_stack.png
:alt: A step chart of every wind unit in the balancing mechanism ordered by its mean January bid, width proportional to capacity: low-strike CfDs bid below zero, merchant wind at zero, most CfDs at 5 to 15 GBP/MWh, ROC wind at 70 to 160 and old high-strike contracts above 100.
:width: 100%

The wind fleet's bids in January 2030, 2030 central case of the CP30 2030 study: each unit's mean contract-value bid (CfD, floor `none`), banding times buy-out (ROC) or zero (merchant), ordered from the cheapest to turn down. Units outside the balancing mechanism are left out. Within each constrained zone the balancing stage turns down from the left.
```

```{figure} /_static/figures/dispatch/cfd_bid_week.png
:alt: Line chart over one January week of hourly bids: three offshore CfD contracts whose bids move opposite to the wholesale price, falling below zero in high-price hours, a flat ROC bid near 156 GBP/MWh and a merchant bid at zero.
:width: 100%

Contract-value bids follow the Stage 1 price hour by hour, $K_i - \lambda_t$, over an example winter week (14-20 January) from the 2030 central case of the CP30 2030 study: in high-price hours a low-strike CfD bids well below zero (NESO receives the payment the unit would have handed back), and when wind drives the price to zero each CfD bids its whole strike. ROC and merchant bids do not depend on the price.
```

#### After the solve: revenue

`support.revenue` adds CfD difference payments ($(K - \lambda) \times$ output) and ROC income
($\text{banding} \times \text{buy-out} \times$ output) to the market summary
(`scripts/market/revenue_tracking.py`). The full settlement of support schemes on the bill is in
{doc}`/model/after_the_solve/consumer_bill/cfd`.

### Scarcity pricing

Scarcity pricing is a Stage 1 feature: a stepped reserve shortage curve that raises the energy price
as the de-rated margin falls ({doc}`wholesale`). It is not a BM price.

## Data

| File | Use |
|------|-----|
| `data/market/lccc/cfd_strike_prices_by_round.csv` | CfD strikes by allocation round |
| `data/market/lccc/checked_cfd_contracts.csv`, `cfd_contract_portfolio_status.csv` | LCCC contracts for named sites |
| `data/market/ofgem_renewables_obligation/` | Ofgem RO register and certificates |
| `data/renewables/repd-q2-jul-2025.csv` | REPD support columns |
| `data/market/elexon_bod/`, `resources/market/elexon/` | ELEXON BOD caches (historical runs and calibration) |
| `data/generators/power_station_dictionary_bmu_crosswalk.csv` | BMU to station crosswalk |

## Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `market.balancing.bid_offer_source` | `auto` | `auto`, `derived`, `elexon`, `csv`, `future_bid_stack` |
| `market.balancing.pricing_basis` | `bid_offer` | `bid_offer` or `at_cost` |
| `market.balancing.default_offer_markup` | `0.10` | Offer markup on marginal cost |
| `market.balancing.default_bid_discount` | `0.10` | Bid discount on marginal cost |
| `market.balancing.derived_bid_convention` | `net_saving` | `net_saving` or `legacy_cost` |
| `market.balancing.carrier_overrides` | by carrier | Per-carrier mode and prices |
| `market.balancing.mc_floors` | by carrier | Marginal-cost floors (GBP/MWh) |
| `market.balancing.storage_bid_reference` | `stage1_price` | `stage1_price`, `stage1_daily_mean` or `fixed` |
| `market.balancing.storage_churn_guard_gbp_per_mwh` | `250.0` | Least cost per MWh destroyed by churn |
| `support.labels.enabled` | `true` | Assign support labels |
| `support.dispatch_costs.enabled` | `true` | Support-aware Stage 1 costs |
| `support.dispatch_costs.roc_buyout_prices` | by year | ROC buy-out (GBP/ROC) |
| `support.bm_bids.enabled` | `true` | Unit-level support bids in the BM |
| `support.bm_bids.mode` | `fixed` | `fixed` or `empirical` |
| `support.bm_bids.contract_value_bids.enabled` | `false` (CP30 source) | Contract-value CfD bids |
| `support.bm_bids.contract_value_bids.bid_basis` | `strike` (CP30 source) | `strike` or `contract_value` |
| `support.bm_bids.contract_value_bids.contract_value_floor` | `zero` (CP30 source) | `zero` or `none` |
| `support.revenue.enabled` | `true` | Post-solve CfD and ROC revenue |

## Validation

- `validate_bm_calibration` (historical ELEXON scenarios) compares each carrier's effective model bid
  and offer prices with ELEXON BOD medians, and model volumes with ELEXON's accepted volumes (BOAV); a
  carrier off by more than `market.balancing.calibration.tolerance` (20%) with enough volume is
  flagged (`{scenario}_bm_calibration.csv/.html`).
- `audit_bm_participation` and `audit_future_bm_participation` report BMU coverage and eligibility.
- Contract-value bids are checked after the solve: the bill's CfD compensation is recomputed from the
  bid audit, and the turn-down re-pricing must reproduce the solve's wind bid cost
  ({doc}`balancing`).

## Limitations

- **Rules, not strategies.** Derived prices are markups calibrated to historical evidence; they do not
  model a participant's bidding strategy or locational market power.
- **One reference price.** CfD bids use the Stage 1 price as the reference; baseload contracts' season-
  ahead reference is approximated by the solve window's demand-weighted mean.
- **Historical calibration carried forward.** Markups and storage margins are fitted to recent years
  and applied to future fleets.
