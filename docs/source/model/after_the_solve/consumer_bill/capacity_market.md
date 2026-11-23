# Capacity Market

## What it does

The Capacity Market pays providers (power stations, storage, interconnectors, demand-side response) to be available
when the system is tight. Agreements are won in auctions held four years (T-4) and one year (T-1) before each delivery
year, which runs October to September. Suppliers fund it through a charge on their customers' weekday demand between
16:00 and 19:00 from November to February. In Ofgem's cap it is a wholesale cost (Annex 2, sheets `4a` to `6e`).

The line does not come from the model's dispatch. It is built from the agreements already signed and a model of the
auctions not yet held: `scripts/market/capacity_market.py` and `scripts/market/build_capacity_market.py` (rule
`build_capacity_market`, outputs in `resources/market/capacity_market/`). Later years use
`scripts/market/build_capacity_market_year.py` (rule `build_capacity_market_year`) and, for 2050,
`build_capacity_market_2050.py`.

## Method and assumptions

### The cost of a calendar year

A calendar year straddles two delivery years. The supplier charge is levied from November to February, so calendar
2030 pays the January-February half of 2029/30 and the November-December half of 2030/31: half of each.

**The contract book** is every live agreement in the NESO Capacity Market Register for each delivery year, at its
auction's clearing price. Clearing prices are in the auction's base-year money and are carried to 2025 with ONS CPI
(D7BT), as the agreements are indexed. Multi-year agreements (up to 15 years for new build) from earlier auctions are
why a delivery year can cost more than its own T-4 alone.

**The auctions not yet held** (for 2030, the 2030/31 T-4 in early 2027) are modelled on DESNZ's published demand
curve. Each type of provider bids according to a distribution anchored on the 2029/30 T-4, with one shape set on the
2028/29 T-4; the model reproduces both held auctions by construction, so this is a calibration, not a test.

The auction's supply is the pathway's own fleet (`cm_t4_2030_31_supply.csv`):

- every eligible unit in the market group's register, de-rated with NESO's 2030/31 T-4 factors (storage by duration,
  interconnectors by cable);
- the cables are CP30's set (`data/interconnectors/cp30_interconnector_composition.csv`), each at CP30's nominal
  rating; MaresConnect (new in 2030) takes Greenlink's Republic of Ireland de-rating factor, as NESO publishes none
  for it;
- plant on a CfD, the RO, a DPA or a RAB cannot hold an agreement; merchant wind, solar and biogas are kept out, as
  almost none entered the 2029/30 auction;
- capacity already holding an agreement for the delivery year is netted off;
- new build is plant commissioned on the pathway's capacity path before the delivery year;
- new long-duration storage on Ofgem's cap and floor enters as a price-taker, and is paid its own award through the
  LDES rates;
- demand-side response, which the model does not hold, is the register's 2029/30 entrants.

T-1 auctions take 1.0 GW at the same year's T-4 price; secondary trading is excluded.

### From the cost to the household

The charge falls on winter weekday evenings, when households use a larger share of electricity than over the year as
a whole. The line therefore carries a household allocation factor:

$$
\text{household line} = \frac{C \times F}{\text{GSP charging volume}} \times 3.1\ \text{MWh}
$$

$F$ = {{ cm_household_factor }} is Ofgem's 2024 Capacity Market allowance for a 3,100 kWh household
(£{{ cm_ofgem_allowance_2024 }}) over a flat share of the same year's cost (£{{ cm_flat_share_2024 }}). It was measured
against transmission-level demand, so it already includes losses, which is why the cost is spread over the GSP
charging volume rather than metered supply, with no further loss factor.

The bill's assumptions file pins the builder's calendar-year total
(`sourced_externals.capacity_market.derivation.agreement_cost_gbp`). `scripts/consumer_bill/sync_capacity_market_pin.py
--write` re-pins it after the auction is rebuilt, and rule `verify_capacity_market_pin` checks it on every bill run.

**Who is paid.** Each technology's calendar-year payments go to its eligible units, as revenue. The LDES floor and the battery revenue top-up both net these payments, so a change in
the Capacity Market moves them pound for pound.

### Assumptions to keep in view

- **The modelled T-4 price** is the main uncertainty. Near the target the demand curve is steep, so a small change in
  eligible supply moves the price a lot. The sensitivities re-clear the auction with one supply input moved: storage
  de-rated at NESO's 2027/28 factors, new LDES kept out, and low and high demand-side response.
- **The household factor** is Ofgem's 2024 ratio held for later years. Heat pumps and electric vehicles could raise the
  household's share of winter peak demand.

## Data

- NESO Capacity Market Register, CMU file, and auction static data (`data/market/capacity_market/`).
- DESNZ, auction parameters letter (July 2026); T-4 2029/30 auction monitor report (March 2026).
- Ofgem, default tariff cap, Annex 2 v1.30, sheets `4a` to `6e`; 2024 cap model (household factor).
- OBR, Economic and Fiscal Outlook, March 2026, detailed receipts table 3.20.
- ONS CPI (D7BT).

## Configuration

`sourced_externals.capacity_market` in the assumptions file: the pinned cost, the household allocation factor and the
T-4 sensitivities. The auction model's settings are in `scripts/market/capacity_market.py`; its inputs are rule
inputs of `build_capacity_market`.

## Validation

**Against NESO's register and OBR, delivery year by delivery year.** The register publishes a forecast cost for each
delivery year (base-year money, before indexation), and OBR forecasts Capacity Market receipts by fiscal year. Where
the auctions have been held, the contract book here follows both, and is above the register's own figures because it
indexes each agreement to 2025 with CPI.

**Against Ofgem's cap.** Ofgem's current allowance covers a delivery year whose auctions cleared at higher prices than
the 2029/30 T-4, so it is above a 2030 line for a known reason.

The worked example below is one run's build-up:

```{include} _generated/cm_delivery_years.md
```

The modelled 2030/31 T-4's supply and its cases:

```{include} _generated/cm_supply.md
```

```{include} _generated/cm_cases.md
```
