# Interconnectors

Great Britain trades electricity with France, Belgium, the Netherlands, Germany, Denmark, Norway and the island of
Ireland over high-voltage DC cables. This page covers which cables are in a scenario, their availability, how foreign
prices are built and how the cables dispatch in each stage.

## What it does

Rule `add_interconnectors_to_network` (`rules/interconnectors.smk`, script
`scripts/interconnectors/add_to_network.py`) adds each cable and a price-taking foreign market at its far end:

| Component | Representation |
|---|---|
| Foreign bus | one per counterparty bidding zone |
| Foreign market | a supply `Generator` (carrier `EU_market`) priced at the import price, and an export sink priced at the export bid |
| Cable (future, and historical back-tests) | two non-negative `Link`s, import and export, each with the cable's loss applied in the direction of flow, and one shared-capacity constraint so both directions together never exceed the rating (`scripts/interconnectors/market_coupling.py`) |
| Cable (historical, default) | a link fixed to the observed ESPENI flow |

The register pipeline before it (rules `ingest_dukes`, `ingest_neso_register`, `combine_datasets`,
`enrich_locations`, `clean`, `map_to_buses`) combines DUKES table 5.13 with the NESO interconnector register, adds
locations and declared overrides, and maps each cable's landing point to a bus of the scenario's own network. Curated
landing buses in `data/interconnectors/bus_mapping.csv` override the coordinate match.

## Method and assumptions

### Which cables

| Scenario type | Cables and ratings |
|---|---|
| Historical | register cables commissioned by the modelled year, at their own ratings |
| FES | register cables scaled uniformly so their total equals FES interconnector capacity (building block Gen_BB022) for the pathway and year; the build fails if the totals differ |
| CP30 | CP30's cable set from `interconnectors.cp30_composition.file`: each register cable at CP30's nominal rating, cables CP30's total implies but the register lacks added, cables outside CP30's count left out. The total must equal ES1 or the build fails; nothing is scaled |

The CP30 composition is a modelling judgement where CP30 does not name a cable; each row records its reason. The same
file feeds the availability profiles, the Capacity Market model and the CP30 checks.

```{figure} /_static/figures/supply/interconnectors_map_2030.png
:alt: Map of Great Britain with a red circle at each interconnector landing, sized by rating, and dashed stubs pointing towards each cable's far end, labelled with cable, country and rating.
:width: 70%
:align: center

The interconnectors of the 2030 build at their GB landing, sized by rating, with a stub towards the far end: CP30's
cable set, 12.45 GW, with MaresConnect added and the Isle of Man cable left out.
```

### Availability

Rule `availability` (`scripts/interconnectors/availability.py`) writes an hourly availability per cable. With
`interconnectors.availability.mode: outage` (default) each cable has planned outages (count, duration, months) and
random forced outages (rate, mean duration), drawn with a fixed seed so rebuilds match. `constant` gives full
availability. Calendars follow the leap-year rule of {doc}`demand/index`.

### Foreign prices

Historical runs with ENTSO-E prices use the year's actual day-ahead prices. Future runs build an hourly import price
and export bid per counterparty from a level and a shape:

| `interconnectors.pricing.foreign_price_source` | Level and shape |
|---|---|
| unset (`entsoe`) | level from `pricing.source` (`derived`: an annual price per country from the FES European generation mix; `entsoe`; `csv`; `cp30`: CP30's published import and export prices); shape from ENTSO-E day-ahead prices of `hourly_shape_year` (`hourly_shape: entsoe`) or flat |
| `cp30_basis` (CP30) | TYNDP 2024 National Trends 2030 zone prices rebased to CP30's gas and carbon, used at their own level (`scripts/interconnectors/cp30_foreign_prices.py`) |
| `tyndp` (FES long range) | TYNDP prices of the modelled year at TYNDP's climate-weighted level in GBP 2024 (`scripts/interconnectors/tyndp_prices.py`) |

Refinements:

- **Wind conditioning** (`shape_conditioning`). Foreign prices fall when GB is windy. A shape taken from another
  year than the weather year loses that link, so the part of each country's price that GB offshore output explains is
  swapped for the weather year's:

  $$
  p'_t = p_t - \beta z^\text{shape}_t + \beta z^\text{weather}_t, \qquad
  \beta = \frac{\operatorname{cov}(p, z^\text{shape})}{\operatorname{var}(z^\text{shape})},
  $$

  with $z$ the standardised GB offshore wind output of each year (`scripts/interconnectors/price_shape.py`).
- **CP30 rebase.** TYNDP's new-CCGT price plateau is mapped to a foreign new CCGT's cost on CP30's fuel and carbon;
  every hour is scaled by that one ratio, so zero-price hours stay zero. Method `refit_2013` instead refits TYNDP's
  price response to residual load on the weather year's wind and solar.
- **Price basis.** `foreign_price_basis: single` uses one hourly price per coupled market (the continent, Norway, the
  single electricity market of Ireland) or, with `single_price_per_country`, per country. `separate` re-centres import
  and export on two annual targets.
- **Trading hurdle.** Import and export prices differ by a bid/ask spread
  (`single_price_transaction_spread_gbp_per_mwh` on the single basis, else `transaction_spread_gbp_per_mwh`) plus the
  cable loss, and by at least `circulation_epsilon_gbp_per_mwh` so power cannot circulate. An export-bid floor
  (`single_price_export_bid_floor_gbp_per_mwh`) stops the hurdle from blocking exports of surplus GB would otherwise
  curtail.
- **Volume tranches** (`volume_tranches`). GB's trade moves the price it trades at. GB's capacity into each coupled
  market is split into equal tranches; each later tranche's import price is one step higher and export price one step
  lower, with step = elasticity x price level / market demand x tranche width (default elasticity 1.0).
- **Sensitivities.** `future_price_case` (`low`, `central`, `high` multipliers), `country_price_overrides_gbp_per_mwh`,
  and `price_floor` / `price_cap` clip the series.

```{figure} /_static/figures/supply/foreign_prices_2030_2050.png
:alt: Horizontal range bars of the hourly import price in GBP/MWh by counterparty country, for 2030 and 2050, with the annual mean marked.
:width: 100%

Hourly import price by counterparty in the 2030 build (CP30 basis, a price per country) and the 2050 build (TYNDP, one
price per coupled market), trading hurdle included: the middle half of hours, the 10th to 90th percentile and the annual
mean.
```

### Dispatch in each stage

| Stage | Behaviour |
|---|---|
| Historical (`historical_flow_mode: fixed`) | flows fixed to ESPENI, so historical dispatch is reproducible |
| Historical back-test (`market_coupled`) | flows chosen against ENTSO-E prices, compared with ESPENI afterwards |
| Single solve | flows optimised against the foreign prices with the full network |
| Stage 1 (wholesale) | flows optimised on the copperplate GB market against the same prices |
| Balancing | flows held at Stage 1 (`market.balancing.fix_interconnectors`), or re-traded at a cost with `market.balancing.interconnector_retrading`; see {doc}`/model/dispatch/interconnector_trading` |

GB-internal DC links (for example the Eastern and Western HVDC links) are part of the network, not interconnectors;
see {doc}`network/topology`.

## Data

| Input | File |
|---|---|
| DUKES table 5.13 | `data/interconnectors/DUKES_5.13_2025.csv` |
| NESO interconnector register | `data/interconnectors/NESO_interconnector_register.csv` |
| Overrides and landing buses | `data/interconnectors/overrides.csv`, `bus_mapping.csv` |
| CP30 cable set | `data/interconnectors/cp30_interconnector_composition.csv` |
| Historical flows | ESPENI (rule `extract_historical_interconnector_flows`) |
| ENTSO-E day-ahead prices | rule `fetch_entsoe_day_ahead_prices` (API token in `.env` or `ENTSOE_API_TOKEN`) |
| European generation mix | rules `download_european_generation_mix`, `process_european_generation_mix` |
| Fuel and carbon prices for the derived price | rule `extract_FES_prices` (the FES workbook's CP1 and CP2 Base Case; FES 2024 publishes no Base Case gas price, so the mean of its High and Low cases) |
| TYNDP prices | `data/interconnectors/tyndp2024/` (CC-BY 4.0; see its README) |

See {doc}`/reference/data/data_sources`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `interconnectors.historical_flow_mode` | `fixed` | `fixed` or `market_coupled` |
| `interconnectors.availability` | `outage`, 1 planned outage of 168 h, 3% forced | cable outages |
| `interconnectors.pricing.source` | `derived` | price level source |
| `interconnectors.pricing.foreign_price_source` | unset (CP30: `cp30_basis`; FES long range: `tyndp`) | own-level price files |
| `interconnectors.pricing.hourly_shape`, `hourly_shape_year` | `entsoe`, weather year | shape source and year |
| `interconnectors.pricing.shape_conditioning.enabled` | `false` (CP30: `true`) | wind conditioning |
| `interconnectors.pricing.foreign_price_basis` | `single` | `single` or `separate` |
| `interconnectors.pricing.single_price_transaction_spread_gbp_per_mwh` | 20.0 | trading hurdle, GBP/MWh |
| `interconnectors.pricing.volume_tranches.enabled` | `false` (CP30: `true`) | price response to GB trade |
| `interconnectors.pricing.cp30_basis` | `rebase_cy2009` (CP30: `refit_2013`) | CP30 rebase method |
| `interconnectors.cp30_composition.file` | CP30 overlay | CP30 cable set |
| `market.balancing.fix_interconnectors`, `interconnector_retrading` | | balancing treatment |

Defaults are in `config/defaults/base/interconnectors.yaml`; CP30 settings in `config/defaults/source/CP30.yaml` and,
for the shape year, wind conditioning and volume tranches, the CP30 2030 study's templates.

## Validation

- CP30 checks compare cable ratings and landing substations with CP30 and the register (rules
  `validate_CP30_locational`, `validate_CP30_fleet_location`), and Stage 1 interconnector energy with ES1 (rule
  `validate_CP30_stage1`, failing beyond 20%).
- `scripts/interconnectors/validate_historical_flows.py` compares market-coupled back-test flows with ESPENI.
- FES long-range years: rule `validate_fes_dispatch` checks interconnector energy against FES ES1, and a diagnostic
  rule reports trade against foreign prices.

## Limitations

- Foreign markets are price takers beyond the volume tranches; there is no model of European dispatch.
- One loss factor per cable; no reactive power or converter limits.
- Foreign price shapes come from another year or a published model, conditioned only on GB offshore wind.
