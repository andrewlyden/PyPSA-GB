## `interconnectors`

File: `config/defaults/base/interconnectors.yaml`. Keys below are relative to `interconnectors`.

| Key | Default | Description |
|---|---|---|
| `historical_flow_mode` | `fixed` | Interconnectors: cross-border cable availability and foreign prices (EU_supply generators and the trade links). Historical scenarios (modelled_year &lt;= 2024) with pricing.source "entsoe" use the year's actual ENTSO-E prices unscaled. historical_flow_mode: "fixed" imposes ESPENI flows so historical dispatch is reproducible; "market_coupled" lets the model choose each link's flow from ENTSO-E prices (an interconnector back-test, compared with ESPENI afterwards). "fixed" \| "market_coupled" (opt-in validation) |
| `availability` | (mapping) | Cable outages, so a long run does not assume perfect cables: planned outages (count, hours, months 1-12) and a forced outage rate (share 0-1) with mean duration in hours; seed fixes the random draw. |
| `availability.mode` | `outage` |  |
| `availability.planned_outages_per_year` | `1` |  |
| `availability.planned_outage_duration_hours` | `168` |  |
| `availability.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `availability.forced_outage_rate` | `0.03` |  |
| `availability.forced_outage_mean_duration_hours` | `24` |  |
| `availability.seed` | `42` |  |
| `pricing` | (mapping) |  |
| `pricing.source` | `derived` | source: "derived" (annual mean per country from the FES European generation-mix merit order, no token), "entsoe" (hourly ENTSO-E day-ahead prices; token in .env or ENTSOE_API_TOKEN, falls back to "derived"), "csv" (user file, csv_file) or "cp30" (NESO CP30 workbook CP.29 annual import/export prices; the CP30 overlay). "derived" \| "entsoe" \| "csv" \| "cp30" |
| `pricing.hourly_shape` | `entsoe` | Future scenarios: "entsoe" takes the hourly ENTSO-E shape of the weather year (or hourly_shape_year) re-centred on the annual target; "flat" uses the annual target in every hour. "entsoe" \| "flat" (future scenarios) |
| `pricing.eur_to_gbp` | `0.86` | EURGBP conversion rate |
| `pricing.gb_price` | `50.0` | Fallback foreign price target (GBP/MWh) for a country with no derived price |
| `pricing.price_floor` | `-500.0` | Lower clip on foreign prices (GBP/MWh); keeps genuine negative prices |
| `pricing.price_cap` | `500.0` | Upper clip on foreign prices (GBP/MWh) |
| `pricing.future_price_case` | `central` | low \| central \| high |
| `pricing.price_case_multipliers` | (mapping) |  |
| `pricing.price_case_multipliers.low` | `0.75` |  |
| `pricing.price_case_multipliers.central` | `1.0` |  |
| `pricing.price_case_multipliers.high` | `1.25` |  |
| `pricing.country_price_overrides_gbp_per_mwh` | `{}` |  |
| `pricing.transaction_spread_gbp_per_mwh` | `1.0` | Bid/ask spread (GBP/MWh of trade, half each side) on top of the cable loss: the ~EUR 1/MWh cross-border hurdle of market models (MacIver, Bukhsh & Bell 2021, Energy Policy). Exports also pay the loss on the price (export slope efficiency^2 x bid). |
| `pricing.circulation_epsilon_gbp_per_mwh` | `0.01` | Minimum gap (GBP/MWh) between import ask and export bid; stops circulation |
| `pricing.foreign_price_basis` | `single` | "single": one hourly price per coupled market (volume_tranches.markets; continental zones capacity-weighted) = ENTSO-E shape + a level offset per market, import and export differing only by the spread and the loss. "separate": import and export re-centred on two annual targets (e.g. CP.29's import and export prices). A flat-shape run keeps separate flat targets. "single" \| "separate" |
| `pricing.single_price_per_country` | `false` | true: one price per counterparty country instead of one per coupled market (own-level price sources only). |
| `pricing.cp30_basis` | (mapping) | TYNDP NT2030 prices rebased to CP30's fuel prices, used when foreign_price_source is "cp30_basis" (scripts/interconnectors/cp30_foreign_prices.py). Prices in EUR/MWh or GBP/MWh as named. |
| `pricing.cp30_basis.method` | `rebase_cy2009` | rebase_cy2009 \| refit_2013 |
| `pricing.cp30_basis.tyndp_new_ccgt_eur` | `82.0` |  |
| `pricing.cp30_basis.foreign_new_ccgt_gbp` | `100.62` |  |
| `pricing.cp30_basis.target_price_cap_eur` | `300.0` | refit_2013: TYNDP scarcity prices capped before the fit |
| `pricing.cp30_basis.ireland_source` | `entsoe_2024` | refit_2013: entsoe_2024 \| refit |
| `pricing.cp30_basis.ireland_offset_gbp` | `7.5` |  |
| `pricing.single_price_level_offsets_gbp_per_mwh` | (mapping) |  |
| `pricing.single_price_level_offsets_gbp_per_mwh.continent` | `0.0` |  |
| `pricing.single_price_level_offsets_gbp_per_mwh.norway` | `0.0` |  |
| `pricing.single_price_level_offsets_gbp_per_mwh.sem` | `0.0` |  |
| `pricing.single_price_transaction_spread_gbp_per_mwh` | `20.0` | Single basis only: the symmetric trading hurdle (full bid/ask spread, GBP/MWh). null = transaction_spread_gbp_per_mwh. |
| `pricing.single_price_export_bid_floor_gbp_per_mwh` | `1.0` | Single basis only: export-bid floor (GBP/MWh). Where the foreign price is positive the export bid is at least min(foreign price, floor), so the hurdle cannot stop GB exporting surplus it would otherwise curtail; imports keep the full hurdle. null = off. |
| `pricing.ramp_limit_per_hour` | `null` | Per-unit link ramp; null is unconstrained |
| `pricing.csv_file` | `null` | Path to user-supplied CSV (source: "csv" only) |
| `pricing.shape_conditioning` | (mapping) | hourly_shape_year (unset = renewables_year) picks the ENTSO-E year whose shape is used. ENTSO-E day-ahead prices start in 2015, so a 2013-weather run must name a later year. A shape laid over another year's weather loses the negative correlation between GB wind and foreign prices. Conditioning swaps the shape year's GB offshore wind term for the weather year's (scripts/interconnectors/price_shape.py). |
| `pricing.shape_conditioning.enabled` | `false` |  |
| `pricing.volume_tranches` | (mapping) | The foreign price responds to GB's summed trade with each coupled market: equal tranches of GB's capacity into the market, each later tranche's import price one step higher and export price one step lower, with step = elasticity x CP.29 level / market demand x tranche width (scripts/interconnectors/volume_tranches.py). Elasticity 1.0 from Swinand and O'Mahoney (2014, MPRA 56246); Cludius et al. (2014) imply about 1.6 for Germany. Market demand from NESO FES 2025 ES2, 2030. |
| `pricing.volume_tranches.enabled` | `false` |  |
| `pricing.volume_tranches.elasticity` | `1.0` |  |
| `pricing.volume_tranches.tranches` | `3` |  |
| `pricing.volume_tranches.demand_file` | `resources/interconnectors/european_generation_mix_2025.csv` |  |
| `pricing.volume_tranches.eu_scenario` | `EU Green` |  |
| `pricing.volume_tranches.pathway_alignment` | `Holistic Transition` |  |
| `pricing.volume_tranches.demand_year` | `2030` |  |
| `pricing.volume_tranches.markets` | (mapping) |  |
| `pricing.volume_tranches.markets.continent` | (mapping) | one coupled market: CWE plus DK1 and NeuConnect's Germany |
| `pricing.volume_tranches.markets.continent.counterparties` | `[France, Belgium, Netherlands, Denmark, Germany]` |  |
| `pricing.volume_tranches.markets.continent.demand_countries` | `[France, Belgium, Netherlands, Germany, Denmark]` |  |
| `pricing.volume_tranches.markets.norway` | (mapping) |  |
| `pricing.volume_tranches.markets.norway.counterparties` | `[Norway]` |  |
| `pricing.volume_tranches.markets.norway.demand_countries` | `[Norway]` |  |
| `pricing.volume_tranches.markets.sem` | (mapping) | all-island market; the Isle of Man is priced on it |
| `pricing.volume_tranches.markets.sem.counterparties` | `[Ireland, Northern Ireland, Isle Of Man]` |  |
| `pricing.volume_tranches.markets.sem.demand_countries` | `[SEM]` |  |
