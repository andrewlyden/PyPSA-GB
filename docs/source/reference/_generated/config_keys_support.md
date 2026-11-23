## `support`

File: `config/defaults/base/support.yaml`. Keys below are relative to `support`.

| Key | Default | Description |
|---|---|---|
| `labels` | (mapping) | Support schemes (CfD, RO, DPA): which generators hold which support, and how support enters dispatch costs, balancing bids and post-solve revenue. labels: assigns support_type (CfD \| ROC \| merchant) to generators during integration. Needed by dispatch_costs, bm_bids and revenue; when disabled every generator is treated as merchant. |
| `labels.enabled` | `true` |  |
| `labels.roc_eligibility_years` | `20` | Years of ROC support from operational_date (standard accreditation period) |
| `labels.dispatchable_support_overrides_file` | `null` | Optional exact-name cross-walks for unit-level support labels. |
| `labels.renewable_support_overrides_file` | `null` |  |
| `labels.historical_station_unit_splits_file` | `null` | Optional sourced replacement of planning-record generators by separately accredited physical units, applied before aggregation. |
| `labels.historical_cfd_tranches` | (mapping) | Historical only: splits shared physical projects into LCCC contract tranches; unsupported capacity is merchant. |
| `labels.historical_cfd_tranches.enabled` | `false` |  |
| `labels.historical_cfd_tranches.mapping_file` | `null` |  |
| `labels.historical_cfd_tranches.actuals_file` | `null` |  |
| `labels.historical_roc_station_labels` | (mapping) | Historical back-tests only: uses the calendar year's Ofgem RO output to correct or add station-evidenced ROC labels before marginal costs. Date-inferred labels keep their own confidence metadata; the post-solve ROC reconciliation remains the strict certificate test. |
| `labels.historical_roc_station_labels.enabled` | `false` |  |
| `labels.historical_roc_station_labels.registry_file` | `null` |  |
| `labels.historical_roc_station_labels.station_overrides_file` | `null` |  |
| `labels.historical_roc_station_labels.residual_representation_assumptions_file` | `null` |  |
| `labels.historical_roc_station_labels.source_register_evidence` | `true` | Builds a source-register crosswalk before the solve (every issuing station against historical REPD extracts and DUKES); only explicit or high-confidence matches may change dispatch labels. |
| `labels.historical_roc_station_labels.buyout_price` | `50.05` | GBP/ROC |
| `labels.roc_closure_date` | `2017-04-01` | Non-CfD renewables operational before this date with no REPD ro_banding are inferred ROC-accredited at default_roc_bandings. The RO closed to new accreditation on 1 April 2017. |
| `labels.default_roc_bandings` | (mapping) | ROC/MWh by carrier where REPD ro_banding is blank (Ofgem banding, final accreditation period); applied only to sites operational before roc_closure_date. |
| `labels.default_roc_bandings.wind_onshore` | `0.9` | Ofgem Band 2 (2013-2017) |
| `labels.default_roc_bandings.wind_offshore` | `2.0` | Ofgem Band 2 (pre-2015 projects, most legacy) |
| `labels.default_roc_bandings.solar_pv` | `1.2` | Ofgem Band 3 (&gt;50 kW, 2013-2015) |
| `labels.default_roc_bandings.small_hydro` | `1.0` | Ofgem Band 2 (up to 20 MW) |
| `labels.default_roc_bandings.large_hydro` | `1.0` | Ofgem Band 2 |
| `labels.default_roc_bandings.tidal_stream` | `5.0` | Ofgem Band 1 (emerging tech) |
| `labels.default_roc_bandings.shoreline_wave` | `5.0` | Ofgem Band 1 (emerging tech) |
| `labels.default_roc_bandings.biomass` | `1.5` | Ofgem dedicated biomass band |
| `labels.cfd_vintage_schedule_file` | `null` | Optional sourced capacity schedule splitting CfD capacity into allocation-round tranches. null: every contract settles at one carrier-wide future_\* strike. |
| `labels.fes_cfd_fractions` | (mapping) | FES-sourced generators (no REPD data): share 0-1 of each carrier's capacity labelled CfD, the rest merchant. Unlisted carriers are all merchant. Nuclear is deliberately absent: CP30 places nuclear at named stations with their own labels (fleet.spatial_proxy.nuclear_station_schedule, data/market/CP30/nuclear_station_schedule_2030.csv), and Sizewell C is financed by a Regulated Asset Base (bill line sizewell_c_rab), not a CfD. |
| `labels.fes_cfd_fractions.wind_offshore` | `1.0` | All new offshore assumed CfD |
| `labels.fes_cfd_fractions.wind_onshore` | `0.3` | 30% CfD, 70% merchant |
| `labels.fes_cfd_fractions.solar_pv` | `0.5` | 50% CfD, 50% merchant |
| `labels.cfd_strike_prices` | (mapping) | CfD strike prices by allocation round (GBP/MWh, no declared price year), for post-solve revenue tracking in historical runs only (not dispatch costs). Legacy values, several known wrong: AR5_offshore is the AR4 clearing price (AR5 drew no offshore bids) and the AR6 entries are administrative strike prices, not the cleared 58.87 / 50.90 / 50.07. Consumer-bill cases do not use this table; they read cleared prices (2012 prices) from data/market/lccc/cfd_strike_prices_by_round.csv. |
| `labels.cfd_strike_prices.AR1_offshore` | `150.0` |  |
| `labels.cfd_strike_prices.AR2_offshore` | `75.0` |  |
| `labels.cfd_strike_prices.AR3_offshore` | `58.0` |  |
| `labels.cfd_strike_prices.AR4_offshore` | `40.0` |  |
| `labels.cfd_strike_prices.AR5_offshore` | `37.35` |  |
| `labels.cfd_strike_prices.AR6_offshore` | `73.0` |  |
| `labels.cfd_strike_prices.AR6_onshore` | `53.0` |  |
| `labels.cfd_strike_prices.AR6_solar` | `47.0` |  |
| `labels.cfd_strike_prices.future_offshore` | `80.0` | Projected for 2030s |
| `labels.cfd_strike_prices.future_onshore` | `55.0` |  |
| `labels.cfd_strike_prices.future_solar` | `50.0` |  |
| `labels.cfd_strike_prices.future_nuclear` | `92.5` | Hinkley Point C's strike (2012 prices) |
| `labels.support_registers` | (mapping) | Scheme registers as the authority for named sites' support labels (scripts/renewables/support_register_labels.py). REPD leaves RO banding blank for most sites and CfD capacity blank for some, and the date rule misses the RO grace period. The RO eligibility test is the bill's RO line's own. Disabled: REPD and the date rule alone. |
| `labels.support_registers.enabled` | `true` |  |
| `labels.support_registers.ro_register` | `data/market/ofgem_renewables_obligation/accredited_stations_ro_curr...` |  |
| `labels.support_registers.ro_certificates` | `data/market/ofgem_renewables_obligation/certificates_ro_current.csv` |  |
| `labels.support_registers.ro_reference_date` | `null` | null: 1 April of modelled_year (the RO obligation year) |
| `labels.support_registers.lccc_portfolio` | `data/market/lccc/cfd_contract_portfolio_status.csv` |  |
| `labels.support_registers.repd` | `data/renewables/repd-q2-jul-2025.csv` |  |
| `labels.support_registers.crosswalk_overrides` | `data/market/support/support_register_crosswalk_overrides.csv` |  |
| `labels.support_registers.checked_cfd_contracts` | `data/market/lccc/checked_cfd_contracts.csv` | Named sites' CfD contracts, checked against the LCCC portfolio one row per round (support_register_labels.apply_checked_cfd_contracts). Part-contracted sites split into their rounds plus a merchant "&lt;site&gt;_non_CfD" remainder; multi-round sites settle each round at its own strike. Sites with terminated or no contracts carry action not_built_refill: left out of the named fleet, their MW refilled by the regional proxy as new capacity, their terminated awards kept netted off their rounds. null: REPD and the matcher alone. |
| `labels.support_registers.dispatchable_proxy_share` | (mapping) | Anonymous CP30 biomass and energy-from-waste proxies take the share of their capacity the RO register holds as live accreditations (certificates issued since min_certificate_year, the bill's activity rule). |
| `labels.support_registers.dispatchable_proxy_share.enabled` | `true` |  |
| `labels.support_registers.dispatchable_proxy_share.carriers` | `[biomass, waste_to_energy]` |  |
| `labels.support_registers.dispatchable_proxy_share.min_certificate_year` | `2023` |  |
| `labels.support_registers.renewable_proxy_share` | (mapping) | Marine (tidal and wave) proxies likewise; no activity rule. |
| `labels.support_registers.renewable_proxy_share.enabled` | `true` |  |
| `labels.support_registers.renewable_proxy_share.carriers` | `[marine]` |  |
| `labels.support_registers.renewable_proxy_share.min_certificate_year` | `null` |  |
| `labels.support_registers.renewable_proxy_share.remainder_support` | `CfD` | Marine proxy capacity the RO register does not hold is tidal stream on CfDs (AR4-AR7a ring-fenced tidal stream, data/market/lccc/cfd_strike_prices_by_round.csv): the CfD vintage schedule assigns those rounds, the rest at the future tidal strike. |
| `bm_bids` | (mapping) | Unit-level support treatment of modelled future renewable balancing bids and offers (ELEXON historical prices are left untouched). Merchant assets keep their technology-specific opportunity-cost fallback. |
| `bm_bids.enabled` | `true` |  |
| `bm_bids.mode` | `fixed` | "fixed": the assumptions below; "empirical": BOD calibration for carrier/support groups with at least min_bmus directly matched BMUs. |
| `bm_bids.case` | `central` |  |
| `bm_bids.min_bmus` | `3` |  |
| `bm_bids.legal_cfd_crosswalk` | (mapping) | Optional bid-only addition for high-confidence LCCC CfD BMUs (ROC excluded until Ofgem accreditation evidence is provided). |
| `bm_bids.legal_cfd_crosswalk.enabled` | `false` |  |
| `bm_bids.legal_cfd_crosswalk.file` | `null` |  |
| `bm_bids.cfd` | (mapping) | Bid and offer prices (GBP/MWh) by support type where evidence is missing. |
| `bm_bids.cfd.offer_price` | `0.0` |  |
| `bm_bids.cfd.bid_price` | `0.0` |  |
| `bm_bids.roc` | (mapping) |  |
| `bm_bids.roc.offer_price` | `0.0` |  |
| `bm_bids.roc.bid_multiplier` | `1.0` | ro_banding x model-year ROC buyout price |
| `bm_bids.exclude_carriers` | `[nuclear, CCS Biomass, biomass]` | Carriers whose support label does not override their carrier bid/offer prices: nuclear keeps its carrier override (a zero CfD offer would let the BM turn Hinkley Point C up for free); CCS Biomass and biomass are fuelled CfD plant, whose bids must reflect the fuel saved when turned down. Must match the code default in market_utils._apply_renewable_support_pricing. |
| `revenue` | (mapping) | Post-solve CfD difference payments and ROC income. Requires market.enabled and labels.enabled. |
| `revenue.enabled` | `true` |  |
| `revenue.include_cfd` | `true` | Calculate CfD top-up/clawback payments |
| `revenue.include_roc` | `true` | Calculate ROC income |
| `revenue.output_format` | `csv` | "csv", "json", or "both" |
| `dispatch_costs` | (mapping) | Support-aware dispatch costs: CfD units at cfd_dispatch_cost (GBP/MWh), ROC units at minus banding x buyout price. |
| `dispatch_costs.enabled` | `true` |  |
| `dispatch_costs.cfd_dispatch_cost` | `0.0` |  |
| `dispatch_costs.cfd_dispatch_cost_exempt_carriers` | `[CCS Biomass, biomass]` | Fuelled CfD carriers that keep their fuel-and-carbon marginal cost instead of cfd_dispatch_cost (the network marginal cost is also the settlement's fuel cost). BECCS runs baseload through min_output instead. biomass: Drax and MGT CfDs |
| `dispatch_costs.dpa_margin_gbp_per_mwh` | `1.0` | Dispatchable Power Agreement plant (Net Zero Teesside Power) is dispatched ahead of unabated gas: its dispatch cost is at most the cheapest unabated CCGT's marginal cost less this margin (GBP/MWh). |
| `dispatch_costs.roc_buyout_prices` | (mapping) | ROC buy-out price (GBP/ROC) by the calendar year in which the April-March obligation period starts. Not the realised value of a ROC, which also includes recycling. |
| `dispatch_costs.roc_buyout_prices.2010` | `36.99` | Ofgem published values. |
| `dispatch_costs.roc_buyout_prices.2011` | `38.69` |  |
| `dispatch_costs.roc_buyout_prices.2012` | `40.71` |  |
| `dispatch_costs.roc_buyout_prices.2013` | `42.02` |  |
| `dispatch_costs.roc_buyout_prices.2014` | `43.3` |  |
| `dispatch_costs.roc_buyout_prices.2015` | `44.33` |  |
| `dispatch_costs.roc_buyout_prices.2016` | `44.77` |  |
| `dispatch_costs.roc_buyout_prices.2017` | `45.58` |  |
| `dispatch_costs.roc_buyout_prices.2018` | `47.22` |  |
| `dispatch_costs.roc_buyout_prices.2019` | `48.78` |  |
| `dispatch_costs.roc_buyout_prices.2020` | `50.05` |  |
| `dispatch_costs.roc_buyout_prices.2021` | `50.8` |  |
| `dispatch_costs.roc_buyout_prices.2022` | `52.88` |  |
| `dispatch_costs.roc_buyout_prices.2023` | `59.01` |  |
| `dispatch_costs.roc_buyout_prices.2024` | `64.73` |  |
| `dispatch_costs.roc_buyout_prices.2025` | `67.06` |  |
| `dispatch_costs.roc_buyout_prices.2026` | `69.34` |  |
| `dispatch_costs.roc_buyout_prices.2027` | `71.42` | Forecast: 3% a year nominal escalation from the 2026-27 value; replace with Ofgem's value once published. |
| `dispatch_costs.roc_buyout_prices.2028` | `73.56` |  |
| `dispatch_costs.roc_buyout_prices.2029` | `75.77` |  |
| `dispatch_costs.roc_buyout_prices.2030` | `78.04` |  |
| `dispatch_costs.roc_buyout_prices.2031` | `80.38` |  |
| `dispatch_costs.roc_buyout_prices.2032` | `82.79` |  |
| `dispatch_costs.roc_buyout_prices.2033` | `85.27` |  |
| `dispatch_costs.roc_buyout_prices.2034` | `87.83` |  |
| `dispatch_costs.roc_buyout_prices.2035` | `90.46` |  |
| `dispatch_costs.roc_buyout_prices.2036` | `93.18` |  |
| `dispatch_costs.roc_buyout_prices.2037` | `95.97` |  |
