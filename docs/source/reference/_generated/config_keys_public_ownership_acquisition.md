## `public_ownership.acquisition`

File: `config/defaults/base/public_ownership/acquisition.yaml`. Keys below are relative to `public_ownership.acquisition`.

| Key | Default | Description |
|---|---|---|
| `interconnectors` | (mapping) | Purchase of the legacy fleet. Acquisition is switched on by any case whose legacy_fleet lever is not private (the fully_nationalised family). The purchase-price basis, thermal basis, annuity term and who pays the annuity are levers (purchase_price_basis, thermal_basis, acquisition_term_years, acquisition_financing: the tariff or general taxation); this block holds only what they need. Interconnectors: every tariff case (one that buys the legacy fleet) also buys the GB halves of the interconnectors, whether or not it buys the wires (GB shares from the bill's interconnector_cap_and_floor table). market_value  the GB half's Stage 1 congestion rent (floored at zero per cable) after corporation tax, times the annuity factor at owner_real_rate over the cable's remaining years (the workbook's post-tax operating rate for interconnectors, Legacy Fleet column AF on their Sc1 hurdle; no capital allowances, as the workbook has no book value for them), annuitised at the public rate over the case's acquisition term explicit      price_gbp by GB-half rent Remaining years per cable: a cap-and-floor cable runs to the end of its regime (regime_years from first operation) or of its asset life, whichever comes first; a cable outside a regime to the end of its asset life. First operation from the bill's regime table, else commissioning_year_by_cable (DUKES 5.13); remaining_life_years_by_cable is a declared override. Code defaults: IC_ACQUISITION_DEFAULTS in nationalised_tariff.py (must match). |
| `interconnectors.purchase_price_basis` | `market_value` |  |
| `interconnectors.owner_real_rate` | `0.0487` |  |
| `interconnectors.corporation_tax_rate` | `0.25` |  |
| `interconnectors.regime_years` | `25.0` |  |
| `interconnectors.asset_life_years` | `40.0` | DNV for NGESO, Network Topology Assessment (6 Mar 2024) p.7: HVDC life ~40 years, major refurbishment ~20 years; National Grid HVDC technical information p.7. |
| `interconnectors.commissioning_year_by_cable` | (mapping) |  |
| `interconnectors.commissioning_year_by_cable.Britned` | `2011` |  |
| `interconnectors.commissioning_year_by_cable.ElecLink` | `2022` |  |
| `interconnectors.remaining_life_years_by_cable` | (mapping) |  |
| `interconnectors.remaining_life_years_by_cable.IFA` | `10.0` | IFA (1986), declared: no retirement date published; converters replaced 2011-12 (GE case study 2018; RTE IPST 2017 'life extension strategy'); a 25-year converter life from 2012 would give 7 years. |
| `interconnectors.price_gbp` | `null` |  |
| `interconnectors.owner_real_rate_by_setting` | (mapping) | Lever interconnector_price: the owner rate the GB halves are valued at (low price = the higher rate). A declared +/- 1 pp on the workbook's rate, which gives no range. |
| `interconnectors.owner_real_rate_by_setting.low` | `0.0587` |  |
| `interconnectors.owner_real_rate_by_setting.high` | `0.0387` |  |
| `hpc_completion` | `data/public_ownership/financing/hpc_completion_cost.csv` | Hinkley Point C completed by the state (in_construction: public_completion): bought at its sunk cost and finished on the public cash rate. The table holds EDF's sourced total and sunk cost (2025 URD). The central case buys it as an operating plant at its contract value. |
| `transaction_cost_share` | (mapping) | Lever acquisition_transaction_costs: advisers', legal and stamp costs as a share of everything bought (purchase, GB interconnector halves, wires). |
| `transaction_cost_share.two_pct` | `0.02` |  |
| `extended_asset_lives` | (mapping) | Lever legacy_asset_lives: extended = extra_years more for these carriers (life extension without repowering), and Sizewell B to EDF's 2055. |
| `extended_asset_lives.extra_years` | `5` |  |
| `extended_asset_lives.carriers` | `[wind_onshore, wind_offshore, solar_pv, marine]` |  |
| `extended_asset_lives.retirement_year_overrides` | (mapping) |  |
| `extended_asset_lives.retirement_year_overrides.Sizewell B` | `2055` |  |
| `repowering_sensitivity` | (mapping) | Report-only sensitivity (scripts/public_ownership/repowering_sensitivity.py): the purchase price's change at the tariff's annuity with `share` of `carrier` repowered. Off: the central valuation has no private life extension, so there is no repowering share to move. Kept outside valuation, which accepts only legacy_valuation.DEFAULT_SETTINGS. |
| `repowering_sensitivity.enabled` | `false` |  |
| `repowering_sensitivity.carrier` | `wind_onshore` |  |
| `repowering_sensitivity.share` | `0.5` |  |
| `valuation` | (mapping) | The legacy fleet's cash flows valued per register row as a declining profile, at the public real rate (what the flows are worth to the state) and at the owners' rate (what the owners would sell for). Flows are held flat at their model-year level until their own expiry: RO accreditation runs ro_accreditation_years from the Ofgem effective date and the scheme closes ro_closure_date; a legacy CfD runs its round's term from its start; plant retires at commissioning plus its carrier life, undated rows at their carrier's median dated commissioning year. Only the values listed in legacy_valuation.DEFAULT_SETTINGS are accepted. Built by snakemake resources/public_ownership/cp30_2030_cp30_network_legacy_valuation_summary.json |
| `valuation.horizon_year` | `life` | 2070 before), and every stream (output, wholesale, balancing, fuel and carbon, Capacity Market) on the solved years' path per MW by technology (anchor_streams all; "price" moved the wholesale price only). legacy_valuation.py. |
| `valuation.anchor_streams` | `all` |  |
| `valuation.ro_register` | `data/market/ofgem_renewables_obligation/accredited_stations_ro_curr...` |  |
| `valuation.ro_closure_date` | `2037-03-31` |  |
| `valuation.ro_accreditation_years` | `20` |  |
| `valuation.cfd_contract_years_from_award` | `18` | Vintaged (2050) registers only: a CfD ends this many years after its round's award. A 2030 register dates each CfD from its start for its round's term. Starts are the LCCC first settlement (or expected start) date per contract phase in cfd_contract_starts (built from data/market/lccc/actual_cfd_generation_and_payments.csv, cfd_contract_portfolio_status.csv and cfd_pypsa_mapping_overrides.csv), else the middle of the commissioning year, else the award year plus cfd_start_years_after_award_undated; a phase covers its contracted MW only (Seagreen: 454 of 1,075 MW). Terms: 15 years for FIDeR and AR1-AR6, 20 from AR7 (DESNZ AR7 final response, July 2025), Hinkley Point C 35, Drax's bridge 4. |
| `valuation.cfd_contract_end_year_overrides` | (mapping) |  |
| `valuation.cfd_contract_end_year_overrides.HPC` | `2065` |  |
| `valuation.cfd_contract_end_year_overrides.DRAX_BRIDGE` | `2031.25` | Drax bridging CfD: April 2027 to March 2031 (DESNZ/Drax, 5 Nov 2025; data/market/CP30/biomass_station_schedule_2030.csv), not award + 18 (2043). |
| `valuation.cfd_contract_starts` | `data/public_ownership/financing/legacy_cfd_contract_starts.csv` |  |
| `valuation.cfd_contract_term_years_by_round` | (mapping) |  |
| `valuation.cfd_contract_term_years_by_round.Investment Contract` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.MGT_IC` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR1` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR2` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR3` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR4` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR5` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.AR6` | `15` |  |
| `valuation.cfd_contract_term_years_by_round.DRAX_BRIDGE` | `4` |  |
| `valuation.cfd_contract_term_years_by_round.HPC` | `35` |  |
| `valuation.cfd_contract_term_years_by_round.POWER_BECCS` | `15` | dual CfD, both legs |
| `valuation.cfd_contract_term_years_default` | `20` |  |
| `valuation.dpa_contract_term_years` | `15` | Net Zero Teesside's DPA, from commissioning                # AR7 on (and the unassigned future rounds) |
| `valuation.cfd_start_years_after_award_undated` | `3` | an undated row ends at award + 3 + 15 = award + 18 |
| `valuation.cfd_first_delivery_year_by_round` | (mapping) | Contracts audit: a round's first delivery year floors a commissioning-dated start (AR8: April 2029). |
| `valuation.cfd_first_delivery_year_by_round.AR8_modelled` | `2029.25` |  |
| `valuation.cm_agreement_floor` | (mapping) | Continuity audit: a gas station runs at least to the end of its last Capacity Market agreement. |
| `valuation.cm_agreement_floor.file` | `data/market/CP30/gas_station_schedule_2030.csv` |  |
| `valuation.cm_agreement_floor.carriers` | `[CCGT, OCGT, gas_engine]` |  |
| `valuation.commissioning_profile_by_carrier` | (mapping) | Continuity audit: the 2030 register's batteries (all on one date) take the fleet's build profile (FES 2025 HT additions 2025-30; the pre-2025 stock on REPD's operational years), not one year; a later year's register keeps its cohorts' own build years. |
| `valuation.commissioning_profile_by_carrier.Battery` | `data/public_ownership/financing/battery_build_profile_2030.csv` |  |
| `valuation.capacity_from_year` | (mapping) | Hinkley Point C: CP30's 2,317 MW in 2030 is unit 1 (1,630 MW net) plus unit 2 prorated from its 31 July 2030 TEC date (data/market/CP30/nuclear_station_schedule_2030.csv); both units (2 x 1,630 MW) run from 2031. |
| `valuation.capacity_from_year.Hinkley Point C` | (mapping) |  |
| `valuation.capacity_from_year.Hinkley Point C.from_year` | `2031` |  |
| `valuation.capacity_from_year.Hinkley Point C.capacity_mw` | `3260.0` |  |
| `valuation.new_contracts` | (mapping) | Sizewell B: "20-year Contract for Difference at GBP 70.50 per megawatt hour in 2025 prices ... starting from 2035, the original closure date", to 2055. Paid as strike x output less its wholesale revenue; 2025 to 2024 money at the bill's CPIH rebasing (132.9 / 138.0). |
| `valuation.new_contracts.Sizewell B` | (mapping) |  |
| `valuation.new_contracts.Sizewell B.from_year` | `2035` |  |
| `valuation.new_contracts.Sizewell B.term_years` | `20` |  |
| `valuation.new_contracts.Sizewell B.strike_gbp_per_mwh` | `70.5` |  |
| `valuation.new_contracts.Sizewell B.rebasing_to_model_price_year` | `0.9630434782608696` |  |
| `valuation.declared_margins` | (mapping) | Drax: its contract years valued at its own margin, not the model's. Drax's published guidance for the bridging CfD: "Targeting average Adj. EBITDA of GBP 100-200 million pa from DPS over agreement period" (April 2027 to March 2031, all four 645 MW biomass units, gain share included; Drax Group plc RNS "Low-carbon dispatchable CfD agreement", 10 Feb 2025; strike at signing GBP 109.90/MWh in 2012 money). The midpoint, GBP 150 m for 2,580 MW, scaled to the modelled capacity and to 2024 money (taken as 2025 money). Drax closes when the contract ends (close_at_end): no merchant months after March 2031. |
| `valuation.declared_margins.Drax Power Station (biomass units)` | (mapping) |  |
| `valuation.declared_margins.Drax Power Station (biomass units).segment_ebitda_gbp` | `150000000.0` |  |
| `valuation.declared_margins.Drax Power Station (biomass units).segment_capacity_mw` | `2580.0` |  |
| `valuation.declared_margins.Drax Power Station (biomass units).through_year` | `2031.25` |  |
| `valuation.declared_margins.Drax Power Station (biomass units).rebasing_to_model_price_year` | `0.9630434782608696` |  |
| `valuation.declared_margins.Drax Power Station (biomass units).close_at_end` | `true` |  |
| `valuation.gate_fee_income` | (mapping) | Energy from waste is paid to take the waste it burns: WRAP Gate Fees Report 2024/25 median GBP 121/t (file), times the tonnes each MWh burns at plant_efficiency (0.25, the marginal-cost breakdown's efficiency_used) and the fuel's 9.62 MJ/kg (Tolvik 2024): 3.6 / (9.62 x 0.25) = 1.50 t/MWh, GBP 181/MWh. Tonnes are capped at the UK's actual EfW throughput, 16.82 Mt in 2024 (Tolvik Consulting, UK Energy from Waste Statistics 2024, section 3), so every EfW row's tonnes scale by 16.82 / (model tonnes). The same capped income is credited in the P3 tariff (nationalised_efw_gate_fee_income). No separate waste-handling cost is added: the register's EfW running costs (technology_costs.csv, BEIS Electricity Generation Costs 2020: fixed O&M GBP 179/kW-yr, VOM GBP 32.9/MWh, 2024 money) are BEIS's whole-plant opex, with gate fees the revenue against it (BEIS/NNFCC, Review of BEIS assumptions for ACT and EfW with CHP, Dec 2020, pp. 9, 17-18); the VOM, about GBP 22 per tonne of input, covers reagents and residue disposal (a declared judgement). |
| `valuation.gate_fee_income.carrier` | `waste_to_energy` |  |
| `valuation.gate_fee_income.file` | `data/public_ownership/financing/efw_gate_fees_2050.json` |  |
| `valuation.gate_fee_income.case` | `central` |  |
| `valuation.gate_fee_income.plant_efficiency` | `0.25` |  |
| `valuation.gate_fee_income.max_tonnes_per_year` | `16820000.0` |  |
| `valuation.owner_value_method` | `workbook_post_tax` | Owners' value: financing workbook v2.0's Approach 2 on each row (scripts/public_ownership/legacy_valuation.py): after-tax flows (25%), plus the capital allowances still to come on the row's book value (18% reducing balance; book value is the workbook's historic capex by cohort, Legacy A1, written down over its accounting life, or the register's as-new capex where the workbook has no cohort), end-of-year discounting at the workbook's post-tax operating rate (Legacy Fleet column AF: offshore 6.1%, calibrated to Brookfield's Dec 2024 purchase of 12.45% of Hornsea 1/2, Walney Extension and Burbo Bank Extension, GBP 4.0 m/MW; the rest by the CEPA hurdle spread: onshore 4.78%, solar 4.52%, nuclear 4.87%, thermal and storage 6.05%), as published. Hinkley Point C, which the workbook does not value, takes the new-build nuclear post-tax rate, 6.30% (owner_rate_named).  owner_rate_technology_by_carrier maps carriers to workbook rows. |
| `valuation.owner_rate_named` | (mapping) |  |
| `valuation.owner_rate_named.Hinkley Point C` | `Large-scale nuclear` |  |
| `valuation.owner_rate_technology_by_carrier` | (mapping) |  |
| `valuation.owner_rate_technology_by_carrier.pumped_hydro` | `Pumped hydro energy storage (PHES)` |  |
| `valuation.owner_rate_technology_by_carrier.CHP` | `Gas generation - unabated` |  |
| `valuation.owner_rate_technology_by_carrier.oil` | `Gas generation - unabated` |  |
| `valuation.owner_rate_technology_by_carrier.biogas` | `Anaerobic digestion (AD)` |  |
| `valuation.close_when_cash_negative` | `true` | The owner's option: a plant closes from the first year after the model year in which its total flow is negative (it is then not a liability that runs on). |
| `valuation.balancing_income_path` | (mapping) | Gas balancing income after the model year follows NESO's constraint-cost forecast. Gas plant (CCGT, OCGT, gas engines: NESO's 'Unabated Gas') keep in year t their 2030 balancing margin (settlement less the variable cost of the energy redispatched) x NESO_t / NESO_2030; renewables, biomass and storage stay flat. Source: NESO, 2025 Annual Balancing Costs Report (June 2025), Figure 30, thermal constraints, FES 2024 Holistic Transition (file, read from the figure's vector paths). The ratio is to the same network's 2030: the expected network (CENTRAL, no_acceleration, GBP 7.22 bn in 2030) by default, the recommended network for network_case cp30 (GBP 3.28 bn); after 2030 NESO's two networks are identical. NESO's projection ends in 2035, so later years hold the 2035 ratio (0.383 expected, 0.843 recommended).  method: neso_constraint_cost \| flat (hold the 2030 margin for life). |
| `valuation.balancing_income_path.method` | `neso_constraint_cost` |  |
| `valuation.balancing_income_path.carriers` | `[CCGT, OCGT, gas_engine]` |  |
| `valuation.balancing_income_path.file` | `data/public_ownership/financing/neso_constraint_cost_path.csv` |  |
| `valuation.balancing_income_path.series` | `ht_expected_network` |  |
| `valuation.balancing_income_path.series_by_network_case` | (mapping) |  |
| `valuation.balancing_income_path.series_by_network_case.cp30` | `ht_recommended_network` |  |
| `valuation.retirement_life_years_by_carrier` | (mapping) | Lives (years) of the legacy fleet: the DESNZ lives the register recovers new build over (technology_costs.csv: solar 38, offshore 30), except onshore wind at 25 (FES E.02 design life). The same rule as the 2050 study. The extended_asset_lives lever is a sensitivity on top. |
| `valuation.retirement_life_years_by_carrier.wind_onshore` | `25` |  |
| `valuation.retirement_life_years_by_carrier.wind_offshore` | `30` |  |
| `valuation.retirement_life_years_by_carrier.solar_pv` | `38` |  |
| `valuation.retirement_life_years_by_carrier.large_hydro` | `60` |  |
| `valuation.retirement_life_years_by_carrier.marine` | `25` |  |
| `valuation.retirement_life_years_by_carrier.biomass` | `25` |  |
| `valuation.retirement_life_years_by_carrier.CCGT` | `25` | Continuity audit: GHD for DESNZ 2025 (unabated gas generation costs, nominal life 25, range 20-35) and FES 2025 E.02 (25), the life the FES 2040/2050 fleet retires CCGTs on; a Capacity Market agreement keeps a station to its end (cm_agreement_floor below). |
| `valuation.retirement_life_years_by_carrier.OCGT` | `25` |  |
| `valuation.retirement_life_years_by_carrier.gas_engine` | `25` |  |
| `valuation.retirement_life_years_by_carrier.waste_to_energy` | `35` |  |
| `valuation.retirement_life_years_by_carrier.oil` | `25` |  |
| `valuation.retirement_life_years_by_carrier.CHP` | `25` |  |
| `valuation.retirement_life_years_by_carrier.Battery` | `15` |  |
| `valuation.retirement_life_years_by_carrier.nuclear` | `60` |  |
| `valuation.retirement_life_years_by_carrier.CCS Biomass` | `25` | the register's recovery life |
| `valuation.retirement_life_years_by_carrier.CCS Gas` | `25` |  |
| `valuation.retirement_year_overrides` | (mapping) | No private life extension: plant runs to its economic life and its site is then rebuilt as new build, so the purchase values it to that life. A life-extension rule belongs to a register bound's fleet_vintages.repowering (applied by legacy_valuation.group_repowering_settings). |
| `valuation.retirement_year_overrides.Hinkley Point C` | `2065` |  |
| `valuation.retirement_year_overrides.Sizewell B` | `2036` | EDF's lifetime extension to 2055 is in the extended-lives sensitivity (extended_asset_lives.retirement_year_overrides below); adopting it centrally needs the FES fleet to keep it. |
| `valuation.undated_commissioning_method` | `carrier_median_dated` |  |
| `valuation.undated_remaining_years_by_carrier` | (mapping) | undated rows with no dated peer in their carrier |
| `valuation.undated_remaining_years_by_carrier.large_hydro` | `30` |  |
| `valuation.undated_remaining_years_by_carrier.CCGT` | `15` |  |
| `valuation.undated_remaining_years_by_carrier.waste_to_energy` | `15` |  |
| `valuation.undated_remaining_years_by_carrier.pumped_hydro` | `40` | Dinorwig, Cruachan, Ffestiniog and Foyers: refurbished civil assets that FES keeps past 2050; 40 runs them to the valuation horizon (2070), as in the 2050 study. |
| `valuation.undated_remaining_years_by_carrier.CCS Biomass` | `25` | Drax BECCS: units converted for capture around 2030, recovered over 25 years in the register. |
| `valuation.undated_remaining_years_default` | `10` |  |
| `valuation.valued_carriers` | `[wind_onshore, wind_offshore, solar_pv, large_hydro, marine, nuclear]` | The headline (valued) group. Thermal, storage and foreign plant are profiled and priced in their own groups but never enter the headline. |
| `valuation.thermal_carriers` | (list of 10 items) |  |
| `valuation.excluded_carriers` | `[EU_market, EU_import, load_shedding]` |  |
| `valuation.not_operating_at_acquisition` | (mapping) |  |
| `valuation.not_operating_at_acquisition.file` | `data/public_ownership/financing/not_operating_at_acquisition.csv` |  |
| `valuation.price_anchors` | (mapping) | The anchored price path: each technology's capture price on a straight line through today and every solved year, flat after the last (scripts/public_ownership/price_anchors.py, rule build_public_ownership_price_anchors). |
| `valuation.price_anchors.solved_groups` | (mapping) | Solved years beyond the defaults (2030: the report group; 2050: fes_ht_2050), e.g. {2040: fes_ht_2040}. |
| `valuation.price_anchors.solved_groups.2040` | `fes_ht_2040` |  |
| `valuation.price_anchors.carriers` | (list of 7 items) | Carriers on the path; others keep the model year's price (a carrier whose role changes, e.g. biomass as a 2050 peaker at GBP 192/MWh, would carry a ratio no existing plant earns). |
| `valuation.price_anchors.today` | (mapping) |  |
| `valuation.price_anchors.today.year` | `2026` |  |
| `valuation.price_anchors.today.level_gbp_per_mwh_nominal` | `95.4` | Ofgem default tariff cap direct-fuel electricity index (Annex 2, before supplier uplifts and losses): Jul-Sep 2026 91.75 (data/market/ofgem_price_cap/2026_Q3/metadata.json) and Oct-Dec 2026 about 99 (cap 17a); the mean, GBP/MWh nominal. Household-shaped, so a little above a flat baseload price. PROVISIONAL. |
| `valuation.price_anchors.today.deflator_to_2024` | `0.943` | 2026 to 2024 prices: 0.967 for 2025 to 2024 (workbook v6.2 Legacy Fleet) over about 2.5% CPI in 2026. |
| `valuation.price_anchors.today.reference_price_files` | `resources/market/CP30_FFR_2030_{month}_stage1_wholesale_price.csv` | The earliest solved year's time-weighted Stage 1 price, computed from these files. |
| `valuation.reference_owner_rates` | `[0.067, 0.083]` | fixed bookend owner rates, for comparison with per-row WACC |
