## `consumer_bill`

File: `config/defaults/base/consumer_bill.yaml`. Keys below are relative to `consumer_bill`.

| Key | Default | Description |
|---|---|---|
| `price_cap` | (mapping) | Electricity-only consumer-bill reporting: the Ofgem default tariff cap (historical) and the future-bill forecast cases, which read completed monthly market outputs and never trigger a solve. |
| `price_cap.enabled` | `false` |  |
| `price_cap.mode` | `official` | Published allowance / validated forward-procurement baseline |
| `price_cap.cap_snapshot` | `2026_Q3` | data/market/ofgem_price_cap/{snapshot} |
| `price_cap.meter_types` | `[single_rate]` | Multi-register meters are not yet implemented |
| `price_cap.payment_methods` | `[direct_debit]` | direct_debit \| standard_credit \| prepayment |
| `price_cap.regions` | `[GB_average]` | GB_average or Ofgem charge-restriction region |
| `price_cap.annual_consumption_kwh` | `[2500]` |  |
| `price_cap.vat_rate` | `0.05` |  |
| `price_cap.cost_stack` | (mapping) |  |
| `price_cap.cost_stack.enabled` | `false` | Published hedge + named retail-stack analysis (2020 only) |
| `price_level_mapping` | (mapping) | Quantile mapping of the model's 2030 hours onto CP30's own price level, NESO CP30 workbook CP.32: each market hour keeps its rank and takes CP.32's price at that rank (scripts/consumer_bill/price_level_mapping.py). Read by the legacy valuation's price path (lever legacy_price_path) and by forecast cases with price_level: cp32 (a static bracket on the price level). Code defaults: price_level_mapping.DEFAULT_SETTINGS (must match). |
| `price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` | Sheet CP.32 of data/CP30/Clean Power 2030 Data Workbook_V003.xlsx (Annex 4, Figure 11, "Price duration curve 2030", Baringa for NESO): 101 points. |
| `price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `price_level_mapping.pathway` | `further_flex_and_renewables` | our pathway; mean GBP 69.89/MWh |
| `price_level_mapping.price_year` | `2024` | 2024 prices: CP.32 states no year; CP30 Annex 4 section 2 rebases all costs to 2024 prices. |
| `price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts` | (mapping) | Future consumer-bill forecast cases (CP30 2030). Each reads twelve solved months (source_scenario_pattern) and their annual aggregate (annual_aggregate_study, a monthly_aggregates entry of config/studies/cp30_2030/study.yaml). |
| `forecasts.cases` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network` | (mapping) | The LOWER network (the recommended CP30 network). Its months bid CfDs at strike minus wholesale with no negative-price rule, and the projector refuses a bill whose CfD rule differs from the bid audit's, so the case reads the cfd_rule_none assumptions. |
| `forecasts.cases.CP30_2030_cp30_network.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` | Sourced recovery denominators: every model-derived component divides by a volume declared here. |
| `forecasts.cases.CP30_2030_cp30_network.annual_aggregate_study` | `CP30_2030_cp30_network` | The twelve-month aggregate that owns the annual accounting arithmetic. |
| `forecasts.cases.CP30_2030_cp30_network.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` | Renewables Obligation scheme parameters, and which Exchequer-relief bookend the bill uses (no default): relief_ends_march_2029 follows government guidance ending the 75% domestic relief in March 2029, making calendar 2030 the first full year of gross recovery. |
| `forecasts.cases.CP30_2030_cp30_network.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` | Fallback only: the assumptions file's benchmark.period_alignment_note wins (project_future_bill.py). |
| `forecasts.cases.CP30_2030_cp30_network.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network.cfd_recovery.method` | `eligible_supplier_sales` | The CfD supplier obligation is levied on eligible supply (net of the EII exemption, from recovery_demand_assumptions), not on total system demand. |
| `forecasts.cases.CP30_2030_cp30_network.max_fallback_strike_share_of_payment` | `0.01` | Largest share 0-1 of the gross CfD payment allowed to settle on a carrier-wide future_\* strike rather than an allocation-round strike; the case fails above it. |
| `forecasts.cases.CP30_2030_cp30_network.tnuos_network_delivery` | (mapping) | TNUoS as NESO publishes it for the recommended CP30 network (2030_tnuos_network_delivery.json, case cp30). |
| `forecasts.cases.CP30_2030_cp30_network.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2` | (mapping) | The same year with the P2 balancing bids: the state-owned share of each unit bids at cost. The months take their parent month's Stage 1 (wholesale_from), so the case differs only in the balancing stage. Recovery denominators belong to the demand year, so it reads CP30_2030_cp30_network's (recovery_case_id). |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p2_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.annual_aggregate_study` | `CP30_2030_cp30_network_p2_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3` | (mapping) | The same year with the P3 balancing bids: every CfD unit is state-owned and bids at cost, ties ordered by a last-in-first-out step that the constraint line and the settlement net out. Its bill is private on the P3 months, the comparator the acquisition tariff on those months replaces. |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p3_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.annual_aggregate_study` | `CP30_2030_cp30_network_p3_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32` | (mapping) | Each case above at CP30's price level (CP.32; price_level_mapping above). Wholesale, procurement, CfD top-ups and interconnector rents move; volumes, the constraint line, RO, LDES and the sourced externals are held as solved (project_future_bill.CP32_HELD_LINES). Each is the forecast case of a price-level twin run group (public_ownership.reference_run_groups.\*_cp32), which re-settles its parent group's register at this level. price_level_parent names the model-price case whose months it re-prices. |
| `forecasts.cases.CP30_2030_cp30_network_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.annual_aggregate_study` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.price_level_parent` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p2_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.annual_aggregate_study` | `CP30_2030_cp30_network_p2_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.price_level_parent` | `CP30_2030_cp30_network_public_bids_p2` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p2_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p3_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.annual_aggregate_study` | `CP30_2030_cp30_network_p3_bids` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.price_level_parent` | `CP30_2030_cp30_network_public_bids_p3` |  |
| `forecasts.cases.CP30_2030_cp30_network_public_bids_p3_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration` | (mapping) | Network bounds: three network-delivery cases, each with its market year, P2 and P3 bid stacks and a CP.32 twin of each: CENTRAL = no_acceleration (NESO's expected network, GBP 7.79 bn of constraint costs): \*_no_acceleration; LOWER   = the recommended CP30 network: CP30_2030_cp30_network\* above; UPPER   = no_upgrades (NESO GBP 12.7 bn): \*_no_upgrades. Every month takes its Stage 1 from the shared stage1 family (wholesale_from), so the cases differ only in the balancing stage, and all read CP30_2030_cp30_network's recovery denominators. tnuos_network_delivery moves the TNUoS line to the case's network (data/market/consumer_bill_benchmarks/ 2030_tnuos_network_delivery.json: nil on CENTRAL and LOWER, whose TO submissions already carry the projects' 2031 delivery dates; a labelled -10% of onshore TO income on UPPER). register_account_group points the battery top-up line at the network's own market group's public account. |
| `forecasts.cases.CP30_2030_no_acceleration.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration.annual_aggregate_study` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.annual_aggregate_study` | `CP30_2030_no_acceleration_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.annual_aggregate_study` | `CP30_2030_no_acceleration_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.annual_aggregate_study` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cp32.price_level_parent` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.annual_aggregate_study` | `CP30_2030_no_acceleration_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p2_cp32.price_level_parent` | `CP30_2030_no_acceleration_public_bids_p2` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.annual_aggregate_study` | `CP30_2030_no_acceleration_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_acceleration_public_bids_p3_cp32.price_level_parent` | `CP30_2030_no_acceleration_public_bids_p3` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low` | (mapping) | The modelled AR8 round's range (bill layer only): the uncontracted 2030 CfD cohort paid at AR6's clearing prices where lower than the modelled AR8 strike (low) or at AR8's administrative strike prices (high), on the CENTRAL network's months. The balancing solves keep the central strike. |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.annual_aggregate_study` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_low.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.annual_aggregate_study` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_ar8_high.register_account_group` | `cp30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd` | (mapping) | The CENTRAL network's year under central dispatch (institution I3, FTI's settlement), the forecast case of the institution twin cp30_2030_no_acceleration_cd (reference_runs.institution_of). It keeps its parent's months: the national price is the market twin's copperplate price, so the wholesale line, demand and annual aggregate are the parent's. The constraint line is the FTI settlement's (fti_settlement: cost recovery + firm access with lost CfD and ROC support + storage and interconnector make-whole + the borrowed forecast-error charge, no re-trading); the CfD line and the RO are metered on the central months (central_scenario_pattern); the cap-and-floor rents move to the central schedule's. counter_trade_premium_on_constraint_line: true adds the central balancing stage's interconnector counter-trade premium to the constraint line; false leaves it in the real-time correction the forecast-error charge replaces (code default true). register_account_group: the twin's own account. |
| `forecasts.cases.CP30_2030_no_acceleration_cd.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.annual_aggregate_study` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.tnuos_network_delivery.case` | `no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.register_account_group` | `cp30_2030_no_acceleration_cd` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.institution` | `central_dispatch` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.institution_parent` | `CP30_2030_no_acceleration` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_central_dispatch` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_acceleration_central_dispatc...` |  |
| `forecasts.cases.CP30_2030_no_acceleration_cd.counter_trade_premium_on_constraint_line` | `true` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd` | (mapping) | The LOWER network's central-dispatch twin, as CENTRAL's above. |
| `forecasts.cases.CP30_2030_cp30_network_cd.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.source_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.annual_aggregate_study` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.tnuos_network_delivery.case` | `cp30` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.institution` | `central_dispatch` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.institution_parent` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.central_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_central_dispatch` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.fti_settlement` | `resources/market/CP30_FFR_2030_year_cp30_network_central_dispatch_f...` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.counter_trade_premium_on_constraint_line` | `true` |  |
| `forecasts.cases.CP30_2030_cp30_network_cd.register_account_group` | `cp30_2030_cp30_network_cd` |  |
| `forecasts.cases.CP30_2030_no_upgrades` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades.annual_aggregate_study` | `CP30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd` | (mapping) | The UPPER network's central-dispatch twin, as CENTRAL's above. |
| `forecasts.cases.CP30_2030_no_upgrades_cd.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.annual_aggregate_study` | `CP30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.register_account_group` | `cp30_2030_no_upgrades_cd` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.institution` | `central_dispatch` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.institution_parent` | `CP30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_central_dispatch` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_upgrades_central_dispatch_ft...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cd.counter_trade_premium_on_constraint_line` | `true` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.annual_aggregate_study` | `CP30_2030_no_upgrades_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.annual_aggregate_study` | `CP30_2030_no_upgrades_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.annual_aggregate_study` | `CP30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_upgrades_cp32.price_level_parent` | `CP30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.annual_aggregate_study` | `CP30_2030_no_upgrades_p2_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p2_cp32.price_level_parent` | `CP30_2030_no_upgrades_public_bids_p2` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.modelled_year` | `2030` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.model_price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.source_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.assumptions` | `data/market/consumer_bill_benchmarks/2030_pypsa_informed_assumption...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.recovery_demand_assumptions` | `data/market/consumer_bill_forecast/2030_recovery_demand_assumptions...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.annual_aggregate_study` | `CP30_2030_no_upgrades_p3_bids` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.ro_scheme_parameters` | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.ro_relief_case` | `relief_ends_march_2029` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.period_alignment_note` | `Calendar 2030, the year the model dispatches. Model lines are calen...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.cfd_recovery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.cfd_recovery.method` | `eligible_supplier_sales` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.max_fallback_strike_share_of_payment` | `0.01` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.tnuos_network_delivery` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.tnuos_network_delivery.file` | `data/market/consumer_bill_benchmarks/2030_tnuos_network_delivery.json` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.tnuos_network_delivery.case` | `no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.recovery_case_id` | `CP30_2030_cp30_network` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.register_account_group` | `cp30_2030_no_upgrades` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level` | `cp32` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping` | (mapping) |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping.source_file` | `data/CP30/cp32_price_duration_curve_2030.csv` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping.source` | `Clean Power 2030 Data Workbook V003 (NESO, Baringa modelling), shee...` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping.pathway` | `further_flex_and_renewables` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping.price_year` | `2024` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_mapping.rebasing_to_model_price_year` | `1.0` |  |
| `forecasts.cases.CP30_2030_no_upgrades_public_bids_p3_cp32.price_level_parent` | `CP30_2030_no_upgrades_public_bids_p3` |  |
