## `public_ownership.reference_run_groups`

File: `config/defaults/base/public_ownership/reference_run_groups.yaml`. Keys below are relative to `public_ownership.reference_run_groups`.

| Key | Default | Description |
|---|---|---|
| `historical_2020` | (mapping) |  |
| `historical_2020.model_year` | `2020` |  |
| `historical_2020.mode` | `twelve_months` |  |
| `historical_2020.scenario_pattern` | `ConsumerBill_{month}2020_Fast` |  |
| `historical_2024` | (mapping) |  |
| `historical_2024.model_year` | `2024` |  |
| `historical_2024.mode` | `full_year` |  |
| `historical_2024.scenario` | `Historical_2024_ETYS_14zone_market` |  |
| `cp30_2030_cp30_network` | (mapping) | LOWER network: CP30's recommended network (network_case cp30). Its scenario_pattern must equal CP30_2030_cp30_network's source_scenario_pattern. This family (the market group, its P2 and P3 bid stacks and their cp32 twins) carries the headline cases only, holds the CENTRAL network's purchase price (purchase_price_from) and has its own institution comparison (its central-dispatch year). The bid months of all three networks bid on this group's state-share and commissioning-year tables, so its verify stamp is the one every network's bid merges read. |
| `cp30_2030_cp30_network.model_year` | `2030` |  |
| `cp30_2030_cp30_network.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `cp30_2030_cp30_network.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network.network_case` | `cp30` |  |
| `cp30_2030_cp30_network.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_cp30_network.case_selection` | (mapping) | The headline cases: every scenario's central case, P2 with state bids at cost, the named cases, the presets and the best/worst bounds, plus each kept case's private comparator (case_selection.comparators, default true). |
| `cp30_2030_cp30_network.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network.institution_comparison` | (mapping) |  |
| `cp30_2030_cp30_network.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_central_dispatch` |  |
| `cp30_2030_cp30_network.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_cp30_network_central_dispatch_f...` |  |
| `cp30_2030_cp30_network_nationalised_dispatch` | (mapping) | The same year under central dispatch at resource cost (institution I3, the nationalised operator). Joins `all` only once its twelve months are in run_scenarios; the nationalised tariff is built on it beside the market-year group, so ownership and institution are never one number. |
| `cp30_2030_cp30_network_nationalised_dispatch.model_year` | `2030` |  |
| `cp30_2030_cp30_network_nationalised_dispatch.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_nationalised_dispatch.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_nationalised_dispatch` |  |
| `cp30_2030_cp30_network_nationalised_dispatch.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_public_bids_p2` | (mapping) | P2 bid stack: the same year with P2's state-owned share of each CfD unit bidding at cost in the balancing mechanism (Stage 1 shared through wholesale_from). Its bill is projected through its own forecast case for private and the cases on its bid stack only; private here reproduces CP30_2030_cp30_network_public_bids_p2, so the bid stack's effect is this group's private less the market group's. |
| `cp30_2030_cp30_network_public_bids_p2.model_year` | `2030` |  |
| `cp30_2030_cp30_network_public_bids_p2.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_public_bids_p2.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p2_bids` |  |
| `cp30_2030_cp30_network_public_bids_p2.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_public_bids_p2.forecast_case` | `CP30_2030_cp30_network_public_bids_p2` |  |
| `cp30_2030_cp30_network_public_bids_p2.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_cp30_network_public_bids_p2.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_public_bids_p2.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p2.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network_public_bids_p2.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network_public_bids_p3` | (mapping) | P3 bid stack: every CfD unit state-owned and bidding at cost, curtailment ordered by a last-in-first-out tie-break. P3 buys the legacy fleet, so its bill is the acquisition tariff: bid_case names the tariff case the months bid on (the case_id in the months' public_ownership_bids block). The group projects `private` on the market bill and tariffs bid_case with every tariff case owning the same fleet (reference_runs. same_fleet_tariff_cases: the fully_nationalised family, which differs only in purchase price basis, thermal basis and annuity term). Each case's stack effect is its tariff here less its tariff on the market group (merge_public_ownership_tariff_balancing_groups), with the purchase price held at the market year. P3's headline is central dispatch, so this re-prices the market-dispatch alternative only. |
| `cp30_2030_cp30_network_public_bids_p3.model_year` | `2030` |  |
| `cp30_2030_cp30_network_public_bids_p3.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_public_bids_p3.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p3_bids` |  |
| `cp30_2030_cp30_network_public_bids_p3.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_public_bids_p3.forecast_case` | `CP30_2030_cp30_network_public_bids_p3` |  |
| `cp30_2030_cp30_network_public_bids_p3.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_cp30_network_public_bids_p3.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_cp30_network_public_bids_p3.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_public_bids_p3.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p3.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network_public_bids_p3.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network_cp32` | (mapping) | Price-level twins (\*_cp32): a group re-settled at CP30's own 2030 price level (CP.32), with no re-solve. A twin (price_level_of) repeats its parent on every key and projects through the parent's forecast case at that level. The asset settlement is the parent's re-settled on the mapped hours (Stage 1 positions and CfD top-ups move; balancing settlement, fuel, ROCs, the DPA and every volume are held as solved); every later stage runs as for the parent under the twin's name. State shares, commissioning years, their check and the bid decomposition are the parent's. The P2/P3 twins are measured against the market group's twin; there is no institution comparison at cp32. Headline cases only. scripts/public_ownership/reference_runs.py (_validate_price_level_twins) enforces this. |
| `cp30_2030_cp30_network_cp32.model_year` | `2030` |  |
| `cp30_2030_cp30_network_cp32.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `cp30_2030_cp30_network_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_cp32.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_cp32.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_cp30_network_cp32.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network_cp32.institution_comparison` | (mapping) |  |
| `cp30_2030_cp30_network_cp32.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_central_dispatch` |  |
| `cp30_2030_cp30_network_cp32.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_cp30_network_central_dispatch_f...` |  |
| `cp30_2030_cp30_network_cp32.price_level_of` | `cp30_2030_cp30_network` |  |
| `cp30_2030_cp30_network_cp32.forecast_case` | `CP30_2030_cp30_network_cp32` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.model_year` | `2030` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p2_bids` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.forecast_case` | `CP30_2030_cp30_network_public_bids_p2_cp32` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network_public_bids_p2_cp32.price_level_of` | `cp30_2030_cp30_network_public_bids_p2` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.model_year` | `2030` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_p3_bids` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.forecast_case` | `CP30_2030_cp30_network_public_bids_p3_cp32` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_cp30_network_public_bids_p3_cp32.price_level_of` | `cp30_2030_cp30_network_public_bids_p3` |  |
| `cp30_2030_no_acceleration` | (mapping) | CENTRAL network: no_acceleration, NESO's expected network (balancing cost GBP 7.79 bn): Norwich to Tilbury (AENC/ATNC) and Sea Link (SCD1) not built by 2030. Its market group carries every case at model price, and the P3 purchase price (purchase_price_from on the other networks) and the auction calibration (cost_calibration.derived_from_group) come from it. Its bid months bid on cp30_2030_cp30_network's share tables; every market and bid month takes the stage1 family's Stage 1. |
| `cp30_2030_no_acceleration.model_year` | `2030` |  |
| `cp30_2030_no_acceleration.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `cp30_2030_no_acceleration.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration.forecast_case` | `CP30_2030_no_acceleration` |  |
| `cp30_2030_no_acceleration.institution_comparison` | (mapping) |  |
| `cp30_2030_no_acceleration.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_central_dispatch` |  |
| `cp30_2030_no_acceleration.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_acceleration_central_dispatc...` |  |
| `cp30_2030_no_acceleration_legislated_2025` | (mapping) | Register bound (register_bound_of): the CENTRAL market group with the legislation year at 2025 ("in force since 2025") instead of 2028, its own register on the same solved months, for the headline cases. Its P2 and P3 state bids are not re-solved: a contract-value view. |
| `cp30_2030_no_acceleration_legislated_2025.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_legislated_2025.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_legislated_2025.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `cp30_2030_no_acceleration_legislated_2025.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_legislated_2025.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_legislated_2025.forecast_case` | `CP30_2030_no_acceleration` |  |
| `cp30_2030_no_acceleration_legislated_2025.institution_comparison` | (mapping) |  |
| `cp30_2030_no_acceleration_legislated_2025.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_central_dispatch` |  |
| `cp30_2030_no_acceleration_legislated_2025.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_acceleration_central_dispatc...` |  |
| `cp30_2030_no_acceleration_legislated_2025.register_bound_of` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_acceleration_legislated_2025.legislated_year` | `2025` |  |
| `cp30_2030_no_acceleration_legislated_2025.case_selection` | (mapping) |  |
| `cp30_2030_no_acceleration_legislated_2025.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_acceleration_legislated_2025.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_acceleration_public_bids_p2` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p2.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_public_bids_p2.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_public_bids_p2.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p2_bids` |  |
| `cp30_2030_no_acceleration_public_bids_p2.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_public_bids_p2.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_public_bids_p2.forecast_case` | `CP30_2030_no_acceleration_public_bids_p2` |  |
| `cp30_2030_no_acceleration_public_bids_p2.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_acceleration_public_bids_p3` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p3.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_public_bids_p3.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_public_bids_p3.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p3_bids` |  |
| `cp30_2030_no_acceleration_public_bids_p3.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_public_bids_p3.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_public_bids_p3.forecast_case` | `CP30_2030_no_acceleration_public_bids_p3` |  |
| `cp30_2030_no_acceleration_public_bids_p3.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_acceleration_public_bids_p3.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_no_acceleration_cp32` | (mapping) |  |
| `cp30_2030_no_acceleration_cp32.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `cp30_2030_no_acceleration_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_cp32.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_cp32.forecast_case` | `CP30_2030_no_acceleration_cp32` |  |
| `cp30_2030_no_acceleration_cp32.institution_comparison` | (mapping) |  |
| `cp30_2030_no_acceleration_cp32.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_central_dispatch` |  |
| `cp30_2030_no_acceleration_cp32.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_acceleration_central_dispatc...` |  |
| `cp30_2030_no_acceleration_cp32.price_level_of` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_acceleration_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_acceleration_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_acceleration_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p2_bids` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.forecast_case` | `CP30_2030_no_acceleration_public_bids_p2_cp32` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.price_level_of` | `cp30_2030_no_acceleration_public_bids_p2` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_acceleration_public_bids_p2_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_p3_bids` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.forecast_case` | `CP30_2030_no_acceleration_public_bids_p3_cp32` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.price_level_of` | `cp30_2030_no_acceleration_public_bids_p3` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_acceleration_public_bids_p3_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_acceleration_cd` | (mapping) | Institution twins (\*_cd, institution_of): private, P1, P2 and their named cases and presets re-projected on the network's central-dispatch year (I3, FTI's settlement), with their own asset settlement (rule build_public_ownership_institution_twin_settlement), register, account, bills, Exchequer ledger and party flows, and no re-solve (cost-based dispatch does not depend on who owns the plant). A twin repeats its parent on every key but forecast_case and case_selection: scenarios, named cases and presets with their private comparators, no grids or bounds. It keeps market-bill cases on the contract-value stack only, so P3 (its own I3 path) and P2's state bids drop out. Each case's I3 bill is set against private under market dispatch by merge_public_ownership_institution_groups ({market group}_consumer_bill_by_institution): institution given owner plus owner given institution, never an ownership saving added to a central-dispatch one. |
| `cp30_2030_no_acceleration_cd.model_year` | `2030` |  |
| `cp30_2030_no_acceleration_cd.mode` | `twelve_months` |  |
| `cp30_2030_no_acceleration_cd.scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration` |  |
| `cp30_2030_no_acceleration_cd.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_acceleration_cd.network_case` | `no_acceleration` |  |
| `cp30_2030_no_acceleration_cd.forecast_case` | `CP30_2030_no_acceleration_cd` |  |
| `cp30_2030_no_acceleration_cd.institution_comparison` | (mapping) |  |
| `cp30_2030_no_acceleration_cd.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_acceleration_central_dispatch` |  |
| `cp30_2030_no_acceleration_cd.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_acceleration_central_dispatc...` |  |
| `cp30_2030_no_acceleration_cd.institution_of` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_acceleration_cd.case_selection` | (mapping) |  |
| `cp30_2030_no_acceleration_cd.case_selection.kinds` | `[scenario, named, preset]` |  |
| `cp30_2030_cp30_network_cd` | (mapping) |  |
| `cp30_2030_cp30_network_cd.model_year` | `2030` |  |
| `cp30_2030_cp30_network_cd.mode` | `twelve_months` |  |
| `cp30_2030_cp30_network_cd.scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network` |  |
| `cp30_2030_cp30_network_cd.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_cp30_network_cd.network_case` | `cp30` |  |
| `cp30_2030_cp30_network_cd.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_cp30_network_cd.case_selection` | (mapping) |  |
| `cp30_2030_cp30_network_cd.case_selection.kinds` | `[scenario, named, preset]` |  |
| `cp30_2030_cp30_network_cd.institution_comparison` | (mapping) |  |
| `cp30_2030_cp30_network_cd.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_cp30_network_central_dispatch` |  |
| `cp30_2030_cp30_network_cd.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_cp30_network_central_dispatch_f...` |  |
| `cp30_2030_cp30_network_cd.institution_of` | `cp30_2030_cp30_network` |  |
| `cp30_2030_cp30_network_cd.forecast_case` | `CP30_2030_cp30_network_cd` |  |
| `cp30_2030_no_upgrades` | (mapping) | UPPER network: no_upgrades (NESO GBP 12.7 bn), no Appendix B work after 2025 built by 2030. Headline cases only; the purchase price is held at the CENTRAL network's. |
| `cp30_2030_no_upgrades.model_year` | `2030` |  |
| `cp30_2030_no_upgrades.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `cp30_2030_no_upgrades.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades.forecast_case` | `CP30_2030_no_upgrades` |  |
| `cp30_2030_no_upgrades.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_upgrades.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades.institution_comparison` | (mapping) |  |
| `cp30_2030_no_upgrades.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_central_dispatch` |  |
| `cp30_2030_no_upgrades.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_upgrades_central_dispatch_ft...` |  |
| `cp30_2030_no_upgrades_public_bids_p2` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p2.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_public_bids_p2.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_public_bids_p2.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p2_bids` |  |
| `cp30_2030_no_upgrades_public_bids_p2.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_public_bids_p2.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_public_bids_p2.forecast_case` | `CP30_2030_no_upgrades_public_bids_p2` |  |
| `cp30_2030_no_upgrades_public_bids_p2.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_upgrades_public_bids_p2.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p2.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades_public_bids_p2.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades_public_bids_p3` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p3.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_public_bids_p3.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_public_bids_p3.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p3_bids` |  |
| `cp30_2030_no_upgrades_public_bids_p3.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_public_bids_p3.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_public_bids_p3.forecast_case` | `CP30_2030_no_upgrades_public_bids_p3` |  |
| `cp30_2030_no_upgrades_public_bids_p3.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_upgrades_public_bids_p3.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_no_upgrades_public_bids_p3.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p3.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades_public_bids_p3.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades_cd` | (mapping) |  |
| `cp30_2030_no_upgrades_cd.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_cd.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_cd.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `cp30_2030_no_upgrades_cd.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_cd.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_cd.forecast_case` | `CP30_2030_no_upgrades_cd` |  |
| `cp30_2030_no_upgrades_cd.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_upgrades_cd.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_cd.case_selection.kinds` | `[scenario, named, preset]` |  |
| `cp30_2030_no_upgrades_cd.institution_comparison` | (mapping) |  |
| `cp30_2030_no_upgrades_cd.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_central_dispatch` |  |
| `cp30_2030_no_upgrades_cd.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_upgrades_central_dispatch_ft...` |  |
| `cp30_2030_no_upgrades_cd.institution_of` | `cp30_2030_no_upgrades` |  |
| `cp30_2030_no_upgrades_cp32` | (mapping) |  |
| `cp30_2030_no_upgrades_cp32.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades` |  |
| `cp30_2030_no_upgrades_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_cp32.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_cp32.forecast_case` | `CP30_2030_no_upgrades_cp32` |  |
| `cp30_2030_no_upgrades_cp32.purchase_price_from` | `cp30_2030_no_acceleration` |  |
| `cp30_2030_no_upgrades_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades_cp32.institution_comparison` | (mapping) |  |
| `cp30_2030_no_upgrades_cp32.institution_comparison.central_scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_central_dispatch` |  |
| `cp30_2030_no_upgrades_cp32.institution_comparison.fti_settlement` | `resources/market/CP30_FFR_2030_year_no_upgrades_central_dispatch_ft...` |  |
| `cp30_2030_no_upgrades_cp32.price_level_of` | `cp30_2030_no_upgrades` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p2_bids` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.forecast_case` | `CP30_2030_no_upgrades_public_bids_p2_cp32` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades_public_bids_p2_cp32.price_level_of` | `cp30_2030_no_upgrades_public_bids_p2` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.model_year` | `2030` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.mode` | `twelve_months` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.scenario_pattern` | `CP30_FFR_2030_{month}_no_upgrades_p3_bids` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.baseline_run_group` | `historical_2024` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.network_case` | `no_upgrades` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.forecast_case` | `CP30_2030_no_upgrades_public_bids_p3_cp32` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.balancing_bids` | `state_at_cost` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.bid_case` | `public_ownership@fully_nationalised_market_value` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.case_selection` | (mapping) |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.case_selection.kinds` | `[scenario, named, preset, bound]` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.case_selection.cases` | `[public_ownership@state_bids_at_cost]` |  |
| `cp30_2030_no_upgrades_public_bids_p3_cp32.price_level_of` | `cp30_2030_no_upgrades_public_bids_p3` |  |
