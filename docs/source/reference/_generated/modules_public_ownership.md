## `scripts/public_ownership/`

Post-solve financing and ownership accounting for public-ownership studies.

| Module | Summary |
|---|---|
| `scripts/public_ownership/acquisition_baseline.py` | What the acquisition year's purchase bought, by technology group. |
| `scripts/public_ownership/acquisition_carry_forward.py` | The 2030 acquisition carried to a later model year. |
| `scripts/public_ownership/aggregate_reference_runs.py` | Safely aggregate annual public-ownership reference-run bundles. |
| `scripts/public_ownership/ancillary_services_2030.py` | The 2030 ancillary services projected on CP30's drivers. |
| `scripts/public_ownership/bill_recovery.py` | Household &lt;-&gt; system conversions for the public-ownership layer and its reports. |
| `scripts/public_ownership/build_asset_finance_register.py` | Create an auditable component/cohort register from solved balancing networks. |
| `scripts/public_ownership/build_asset_settlement.py` | WP-D command line: annual market and support settlement per asset. |
| `scripts/public_ownership/build_cfd_strike_adjustments.py` | WP-B: per-case CfD strike adjustments reconciling exactly to the finance register. |
| `scripts/public_ownership/build_consumer_bill_financing_bridge.py` | Show how public-ownership finance savings relate to consumer-bill categories. |
| `scripts/public_ownership/build_public_account.py` | WP-D command line: the public account by case and group, and what each rule routes. |
| `scripts/public_ownership/build_vintage_technology_costs.py` | Build the 2025-2050 vintage technology cost table (FES 2050 plan, task Y2). |
| `scripts/public_ownership/build_workbook_esm_feedback.py` | Items J4 and J5 on workbook v2.0: what the model hands back to the financing workbook. |
| `scripts/public_ownership/calculate_financing_scenarios.py` | Build fixed-dispatch ownership/financing accounts from reviewed registers. |
| `scripts/public_ownership/capacity_market_design.py` | The Capacity Market design under public ownership. |
| `scripts/public_ownership/capacity_market_state_bids.py` | (no module docstring) |
| `scripts/public_ownership/capacity_procurement.py` | (no module docstring) |
| `scripts/public_ownership/central_strike_diagnostic_2050.py` | (no module docstring) |
| `scripts/public_ownership/cfd_repricing.py` | Route the financing saving to the consumer bill: CfD re-pricing and reconciliation. |
| `scripts/public_ownership/commissioning_years.py` | (no module docstring) |
| `scripts/public_ownership/compare_outputs.py` | Compare two trees of bill and ownership outputs, file by file. |
| `scripts/public_ownership/contract_book_2050.py` | The 2050 support contracts as one book, settled through one path (task SETTLE2050). |
| `scripts/public_ownership/declared_items.py` | (no module docstring) |
| `scripts/public_ownership/derive_cost_calibration.py` | Derive, or check, the private cost calibration table. |
| `scripts/public_ownership/electrified_household.py` | The electrified household beside the 3,100 kWh benchmark. |
| `scripts/public_ownership/exchequer_ledger.py` | The Exchequer ledger beside the bill headline, per ownership case. |
| `scripts/public_ownership/extract_workbook_v5.py` | Export the financing inputs of the financing workbook (v6.2, ``GB_Power_Model_v6.2.xlsx``) to a YAML snapshot. |
| `scripts/public_ownership/finance.py` | Pure finance functions used by the public-ownership accounting layer. |
| `scripts/public_ownership/fingerprints.py` | What each heavy stage's per-case result depends on, as a key. |
| `scripts/public_ownership/fleet_continuity.py` | Fleet continuity between the acquisition year and a later model year. |
| `scripts/public_ownership/fleet_vintages.py` | The vintaged 2050 fleet: a stock-flow build from the FES annual trajectory (FES 2050 plan, task Y10). |
| `scripts/public_ownership/headline_moves.py` | Every headline move against the last accepted results has a declared reason. |
| `scripts/public_ownership/hpc_schedule.py` | Bill-level sensitivity: Hinkley Point C on EDF's current schedule. |
| `scripts/public_ownership/investor_ledger.py` | The private-investor column and the three-party identity. |
| `scripts/public_ownership/legacy_valuation.py` | Value the legacy fleet's cash flows as a declining profile. |
| `scripts/public_ownership/legislation_year.py` | (no module docstring) |
| `scripts/public_ownership/lever_workbook.py` | (no module docstring) |
| `scripts/public_ownership/levers.py` | Levers, scenario bundles and the sensitivity expansion every stage shares (item K1). |
| `scripts/public_ownership/merge_balancing_groups.py` | Put the balancing-bid cases beside the market cases, with the bid stack's effect split out. |
| `scripts/public_ownership/merge_institution_groups.py` | P1 and P2 under central dispatch beside the market year: three parts per case and the headline. |
| `scripts/public_ownership/merge_tariff_balancing_groups.py` | Put a tariff case on its own balancing-bid stack beside the same case on the market months. |
| `scripts/public_ownership/model_year.py` | The model year of a finance and bill run, and the year-dependent names derived from it (item Y4). |
| `scripts/public_ownership/nationalised_institutions.py` | (no module docstring) |
| `scripts/public_ownership/nationalised_tariff.py` | The nationalised tariff. |
| `scripts/public_ownership/network_finance_account.py` | The network finance account. |
| `scripts/public_ownership/network_rav_roll_forward.py` | Network RAV roll-forward past the determined price controls (FES 2050 plan, task Y12). |
| `scripts/public_ownership/offshore_deal_multiple.py` | Legacy offshore wind bought at observed deal multiples. |
| `scripts/public_ownership/ownership_shares.py` | Per-generator state ownership shares, for the balancing bid stack. |
| `scripts/public_ownership/party_flows.py` | Itemised money flows between the parties, per case against private. |
| `scripts/public_ownership/pathway_2030_2050.py` | Pathway-lite 2030 to 2050: the public ownership benefits year by year, without solving a network. |
| `scripts/public_ownership/post_solve_2050.py` | Post-solve pipeline for the consolidated 2050 re-solve: one checked launch. |
| `scripts/public_ownership/post_solve_year.py` | Post-solve steps for a long-range FES year, in order. |
| `scripts/public_ownership/price_anchors.py` | The legacy fleet's price path anchors: today, and every solved model year. |
| `scripts/public_ownership/price_level_settlement.py` | The asset settlement at CP30's 2030 price level, CP.32. |
| `scripts/public_ownership/project_public_ownership_consumer_bill.py` | WP-C: one physical run, several bills. |
| `scripts/public_ownership/public_account.py` | WP-D, WP-I and WP-E: the public account and what it routes to the bill. |
| `scripts/public_ownership/reference_runs.py` | Shared reference-run expansion, case expansion and decision-register validation. |
| `scripts/public_ownership/register_io.py` | The per-case asset finance register on disk: Parquet body plus a case-settings table. |
| `scripts/public_ownership/repowering_sensitivity.py` | (no module docstring) |
| `scripts/public_ownership/results_audit.py` | Results audit: invariants checked on the real outputs of a bills run. |
| `scripts/public_ownership/run_financing.py` | Run one configured public-ownership financing bundle without rebuilding PyPSA. |
| `scripts/public_ownership/sensitivity_axes.py` | (no module docstring) |
| `scripts/public_ownership/settlement.py` | WP-D: link each register asset to its market and support settlement. |
| `scripts/public_ownership/share_coverage.py` | Check that an ownership share table names the network it is about to bid on. |
| `scripts/public_ownership/sizewell_c_rab.py` | (no module docstring) |
| `scripts/public_ownership/stage_timing.py` | Opt-in wall-clock timing of the ownership layer's hot loops. |
| `scripts/public_ownership/study_config.py` | A study's finance and bill settings on its parent scenario entry. |
| `scripts/public_ownership/support_lines_2050.py` | Support lines for the 2050 private counterfactual's unsupported plant. |
| `scripts/public_ownership/support_regime_2050.py` | The private support regime in calendar 2050. |
| `scripts/public_ownership/verify_bill_invariants.py` | Money invariants of a group's bills that a ledger closure cannot see. |
| `scripts/public_ownership/verify_ownership_tables.py` | Check that the ownership tables the public-bids BM solves read still match the register. |
| `scripts/public_ownership/verify_ownership_tables_2050.py` | (no module docstring) |
| `scripts/public_ownership/vintage_capital_2050.py` | How a 2050 vintage recovers its capital: the strike basis, the register's capital charge, and the calibration. |
| `scripts/public_ownership/vintage_costs.py` | Loader for the 2025-2050 vintage technology cost table (FES 2050 plan, task Y2). |
| `scripts/public_ownership/workbook_variants.py` | Recalculate the financing workbook's own alternative cases in Excel (Windows only). |
