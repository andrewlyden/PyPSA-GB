## `scripts/market/`

| Module | Summary |
|---|---|
| `scripts/market/adequacy_controls.py` | (no module docstring) |
| `scripts/market/aggregate_neso_comparator.py` | Aggregate complete monthly CP30/NESO comparator market summaries. |
| `scripts/market/analyze_consumer_bill_cost_stack.py` | Create a transparent annual Ofgem/PyPSA electricity-bill decomposition. |
| `scripts/market/analyze_market.py` | Market Simulation Analysis — Post-processing for Two-Stage Dispatch |
| `scripts/market/audit_consumer_bill_methodology.py` | Audit which consumer-bill components are evidenced for a PyPSA scenario. |
| `scripts/market/audit_cp30_offshore_links.py` | Audit capacity and realised use of CP30 explicit offshore export links. |
| `scripts/market/bm_calibration_scorecard.py` | BM Calibration Scorecard — Per-Carrier Model-vs-ELEXON Price & Volume Audit |
| `scripts/market/bm_elexon_fit.py` | Per-Carrier ELEXON Price Fit for Fallback Pricing |
| `scripts/market/bm_participation_audit.py` | Audit BM eligibility and model redispatch coverage for historical ELEXON runs. |
| `scripts/market/bm_storage_churn.py` | Simultaneous charge and discharge in the balancing stage, by storage carrier and month. |
| `scripts/market/build_ar8_modelled_strikes.py` | The modelled AR8 CfD round of the 2030 CP30 chain. |
| `scripts/market/build_bmu_support_crosswalk.py` | Build an evidence-scored BMU-to-CfD/ROC crosswalk from curated IDs. |
| `scripts/market/build_capacity_market.py` | Build the calendar-2030 Capacity Market: contract book, auction model, assets. |
| `scripts/market/build_capacity_market_2050.py` | The calendar-2050 Capacity Market with V2G bidding in as DSR, centrally. |
| `scripts/market/build_capacity_market_year.py` | The Capacity Market of any long-range model year, 2031 to 2050. |
| `scripts/market/build_consumer_bill_component_register.py` | Build the fully itemised Ofgem benchmark component register. |
| `scripts/market/build_consumer_bill_forecast_registry.py` | Classify a detailed bill register for transparent future-scenario forecasting. |
| `scripts/market/build_household_bill_component_register.py` | Project the published detailed register onto one selected consumer bill. |
| `scripts/market/calculate_consumer_bill.py` | Snakemake entry point for Ofgem-structured electricity consumer bills. |
| `scripts/market/calibrate_scarcity_curve.py` | (no module docstring) |
| `scripts/market/capacity_market.py` | The GB Capacity Market in calendar 2030: contract book and auction model. |
| `scripts/market/capacity_market_2050.py` | The GB Capacity Market in calendar 2050 on FES 2025 Holistic Transition (task Y11). |
| `scripts/market/central_constraint_screening.py` | Exact branch-constraint screening for a central dispatch Stage 1. |
| `scripts/market/cfd_bm_bids.py` | Contract-value balancing bids for supported generation (backlog item B1). |
| `scripts/market/cfd_strike_table.py` | Contract-aware CfD strike prices, indexed from published 2012-price results. |
| `scripts/market/cfd_strikes.py` | CfD strikes and live contracts in a model year. |
| `scripts/market/cfd_strikes_2050.py` | Old name of :mod:`scripts.market.cfd_strikes`. |
| `scripts/market/commitment_proxy.py` | A linear proxy for unit commitment in the real-time stage. |
| `scripts/market/compare_tyndp26_year_2050.py` | (no module docstring) |
| `scripts/market/consumer_bill_breakdown.py` | Create a readable, auditable consumer-electricity-bill breakdown. |
| `scripts/market/copperplate.py` | Exact copperplate for the wholesale stage: collapse the mainland to one bus. |
| `scripts/market/curtailment_order.py` | The order in which the wholesale curtails zero-cost renewables. |
| `scripts/market/decompose_public_bm_bids.py` | Decompose a public-ownership balancing run against its matched baseline. |
| `scripts/market/diagnose_interconnector_pricing.py` | What GB's cross-border trade actually earned and paid, market by market (item IC1). |
| `scripts/market/diagnose_storage_degeneracy.py` | Separate storage round-trip loss from zero-price solver degeneracy (item B2). |
| `scripts/market/elexon_data.py` | Fetch ELEXON BMRS bid/offer data for historical market scenarios. |
| `scripts/market/extract_ofgem_2020_adjustment_components.py` | Extract the 2020 Annex 8 adjustment allowance and its non-additive lineage. |
| `scripts/market/extract_ofgem_2020_network_components.py` | Extract detailed 2020 Ofgem Annex 3 electricity-network allowances. |
| `scripts/market/extract_ofgem_2020_policy_components.py` | Extract the published 2020 Ofgem policy-cost components from Annex 4. |
| `scripts/market/extract_ofgem_2020_smart_meter_components.py` | Extract the published 2020 smart-metering net-cost components. |
| `scripts/market/fetch_neso_cp2030_annex4_costs.py` | Fetch and parse NESO Clean Power 2030 Annex 4 commodity assumptions. |
| `scripts/market/forecast_error.py` | Day-ahead forecast error for the scheduling stage. |
| `scripts/market/forecast_error_charge.py` | Carry FTI's forecast-error cost onto this system, because our model produces none. |
| `scripts/market/fti_settle_year.py` | Settle every month of a central dispatch year against the market year, and total it. |
| `scripts/market/fti_settlement.py` | Settle a central dispatch case the way FTI settle it, and set it beside its market twin. |
| `scripts/market/future_bid_stack.py` | Build and apply empirical future balancing bid/offer residuals. |
| `scripts/market/generate_market_analysis_notebook.py` | Generate a per-scenario Jupyter notebook for two-stage market dispatch analysis. |
| `scripts/market/generate_wholesale_notebook.py` | Generate a per-scenario Jupyter notebook for wholesale market dispatch analysis. |
| `scripts/market/ic1_month_diagnostics.py` | IC1 interconnector diagnostics for solved market months. |
| `scripts/market/interconnector_counter_trade.py` | Interconnector counter-trading in the balancing stage: its price and its limits. |
| `scripts/market/interconnector_hurdle.py` | Stage 1 interconnector trading-hurdle override. |
| `scripts/market/internal_dc_tie_break.py` | Tie-break cost on GB-internal HVDC flow. |
| `scripts/market/marginal_cost_basis.py` | Stage 1 marginal-cost basis. |
| `scripts/market/market_accounting.py` | Definitions and reconciliation for two-stage market output accounting. |
| `scripts/market/market_utils.py` | Market Simulation Utilities for PyPSA-GB |
| `scripts/market/must_take.py` | Must-take generation: units that are never curtailed or redispatched. |
| `scripts/market/neso_zones.py` | NESO's 17 CP.08 economic zones on a saved network. |
| `scripts/market/network_delivery.py` | Network delivery cases for the balancing stage. |
| `scripts/market/ofgem_hedging.py` | Published Ofgem wholesale-allowance building blocks. |
| `scripts/market/ofgem_price_cap.py` | Ofgem default-tariff-cap electricity bill utilities. |
| `scripts/market/pricing_basis.py` | The balancing stage's pricing basis: bids and offers, or every unit at its marginal cost. |
| `scripts/market/reconcile_bsuos_redispatch.py` | Compare the published BSUoS bill allowance with modelled redispatch activity. |
| `scripts/market/revenue_tracking.py` | Revenue tracking for CfD and ROC subsidy schemes. |
| `scripts/market/scarcity_diagnostic.py` | (no module docstring) |
| `scripts/market/scarcity_pricing.py` | (no module docstring) |
| `scripts/market/seasonal_storage.py` | Seasonal (hydrogen) storage across independently solved months: the two-pass approach. |
| `scripts/market/solve_balancing.py` | Solve Balancing Mechanism  Stage 2 of Two-Stage Market Dispatch |
| `scripts/market/solve_seasonal_storage_pass.py` | Snakemake entry for rule solve_seasonal_storage_pass. |
| `scripts/market/solve_wholesale.py` | Solve Wholesale Market — Stage 1 of Two-Stage Market Dispatch |
| `scripts/market/station_identity_registry.py` | Build a provenance-preserving public station-identity registry. |
| `scripts/market/support_bid_stack.py` | Calibrate CfD and ROC BM prices from historical ELEXON BOD panels. |
| `scripts/market/support_bids.py` | Balancing bids and offers for a long-range fleet from its support labels. |
| `scripts/market/support_labels.py` | Support labels for a long-range model year's balancing stage. |
| `scripts/market/support_labels_2050.py` | Old name of :mod:`scripts.market.support_labels`. |
| `scripts/market/temporal_resolution.py` | Stage 1 at a coarser temporal resolution. |
| `scripts/market/turn_down_allocation.py` | Pro rata zonal turn-down re-pricing of the balancing stage's wind bids. |
| `scripts/market/validate_bm.py` | BM Validation — Compare PyPSA-GB Balancing Mechanism Against ELEXON Actuals |
| `scripts/market/validate_neso_constraints.py` | NESO Constraint Validation — Compare Model BM vs NESO Thermal Constraints |
| `scripts/market/wholesale_commitment.py` | Stage 1 unit commitment with prices. |
