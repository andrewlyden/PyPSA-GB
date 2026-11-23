## `scripts/consumer_bill/`

| Module | Summary |
|---|---|
| `scripts/consumer_bill/aggregate_monthly_generator_registers.py` | Build one annual generator/BMU register from monthly scenario outputs. |
| `scripts/consumer_bill/aggregate_physical_network_losses.py` | Aggregate AC transmission losses from a complete set of physical PyPSA runs. |
| `scripts/consumer_bill/align_historical_procurement.py` | Extract comparable pre-uplift forward indices from pinned Annex 2 releases. |
| `scripts/consumer_bill/archetype_cost_to_serve.py` | What a heat pump or an electric vehicle costs to serve (item G6 follow-up). |
| `scripts/consumer_bill/audit_lccc_profile_calibration.py` | Audit raw renewable profiles against calendar-year LCCC CfD generation. |
| `scripts/consumer_bill/audit_pypsa_roc_capacity_coverage.py` | Account for every PyPSA renewable-capacity block in the RO mapping audit. |
| `scripts/consumer_bill/audit_renewable_performance_factors.py` | Audit renewable performance factors against named LCCC CfD outturn. |
| `scripts/consumer_bill/audit_unresolved_roc_station_sources.py` | Audit unresolved Ofgem ROC stations against historical source registers. |
| `scripts/consumer_bill/bank_holidays.py` | England and Wales bank holidays, the days the Elexon peak block leaves out. |
| `scripts/consumer_bill/bill_groupings.py` | How bill lines are grouped for presentation: dated data, not code. |
| `scripts/consumer_bill/bill_scope.py` | Shared year and provenance labels for consumer-bill outputs. |
| `scripts/consumer_bill/build_2020_backtest.py` | Build an honest annual historical-vs-reconstructed consumer-bill backtest. |
| `scripts/consumer_bill/build_2050_bill_inputs.py` | The three 2050 bill input files: the generic long-range builder for 2050 (task Y13). |
| `scripts/consumer_bill/build_annual_support_to_bill_bridge.py` | Connect annual CfD/ROC reconciliation evidence to the consumer-bill stack. |
| `scripts/consumer_bill/build_bus_dno_mapping.py` | Map PyPSA demand buses to DNO/PES areas using NESO's GSP lookup. |
| `scripts/consumer_bill/build_domestic_demand_profile.py` | Materialise a domestic consumption shape from Ofgem's base/peak allocation. |
| `scripts/consumer_bill/build_domestic_pes_weights.py` | Aggregate official domestic electricity consumption to PES/DNO areas. |
| `scripts/consumer_bill/build_fes_bill_inputs.py` | Build the three bill input files of a long-range FES model year (one builder for every year). |
| `scripts/consumer_bill/build_historical_alignment_scorecard.py` | Create a like-for-like monthly alignment scorecard for a historic run. |
| `scripts/consumer_bill/build_historical_consumer_bill_bundle.py` | Build a traceable annual consumer-bill input bundle from Ofgem workbooks. |
| `scripts/consumer_bill/build_historical_evidence_panel.py` | Reconcile saved Ofgem component evidence and audit procurement transferability. |
| `scripts/consumer_bill/build_historical_report.py` | Build the validation-first historical consumer-bill reporting directory. |
| `scripts/consumer_bill/build_historical_report_index.py` | Create the maintained index for historical consumer-bill reports. |
| `scripts/consumer_bill/build_historical_support_outturn.py` | Create a year-agnostic CfD/RO evidence report for a historic bill run. |
| `scripts/consumer_bill/build_lccc_cfd_mapping.py` | Build an auditable LCCC-to-PyPSA CfD contract mapping. |
| `scripts/consumer_bill/build_network_cost_layers.py` | Build tariff-ready network-cost layers and a future allowed-revenue projection. |
| `scripts/consumer_bill/build_network_cost_to_bill_bridge.py` | Build a 2020 network-cost bridge without treating PyPSA flows as tariffs. |
| `scripts/consumer_bill/build_ofgem_forward_procurement_bridge.py` | Reconstruct the 2020 Ofgem electricity forward-procurement allowance. |
| `scripts/consumer_bill/build_ofgem_roc_residual_cohorts.py` | Build an evidence-led ROC ledger from Ofgem's station-level outturn. |
| `scripts/consumer_bill/build_ofgem_roc_station_overlay.py` | Create a station-first historical ROC overlay without double-counting PyPSA. |
| `scripts/consumer_bill/build_pypsa_spot_procurement_bridge.py` | Build a historical wholesale backtest using the PyPSA spot price. |
| `scripts/consumer_bill/build_recovery_bridges.py` | Build transparent consumer-bill recovery bridges from PyPSA and regulatory inputs. |
| `scripts/consumer_bill/build_regional_network_tariff_replay.py` | Replay Ofgem's 2020 regional network allowances against domestic geography. |
| `scripts/consumer_bill/build_repd_roc_station_mapping.py` | Create an auditable REPD-to-Ofgem RO accredited-station crosswalk. |
| `scripts/consumer_bill/build_ro_eligible_fleet.py` | Estimate the 2030 Renewables Obligation scheme cost from the Ofgem register. |
| `scripts/consumer_bill/build_roc_mapping_priority_worklist.py` | Create a review worklist for high-volume, officially unmapped ROC stations. |
| `scripts/consumer_bill/build_roc_station_source_evidence.py` | Build a reusable Ofgem ROC-station evidence crosswalk from source registers. |
| `scripts/consumer_bill/calibrate_offshore_availability_factors.py` | Derive resource-region availability factors from LCCC outturn plus BM curtailment. |
| `scripts/consumer_bill/compare_historical_components.py` | Annual Ofgem/model comparison with explicit recovery and validation boundaries. |
| `scripts/consumer_bill/complete_roc_station_representation.py` | Complete the ROC station ledger with explicit aggregate representations. |
| `scripts/consumer_bill/component_ledger.py` | Versioned, validation-first accounting ledger for consumer bills. |
| `scripts/consumer_bill/duos_cp30_growth.py` | DUoS carried into RIIO-ED3 on CP30's distribution investment. |
| `scripts/consumer_bill/extract_central_dispatch_costs.py` | The bill's constraint line under central dispatch (institution I3), from the year FTI settlement. |
| `scripts/consumer_bill/extract_future_balancing_costs.py` | Extract per-month constraint costs, including CfD curtailment compensation. |
| `scripts/consumer_bill/extract_neso_balancing_cost_split.py` | Split published NESO BSUoS expenditure into constraint and non-constraint parts. |
| `scripts/consumer_bill/extract_network_price_control_building_blocks.py` | Extract the network price-control revenue building blocks. |
| `scripts/consumer_bill/extract_ofgem_roc_calendar_outturn.py` | Extract a calendar-year GB ROC issuance outturn from Ofgem's public export. |
| `scripts/consumer_bill/extract_pypsa_bill_signals.py` | Extract bill-relevant system signals from a solved PyPSA market network. |
| `scripts/consumer_bill/fes_bill_figures.py` | The FES 2025 figures a long-range bill year reads, for any model year. |
| `scripts/consumer_bill/household_archetypes.py` | Household archetypes with a standing and a unit split (item G6). |
| `scripts/consumer_bill/line_recovery.py` | Per-line recovery: how one household line converts to and from its system cost. |
| `scripts/consumer_bill/map_pypsa_generators_to_ofgem_roc.py` | Map direct PyPSA generators to stations that received calendar-year ROCs. |
| `scripts/consumer_bill/must_run_support_sensitivity.py` | Bill sensitivity: the support CP30's must-run plant without a contract would need. |
| `scripts/consumer_bill/price_level_mapping.py` | The model's 2030 hours priced at CP30's own price level, CP.32. |
| `scripts/consumer_bill/project_future_bill.py` | Project a granular future bill from complete hourly model results and assumptions. |
| `scripts/consumer_bill/reconcile_lccc_cfd_model_dispatch.py` | Reconcile mapped PyPSA CfD dispatch with LCCC contract-level outturn. |
| `scripts/consumer_bill/reconcile_roc_model_dispatch.py` | Reconcile a solved PyPSA ROC proxy against calendar-year Ofgem outturn. |
| `scripts/consumer_bill/refresh_recovery_demand.py` | Refresh the recovery-demand volumes of a bill case from its solved months. |
| `scripts/consumer_bill/review_consumer_bill.py` | One review page for the selected bill, component provenance and model evidence. |
| `scripts/consumer_bill/sync_capacity_market_pin.py` | Keep the 2030 bill's Capacity Market pin equal to the auction builder's summary. |
| `scripts/consumer_bill/validate_consumer_bill_input_bundle.py` | Validate the portable annual input bundle used by a consumer-bill case. |
