## `market.wholesale`

File: `config/defaults/base/market/wholesale.yaml`. Keys below are relative to `market.wholesale`.

| Key | Default | Description |
|---|---|---|
| `mode` | `rolling_day_ahead` | Stage 1: Wholesale Market Dispatch mode: "single"              - Solve entire period in one LP (code default) "rolling_day_ahead"   - Solve in rolling windows (day-ahead market) Each window is optimised independently; storage SoC carries between windows when carry_soc is true. Generators and storage see only window_hours (plus any look-ahead) ahead. |
| `window_hours` | `24` | Hours per window (rolling_day_ahead mode) |
| `carry_soc` | `true` | Carry storage SoC between windows |
| `temporal_resolution_hours` | `1` | temporal_resolution_hours: Stage 1 solves on blocks of this many hours (block-mean inputs, snapshot weightings = block hours, ramp limits x hours capped at 1); outputs are written hourly, power held within each block and storage / store levels linear inside it, so the BM runs hourly on the held block positions. 1 = hourly; 3 matches NESO's simplified 3-hour PLEXOS market run (CP30 Annex 4, CP.28-30). Rolling windows and look-ahead must be whole blocks; not compatible with Stage 1 unit commitment. scripts/market/temporal_resolution.py. |
| `scarcity_pricing` | (mapping) | Scarcity pricing: a stepped reserve shortage curve in the Stage 1 LP. Margin = spare GB generator capacity + storage deliverable for an hour + spare import capacity; below the curve's thresholds the LP buys shortfall at rising prices, which lift the energy price. Calibrated on the 2024 gap between N2EX day-ahead prices and the model against NESO's de-rated margin (scripts/market/calibrate_scarcity_curve.py), with the BSC Reserve Scarcity Price (LoLP x VoLL) below 1 GW. On by default; always off under market.dispatch: central. |
| `scarcity_pricing.enabled` | `true` |  |
| `scarcity_pricing.curve_file` | `data/market/scarcity/scarcity_curve_2024.json` |  |
| `scarcity_pricing.excluded_carriers` | `[load_shedding, EU_market, EU_import]` |  |
| `scarcity_pricing.max_drm_threshold_mw` | `null` | De-rated margin cap (MW): steps whose threshold is above it are dropped, e.g. 8000 keeps only the steps below 8 GW. null keeps every step. |
| `copperplate_mode` | `collapse` | How the single GB price is produced: "collapse" - exact copperplate: the mainland synchronous sub-network (every load) is collapsed onto one bus just before the solve; islands, External buses and every link keep their ratings, and results are expanded back onto the full network for the saved .nc. Same objective as "relax" to ~1e-6 relative and much faster (scripts/market/copperplate.py). "relax"    - legacy copperplate: every line/transformer keeps its place with s_nom set to transmission_relaxation. Slow and ill-scaled. "none"     - no copperplate: Stage 1 solves on the real network (central dispatch). |
| `relax_island_hvdc` | `true` | Island HVDC in the copperplate. The collapse keeps every link's rating, so without this the day-ahead market would see island HVDC limits (e.g. Shetland's 600 MW) and curtail island wind in Stage 1. When true, every GB-internal link of island_hvdc_carriers with an end off the mainland is lifted (to what all GB islands could push through it) in "collapse" and "relax"; the balancing stage keeps the real ratings. Not applied under copperplate_mode "none". |
| `island_hvdc_carriers` | `[DC]` |  |
| `forecast_error` | (mapping) | Day-ahead forecast error for the scheduling stage (off by default). When enabled, Stage 1 schedules on a forecast rather than the outturn: one national AR(1) error per technology, added to every unit's capacity factor after shaping by its availability (4\*CF\*(1-CF), so no error where there is no resource and the forecast stays unbiased). The number per technology is the standard deviation before shaping (share of capacity); the realised national RMSE is lower by the RMS of the shape (about 0.8 for wind, 0.55 for solar) and is logged. autocorrelation_hours: AR(1) time constant; demand_rmse: demand error as a share of demand. Stage 2 always sees the outturn, so its redispatch against the Stage 1 schedule is the cost of resolving the error, which the FTI/NESO scheduling and dispatch assessment (Feb 2026) identifies as the main consumer cost of central dispatch. The series is generated for the whole calendar year from the seed and sliced to the solve period, so a week and the month containing it see identical forecasts. Calibration: the NESO day-ahead wind forecast has a capacity-weighted MAE of about 4.2% (Significance, 2024), roughly 5.3% RMSE; the defaults realise about 5.5% for wind and 5% for solar. scripts/market/forecast_error.py. |
| `forecast_error.enabled` | `false` |  |
| `forecast_error.seed` | `2030` |  |
| `forecast_error.autocorrelation_hours` | `6.0` |  |
| `forecast_error.technologies` | (mapping) |  |
| `forecast_error.technologies.wind_offshore` | `0.07` |  |
| `forecast_error.technologies.wind_onshore` | `0.07` |  |
| `forecast_error.technologies.solar_pv` | `0.09` |  |
| `forecast_error.demand_rmse` | `0.015` |  |
| `transmission_relaxation` | `1.0e6` | s_nom value (MVA) applied to ALL lines and transformers in "relax" mode. Must be &gt;&gt; max line flow. |
| `storage_marginal_cost_floor` | `1.0` | Minimum storage dispatch cost (GBP/MWh discharged) enforced before the wholesale solve. Breaks the zero-price charge/discharge degeneracy that a barrier solve without crossover otherwise exploits (see solve.solver.crossover). Units already carrying a higher marginal_cost are left alone; 0 disables. |
| `interconnector_hurdle_override` | (mapping) | Stage 1 calibration blocks, off by default; their other keys default in code. interconnector_hurdle_override (scripts/market/interconnector_hurdle.py, DEFAULT_HURDLE_OVERRIDE): when enabled, Stage 1 re-prices the single-basis interconnector links before the solve as if built with spread_gbp_per_mwh [0.0] and export_bid_floor_gbp_per_mwh [null = no floor] in place of interconnectors.pricing single_price_transaction_spread / single_price_export_bid_floor (GBP/MWh). The IC1 volume tranches, the cable loss and the circulation epsilon are kept. level_offset_gbp_per_mwh [0.0] adds a constant to every foreign reference price; apply_in_balancing [false] re-prices the balancing stage's network the same way, so the BM values IC moves on Stage 1's foreign prices. churn_sink (market_utils._churn_sink_frame): a negative-only generator at each storage bus, share_of_storage_power (share 0-1) x that bus's storage power, priced just above the AR4+ curtailment bid, that absorbs surplus where the market would otherwise curtail; the balancing stage holds it at its Stage 1 output. central_constraint_screening (scripts/market/central_constraint_screening.py, DEFAULT_CONFIG): a central-dispatch Stage 1 solves each window with thermal limits on a monitored subset of branches and adds violated branches until none is exceeded; same optimum. Related: market.hydrogen_offtake_scale [absent = the build's offtake] scales the GB_H2 offtake load in both stages (market_utils.apply_hydrogen_offtake_scale). |
| `interconnector_hurdle_override.enabled` | `false` |  |
| `churn_sink` | (mapping) |  |
| `churn_sink.enabled` | `false` |  |
| `central_constraint_screening` | (mapping) |  |
| `central_constraint_screening.enabled` | `false` |  |
| `marginal_cost_basis` | (mapping) | Marginal cost basis for Stage 1 (as_built \| resource_cost). as_built: Stage 1 schedules on the network's market-shaped costs (ROC plant at -(banding x buyout), CfD plant at zero), which is what the plant bids. resource_cost: a public operator dispatching on fuel, carbon and variable O&M alone; renewables go to renewable_floor (GBP/MWh) and, using the build-time cost audit, any other generator whose cost carries a subsidy layer goes to fuel + carbon + VOM. Nuclear, thermal formula costs, storage and load shedding are untouched. scripts/market/marginal_cost_basis.py. |
| `marginal_cost_basis.mode` | `as_built` |  |
| `marginal_cost_basis.renewable_floor` | `0.01` |  |
| `marginal_cost_basis.support_order` | (mapping) | Under resource_cost, renewables at the floor are ordered by the support they lose when turned down (merchant first, the highest CfD strike or ROC value last), spread over spread_gbp_per_mwh (Ofgem 2023, A1.18). Code default DEFAULT_SUPPORT_ORDER. |
| `marginal_cost_basis.support_order.enabled` | `true` |  |
| `marginal_cost_basis.support_order.spread_gbp_per_mwh` | `1.0` |  |
| `renewable_curtailment_order` | (mapping) | The order in which zero-cost renewables give way when supply exceeds demand (as_built \| contract_aware). as_built: every CfD and merchant unit at 0, a tie the barrier solver splits arbitrarily. contract_aware: each unit backs off where its support terms make it: merchant renewables (merchant_carriers) at their cost plus merchant_variable_cost_gbp_per_mwh, so they are curtailed first; AR4+ CfDs at ar4_plus_bid_gbp_per_mwh (0: the contract pays whenever the price is not negative, so 0 is the reservation price); AR1-AR3, Investment Contracts and HPC at minus strike, so only they and ROC plant take the price below zero. Market wholesale only: not applied under copperplate_mode none (central dispatch) or marginal_cost_basis resource_cost. scripts/market/curtailment_order.py. |
| `renewable_curtailment_order.mode` | `as_built` |  |
| `renewable_curtailment_order.ar4_plus_bid_gbp_per_mwh` | `0.0` |  |
| `renewable_curtailment_order.merchant_variable_cost_gbp_per_mwh` | `0.5` |  |
| `renewable_curtailment_order.merchant_carriers` | (list of 6 items) |  |
| `unit_commitment` | (mapping) | Optional Stage 1 unit commitment overlay, independent of solve.mode, so the wholesale schedule can be MILP while the BM remains an anchored LP. Disabled by default (slower). Further keys, not set here (code defaults in brackets; scripts/market/wholesale_commitment.py): method [milp]: milp = one MILP per window, no marginal prices; milp_fixed = MILP commitment, then the LP with that commitment held as output bounds (status x floor &lt;= p &lt;= status x available), whose duals are the prices (restricted pricing); linearized = PyPSA's linearised relaxation (Hua et al. 2017), one LP. carrier_parameters.&lt;carrier&gt;.module_mw [null]: a committed unit's floor is p_min_pu x min(1, module_mw / p_nom) x p_max_pu(t) and a start costs start_up_cost_per_mw x min(module_mw, p_nom) (stations committed one module at a time). milp_options [{MIPGap: 0.005, TimeLimit: 600}]: Gurobi options of the MILP pass. initial_status also takes "free" (no initial state); milp_fixed carries each window's commitment into the next and applies initial_status to the first window only. Units with a positive minimum output (must-run) are never committed. Units: min_p_nom_mw in MW; ramp limits as a share of p_nom per hour; min up/down times in hours; start-up and shut-down costs in GBP/MW. |
| `unit_commitment.enabled` | `false` |  |
| `unit_commitment.min_p_nom_mw` | `50.0` |  |
| `unit_commitment.initial_status` | `off` | "off" \| "on" \| "preserve" |
| `unit_commitment.default_ramp_limit_up` | `1.0` |  |
| `unit_commitment.default_ramp_limit_down` | `1.0` |  |
| `unit_commitment.default_ramp_limit_start_up` | `1.0` |  |
| `unit_commitment.default_ramp_limit_shut_down` | `1.0` |  |
| `unit_commitment.exclude_carriers` | (list of 7 items) |  |
| `unit_commitment.carrier_parameters` | (mapping) |  |
| `unit_commitment.carrier_parameters.CCGT` | (mapping) |  |
| `unit_commitment.carrier_parameters.CCGT.enabled` | `true` |  |
| `unit_commitment.carrier_parameters.CCGT.p_min_pu` | `0.35` |  |
| `unit_commitment.carrier_parameters.CCGT.min_up_time` | `4` |  |
| `unit_commitment.carrier_parameters.CCGT.min_down_time` | `3` |  |
| `unit_commitment.carrier_parameters.CCGT.ramp_limit_up` | `0.5` |  |
| `unit_commitment.carrier_parameters.CCGT.ramp_limit_down` | `0.5` |  |
| `unit_commitment.carrier_parameters.CCGT.start_up_cost_per_mw` | `45.0` |  |
| `unit_commitment.carrier_parameters.CCGT.shut_down_cost_per_mw` | `0.0` |  |
| `unit_commitment.carrier_parameters.OCGT` | (mapping) |  |
| `unit_commitment.carrier_parameters.OCGT.enabled` | `true` |  |
| `unit_commitment.carrier_parameters.OCGT.p_min_pu` | `0.2` |  |
| `unit_commitment.carrier_parameters.OCGT.min_up_time` | `1` |  |
| `unit_commitment.carrier_parameters.OCGT.min_down_time` | `1` |  |
| `unit_commitment.carrier_parameters.OCGT.ramp_limit_up` | `1.0` |  |
| `unit_commitment.carrier_parameters.OCGT.ramp_limit_down` | `1.0` |  |
| `unit_commitment.carrier_parameters.OCGT.start_up_cost_per_mw` | `20.0` |  |
| `unit_commitment.carrier_parameters.OCGT.shut_down_cost_per_mw` | `0.0` |  |
| `unit_commitment.carrier_parameters.gas_engine` | (mapping) | reciprocating engines: fast, flexible |
| `unit_commitment.carrier_parameters.gas_engine.enabled` | `true` |  |
| `unit_commitment.carrier_parameters.gas_engine.p_min_pu` | `0.2` |  |
| `unit_commitment.carrier_parameters.gas_engine.min_up_time` | `1` |  |
| `unit_commitment.carrier_parameters.gas_engine.min_down_time` | `1` |  |
| `unit_commitment.carrier_parameters.gas_engine.ramp_limit_up` | `1.0` |  |
| `unit_commitment.carrier_parameters.gas_engine.ramp_limit_down` | `1.0` |  |
| `unit_commitment.carrier_parameters.gas_engine.start_up_cost_per_mw` | `10.0` |  |
| `unit_commitment.carrier_parameters.gas_engine.shut_down_cost_per_mw` | `0.0` |  |
| `unit_commitment.carrier_parameters.coal` | (mapping) |  |
| `unit_commitment.carrier_parameters.coal.enabled` | `true` |  |
| `unit_commitment.carrier_parameters.coal.p_min_pu` | `0.4` |  |
| `unit_commitment.carrier_parameters.coal.min_up_time` | `8` |  |
| `unit_commitment.carrier_parameters.coal.min_down_time` | `6` |  |
| `unit_commitment.carrier_parameters.coal.ramp_limit_up` | `0.35` |  |
| `unit_commitment.carrier_parameters.coal.ramp_limit_down` | `0.35` |  |
| `unit_commitment.carrier_parameters.coal.start_up_cost_per_mw` | `80.0` |  |
| `unit_commitment.carrier_parameters.coal.shut_down_cost_per_mw` | `0.0` |  |
| `unit_commitment.carrier_parameters.oil` | (mapping) |  |
| `unit_commitment.carrier_parameters.oil.enabled` | `true` |  |
| `unit_commitment.carrier_parameters.oil.p_min_pu` | `0.25` |  |
| `unit_commitment.carrier_parameters.oil.min_up_time` | `2` |  |
| `unit_commitment.carrier_parameters.oil.min_down_time` | `2` |  |
| `unit_commitment.carrier_parameters.oil.ramp_limit_up` | `0.8` |  |
| `unit_commitment.carrier_parameters.oil.ramp_limit_down` | `0.8` |  |
| `unit_commitment.carrier_parameters.oil.start_up_cost_per_mw` | `30.0` |  |
| `unit_commitment.carrier_parameters.oil.shut_down_cost_per_mw` | `0.0` |  |
| `look_ahead_hours` | `24` | Look-ahead (rolling_day_ahead only): each window is optimised over window_hours + look_ahead_hours (cut at the solve period's end) and only its first window_hours are kept; the next window starts from the state at the last kept hour. Storage then values what it holds at the window's end at the next day's prices instead of at nothing. 0 or null = no look-ahead. Rationale and cost: scripts/market/solve_wholesale.py, "Look-ahead for the rolling day-ahead stage". |
| `look_ahead_resolution_hours` | `6` | The look-ahead hours as blocks of this many hours (block-mean inputs, snapshot weightings equal to the block length); the kept hours stay at full resolution. null = hourly. 6-hour blocks keep about 98% of the hourly look-ahead's storage arbitrage gain at a much smaller LP. |
