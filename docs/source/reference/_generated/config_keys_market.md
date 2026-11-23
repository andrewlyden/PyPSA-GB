## `market`

File: `config/defaults/base/market.yaml`. Keys below are relative to `market`.

| Key | Default | Description |
|---|---|---|
| `enabled` | `false` | MARKET SIMULATION (disabled by default) Two-stage market dispatch: wholesale clearing + balancing mechanism redispatch. Stage 1 (Wholesale): copperplate dispatch, no network constraints, uniform price. Stage 2 (Balancing): redispatch with full network constraints; generators and storage adjust from their wholesale position at bid/offer prices. Requires a finalized network ({scenario}.nc) from the main workflow. Runs in the main workflow when a scenario sets market.enabled: true. Master toggle for market simulation |
| `dispatch` | `self` | self: the market (Stage 1 clears, the BM pays bids and offers). central: the system operator dispatches at cost; it attaches the overlay config/defaults/dispatch/central.yaml. |
| `wholesale_only` | `false` | If true, run only wholesale (no balancing/analysis) |
| `storage_soc` | (mapping) | Storage state of charge at the month boundary. Both stages start storage at the same level (initial_fraction, share 0-1; null = the network's own, 50%). month_end: at_least_initial \| free; at_least_initial makes Stage 1's last window end each unit at or above its starting level (capped at what it can recharge in that window), so a month cannot sell free starting energy. carriers: storage carriers it applies to (null = all). |
| `storage_soc.initial_fraction` | `null` |  |
| `storage_soc.month_end` | `at_least_initial` |  |
| `storage_soc.carriers` | `null` |  |
| `hydrogen_store` | (mapping) | Hydrogen store level. carry_level: the store level is carried across rolling windows in both stages, like storage state of charge, instead of each window being cyclic. initial_fraction: starting level as a share of e_nom (null = the network's e_initial). month_end (at_least_initial \| free): at_least_initial holds the last window's end at or above the start, capped at the window's starting level plus what the electrolysers can make in it net of the offtake. carriers: the store carriers carried. See scripts/market/market_utils.py (prepare_carried_stores). end_of_window_value_gbp_per_mwh_h2: value V (GBP/MWh H2) credited on each carried store's level at the end of every Stage 1 rolling window (objective only, stripped from the reported objective), so a day-ahead window values hydrogen it leaves for the next and electrolyses whenever electricity is below efficiency x V. Not used by the BM or in single mode. null = off; set in source/CP30.yaml. |
| `hydrogen_store.carry_level` | `true` |  |
| `hydrogen_store.initial_fraction` | `0.5` |  |
| `hydrogen_store.month_end` | `at_least_initial` |  |
| `hydrogen_store.carriers` | `[H2_gas]` |  |
| `hydrogen_store.end_of_window_value_gbp_per_mwh_h2` | `null` |  |
| `internal_dc_tie_break_gbp_per_mwh` | `0.75` | Internal HVDC tie-break (GBP/MWh). GB-internal DC links (EGL1/2, Western HVDC, the Caithness-Moray-Shetland legs, ...) are lossless and free both ways, so their flow beside the AC network is otherwise a degenerate tie. Both stages add this cost on \|flow\| (a proxy for converter losses); objective only, stripped from the Stage 1 objective and the BM cost and reported as internal_dc_tie_break_cost_GBP in the market summary. Cross-border interconnectors and hydrogen links are not touched. 0 or null = off. scripts/market/internal_dc_tie_break.py. |
| `must_take_generation` | (mapping) | Must-take generation: generators, by the data_source the build stamped, that run at their availability in both stages: never curtailed in Stage 1 (p_min_pu = p_max_pu) and held at their Stage 1 position, out of the balancing mechanism. Set at market level because both stages read it. Example: CP30_small_scale_solar_gsp (CP30 rooftop G98/G83 solar). scripts/market/must_take.py; empty list = none. |
| `must_take_generation.data_sources` | `[]` |  |
