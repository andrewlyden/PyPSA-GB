## `storage`

File: `config/defaults/base/storage.yaml`. Keys below are relative to `storage`.

| Key | Default | Description |
|---|---|---|
| `round_trip_efficiencies` | (mapping) | Round-trip efficiency (share 0-1) of each storage carrier built from the FES building blocks (add_storage.py). Each leg (efficiency_store, efficiency_dispatch) is the square root of the value. Code defaults in add_storage.py must match. Diagnose fleet round trips with scripts/market/diagnose_storage_degeneracy.py. Changing a value requires a re-solve. |
| `round_trip_efficiencies.Battery` | `0.8464` | 0.92 per leg, grid-scale lithium |
| `round_trip_efficiencies.Domestic Battery` | `0.8464` | same cells, behind the meter |
| `round_trip_efficiencies.Pumped Storage Hydroelectricity` | `0.7569` | 0.87 per leg |
| `round_trip_efficiencies.LAES` | `0.55` | storage_parameters.py LAES RTE |
| `round_trip_efficiencies.LDES` | `0.7` | CP30's long-duration storage ("other storage", mostly existing pumped hydro): its implied round trip, 10,441.5 / 14,848.7 GWh = 0.703. |
| `dispatch_costs` | (mapping) | Storage dispatch cost by carrier (GBP/MWh discharged) where the storage data has no marginal_cost: variable O&M and degradation, and a tie-break against the zero-price degeneracy in which an interior-point solve charges and discharges a unit at once (see solve.solver.crossover). Applied in add_storage.py and enforced as a floor in the wholesale solve (market.wholesale.storage_marginal_cost_floor). |
| `dispatch_costs.default` | `1.0` |  |
| `dispatch_costs.Battery` | `1.0` |  |
| `dispatch_costs.battery` | `1.0` |  |
| `dispatch_costs.LAES` | `1.5` |  |
| `dispatch_costs.CAES` | `1.0` |  |
| `dispatch_costs.pumped_hydro` | `0.5` |  |
| `dispatch_costs.Pumped Storage Hydroelectricity` | `0.5` |  |
| `standing_losses` | (mapping) | Standing loss (share of stored energy lost per hour) by storage carrier; carriers not listed take default. Code defaults: add_storage.py DEFAULT_STORAGE_STANDING_LOSSES. |
| `standing_losses.default` | `0.0` |  |
| `standing_losses.Battery` | `5e-05` |  |
| `standing_losses.Domestic Battery` | `5e-05` |  |
| `standing_losses.battery` | `5e-05` |  |
| `standing_losses.LAES` | `0.0` |  |
| `standing_losses.Liquid Air Energy Storage` | `0.0` |  |
| `standing_losses.Pumped Storage Hydroelectricity` | `0.0` |  |
| `standing_losses.pumped_hydro` | `0.0` |  |
| `placement` | (mapping) | Capacity-aware storage placement. After nearest-bus mapping no bus may hold more storage power than max_loading x the rating of the lines, transformers and links touching it (MW per MVA); the excess goes to the same substation's other busbars with headroom, else to the nearest bus of at least min_target_voltage_kv (kV) within search_radius_km. Code defaults: add_storage.py |
| `placement.enabled` | `true` |  |
| `placement.max_loading` | `1.0` |  |
| `placement.search_radius_km` | `40.0` |  |
| `placement.min_target_voltage_kv` | `132.0` |  |
