## `scripts/CP30/`

Clean Power 2030 data extraction helpers.

| Module | Summary |
|---|---|
| `scripts/CP30/assumptions.py` | Explicit CP30 operational assumptions. |
| `scripts/CP30/audit_offshore_landing_mapping.py` | Create a transparent evidence/review register for offshore landing mappings. |
| `scripts/CP30/audit_offshore_project_crosswalk.py` | Audit whether a CP30 landing map has a project and TEC evidence chain. |
| `scripts/CP30/balancing_validation.py` | Compare a solved CP30 2030 balancing family against NESO's published figures. |
| `scripts/CP30/brownfield_fleet.py` | Build the CP30 renewable fleet from named projects, proxying only the residual. |
| `scripts/CP30/build_gas_station_schedule.py` | Write the declared CP30 gas station schedule. |
| `scripts/CP30/build_reinforcement_candidate_crosswalk.py` | Produce a \*candidate\* CP30-to-ETYS reinforcement crosswalk. |
| `scripts/CP30/cfd_station_schedule.py` | Named CfD stations carved out of CP30's anonymous proxies. |
| `scripts/CP30/committed_overlay.py` | Committed plant that CP30 omits, added as a declared overlay. |
| `scripts/CP30/connection_allocation.py` | Source-backed spatial allocation for CP30 future connection capacity. |
| `scripts/CP30/CP30_data.py` | Extract Clean Power 2030 workbook tables into normalized CSV resources. |
| `scripts/CP30/energy_validation.py` | CP30 energy validation: build-time availability and post-solve dispatch. |
| `scripts/CP30/fleet_location_validation.py` | CP30 fleet location check: installed capacity by CP30 ES5 region, NESO CP.08 zone and ETYS boundary side. |
| `scripts/CP30/gas_schedule.py` | Build the CP30 gas fleet from named stations instead of proxy buses. |
| `scripts/CP30/generation.py` | Clean Power 2030 generation capacity helpers. |
| `scripts/CP30/hnd_hybrid.py` | Fail-closed HND additions to an ETYS network and CP30 capacity allocation. |
| `scripts/CP30/locational_validation.py` | Locational and network validation of a built CP30 network. |
| `scripts/CP30/nuclear_schedule.py` | Place CP30 nuclear capacity at named stations instead of proxy buses. |
| `scripts/CP30/reinforcement_layer.py` | (no module docstring) |
| `scripts/CP30/reinforcements.py` | (no module docstring) |
| `scripts/CP30/spatial_allocation.py` | CP30 ES5 regional spatial allocation helpers. |
| `scripts/CP30/stage1_validation.py` | Validate a CP30 calendar year's Stage 1 (wholesale) dispatch against CP30. |
| `scripts/CP30/storage_interconnectors.py` | Clean Power 2030 storage and interconnector helpers. |
| `scripts/CP30/storage_regions.py` | CP30 ES5 regional allocation for storage. |
| `scripts/CP30/validate_cp30_dispatch.py` | Validate a solved CP30 calendar year's FINAL dispatch against CP30 energy. |
| `scripts/CP30/validate_cp30_network.py` | Write CP30 network validation outputs for a built PyPSA network. |
| `scripts/CP30/validation.py` | CP30 network/reference validation helpers. |
