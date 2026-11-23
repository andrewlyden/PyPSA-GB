## `network`

File: `config/defaults/base/network.yaml`. Keys below are relative to `network`.

| Key | Default | Description |
|---|---|---|
| `clustering` | (mapping) | Network clustering (off by default). Presets and strategies: config/clustering.yaml. |
| `clustering.enabled` | `false` |  |
| `component_aggregation` | (mapping) | Merges identical generators, storage units and loads at the same bus, summing their capacities. Components merge only when their attributes and time series match. Independent of clustering. |
| `component_aggregation.enabled` | `false` |  |
| `component_aggregation.include_generators` | `false` | Aggregate identical thermal generators |
| `component_aggregation.include_storage_units` | `false` | Aggregate storage units when possible |
| `component_aggregation.include_loads` | `false` | Aggregate loads per bus |
| `component_aggregation.include_stores` | `false` | Aggregate Store components (rarely used) |
| `component_aggregation.include_committable` | `false` | Allow committable unit merging |
| `component_aggregation.tolerance` | `1e-09` | Float comparison tolerance |
| `transmission` | (mapping) | Transmission constraints: options to relax published ratings, the balancing-stage voltage floor, seasonal ratings, per-line corrections and outage schedules. |
| `transmission.min_line_s_nom` | `0` | Floor on line capacity (MVA); lines below it are raised to it. 0 = published ratings. |
| `transmission.min_transformer_s_nom` | `0` | Floor on transformer capacity (MVA). 0 = published ratings. |
| `transmission.capacity_scale` | `1.0` | Scaling factor on every line and transformer capacity (1.0 = unchanged). |
| `transmission.min_bm_constraint_voltage_kv` | `220` | Balancing-stage voltage floor (kV). Every line and transformer with an end strictly below it has s_nom set to 9999 MVA (unconstrained) before the BM solve; transformer impedance is preserved. 400 and 275 kV lines and 400/275 kV transformers keep their ratings (solve_network.bm_voltage_floor_branches). Why: ETYS Appendix B carries individual 132 kV trunk segments but not the parallel 33/66 kV mesh and ring paths, so a DC power flow saturates them at unrealistically low flows. NESO manages 132 kV loop flows with operational tools this model does not represent, and publishes boundary flows at inter-zone (275 kV) level. 0 = published ratings at every voltage; 220 = relax 132 kV and below; 275 = also relax 220 kV. |
| `transmission.bm_voltage_floor` | (mapping) | Scottish exception: SHE Transmission and SPT own the 132 kV network, which NESO manages in the BM, so SHE/SPT branches (ETYS licensee of the Appendix B sheet, else substation or location) whose lower voltage is at least scottish_min_constrained_kv (kV) keep their ratings. Their 33/66 kV collector networks and E&W sub-220 kV (DNO distribution) stay relaxed. |
| `transmission.bm_voltage_floor.scottish_transmission_constrained` | `true` |  |
| `transmission.bm_voltage_floor.scottish_min_constrained_kv` | `132` |  |
| `transmission.seasonal_ratings` | (mapping) | ETYS seasonal circuit ratings (Appendix B B-2-1a/b/c, B-2-2a/b/c). s_nom is the winter rating; lines without published seasonal values take their voltage class's median season/winter ratio. Network-constrained solves (BM, central dispatch, network solve) apply them as a monthly factor on lines_t.s_max_pu; relaxed 9,999 MVA lines, transformers (one rating) and the Stage 1 copperplate are unaffected. season_months (month numbers) are the ER P27 / ACE 104 overhead-line seasons as tabulated in Northern Powergrid IMP/001/011 v4.0 Table 1. The secured boundary limits (scripts/solve/secured_boundaries.py) use NOA seasons on a separate constraint. |
| `transmission.seasonal_ratings.enabled` | `true` |  |
| `transmission.seasonal_ratings.season_months` | (mapping) |  |
| `transmission.seasonal_ratings.season_months.winter` | `[11, 12, 1, 2]` |  |
| `transmission.seasonal_ratings.season_months.spring` | `[3, 4]` |  |
| `transmission.seasonal_ratings.season_months.summer` | `[5, 6, 7, 8]` |  |
| `transmission.seasonal_ratings.season_months.autumn` | `[9, 10]` |  |
| `transmission.line_rating_overrides` | (mapping) | Per-line s_nom overrides (MVA) for known ETYS data errors: {line_id: s_nom_mva}. Applied in the network solve and the BM after loading; they take precedence over min_bm_constraint_voltage_kv. Checked against ETYS 2022-2025 Appendix B and left at their published ratings: CHAP1-_ECCF1J_0 / _ECCF1K_0 (radial stubs), GLLU1Q/R_NETS1-_0 and SFEE1A_SFEE1C_0 / SFEE1B_SFEE1D_0. |
| `transmission.line_rating_overrides.GHPS1-_GLGL1-_ETYS_UPG_2027` | `482` | Glenmuckloch-Glenglass 132 kV: ETYS 2024 B-2-2b gives 12/12/11/12 MVA; ETYS 2022 and 2023 list the identical circuits at 482/445/385/445 MVA, matching the 3 x 360 MVA SGTs at GHPS. Winter 482 from the earlier editions; the seasonal factor keeps the 2024 ratios. |
| `transmission.line_rating_overrides.GHPS1-_GLGL1-_ETYS_UPG_2027_1` | `482` |  |
| `transmission.line_rating_overrides.MAHI3-_TRLO3-_0` | `9999` | 33kV artifact: 81 MW wind behind 40 MVA sub-transmission |
| `transmission.line_rating_overrides.GLGL3-_WHLL3-_0` | `9999` | 33kV artifact: 70 MW wind behind 30 MVA sub-transmission |
| `transmission.line_rating_overrides.HARR3-_KYPE3A_0` | `9999` | 33kV artifact: 72 MW wind behind 40 MVA sub-transmission |
| `transmission.line_rating_overrides.ELDE1Q_JOHN1Q_0` | `149` | Elderslie-Johnstone 132 kV cables: ETYS 2024 B-2-2b "Change" rows give 1 MVA, a placeholder; B-2-1b and the later change give 149 MVA. At 1 MVA the 46 MW Johnstone load (JOHN3-) is islanded. |
| `transmission.line_rating_overrides.ELDE1R_JOHN1R_0` | `149` |  |
| `transmission.outage_schedule` | (mapping) | Transmission outage schedule. Without outages the model sees the intact network and under-estimates constraint costs (NESO 2020 thermal constraint costs GBP 596m against GBP 126m modelled intact-only). source: "csv" (user-supplied file), "synthetic" (stochastic outages, parameters below) or "neso" (NESO day-ahead boundary limits). |
| `transmission.outage_schedule.enabled` | `false` |  |
| `transmission.outage_schedule.source` | `neso` | "csv" \| "synthetic" \| "neso" |
| `transmission.outage_schedule.neso` | (mapping) | NESO Day-Ahead Constraint Flows and Limits (https://www.neso.energy/data-portal/day-ahead-constraint-flows-and-limits). The limits already include N-1/N-2 security margins and real outages. constraint_mode: aggregate_boundary: per-line thermal limits kept, plus a constraint holding the signed total flow across each mapped boundary within the NESO limit (closer to what NESO publishes). uniform_s_max_pu: each boundary limit becomes one percentage derate on every mapped line. |
| `transmission.outage_schedule.neso.boundary_mapping` | `data/network/neso_boundary_mapping.yaml` |  |
| `transmission.outage_schedule.neso.resource_id` | `38a18ec1-9e40-465d-93fb-301e80fd1352` |  |
| `transmission.outage_schedule.neso.api_base` | `https://api.neso.energy/api/3/action/datastore_search_sql` |  |
| `transmission.outage_schedule.neso.cache_dir` | `resources/neso_cache` | Read order: a complete cache window, then committed_cache, then the NESO API only if allow_api_fetch (a boundary found in none of them raises). |
| `transmission.outage_schedule.neso.committed_cache` | `data/network/neso_day_ahead_constraint_limits.csv.gz` | Committed snapshot of NESO day-ahead boundary limits, 2019-10 to 2026-04 (scripts/solve/build_neso_boundary_limit_cache.py). |
| `transmission.outage_schedule.neso.allow_api_fetch` | `false` |  |
| `transmission.outage_schedule.neso.constraint_mode` | `aggregate_boundary` | "uniform_s_max_pu" \| "aggregate_boundary" |
| `transmission.outage_schedule.neso.nearest_tolerance_minutes` | `45` | Direct timestamp match window before gap filling |
| `transmission.outage_schedule.neso.gap_fill_mode` | `interpolate` | "interpolate" \| "nearest_available" \| "unconstrained" \| "fail" |
| `transmission.outage_schedule.neso.min_s_max_pu` | `0.05` | interpolate = use surrounding NESO data, then edge-fill if needed Uniform mode only: floor to prevent near-zero line capacity |
| `transmission.outage_schedule.neso.soft_boundaries` | `{}` | Option: aggregate-boundary slack, keyed by boundary name, e.g. |
| `transmission.outage_schedule.synthetic` | (mapping) | enabled: true penalty_gbp_per_mwh: 1000 max_violation_mw: 2500 Synthetic maintenance and forced outages, on lines and transformers of at least min_s_nom_mva. |
| `transmission.outage_schedule.synthetic.seed` | `42` | Random seed for reproducibility |
| `transmission.outage_schedule.synthetic.min_s_nom_mva` | `100` | Only schedule outages on lines &gt;= this rating |
| `transmission.outage_schedule.synthetic.forced_outage_rate` | (mapping) | Forced outage rate by voltage (kV): share of hours unavailable per year |
| `transmission.outage_schedule.synthetic.forced_outage_rate.400` | `0.03` | 400kV: ~3% unavailability (263 hours/year) |
| `transmission.outage_schedule.synthetic.forced_outage_rate.275` | `0.04` | 275kV: ~4% |
| `transmission.outage_schedule.synthetic.forced_outage_rate.132` | `0.05` | 132kV: ~5% |
| `transmission.outage_schedule.synthetic.maintenance_duration_hours` | `168` | Average maintenance window duration (hours) 1 week typical |
| `transmission.outage_schedule.synthetic.seasonal_weights` | (mapping) | Relative maintenance weight by month (more maintenance in low-demand months) |
| `transmission.outage_schedule.synthetic.seasonal_weights.1` | `0.4` | January: winter, minimal maintenance |
| `transmission.outage_schedule.synthetic.seasonal_weights.2` | `0.5` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.3` | `0.8` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.4` | `1.2` | April: spring shoulder |
| `transmission.outage_schedule.synthetic.seasonal_weights.5` | `1.5` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.6` | `1.8` | June: summer, peak maintenance |
| `transmission.outage_schedule.synthetic.seasonal_weights.7` | `1.8` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.8` | `1.5` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.9` | `1.2` | September |
| `transmission.outage_schedule.synthetic.seasonal_weights.10` | `0.8` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.11` | `0.5` |  |
| `transmission.outage_schedule.synthetic.seasonal_weights.12` | `0.3` | December: winter, minimal maintenance |
| `etys` | (mapping) | ETYS publication year for the base topology (ETYS network model only): 2022, 2023 or 2024. |
| `etys.year` | `2024` | ETYS publication year |
| `upgrades` | (mapping) | Planned network upgrades from the selected ETYS Appendix B: circuit additions and removals, transformer changes and new HVDC links (network_model: "ETYS" only). |
| `upgrades.enabled` | `true` | Set false to not apply network upgrades |
| `upgrades.upgrade_year` | `null` | Apply upgrades through this year (null = modelled_year) null, or specific year like 2030 |
| `upgrades.documented_topology_corrections` | (mapping) | Source-backed corrections; each can be disabled per scenario without altering ETYS data. |
| `upgrades.documented_topology_corrections.little_horsted_2025_bus_section` | `true` |  |
| `upgrades.documented_topology_corrections.yorkshire_green_2027_terminal_reconciliation` | `true` |  |
| `upgrades.documented_topology_corrections.connahs_quay_2027_terminal_reconciliation` | `true` | CONQ42 in the 2027 North Wales rows is the base busbar CONQ41. |
| `upgrades.documented_topology_corrections.glenglass_2025_terminal_reconciliation` | `true` | GLGL1B in the 2029/2033 rows is the base busbar GLGL1Q |
| `upgrades.topology_sensitivities` | `[]` | Explicit non-default internal-topology sensitivities. Keep empty for base runs. |
| `upgrades.busbar_alias_fallback` | `true` | Resolve an ETYS busbar-specific node (e.g. COCK21) to the site's generic busbar at the same voltage (COCK2-) when only the latter exists in the base network (e.g. Inch Cape's export cables to Cockenzie, which otherwise leave the INCO/INCW buses as a dead island). |
| `upgrades.prune_dead_islands` | `true` | After removals, drop small islands that carry no generation, load or storage (e.g. Robin Rigg West's HARK12 pocket after the 2027 Harker 275/132 kV transformer removal). Islands with attached components are kept. |
| `upgrades.engineering_fallbacks` | (mapping) | Option: engineering ratings (MVA) for named ETYS additions whose electrical data is TBC. Off: incomplete rows are audited but not added, so the base case is strictly source-backed. |
| `upgrades.engineering_fallbacks.enabled` | `false` |  |
| `upgrades.engineering_fallbacks.circuit_rating_mva_by_voltage_kv` | (mapping) |  |
| `upgrades.engineering_fallbacks.circuit_rating_mva_by_voltage_kv.400` | `1000` |  |
| `upgrades.engineering_fallbacks.circuit_rating_mva_by_voltage_kv.275` | `400` |  |
| `upgrades.engineering_fallbacks.circuit_rating_mva_by_voltage_kv.220` | `300` |  |
| `upgrades.engineering_fallbacks.circuit_rating_mva_by_voltage_kv.132` | `150` |  |
| `upgrades.engineering_fallbacks.transformer_rating_mva_by_voltage_kv` | (mapping) |  |
| `upgrades.engineering_fallbacks.transformer_rating_mva_by_voltage_kv.400` | `600` |  |
| `upgrades.engineering_fallbacks.transformer_rating_mva_by_voltage_kv.275` | `350` |  |
| `upgrades.engineering_fallbacks.transformer_rating_mva_by_voltage_kv.132` | `120` |  |
| `upgrades.engineering_fallbacks.default_circuit_rating_mva` | `500` |  |
| `upgrades.engineering_fallbacks.default_transformer_rating_mva` | `300` |  |
| `spatial_mapping` | (mapping) | How generators and other sites are mapped to network buses (all generator types). |
| `spatial_mapping.weak_bus_relief` | (mapping) | Renewable placements above max_loading x a bus's branch capacity (MW per MVA) are moved, proxy tranches first, to the same substation, else the nearest bus of at least min_target_voltage_kv (kV) within search_radius_km that has room. Capacity, profiles and CP30 regions are unchanged. |
| `spatial_mapping.weak_bus_relief.enabled` | `true` |  |
| `spatial_mapping.weak_bus_relief.max_loading` | `1.5` |  |
| `spatial_mapping.weak_bus_relief.search_radius_km` | `40.0` |  |
| `spatial_mapping.weak_bus_relief.min_target_voltage_kv` | `132.0` |  |
| `spatial_mapping.export_relief` | (mapping) | export_relief: CP30 proxy increments of the listed carriers are moved out of any pocket whose generation less minimum load cannot reach the &gt;= sink_kv (kV) grid over the branches the BM constrains (transmission.min_bm_constraint_voltage_kv and bm_voltage_floor), to the nearest bus with export headroom within search_radius_km. Named farms stay at their connection points. |
| `spatial_mapping.export_relief.enabled` | `true` |  |
| `spatial_mapping.export_relief.carriers` | `[wind_onshore, solar_pv]` |  |
| `spatial_mapping.export_relief.search_radius_km` | `60.0` |  |
| `spatial_mapping.export_relief.min_target_voltage_kv` | `132.0` |  |
| `spatial_mapping.export_relief.sink_kv` | `275.0` |  |
| `spatial_mapping.placement_exclusions` | (mapping) | placement_exclusions (ETYS networks): buses that nearest-bus matching never gives onshore wind, solar or storage: DC-only hubs, offshore platforms (outside the GSP land boundary plus land_buffer_m, metres), OFTO onshore export substations radial to an offshore platform, and buses more than coordinate_outlier_km from the rest of their substation. Rows on such a bus move to the nearest allowed bus. |
| `spatial_mapping.placement_exclusions.enabled` | `true` |  |
| `spatial_mapping.placement_exclusions.land_boundary_file` | `data/network/GSP/GSP_regions_27700_20250109.geojson` |  |
| `spatial_mapping.placement_exclusions.land_buffer_m` | `2000.0` |  |
| `spatial_mapping.placement_exclusions.coordinate_outlier_km` | `30.0` |  |
| `spatial_mapping.placement_exclusions.cluster_km` | `5.0` |  |
| `spatial_mapping.coordinate_detection` | (mapping) | Coordinate system detection thresholds (OSGB36 metres vs WGS84 degrees) |
| `spatial_mapping.coordinate_detection.osgb36_min_value` | `1000` | x or y &gt; 1000: OSGB36 (metres) |
| `spatial_mapping.coordinate_detection.wgs84_max_absolute` | `100` | x and y &lt; 100: WGS84 (degrees) |
| `spatial_mapping.transmission_regions` | (mapping) | Transmission operator latitude boundaries (degrees N, WGS84): SHETL lat &gt; shetl_lat_min; SPTL sptl_lat_min &lt; lat &lt;= shetl_lat_min; NGET lat &lt;= sptl_lat_min. |
| `spatial_mapping.transmission_regions.shetl_lat_min` | `57.0` |  |
| `spatial_mapping.transmission_regions.sptl_lat_min` | `55.5` |  |
| `spatial_mapping.distance_km` | (mapping) | Bus-mapping distances (km). Sites beyond them still map to the nearest bus (no hard cutoff). |
| `spatial_mapping.distance_km.site_to_bus` | `200.0` | General site to nearest bus |
| `spatial_mapping.distance_km.gsp_to_bus` | `100.0` | GSP coordinate to nearest bus |
| `spatial_mapping.distance_km.thermal_generator` | `150.0` | Thermal generator (OSGB36) to nearest bus |
| `spatial_mapping.fuzzy_threshold` | `0.8` | Fuzzy string matching score threshold (0-1) |
| `spatial_mapping.fallback_major_bus_count` | `10` | Fallback: top-N most-connected buses used when no major bus is found |
| `spatial_mapping.generator_voltage` | (mapping) | When a generator is moved to a higher-voltage bus |
| `spatial_mapping.generator_voltage.min_check_mw` | `50` | Skip bus check below this capacity (MW) |
| `spatial_mapping.generator_voltage.very_large_mw` | `300` | At or above this (MW), connect at &gt;= min_preferred_kv |
| `spatial_mapping.generator_voltage.large_low_voltage_mw` | `200` | At or above this (MW), never connect at &lt;= low_voltage_kv |
| `spatial_mapping.generator_voltage.medium_mw` | `100` | At or above this (MW), check transformer capacity |
| `spatial_mapping.generator_voltage.max_transformer_ratio` | `0.8` | Max generator / transformer s_nom ratio |
| `spatial_mapping.generator_voltage.min_preferred_kv` | `275` | Preferred minimum connection voltage (kV) |
| `spatial_mapping.generator_voltage.high_voltage_kv` | `400` | 400 kV-only search threshold (kV) |
| `spatial_mapping.generator_voltage.low_voltage_kv` | `33` | Low-voltage threshold (kV) |
| `spatial_mapping.generator_voltage.fallback_hv_radius_km` | `80` | Max km for post-aggregation nearest HV bus search |
| `spatial_mapping.earth_radius_km` | `6371.0` | Earth radius (km) for haversine distances |
