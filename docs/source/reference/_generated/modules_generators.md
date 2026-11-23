## `scripts/generators/`

| Module | Summary |
|---|---|
| `scripts/generators/aggregate_renewable_generators.py` | Aggregate renewable generators by (bus, carrier) to reduce problem size. |
| `scripts/generators/analyze_unmapped_generators.py` | Unmapped Generator Analysis for PyPSA-GB |
| `scripts/generators/apply_marginal_costs.py` | Apply marginal costs to generators in PyPSA network. |
| `scripts/generators/build_bmu_mapping.py` | Build BMU-to-generator name mapping for ELEXON bid/offer integration. |
| `scripts/generators/build_elexon_monthly_availability.py` | Monthly station availability for a historical validation year from ELEXON physical notifications. |
| `scripts/generators/build_fes_station_schedule.py` | Build one year's FES declared station schedule. |
| `scripts/generators/build_power_station_dictionary_crosswalk.py` | Build a local BMU crosswalk from the OSUKED Power Station Dictionary. |
| `scripts/generators/calibrate_daily_mc.py` | Calibrate daily per-generator empirical marginal costs from ELEXON data. |
| `scripts/generators/calibrate_renewable_mc.py` | Calibrate renewable marginal costs using ELEXON bid data. |
| `scripts/generators/calibrate_thermal_mc.py` | Calibrate thermal marginal costs using ELEXON day-ahead price data. |
| `scripts/generators/calibrate_wind_profiles.py` | Diagnostic: compare ERA5 wind profiles against ESPENI observed generation. |
| `scripts/generators/cfd_vintage_labels.py` | Assign allocation-round labels to CP30 CfD capacity by declared schedule. |
| `scripts/generators/create_generators_full.py` | Create Full Generators Dataset ============================= |
| `scripts/generators/declared_buses.py` | Declared connection buses for historical builds: the 2030 placement audit applied to past years. |
| `scripts/generators/deduplicate_tec_repd.py` | TEC vs REPD Deduplication Script - Simplified Approach |
| `scripts/generators/DUKES_generator_data.py` | Extract historical generator capacity data from DUKES 5.11 |
| `scripts/generators/enhance_locations_with_wikipedia.py` | Enhance generator locations using a Wikipedia coordinate database. |
| `scripts/generators/enhance_tec_locations_with_geocoding.py` | Enhanced TEC Location Mapping with Geocoding Backup for PyPSA-GB |
| `scripts/generators/export_correction_factors.py` | Export carrier correction factors from historical ELEXON-calibrated runs. |
| `scripts/generators/export_network_to_csv.py` | (no module docstring) |
| `scripts/generators/extract_repd_dispatchable_sites.py` | Extract Dispatchable Thermal Sites from REPD for PyPSA-GB |
| `scripts/generators/fes_supply.py` | (no module docstring) |
| `scripts/generators/finalize_generator_integration.py` | (no module docstring) |
| `scripts/generators/historical_dispatch_anchors.py` | Transparent validation-only anchors for historically committed generation. |
| `scripts/generators/historical_roc_physical_overlay.py` | Apply reviewed, physical 2020 ROC station corrections before dispatch. |
| `scripts/generators/integrate_dukes_locations.py` | DUKES 5.11 Power Station Location Integration for PyPSA-GB |
| `scripts/generators/integrate_renewable_generators.py` | Integrate renewable generators into PyPSA network. |
| `scripts/generators/integrate_thermal_generators.py` | Integrate thermal generators into PyPSA network with hybrid data source routing. |
| `scripts/generators/map_dispatchable_generator_locations.py` | Dispatchable Generator Location Mapping for PyPSA-GB |
| `scripts/generators/map_final_generators.py` | Final generator mapping strategies to complete the database. |
| `scripts/generators/marginal_costs.py` | Marginal Cost Calculations for PyPSA-GB - STANDALONE UTILITY |
| `scripts/generators/ofgem_roc_capacity_overlay.py` | Allocate the full known Ofgem ROC DNC into the historical PyPSA fleet. |
| `scripts/generators/prepare_dispatchable_generator_sites.py` | Prepare Dispatchable Generator Site Data for PyPSA-GB |
| `scripts/generators/process_tec_generators.py` | Standalone TEC Generator Processing for PyPSA-GB |
| `scripts/generators/proxy_profiles.py` | Regional weather profiles for proxy renewable units. |
| `scripts/generators/renewable_performance_factors.py` | Technology and resource-region renewable performance-factor registry. |
| `scripts/generators/single_word_tec_repd_matching.py` | Single Word Matching for TEC-REPD Sites |
| `scripts/generators/station_unit_splits.py` | Replace one model generator with evidenced separately-accredited units. |
| `scripts/generators/support_scheme_tranches.py` | Represent historical CfD contract tranches explicitly in a PyPSA network. |
| `scripts/generators/thermal_availability.py` | Reproducible non-UC thermal planned and forced availability profiles. |
| `scripts/generators/thermal_ramping.py` | LP-compatible thermal ramp-rate assumptions for future market scenarios. |
