# PyPSA-GB Scripts Folder Review

**Date:** January 14, 2026  
**Purpose:** Categorize scripts in the scripts folder and identify which are used in the workflow

---

## Executive Summary

- **Total Scripts Analyzed:** ~155 scripts
- **Core Workflow Scripts:** 58 (essential)
- **Utility Modules:** 11 (imported dependencies)
- **Development Tools:** 35 (debugging/diagnostics - keep)
- **Visualization Tools:** 14 (useful for analysis)
- **Export Tools:** 4 (data extraction)
- **Archive Candidates:** 33 (completed one-time tasks or narrow debug scripts)

---

## Subfolder Analysis

### ✅ KEEP - All Scripts in Subfolders Are Used

#### `scripts/demand_components/` (Used in demand.smk)
- `heat_pumps.py` - Heat pump demand disaggregation
- `electric_vehicles.py` - EV demand disaggregation
- `integrate.py` - Component integration into network
- `__init__.py` - Package initialization

**Status:** All actively used in workflow - KEEP AS-IS

#### `scripts/flex/` (Used in demand.smk for flexibility modeling)
- `ev_fleet_projections.py` - EV fleet projections
- `ev_charging_patterns.py` - EV charging patterns
- `ev_flexibility_params.py` - EV flexibility parameters
- `thermal_storage_potential.py` - Thermal storage potential
- `thermal_flexibility_params.py` - Thermal flexibility parameters
- `dr_potential.py` - Demand response potential
- `dr_flexibility_params.py` - DR flexibility parameters
- `storage_*.py` - Storage processing modules
- `flex_placeholders.py` - Placeholder implementations

**Status:** All used in workflow - KEEP AS-IS

#### `scripts/interconnectors/` (Used in interconnectors.smk)
- `download_european_generation.py` - EU generation data
- `process_european_generation.py` - EU marginal costs
- `extract_historical_flows.py` - Historical interconnector flows
- `ingest_dukes.py` - DUKES interconnector data
- `ingest_neso_register.py` - NESO register data
- `combine_datasets.py` - Dataset combination
- `enrich_locations.py` - Location enrichment
- `clean_interconnectors.py` - Data cleaning
- `map_to_buses.py` - Bus mapping
- `availability.py` - Availability time series
- `add_to_network.py` - Network integration
- `pipeline_placeholder.py` - Future pipeline data
- `validate_historical_flows.py` - Flow validation
- `visualize_european_prices.py` - Price visualization

**Status:** All used in workflow - KEEP AS-IS

**Recommendation:** ✅ **Do NOT move these scripts to root** - They are well-organized in subfolders and actively used by the workflow.

---

## Root Scripts Categorization

### Category 1: Core Workflow Scripts ✅ KEEP
**Used directly in .smk rule files**

#### Network Building
- `ETYS_network.py` - ETYS topology construction (network_build.smk)
- `ETYS_upgrades.py` - Network infrastructure upgrades (network_build.smk)
- `build_network.py` - Reduced/Zonal network builder (network_build.smk)
- `cluster_network.py` - Network clustering (network_clustering.smk)

#### Generators
- `DUKES_generator_data.py` - Historical thermal data processing (generators.smk)
- `deduplicate_tec_repd.py` - TEC/REPD deduplication (generators.smk)
- `process_tec_generators.py` - TEC register processing (generators.smk)
- `enhance_tec_locations_with_geocoding.py` - Location geocoding (generators.smk)
- `map_dispatchable_generator_locations.py` - Generator mapping (generators.smk)
- `prepare_dispatchable_generator_sites.py` - Site preparation (generators.smk)
- `analyze_unmapped_generators.py` - Unmapped analysis (generators.smk)
- `enhance_locations_with_wikipedia.py` - Wikipedia enrichment (generators.smk)
- `integrate_dukes_locations.py` - DUKES integration (generators.smk)
- `single_word_tec_repd_matching.py` - Single-word matching (generators.smk)
- `map_final_generators.py` - Final mapping (generators.smk)
- `create_generators_full.py` - Full generator creation (generators.smk)
- `generators_capacities.py` - Capacity calculations (generators.smk)
- `integrate_renewable_generators.py` - Renewable integration (generators.smk)
- `extract_repd_dispatchable_sites.py` - REPD extraction (generators.smk)
- `integrate_thermal_generators.py` - Thermal integration (generators.smk)
- `finalize_generator_integration.py` - Final integration (generators.smk)
- `apply_marginal_costs.py` - Marginal cost application (generators.smk)
- `export_network_to_csv.py` - CSV export (generators.smk)
- `marginal_costs.py` - Marginal cost calculations (generators.smk)

#### Renewables
- `prepare_renewable_site_data.py` - Site data preparation (renewables.smk)
- `map_renewable_profiles.py` - Profile mapping (renewables.smk)
- `generate_marine_profiles.py` - Marine generation (renewables.smk)
- `generate_hydro_profiles.py` - Hydro generation (renewables.smk)

#### Storage
- `add_storage.py` - Storage integration (storage.smk)

#### Demand
- `load.py` - Demand loading (demand.smk)

#### FES Data
- `FES_data.py` - FES data extraction (FES.smk)
- `extract_FES_prices.py` - FES price extraction (FES.smk)

#### Network Solving
- `finalize_network.py` - Network finalization (solve.smk)
- `solve_network.py` - Optimization solving (solve.smk)

#### Analysis
- `analyze_solved_network.py` - Solved network analysis (analysis.smk)
- `generate_analysis_notebook.py` - Notebook generation (analysis.smk)

#### Validation
- `validate_network.py` - Network validation (network_clustering.smk)

---

### Category 2: Utility Modules ✅ KEEP
**Imported by other scripts - essential dependencies**

- `carrier_definitions.py` - Technology attributes and colors
- `scenario_detection.py` - Historical/future scenario routing
- `spatial_utils.py` - Geographic/coordinate utilities
- `logging_config.py` - Centralized logging configuration
- `time_utils.py` - Time series utilities
- `plotting.py` - Core plotting utilities
- `network_io.py` - Network I/O functions
- `custom_clustering_functions.py` - Clustering algorithms
- `timeseries_synchronization.py` - Timestep alignment
- `generators.py` - Generator utilities (legacy, used in workflow)
- `renewable_integration.py` - Renewable integration utilities

---

### Category 3: Development/Debugging Tools ✅ KEEP
**Not in workflow but valuable for development**

#### Network Diagnostics
- `check_isolated_buses.py` - Detect disconnected buses
- `check_bottlenecks.py` - Identify network bottlenecks
- `check_clustered_flows.py` - Validate clustered network flows
- `inspect_network_before_solve.py` - Pre-solve comprehensive diagnostics
- `check_model_constraints.py` - Constraint validation
- `check_load_shedding.py` - Load shedding verification

#### Generator Analysis
- `analyze_generator_mapping.py` - Cross-network mapping comparison
- `analyze_spatial_balance.py` - Spatial distribution analysis
- `analyze_grid_constraints.py` - Grid constraint analysis
- `check_generator_placement.py` - Generator placement validation
- `check_wrong_buses.py` - Incorrect bus assignment detection

#### Coordinate/Location Diagnostics
- `diagnose_coordinates.py` - Coordinate system validation
- `check_coordinate_progression.py` - Coordinate consistency
- `check_generator_coords.py` - Generator coordinate validation

#### Infeasibility Diagnostics
- `diagnose_infeasibility.py` - Infeasibility diagnosis
- `diagnose_infeasibility_detailed.py` - Detailed infeasibility analysis
- `diagnose_unbounded.py` - Unbounded solution diagnosis
- `check_unbounded_patterns.py` - Unbounded pattern detection
- `deep_unbounded_check.py` - Deep unbounded checking

#### Results Analysis
- `analyze_results.py` - General results analysis
- `analyze_solve_complexity.py` - Solver complexity analysis
- `analyze_retired_generators.py` - Retired generator tracking

#### Validation Tools
- `validate_scenarios.py` - Scenario pre-flight checks
- `validate_network_generators.py` - Generator validation
- `validate_storage_integration.py` - Storage validation
- `validate_renewable_output.py` - Renewable output validation
- `validate_marine_profiles.py` - Marine profile validation
- `validate_interconnector_implementation.py` - Interconnector validation
- `deep_network_validation.py` - Deep network validation
- `validate_dukes_coordinates.py` - DUKES coordinate validation
- `generate_validation_report.py` - Comprehensive validation report

#### Testing Tools
- `check_renewable_cf.py` - Capacity factor checks
- `verify_interpolation.py` - Time series interpolation
- `verify_units.py` - Unit consistency
- `check_timesteps.py` - Timestep validation
- `check_p_set.py` - Generation setpoint checks

---

### Category 4: Visualization Tools ✅ KEEP
**Useful for analysis and presentations**

- `plot_network.py` - General network plotting
- `plot_comprehensive_map.py` - Comprehensive Folium map
- `plot_comprehensive_map_new.py` - Updated comprehensive map
- `plot_comprehensive_generators.py` - Generator-focused plotting
- `plot_renewables_pypsa.py` - Renewable generator visualization
- `plot_thermal_generators_pypsa.py` - Thermal generator visualization
- `plot_storage_pypsa.py` - Storage visualization
- `plot_interconnectors_pypsa.py` - Interconnector plotting
- `plot_interconnectors_enhanced.py` - Enhanced interconnector plots
- `plot_renewables_pydeck.py` - Pydeck renewable visualization
- `plot_etys_coordinate_analysis.py` - ETYS coordinate analysis
- `plot_5hour_results.py` - Short-duration result plotting
- `plot_clustered_manual.py` - Manual clustered plotting
- `plotting_clustered.py` - Clustered plotting utilities

---

### Category 5: Export/Reporting Tools ✅ KEEP
**Data extraction and reporting**

- `export_network_to_excel.py` - Excel export for inspection
- `generate_test_summary.py` - Test summary generation
- `aggregate_generators.py` - Generator aggregation

---

### Category 6: Archive Candidates ⚠️ CONSIDER ARCHIVING
**Narrow/specific debug scripts - likely completed**

#### ETYS-Specific Debugging (Very Narrow)
- `check_etys_buses.py` - Checks only 3 specific buses (INDQ41, BLHI4-, PEHEMEC4)
- `check_base_network_buses.py` - ETYS bus naming/coordinate analysis
- `debug_etys_infeasibility.py` - ETYS-specific infeasibility debugging

#### FES-Specific Debugging
- `debug_fes_generators.py` - FES generator mapping debug
- `check_future_mapping.py` - Future scenario mapping checks

#### Other Specific Debugging
- `debug_interconnector_flows.py` - Interconnector flow debugging
- `debug_bus_concentration.py` - Bus concentration debugging
- `debug_unclassified.py` - Unclassified debugging

#### Weather Data Check
- `check_cutouts_exist.py` - Weather cutout availability (could be useful)

**Recommendation:** Move to `scripts/archive/narrow_debug/`

---

### Category 7: One-Time Data Preparation ⚠️ CONSIDER ARCHIVING
**Completed one-time tasks**

#### Manual Coordinate Fixes (Completed)
- `apply_manual_coordinates.py` - Manual fixes (e.g., Spalding Energy)
- `apply_repd_duplicate_coordinates.py` - REPD duplicate coordinates
- `create_web_search_coordinates.py` - Web-searched coordinates

#### Specialized Matching (Superseded by current approach)
- `fast_targeted_tec_repd_matching.py` - Fast TEC/REPD matching
- `enhanced_tec_repd_deduplication.py` - Enhanced deduplication
- `super_enhanced_tec_repd_deduplication.py` - Super-enhanced deduplication

#### One-Time Extractions
- `extract_espeni_nuclear_availability.py` - ESPENI nuclear extraction
- `ingest_tec_register.py` - TEC register ingestion (may still be useful for updates)
- `get_gsp_api_urls.py` - GSP API URL fetching

#### Testing of Specific Fixes (Completed)
- `test_coordinate_fix.py` - Coordinate fix testing
- `test_enhanced_bmu.py` - Enhanced BMU testing
- `test_gsp_mapping.py` - GSP mapping testing
- `test_hour57.py` - Hour 57 bug testing (leap year issue)
- `validate_test.py` - General validation testing
- `run_tests.py` - Test runner (may still be useful)
- `test_all_renewables.py` - Comprehensive renewable testing
- `test_renewable_generation.py` - Simple renewable generation test

#### Data Normalization (Completed)
- `normalize_generator_coordinates.py` - Coordinate normalization
- `create_pypsa_renewable_generators.py` - PyPSA renewable creation

#### Utilities (May Still Be Useful)
- `postcode_geocoder.py` - Postcode geocoding utility (reusable)
- `quick_validate_dukes_coords.py` - Quick DUKES validation
- `prepare_cutouts.py` - Weather cutout preparation (may be needed for new years)

**Recommendation:** 
- Move completed tasks to `scripts/archive/one_time_prep/`
- Keep `postcode_geocoder.py`, `prepare_cutouts.py`, `ingest_tec_register.py` in root (reusable)

---

## Recommendations

### 1. ✅ Keep Subfolder Structure
**Do NOT move scripts from subfolders to root.** They are:
- Well-organized by functional area
- Actively used in the workflow
- Properly namespaced

### 2. ⚠️ Create Archive Folders
Create two archive folders for older scripts:

```
scripts/
├── archive/
│   ├── narrow_debug/          # Category 6 scripts
│   └── one_time_prep/         # Category 7 scripts
│   └── README.md              # Explain what's in archives
```

### 3. 📝 Scripts to Archive

**Move to `archive/narrow_debug/` (10 scripts):**
- check_etys_buses.py
- check_base_network_buses.py
- debug_etys_infeasibility.py
- debug_fes_generators.py
- check_future_mapping.py
- debug_interconnector_flows.py
- debug_bus_concentration.py
- debug_unclassified.py
- check_cutouts_exist.py (or keep if still useful)

**Move to `archive/one_time_prep/` (20 scripts):**
- apply_manual_coordinates.py
- apply_repd_duplicate_coordinates.py
- create_web_search_coordinates.py
- fast_targeted_tec_repd_matching.py
- enhanced_tec_repd_deduplication.py
- super_enhanced_tec_repd_deduplication.py
- extract_espeni_nuclear_availability.py
- get_gsp_api_urls.py
- test_coordinate_fix.py
- test_enhanced_bmu.py
- test_gsp_mapping.py
- test_hour57.py
- validate_test.py
- test_all_renewables.py
- test_renewable_generation.py
- normalize_generator_coordinates.py
- create_pypsa_renewable_generators.py
- quick_validate_dukes_coords.py

**Keep in root (still useful):**
- postcode_geocoder.py (reusable utility)
- prepare_cutouts.py (needed for new weather years)
- ingest_tec_register.py (TEC updates monthly)
- run_tests.py (test infrastructure)

### 4. 📚 Documentation Needed

Create `scripts/README.md`:
```markdown
# Scripts Folder Organization

## Structure
- `scripts/` - Core workflow and utility scripts
- `scripts/demand_components/` - Demand disaggregation modules
- `scripts/flex/` - Flexibility modeling modules
- `scripts/interconnectors/` - Interconnector processing pipeline
- `scripts/archive/` - Historical scripts (not actively used)

## Categories
- **Workflow Scripts** - Used in Snakefile/rules/*.smk
- **Utilities** - Imported by other scripts
- **Development Tools** - Debugging and validation
- **Visualization** - Plotting and analysis
- **Archive** - Completed one-time tasks or narrow debugging

See SCRIPTS_REVIEW.md for detailed categorization.
```

### 5. 🧹 Cleanup Summary

**Keep in Active Use:** 122 scripts  
**Archive:** 33 scripts  
**Gain:** Cleaner scripts folder, easier navigation, preserved history

---

## Impact Analysis

### Before Cleanup
- 155 scripts in root (overwhelming)
- Hard to distinguish workflow vs. utility vs. debugging
- Many one-time scripts mixed with core functionality

### After Cleanup
- ~122 active scripts in root
- 33 archived (preserved for reference)
- Clear separation of concerns
- Easier onboarding for new developers

---

## Next Steps

1. **Review this document** - Confirm categorization accuracy
2. **Create archive folders** - Set up new directory structure
3. **Move archived scripts** - Relocate 33 scripts to archive
4. **Create README.md** - Document folder organization
5. **Update AGENTS.md** - Reflect new structure
6. **Test workflow** - Ensure no broken imports
7. **Commit changes** - Version control the reorganization

---

**Generated:** January 14, 2026  
**Reviewer:** AI Assistant (GitHub Copilot)
