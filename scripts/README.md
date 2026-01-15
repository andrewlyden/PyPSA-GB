# Scripts Folder Organization

This folder contains all Python scripts used in the PyPSA-GB workflow, organized by functional area.

## Folder Structure

```
scripts/
├── FES/                    # FES (Future Energy Scenarios) data processing
├── generators/             # Generator integration (thermal, renewable)
├── renewables/             # Renewable profile generation
├── storage/                # Storage integration
├── demand/                 # Demand loading and disaggregation
│   ├── flex/              # Demand flexibility modeling
│   ├── electric_vehicles.py
│   ├── heat_pumps.py
│   └── integrate.py
├── interconnectors/        # Interconnector processing pipeline
├── network_build/          # Network construction (ETYS/Reduced/Zonal)
├── network_clustering/     # Network clustering algorithms
├── solve/                  # Network optimization and solving
├── analysis/               # Results analysis and visualization
├── utilities/              # Shared utility modules
└── archive/                # Historical/deprecated scripts
    ├── narrow_debug/      # Specific debugging scripts
    └── one_time_prep/     # Completed data preparation tasks
```

## Folder Details

### FES/
Scripts for downloading and processing NESO Future Energy Scenarios data:
- `FES_data.py` - Download FES datasets via API
- `extract_FES_prices.py` - Extract fuel and carbon prices

### generators/
Generator integration pipeline:
- DUKES historical thermal data processing
- TEC/REPD deduplication and location mapping
- Renewable generator integration (historical and future)
- Thermal generator integration
- Marginal cost application
- Network CSV export

**Core scripts:**
- `DUKES_generator_data.py` - Process DUKES thermal capacity
- `integrate_renewable_generators.py` - Add renewables to network
- `integrate_thermal_generators.py` - Add thermal generators to network
- `finalize_generator_integration.py` - Final integration step
- `apply_marginal_costs.py` - Apply fuel and carbon costs

### renewables/
Renewable generation profile creation:
- `prepare_renewable_site_data.py` - Prepare site data from REPD
- `map_renewable_profiles.py` - Map Atlite profiles to sites
- `generate_marine_profiles.py` - Generate marine (tidal/wave) profiles
- `generate_hydro_profiles.py` - Generate hydro profiles

### storage/
Storage unit integration:
- `add_storage.py` - Add storage units (batteries, pumped hydro) to network

### demand/
Demand processing and disaggregation:
- `load.py` - Load base electricity demand
- `electric_vehicles.py` - EV demand disaggregation
- `heat_pumps.py` - Heat pump demand disaggregation
- `integrate.py` - Integrate demand components
- **flex/** - Demand flexibility parameters and projections

### interconnectors/
Interconnector data processing pipeline:
- DUKES and NESO register ingestion
- Location enrichment and bus mapping
- Historical flow extraction
- European generation mix integration
- Network integration

**Key scripts:**
- `add_to_network.py` - Add interconnectors to PyPSA network
- `map_to_buses.py` - Map interconnectors to network buses
- `process_european_generation.py` - Calculate European marginal costs

### network_build/
Network topology construction:
- `ETYS_network.py` - Build ETYS high-resolution network
- `ETYS_upgrades.py` - Apply planned network upgrades
- `build_network.py` - Build Reduced/Zonal networks

### network_clustering/
Network spatial aggregation:
- `cluster_network.py` - Cluster network using custom algorithms
- `validate_network.py` - Validate clustered network topology

### solve/
Network optimization:
- `finalize_network.py` - Prepare network for solving
- `solve_network.py` - Optimize network using Gurobi/HiGHS

### analysis/
Post-solve analysis:
- `analyze_solved_network.py` - Comprehensive results analysis
- `generate_analysis_notebook.py` - Generate Jupyter notebooks

### utilities/
Shared utility modules imported by other scripts:
- `carrier_definitions.py` - Technology attributes and colors
- `scenario_detection.py` - Historical vs future scenario routing
- `spatial_utils.py` - Geographic/coordinate utilities
- `logging_config.py` - Centralized logging configuration
- `plotting.py` - Core plotting utilities
- `network_io.py` - Network I/O functions
- `time_utils.py` - Time series utilities
- Plus validation, debugging, and visualization tools

### archive/
Historical scripts no longer in active use:
- **narrow_debug/** - Specific debugging scripts for completed issues
- **one_time_prep/** - Completed data preparation/migration tasks

## Usage Patterns

### Workflow Scripts
Scripts called directly by Snakemake rules (via `script:` directive):
- Located in functional folders (FES/, generators/, etc.)
- Receive inputs via `snakemake.input`, `snakemake.output`, `snakemake.params`
- Use centralized logging from `utilities/logging_config.py`

### Utility Modules
Shared code imported by other scripts:
- Located in `utilities/`
- Import as: `from scripts.utilities.module_name import function`
- Examples: `scenario_detection`, `spatial_utils`, `carrier_definitions`

### Standalone Tools
Development/analysis scripts run independently:
- Validation scripts: `validate_scenarios.py`, `validate_network_generators.py`
- Visualization: `plot_network.py`, `plot_comprehensive_map.py`
- Analysis: `analyze_spatial_balance.py`, `diagnose_infeasibility.py`

## Import Conventions

When scripts import from other scripts:

```python
# Correct - explicit path
from scripts.utilities.logging_config import setup_logging
from scripts.utilities.spatial_utils import map_sites_to_buses
from scripts.utilities.scenario_detection import is_historical_scenario

# Used in Snakemake rule files
from scripts.utilities.scenario_detection import auto_configure_scenario
```

## Development Workflow

1. **Adding New Scripts:**
   - Place in appropriate functional folder
   - Use imports from `utilities/` for shared code
   - Add `__init__.py` if creating new submodule
   - Update corresponding `.smk` rule file

2. **Modifying Imports:**
   - All utility imports go through `scripts.utilities.`
   - Maintain consistent import patterns across scripts
   - Test with `snakemake -n` to verify imports resolve

3. **Debugging:**
   - Use tools in `utilities/` for validation and diagnostics
   - Check `logs/` folder for execution logs
   - Run standalone scripts directly for testing

## Key Files

| File | Purpose |
|------|---------|
| `utilities/scenario_detection.py` | Routes historical vs future data sources |
| `utilities/carrier_definitions.py` | Defines technology colors and attributes |
| `utilities/spatial_utils.py` | Maps sites to network buses |
| `utilities/logging_config.py` | Centralized logging setup |
| `generators/integrate_renewable_generators.py` | Main renewable integration logic |
| `generators/integrate_thermal_generators.py` | Main thermal integration logic |
| `solve/solve_network.py` | Network optimization solver |

## Migration Notes (January 2026)

All scripts were reorganized from flat structure to functional folders:
- Scripts moved from `scripts/*.py` to `scripts/{category}/*.py`
- Imports updated to use `scripts.utilities.` prefix
- Workflow (.smk files) updated with new paths
- No functionality changes - pure reorganization

**Testing:** Run `snakemake -n` to validate all paths resolve correctly.

---

*Last Updated: January 14, 2026*
*Maintained by: PyPSA-GB Team*
