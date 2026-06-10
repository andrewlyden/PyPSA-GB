# Snakemake Workflow

PyPSA-GB uses [Snakemake](https://snakemake.github.io/) to orchestrate the model building and solving process. This page explains the workflow structure.

## Workflow Overview

The main workflow is defined in `Snakefile` and consists of modular rules in `rules/*.smk`:

```{mermaid}
flowchart TB
    subgraph Data["Data Processing"]
        FES["Process FES Data"]
        DUKES["Process DUKES"]
        REPD["Process REPD"]
    end
    
    subgraph Network["Network Assembly"]
        BASE["Build Base Network"]
        DEMAND["Add Demand"]
        FLEX["Add Demand Flexibility"]
        RENEW["Add Renewables"]
        THERMAL["Add Thermal Generators"]
        STORAGE["Add Storage"]
        HYDRO["Add Hydrogen System"]
        INTER["Add Interconnectors"]
    end
    
    subgraph Cluster["Optional Transforms"]
        CLUST["Cluster Network"]
        UPGRADES["Apply ETYS Upgrades"]
    end
    
    subgraph Solve["Optimization"]
        FINALIZE["Finalize Network"]
        SOLVE["Solve Network"]
        ANALYZE["Analyze Results"]
    end

    subgraph Market["Optional Market Dispatch"]
        WHOLESALE["Solve Wholesale Market"]
        BM["Solve Balancing Mechanism"]
        MARKET_ANALYZE["Analyze Market Results"]
    end
    
    FES --> RENEW
    FES --> THERMAL
    FES --> STORAGE
    FES --> HYDRO
    DUKES --> THERMAL
    REPD --> RENEW
    
    BASE --> DEMAND
    DEMAND --> FLEX
    FLEX --> RENEW
    RENEW --> THERMAL
    THERMAL --> STORAGE
    STORAGE --> HYDRO
    HYDRO --> INTER
    INTER --> CLUST
    INTER --> UPGRADES
    CLUST --> FINALIZE
    UPGRADES --> FINALIZE
    INTER --> FINALIZE
    FINALIZE --> SOLVE
    SOLVE --> ANALYZE
    FINALIZE --> WHOLESALE
    WHOLESALE --> BM
    BM --> MARKET_ANALYZE
```

## Two Workflows

PyPSA-GB has two separate Snakemake workflows:

### Main Workflow (`Snakefile`)

Runs the core model - assumes weather cutouts exist.

```bash
snakemake --cores 4
```

### Cutouts Workflow (`Snakefile_cutouts`)

Acquires weather cutouts using a **tiered strategy** to minimize wait times:

1. Check `data/atlite/cutouts/` for cached files
2. Download from Zenodo (~5-10 minutes for years 2010-2024)  
3. Generate from ERA5 via atlite (~2-4 hours per year)

```bash
# Configure years in config/cutouts_config.yaml, then run:
snakemake -s Snakefile_cutouts --cores 1
```

**Key Features**:
- Automatic Zenodo download for years 2010-2024 (no CDS credentials needed)
- MD5 checksum verification
- Falls back to ERA5 API for other years or if Zenodo fails

See {doc}`../getting_started/installation` for detailed setup instructions.

## Rule Files

Rules are organized by function in `rules/`:

| File | Purpose |
|------|---------|
| `network_build.smk` | Base network construction (ETYS/Reduced/Zonal) |
| `demand.smk` | Demand extraction, profiles, disaggregation, and demand-side flexibility integration |
| `renewables.smk` | Renewable generation profiles (wind/solar profiles from ERA5) |
| `generators.smk` | Thermal generator integration (DUKES historical + FES future) |
| `storage.smk` | Storage unit integration (battery, pumped hydro, LAES) |
| `hydrogen.smk` | Hydrogen system (electrolysis, H2 storage, H2 turbines) |
| `interconnectors.smk` | Cross-border connections (GB-Europe, GB-Ireland, etc.) |
| `network_clustering.smk` | Network reduction/clustering (spatial/k-means presets) |
| `FES.smk` | FES data downloading and processing |
| `solve.smk` | Network finalization and optimization solving |
| `market.smk` | Optional wholesale market, balancing mechanism, validation, and market notebooks |
| `analysis.smk` | Results analysis, spatial plots, dashboards, notebooks |

## Key Rules

### `build_network`

Creates the base PyPSA network from ETYS/Reduced/Zonal topology.

For the **ETYS** network model, this is a **two-stage pipeline**:

1. **`process_ETYS_data`** — Parses the raw ETYS Appendix B Excel file into standardized intermediate CSVs:
   - `resources/network/ETYS/ETYS_{year}_components.csv` (circuits, transformers, HVDC)
   - `resources/network/ETYS/ETYS_{year}_buses.csv` (bus names, coordinates, voltage levels)
   - Uses `GB_network.xlsx` for offshore wind farm edges and demand node mappings
   - Uses `substation_coordinates.csv` for bus coordinate lookups
   - Script: `scripts/network_build/process_ETYS_data.py`

2. **`build_ETYS_base_network`** — Assembles the intermediate CSVs into a PyPSA network:
   - Resolves missing bus coordinates using a multi-tier strategy (GSP mapping → substation lookup → prefix fallback → distance-weighted guessing)
   - Validates bus locations against GB land boundaries (GSP region GeoJSON)
   - Identifies offshore buses (OFTO wind farm connection nodes)
   - Applies ETYS network upgrades if `etys_upgrades.enabled: true`
   - Script: `scripts/network_build/ETYS_network.py`

For **Reduced** and **Zonal** networks, a single rule reads the CSV files directly.

**Input**: Network topology files in `data/network/`  
**Output**: `resources/network/{model}_base_network.nc` (then copied to `{scenario}_network.nc`)

### `integrate_renewable_generators`

Adds renewable generators with capacity and profiles.

**Input**: REPD sites (historical) or FES projections (future), demand network  
**Output**: `resources/network/{scenario}_network_demand_renewables.pkl`

### `solve_network`

Runs the linear optimal power flow optimization.

**Input**: Complete unsolved network  
**Output**: `resources/network/{scenario}_solved.nc`

### `solve_wholesale_market`

Runs the optional Stage 1 market solve when `market.enabled: true`. It relaxes
line and transformer capacities to create a copperplate wholesale schedule and
uniform demand-bus wholesale price.

**Input**: Finalized network (`resources/network/{scenario}.nc`)  
**Output**: `resources/market/{scenario}_wholesale.nc` and wholesale CSVs

### `solve_balancing_mechanism`

Runs the optional Stage 2 market solve unless `market.wholesale_only: true`. It
restores network constraints and redispatches from the wholesale position using
configured bid and offer prices.

**Input**: Finalized network and wholesale outputs  
**Output**: `resources/market/{scenario}_balancing.nc`, redispatch, congestion, and price comparison CSVs

See {doc}`market` for configuration and output details.

## Running the Workflow

### Basic Commands

```bash
# Run active scenarios from config/config.yaml
snakemake --cores 4

# Dry run (show what would execute)
snakemake --cores 4 -n -p

# Run one scenario without editing config/config.yaml
snakemake --cores 4 --config scenario=HT35

# Force re-run of specific rule and downstream
snakemake -R solve_network --cores 4
```

### Parallel Execution

The `--cores` flag controls parallelism:

```bash
# Single-threaded (sequential)
snakemake --cores 1

# Use 4 cores
snakemake --cores 4

# Use all available cores
snakemake --cores all
```

### Viewing the DAG

Generate a visualization of the workflow:

```bash
snakemake --dag | dot -Tpng > dag.png
```

## Intermediate Files

The workflow generates intermediate files that can be inspected. Files use `.pkl` format for fast I/O during development, switching to `.nc` at final assembly:

```
resources/network/
├── ETYS/
│   ├── ETYS_2024_components.csv                    # Stage 1: parsed circuits/transformers/HVDC
│   └── ETYS_2024_buses.csv                        # Stage 1: bus names and coordinates
├── ETYS_2024_base_network.nc                       # Stage 2: assembled PyPSA network
├── HT35_network.nc                                 # Base network (scenario copy)
├── HT35_network_demand.pkl                         # + demand
├── HT35_network_demand_renewables.pkl                 # + renewables
├── HT35_network_demand_renewables_thermal_generators.pkl  # + thermal
├── HT35_network_demand_renewables_thermal_generators_storage.pkl  # + storage
├── HT35_network_demand_renewables_thermal_generators_storage_hydrogen.pkl  # + H2
├── HT35_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  # final
├── HT35_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  # (optional)
└── HT35_solved.nc                                    # Solved (final results)
```

Market-enabled scenarios also add outputs under `resources/market/`, including
the wholesale network, wholesale price CSV, balancing network, redispatch
summary, congestion diagnostics, and price comparison files. Market dashboards
and notebooks are written under `resources/analysis/`.

## Logs

Each rule writes logs to `logs/`:

```
logs/
├── network_build/HT35.log
├── solve/HT35.log
├── plotting/HT35_spatial.log
└── ...
```

Market rules write logs under `logs/market/`, for example
`logs/market/solve_wholesale_{scenario}.log` and
`logs/market/solve_balancing_{scenario}.log`.

## Configuration Flow

```{mermaid}
flowchart LR
    CONFIG["config/config.yaml"] --> SCENARIOS["config/scenarios.yaml"]
    DEFAULTS["config/defaults.yaml"] --> SCENARIOS
    SCENARIOS --> SNAKEMAKE["Snakemake"]
    SNAKEMAKE --> RULES["rules/*.smk"]
    RULES --> SCRIPTS["scripts/*.py"]
```

## Troubleshooting

### Rule Failed

Check the log file:
```bash
cat logs/solve/HT35.log
```

### Re-run Failed Rule

Force re-execution:
```bash
snakemake -R failed_rule_name --cores 4
```

### Missing Input

Snakemake will tell you which input is missing. Generate it first:
```bash
snakemake --cores 4
```

### Clean and Restart

Remove outputs for a scenario:
```bash
rm -rf resources/network/HT35*.nc
snakemake --cores 4 --config scenario=HT35
```

## Advanced Usage

### Running Specific Rules

```bash
# Only run data processing (no solving)
snakemake resources/network/HT35_finalized.nc --cores 4

# Only run analysis on existing solved network
snakemake resources/analysis/HT35_spatial.html --cores 1
```

### Profile Performance

```bash
snakemake --profile slurm  # For cluster execution
```

### Using Conda Environments

```bash
snakemake --cores 4 --use-conda  # Use rule-specific envs
```
