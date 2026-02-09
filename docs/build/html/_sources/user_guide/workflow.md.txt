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
    
    FES --> RENEW
    FES --> THERMAL
    FES --> STORAGE
    FES --> HYDRO
    DUKES --> THERMAL
    REPD --> RENEW
    
    BASE --> DEMAND
    DEMAND --> RENEW
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
```

## Two Workflows

PyPSA-GB has two separate Snakemake workflows:

### Main Workflow (`Snakefile`)

Runs the core model - assumes weather cutouts exist.

```bash
snakemake resources/network/HT35_solved.nc -j 4
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
| `demand.smk` | Demand extraction, profiles, and flexibility options |
| `renewables.smk` | Renewable generation profiles (wind/solar profiles from ERA5) |
| `generators.smk` | Thermal generator integration (DUKES historical + FES future) |
| `storage.smk` | Storage unit integration (battery, pumped hydro, LAES) |
| `hydrogen.smk` | Hydrogen system (electrolysis, H2 storage, H2 turbines) |
| `interconnectors.smk` | Cross-border connections (GB-Europe, GB-Ireland, etc.) |
| `network_clustering.smk` | Network reduction/clustering (spatial/k-means presets) |
| `FES.smk` | FES data downloading and processing |
| `solve.smk` | Network finalization and optimization solving |
| `analysis.smk` | Results analysis, spatial plots, dashboards, notebooks |

## Key Rules

### `build_network`

Creates the base PyPSA network from ETYS/Reduced/Zonal topology.

**Input**: Network topology files in `data/network/`  
**Output**: `resources/network/{scenario}_network.nc`

### `integrate_renewable_generators`

Adds renewable generators with capacity and profiles.

**Input**: REPD sites (historical) or FES projections (future), demand network  
**Output**: `resources/network/{scenario}_network_demand_renewables.pkl`

### `solve_network`

Runs the linear optimal power flow optimization.

**Input**: Complete unsolved network  
**Output**: `resources/network/{scenario}_solved.nc`

## Running the Workflow

### Basic Commands

```bash
# Run to a specific target
snakemake resources/network/HT35_solved.nc -j 4

# Dry run (show what would execute)
snakemake resources/network/HT35_solved.nc -n -p

# Force re-run of specific rule and downstream
snakemake -R solve_network -j 4

# Run all active scenarios
snakemake -j 4
```

### Parallel Execution

The `-j` flag controls parallelism:

```bash
# Single-threaded (sequential)
snakemake target.nc -j 1

# Use 4 cores
snakemake target.nc -j 4

# Use all available cores
snakemake target.nc -j
```

### Viewing the DAG

Generate a visualization of the workflow:

```bash
snakemake --dag resources/network/HT35_solved.nc | dot -Tpng > dag.png
```

## Intermediate Files

The workflow generates intermediate files that can be inspected. Files use `.pkl` format for fast I/O during development, switching to `.nc` at final assembly:

```
resources/network/
├── HT35_network.nc                                    # Base network
├── HT35_network_demand.pkl                            # + demand
├── HT35_network_demand_renewables.pkl                 # + renewables
├── HT35_network_demand_renewables_thermal_generators.pkl  # + thermal
├── HT35_network_demand_renewables_thermal_generators_storage.pkl  # + storage
├── HT35_network_demand_renewables_thermal_generators_storage_hydrogen.pkl  # + H2
├── HT35_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  # final
├── HT35_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc  # (optional)
└── HT35_solved.nc                                    # Solved (final results)
```

## Logs

Each rule writes logs to `logs/`:

```
logs/
├── network_build/HT35.log
├── solve/HT35.log
├── plotting/HT35_spatial.log
└── ...
```

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
snakemake -R failed_rule_name -j 4
```

### Missing Input

Snakemake will tell you which input is missing. Generate it first:
```bash
snakemake missing_file.nc -j 4
```

### Clean and Restart

Remove outputs for a scenario:
```bash
rm -rf resources/network/HT35*.nc
snakemake resources/network/HT35_solved.nc -j 4
```

## Advanced Usage

### Running Specific Rules

```bash
# Only run data processing (no solving)
snakemake resources/network/HT35_finalized.nc -j 4

# Only run analysis on existing solved network
snakemake resources/analysis/HT35_spatial.html -j 1
```

### Profile Performance

```bash
snakemake --profile slurm  # For cluster execution
```

### Using Conda Environments

```bash
snakemake -j 4 --use-conda  # Use rule-specific envs
```
