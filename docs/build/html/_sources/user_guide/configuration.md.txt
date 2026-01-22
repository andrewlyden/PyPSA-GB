# Configuration Reference

PyPSA-GB uses YAML configuration files to control all aspects of the model.

## Configuration Files

```
config/
├── config.yaml       # Active scenarios and global overrides
├── scenarios.yaml    # Scenario definitions
├── defaults.yaml     # Default values for all parameters
└── clustering.yaml   # Network clustering presets
```

## config.yaml

The main configuration file. Specifies which scenarios to run and any global overrides.

```yaml
# Active scenarios to run
scenarios:
  - HT35
  - HT50

# Global overrides (apply to all scenarios)
solver:
  name: "gurobi"
  
# Logging level
logging_level: "INFO"
```

## scenarios.yaml

Defines individual scenarios. Each scenario inherits from `defaults.yaml` and can override any parameter.

### Full Scenario Example

```yaml
HT35:
  # Descriptive information
  description: "Holistic Transition 2035 - Full ETYS network"
  
  # Core temporal settings
  modelled_year: 2035           # Target year for the model
  renewables_year: 2019         # Weather year for renewable profiles
  demand_year: 2035             # Demand profile year
  
  # Network configuration
  network_model: "ETYS"         # ETYS | Reduced | Zonal
  
  # FES data source (for future years)
  FES_year: 2024                # FES publication year
  FES_scenario: "Holistic Transition"
  
  # Time period to solve
  solve_period:
    enabled: true
    start: "2035-01-01 00:00"
    end: "2035-01-07 23:00"
  
  # Solver configuration
  solver:
    name: "gurobi"
    method: 2                   # 0=primal, 1=dual, 2=barrier
    crossover: 0                # 0=off, -1=auto
    threads: 4
  
  # Resolution
  timestep_minutes: 60          # 30 or 60
```

### Minimal Scenario

Inherits most settings from defaults:

```yaml
MyQuickTest:
  description: "Quick test run"
  modelled_year: 2030
  network_model: "Reduced"      # Fast 32-bus model
  FES_scenario: "Holistic Transition"
```

## defaults.yaml

Default values used when not specified in a scenario.

```yaml
# Temporal defaults
timestep_minutes: 60
modelled_year: 2035
renewables_year: 2019

# Network defaults
network_model: "ETYS"

# FES defaults
FES_year: 2024
FES_scenario: "Holistic Transition"

# Economic parameters
voll: 6000.0                    # Value of Lost Load (£/MWh)
discount_rate: 0.07

# Solver defaults
solver:
  name: "gurobi"
  method: 2
  crossover: 0
  threads: 0                    # 0 = auto
  
solve_mode: "LP"                # LP | MILP

# Solve period (if enabled)
solve_period:
  enabled: false
  start: null
  end: null

# Clustering defaults (disabled unless scenario opts in)
clustering:
  enabled: false
  aggregate_components:
    enabled: false
    include_storage_units: false
    include_stores: false
```

## Key Parameters

### Modelled Year

```yaml
modelled_year: 2035
```

- **Historical** (≤2024): Uses DUKES for thermal, REPD for renewables, ESPENI for demand
- **Future** (>2024): Uses FES projections for capacity, requires `FES_scenario`

### Network Model

```yaml
network_model: "ETYS"  # or "Reduced" or "Zonal"
```

| Model | Buses | Lines | Use Case |
|-------|-------|-------|----------|
| ETYS | ~2000 | ~3000 | Full transmission detail |
| Reduced | 32 | 64 | Testing, quick analysis |
| Zonal | 17 | ~30 | Aggregate regional flows |

### FES Scenario

```yaml
FES_year: 2024
FES_scenario: "Holistic Transition"
```

Available FES 2024 scenarios:
- `"Holistic Transition"` - Balanced approach
- `"Electric Engagement"` - High electrification
- `"Hydrogen Evolution"` - Hydrogen-focused
- `"Falling Short"` - Slower progress

### Solve Period

Limit the time window to reduce computation:

```yaml
solve_period:
  enabled: true
  start: "2035-01-01 00:00"
  end: "2035-01-07 23:00"
```

```{tip}
For full-year runs, set `enabled: false` or omit `solve_period`.
```

### Solver Configuration

```yaml
solver:
  name: "gurobi"        # gurobi | highs
  method: 2             # 0=primal, 1=dual, 2=barrier (recommended)
  crossover: 0          # 0=off (faster), -1=auto (more accurate)
  threads: 4            # 0=auto
  BarConvTol: 1e-6      # Barrier convergence tolerance
  FeasibilityTol: 1e-6  # Constraint feasibility
```

For HiGHS (open-source alternative):
```yaml
solver:
  name: "highs"
```

### Timestep Resolution

```yaml
timestep_minutes: 60  # or 30
```

Half-hourly (30 min) doubles computation time but captures faster dynamics.

## Clustering Configuration

In `config/clustering.yaml`:

```yaml
# Clustering presets (example)
presets:
  gsp_spatial:
    method: spatial
    boundaries_path: "data/network/GSP/GSP_regions_27700_20250109.geojson"
    cluster_column: "GSPs"
  kmeans_10:
    method: kmeans
    n_clusters: 10
```

Use in scenario:
```yaml
HT35_clustered:
  clustering:
    preset: "gsp_spatial"   # or inline: { method: kmeans, n_clusters: 10 }

    # Optional: aggregate identical components after clustering
    aggregate_components:
      enabled: true
      include_storage_units: true   # merge identical StorageUnits
      include_stores: false         # merge Stores (rarely used)
```

## Environment Variables

Some settings can be overridden via environment variables:

```bash
export PYPSA_GB_SOLVER=highs
export PYPSA_GB_THREADS=8
```

## Validation

Validate your configuration:

```bash
python scripts/validate_scenarios.py
```

This checks:
- Required fields present
- FES data available
- Cutouts exist for weather year
- No conflicting settings

## Example Configurations

### Development/Testing

```yaml
DevTest:
  modelled_year: 2030
  network_model: "Reduced"
  solve_period:
    enabled: true
    start: "2030-01-01 00:00"
    end: "2030-01-01 23:00"  # Single day
  solver:
    name: "highs"
```

### Production Run

```yaml
Production_HT35:
  modelled_year: 2035
  network_model: "ETYS"
  FES_scenario: "Holistic Transition"
  solve_period:
    enabled: false  # Full year
  solver:
    name: "gurobi"
    method: 2
    threads: 0
```

### Sensitivity Analysis

```yaml
HT35_high_voll:
  modelled_year: 2035
  voll: 10000.0  # Higher value of lost load

HT35_low_voll:
  modelled_year: 2035
  voll: 3000.0
```
