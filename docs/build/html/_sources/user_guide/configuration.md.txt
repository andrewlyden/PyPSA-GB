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
run_scenarios:
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
  demand_year: 2020             # Historical demand profile year
  
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

# Component aggregation — runs at finalization for ALL scenarios (disabled by default)
component_aggregation:
  enabled: false
  include_storage_units: false
  include_stores: false

# Transmission constraint settings (ETYS network only)
transmission:
  min_line_s_nom: 0             # 0 = use actual ratings
  min_transformer_s_nom: 0      # 0 = use actual ratings
  capacity_scale: 1.0           # 1.0 = no scaling

# ETYS data source (ETYS network only)
etys:
  year: 2024                    # ETYS publication year (2022, 2023, 2024)

# ETYS network upgrades (ETYS network only)
etys_upgrades:
  enabled: true                 # Apply planned upgrades
  upgrade_year: null            # null = use modelled_year

# Demand flexibility defaults (disabled unless scenario opts in)
demand_flexibility:
  enabled: false
  heat_pumps:
    enabled: false
    mode: "MIXED"
    flex_share: 0.2
  electric_vehicles:
    enabled: false
    tariff: "MIXED"
    flex_share: 0.2
  event_response:
    enabled: false
    mode: "both"
    dsr_capacity_mw: 5000
```

## Key Parameters

### Modelled Year

```yaml
modelled_year: 2035
```

- **Historical** (≤2024): Uses DUKES for thermal, REPD for renewables, ESPENI for demand
- **Future** (>2024): Uses FES projections for capacity and ED1-scaled demand, requires `FES_scenario`

### ETYS Configuration

These settings apply only when `network_model: "ETYS"`.

#### ETYS Publication Year

```yaml
etys:
  year: 2024  # Available: 2022, 2023, 2024
```

Selects which ETYS data edition to use for the base network topology. Each publication year maps to a specific Appendix B Excel file — the mapping is managed by `scripts/network_build/etys_file_registry.py`.

#### ETYS Network Upgrades

```yaml
etys_upgrades:
  enabled: true       # Apply planned network reinforcements
  upgrade_year: null   # null = use modelled_year, or specify e.g. 2030
```

When enabled, applies circuit additions/removals, transformer changes, and HVDC additions from the ETYS upgrade sheets. The `upgrade_year` controls how far into the future to apply upgrades — set to `null` to apply all upgrades through the scenario's `modelled_year`.

#### Transmission Constraints

```yaml
transmission:
  min_line_s_nom: 0         # Minimum line capacity floor (MVA), 0 = use actual
  min_transformer_s_nom: 0  # Minimum transformer capacity floor (MVA)
  capacity_scale: 1.0       # Global scaling factor for all line/transformer capacities
```

These settings can relax transmission constraints for feasibility or sensitivity analysis. For example, `capacity_scale: 1.5` increases all ratings by 50%.

### Network Model

```yaml
network_model: "ETYS"  # or "Reduced" or "Zonal"
```

| Model | Buses | Lines | Use Case |
|-------|-------|-------|----------|
| ETYS | ~2000 | ~3000 | Full transmission detail |
| Reduced | 32 | 64 | Testing, quick analysis |
| Zonal | 17 | ~30 | Aggregate regional flows |

### Demand Data

```yaml
demand_timeseries: "ESPENI"     # temporal profile source
demand_year: 2020               # profile year used for future scaling
```

Historical scenarios (`modelled_year <= 2024`) use ESPENI directly for the annual demand and profile. Future scenarios (`modelled_year > 2024`) use the configured `demand_timeseries` and `demand_year` only for the temporal shape. The annual demand target comes from the FES workbook `ED1` sheet as total consumer electricity demand. The workflow uses FES `Dem_BB003` as the GSP spatial distribution and scales those GSP shares to the ED1 total.

This means future demand includes direct transmission-connected demand in the national total and does not treat raw `Dem_BB003` as total electricity demand.

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

### Market Dispatch

Market dispatch is optional and configured under `market`. It runs after the
final network has been built.

```yaml
market:
  enabled: true
  wholesale_only: false
  wholesale:
    mode: "rolling_day_ahead"    # single | rolling_day_ahead
    window_hours: 24
    carry_soc: true
    transmission_relaxation: 1.0e6
  balancing:
    mode: "rolling"              # full_period | rolling
    window_hours: 1
    bid_offer_source: "derived"  # auto | derived | elexon | csv
    fix_interconnectors: true
```

Use `wholesale_only: true` to stop after the copperplate wholesale solve and
generate a wholesale notebook. Use the full two-stage mode to add balancing
redispatch, congestion, and constraint-cost outputs. For the full explanation,
see {doc}`market`.

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
  # Component aggregation — runs at finalization for ALL scenarios (independent of clustering)
  component_aggregation:
    enabled: true
    include_loads: true            # merge loads per bus
    include_storage_units: true   # merge identical StorageUnits
    include_stores: false         # merge Stores (rarely used)
```

## Renewable Generator Aggregation

Optionally aggregate renewable generators per (bus, carrier) group after integration, reducing model size while conserving capacity and energy:

```yaml
renewable_aggregation:
  enabled: true              # false by default
  carriers:                  # Carriers to aggregate
    - wind_onshore
    - wind_offshore
    - solar_pv
    - large_hydro
    - small_hydro
    - tidal_stream
    - shoreline_wave
    - tidal_lagoon
```

When enabled, all generators sharing the same bus and carrier are merged into one:
- **`p_nom`** is summed exactly (capacity conserved)
- **`p_max_pu`** time series is capacity-weighted averaged (energy conserved)
- Groups with a single generator are untouched
- Non-renewable generators are never affected

This runs inline as part of `integrate_renewable_generators` — no extra pipeline step.

```{tip}
Aggregation is most beneficial on large ETYS networks (2000+ buses) where many GSPs share a bus, reducing generator count by 50-90% with negligible accuracy loss.
```

## Demand Flexibility

Configure demand-side flexibility under `demand_flexibility`:

```yaml
demand_flexibility:
  enabled: true                  # Master switch
  heat_pumps:
    enabled: true
    mode: "MIXED"                # TANK, COSY, or MIXED
    flex_share: 0.2
  electric_vehicles:
    enabled: true
    tariff: "MIXED"              # GO, INT, V2G, or MIXED
    flex_share: 0.2
  event_response:
    enabled: true
    mode: "both"                 # regular, winter, or both
    dsr_capacity_mw: 5000
```

Each flexibility type has its own `enabled` flag and can be configured independently. For the full parameter reference, see {doc}`demand_flexibility`.

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
