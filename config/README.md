# PyPSA-GB Configuration System

This document explains the reorganized configuration structure.

## File Structure

```
config/
├── config.yaml          # What to run + global overrides (EDIT THIS)
├── scenarios.yaml       # Scenario definitions (EDIT THIS)
├── defaults.yaml        # Default values (rarely edit)
├── clustering.yaml      # Clustering presets (rarely edit)
├── config_loader.py     # Python utility for loading configs
```

## Quick Start

### 1. Define a New Scenario

Add to `scenarios.yaml` - **only specify what differs from defaults**:

```yaml
# Historical scenario (≤ 2024)
My_Historical_2020:
  description: "My custom 2020 scenario"
  modelled_year: 2020
  renewables_year: 2020
  demand_year: 2020
  network_model: "Reduced"    # Or "ETYS", "Zonal"

# Future scenario (> 2024) - requires FES config
My_Future_2035:
  description: "My future scenario"
  modelled_year: 2035
  FES_year: 2024
  FES_scenario: "Holistic Transition"
  renewables_year: 2020       # Weather proxy year
  demand_year: 2020           # Demand baseline
```

### 2. Enable Clustering

Reference a preset by name:

```yaml
My_Clustered_Scenario:
  modelled_year: 2020
  renewables_year: 2020
  demand_year: 2020
  clustering: gsp_spatial     # Use preset from clustering.yaml
```

Or define inline:

```yaml
My_Custom_Clustering:
  modelled_year: 2020
  renewables_year: 2020
  demand_year: 2020
  clustering:
    method: kmeans
    n_clusters: 8
    random_state: 42
```

### 3. Run Scenarios

Edit `config.yaml`:

```yaml
run_scenarios:
  - My_Historical_2020
  - My_Future_2035
  # Comment out scenarios you don't want to run
```

### 4. Validate

```bash
python config/config_loader.py --validate
python config/config_loader.py --scenario My_Historical_2020
```

## Configuration Hierarchy

Settings are merged in this order (later overrides earlier):

1. **defaults.yaml** - Base settings for all scenarios
2. **config.yaml** (global overrides) - Override defaults for all scenarios
3. **scenarios.yaml** (scenario-specific) - Override for specific scenario

Example: If `defaults.yaml` has `timestep_minutes: 60` but your scenario specifies `timestep_minutes: 30`, your scenario uses 30.

## What Goes Where

| Setting | File | Example |
|---------|------|---------|
| Solver config | `defaults.yaml` | Gurobi threads, tolerances |
| Timestep resolution | `defaults.yaml` | 30 or 60 minutes |
| VOLL | `defaults.yaml` | £6000/MWh |
| Generator aggregation | `defaults.yaml` | Enabled/disabled |
| Scenarios to run | `config.yaml` | List of scenario IDs |
| Logging config | `config.yaml` | Level, directory |
| Scenario definitions | `scenarios.yaml` | Years, network model |
| Clustering presets | `clustering.yaml` | GSP spatial, k-means, etc. |

## Available Clustering Presets

| Name | Method | Description |
|------|--------|-------------|
| `gsp_spatial` | spatial | Cluster to GSP regions (~30 zones) |
| `admin_regions` | spatial | Cluster to admin regions |
| `kmeans_10` | kmeans | 10 geographical clusters |
| `kmeans_5` | kmeans | 5 geographical clusters |
| `hierarchical_15` | hierarchical | 15 hierarchical clusters |

## FES Scenario Options

| FES Year | Available Scenarios |
|----------|---------------------|
| 2025 | Counterfactual, Electric Engagement, Holistic Transition, Hydrogen Evolution |
| 2024 | Counterfactual, Electric Engagement, Holistic Transition, Hydrogen Evolution |
| 2023 | Falling Short, Consumer Transformation, System Transformation, Leading the Way |
| 2022 | Falling Short, Consumer Transformation, System Transformation, Leading the Way |
| 2021 | Steady Progression, Consumer Transformation, System Transformation, Leading the Way |

## CLI Usage

```bash
# List all scenarios
python config/config_loader.py --list

# Show active scenarios (from run_scenarios)
python config/config_loader.py --active

# Show specific scenario (fully resolved)
python config/config_loader.py --scenario HT35

# Show as JSON (for debugging)
python config/config_loader.py --scenario HT35 --json

# Validate all scenarios
python config/config_loader.py --validate
```

## Using in Python

```python
from config.config_loader import load_config, get_scenario, get_active_scenarios

# Load everything
config = load_config()

# Get specific scenario (fully resolved with defaults)
scenario = get_scenario("HT35")
print(scenario["modelled_year"])  # 2035
print(scenario["solver"]["name"])  # gurobi (inherited from defaults)

# Get all active scenarios
for scenario in get_active_scenarios():
    print(f"Running: {scenario['_scenario_id']}")
```
