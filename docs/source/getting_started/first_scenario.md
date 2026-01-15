# Your First Custom Scenario

This tutorial walks you through creating and running a custom scenario.

## Understanding Scenarios

A **scenario** in PyPSA-GB defines:
- The modelled year (historical or future)
- Which FES pathway to use
- Network model type (ETYS, Reduced, Zonal)
- Solve settings and time period

Scenarios are defined in `config/scenarios.yaml` and activated in `config/config.yaml`.

## Step 1: View Existing Scenarios

Open `config/scenarios.yaml` to see example scenarios:

```yaml
# Example scenario definition
HT35:
  description: "Holistic Transition 2035"
  modelled_year: 2035
  renewables_year: 2019
  demand_year: 2035
  network_model: "ETYS"
  FES_year: 2024
  FES_scenario: "Holistic Transition"
  solve_period:
    enabled: true
    start: "2035-01-01 00:00"
    end: "2035-01-07 23:00"
```

## Step 2: Create Your Scenario

Add a new scenario to `config/scenarios.yaml`:

```yaml
# My custom scenario
MyScenario_2040:
  description: "Custom 2040 analysis - winter week"
  modelled_year: 2040
  renewables_year: 2018          # Weather year for renewables
  demand_year: 2040
  network_model: "Reduced"       # Use 32-bus for faster solving
  FES_year: 2024
  FES_scenario: "Electric Engagement"
  solve_period:
    enabled: true
    start: "2040-01-13 00:00"    # A winter week
    end: "2040-01-19 23:00"
  solver:
    name: "gurobi"
    method: 2                    # Barrier method
```

### Key Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `modelled_year` | Target year to model | 2010-2050 |
| `renewables_year` | Weather data year | 2010-2023 (must have cutout) |
| `network_model` | Network resolution | `ETYS`, `Reduced`, `Zonal` |
| `FES_year` | FES release year | 2021, 2022, 2023, 2024, 2025 |
| `FES_scenario` | FES pathway name | Varies by FES release |
| `solve_period` | Time window to solve | Any date range |

### Available FES Years and Scenarios

FES releases are updated annually by NESO. You can use different FES release years to:
- Compare scenario projections across different releases
- Access historical FES assumptions from earlier releases
- Model the evolution of capacity expectations over time

**Note**: Different newer FES releases may have different scenario names and parameters. Check available data before running.

## Step 3: Activate Your Scenario

Edit `config/config.yaml` to add your scenario:

```yaml
scenarios:
  - MyScenario_2040
  # - HT35  # Comment out others if not needed
```

## Step 4: Validate Configuration

Check your scenario is valid:

```bash
python scripts/validate_scenarios.py
```

This checks:
- All required fields are present
- FES data exists for the specified year
- Weather cutouts are available

## Step 5: Run Your Scenario

```bash
# Dry run first to see what will execute
snakemake resources/network/MyScenario_2040_solved.nc -n -p

# Full run
snakemake resources/network/MyScenario_2040_solved.nc -j 4
```

## Step 6: Analyze Results

Load and analyze your results:

```python
import pypsa
import pandas as pd
import matplotlib.pyplot as plt

# Load network
n = pypsa.Network("resources/network/MyScenario_2040_solved.nc")

# Generation dispatch over time
gen = n.generators_t.p.groupby(n.generators.carrier, axis=1).sum()
gen.plot.area(figsize=(12, 6), title="Generation Dispatch")
plt.ylabel("Power (MW)")
plt.tight_layout()
plt.savefig("my_scenario_dispatch.png")

# Curtailment analysis
renewable_carriers = ['wind_onshore', 'wind_offshore', 'solar_pv']
for carrier in renewable_carriers:
    gens = n.generators[n.generators.carrier == carrier]
    if len(gens) > 0:
        available = n.generators_t.p_max_pu[gens.index].sum(axis=1) * gens.p_nom.sum()
        dispatched = n.generators_t.p[gens.index].sum(axis=1)
        curtailment = (available - dispatched) / available * 100
        print(f"{carrier}: {curtailment.mean():.1f}% average curtailment")
```

## Scenario Variations

### Historical Scenario

For historical validation (uses real data, not FES):

```yaml
Historical_2022:
  description: "Historical year 2022 validation"
  modelled_year: 2022
  renewables_year: 2022
  demand_year: 2022
  network_model: "ETYS"
  # No FES_scenario needed for historical years
```

### High-Resolution Solve

For detailed analysis of specific events:

```yaml
StormAnalysis_2035:
  description: "Storm period analysis"
  modelled_year: 2035
  solve_period:
    enabled: true
    start: "2035-02-01 00:00"
    end: "2035-02-03 23:00"  # 3-day detailed analysis
  timestep_minutes: 30  # Half-hourly resolution
```

### Clustered Network

For faster solving with reasonable accuracy:

```yaml
HT35_clustered:
  description: "Holistic Transition 2035 - clustered"
  modelled_year: 2035
  network_model: "ETYS"
  clustering:
    enabled: true
    n_clusters: 100  # Reduce to 100 buses
```

### Comparing FES Releases

Compare how projections change across FES releases:

```yaml
HT35_FES2024:
  description: "HT scenario 2035 - FES 2024"
  modelled_year: 2035
  FES_year: 2024
  FES_scenario: "Holistic Transition"
  network_model: "Reduced"

HT35_FES2023:
  description: "HT scenario 2035 - FES 2023"
  modelled_year: 2035
  FES_year: 2023                    # Different FES release
  FES_scenario: "Holistic Transition"
  network_model: "Reduced"
```

Run both scenarios to compare results and understand how NESO's capacity expectations evolved between releases.

## Common Issues

### "Cutout not found"

Ensure weather data exists for your `renewables_year`:
```bash
ls resources/atlite/
```

If missing, generate it:
```bash
snakemake -s Snakefile_cutouts resources/atlite/GB_2018.nc -j 2
```

### "Infeasible optimization"

Check that generation capacity meets demand:
```python
n = pypsa.Network("resources/network/MyScenario_unsolved.nc")
print(f"Generation: {n.generators.p_nom.sum()/1000:.1f} GW")
print(f"Peak demand: {n.loads_t.p_set.sum(axis=1).max()/1000:.1f} GW")
```

## Next Steps

- {doc}`../user_guide/configuration` - Full configuration reference
- {doc}`../user_guide/scenarios` - Advanced scenario design
- {doc}`../data_reference/data_sources` - Understanding input data
