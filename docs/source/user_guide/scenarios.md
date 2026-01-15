# Scenario Design

This guide covers how to design effective scenarios for your analysis.

## Scenario Types

### Historical Scenarios

Model past years using actual data (not FES projections).

```yaml
Historical_2022:
  description: "Validation against 2022 outturn"
  modelled_year: 2022
  renewables_year: 2022    # Must match for consistency
  demand_year: 2022
  network_model: "ETYS"
  # No FES_scenario needed
```

**Use cases**:
- Model validation against historical outturn
- Understanding past system behavior
- Baseline for comparison

**Data sources**:
- Thermal generation: DUKES statistics
- Renewables: REPD (Renewable Energy Planning Database)
- Demand: ESPENI profiles

### Future Scenarios

Project future years using NESO Future Energy Scenarios.

```yaml
Future_2035:
  description: "2035 under Holistic Transition"
  modelled_year: 2035
  renewables_year: 2019    # Historical weather pattern
  demand_year: 2035
  network_model: "ETYS"
  FES_year: 2024
  FES_scenario: "Holistic Transition"
```

**Data sources**:
- Thermal/Renewables/Storage: FES projections
- Demand: FES annual totals with historical profiles
- Network: ETYS planned upgrades (optional)

## FES Pathway Selection

### Holistic Transition

Balanced approach with multiple low-carbon vectors.

```yaml
FES_scenario: "Holistic Transition"
```

- Moderate electrification
- Green hydrogen development
- Biomass and CCS
- Good for "central" scenarios

### Electric Engagement

Maximum electrification pathway.

```yaml
FES_scenario: "Electric Engagement"
```

- Very high electricity demand
- Extensive heat pump deployment
- Limited hydrogen role
- Tests grid capacity limits

### Hydrogen Evolution

Hydrogen-centric decarbonization.

```yaml
FES_scenario: "Hydrogen Evolution"
```

- Blue hydrogen from natural gas + CCS
- Hydrogen for heat and transport
- Lower electricity demand growth
- Industrial hydrogen clusters

### Falling Short

Slower progress scenario.

```yaml
FES_scenario: "Falling Short"
```

- Misses 2050 net zero
- Tests system resilience
- Useful for stress testing

## Weather Year Selection

The `renewables_year` determines the weather pattern for renewable generation.

```yaml
renewables_year: 2019  # Choose based on your analysis goals
```

### High Wind Years
- 2015, 2019: Above-average wind output

### Low Wind Years  
- 2010, 2021: Below-average wind

### Extreme Events
- 2018: "Beast from the East" cold snap
- Check ERA5 data for specific events

```{tip}
For robust analysis, run multiple weather years and compare results.
```

## Time Period Selection

### Full Year

For complete annual analysis:

```yaml
solve_period:
  enabled: false
```

### Representative Weeks

For faster analysis:

```yaml
solve_period:
  enabled: true
  start: "2035-01-13 00:00"   # Winter week
  end: "2035-01-19 23:00"
```

### Typical Periods

| Period | Dates | Characteristics |
|--------|-------|-----------------|
| Winter peak | Jan 10-17 | High demand, low solar |
| Summer low | Aug 1-8 | Low demand, high solar |
| Autumn wind | Nov 1-8 | High wind, moderate demand |
| Spring transition | Apr 15-22 | Variable conditions |

### Stress Testing

For extreme conditions:

```yaml
# Find the peak demand day in your weather year
solve_period:
  enabled: true
  start: "2035-01-15 00:00"
  end: "2035-01-22 23:00"
```

## Sensitivity Analysis

### Capacity Sensitivities

Create variants with modified assumptions:

```yaml
HT35_baseline:
  modelled_year: 2035
  FES_scenario: "Holistic Transition"

HT35_more_wind:
  modelled_year: 2035
  FES_scenario: "Holistic Transition"
  capacity_scaling:
    wind_offshore: 1.2  # 20% more offshore wind

HT35_less_nuclear:
  modelled_year: 2035
  FES_scenario: "Holistic Transition"
  capacity_scaling:
    nuclear: 0.5  # 50% less nuclear
```

### Economic Sensitivities

```yaml
HT35_high_gas:
  fuel_prices:
    natural_gas: 150  # £/MWh (vs default ~80)

HT35_high_carbon:
  carbon_price: 200  # £/tCO2 (vs default ~100)
```

## Multi-Scenario Comparison

### Defining a Suite

```yaml
# In config/scenarios.yaml
HT35_suite:
  - HT35_baseline
  - HT35_high_wind
  - HT35_low_nuclear
  - HT35_high_demand
```

### Running All

```yaml
# In config/config.yaml
scenarios:
  - HT35_baseline
  - HT35_high_wind
  - HT35_low_nuclear
```

```bash
snakemake -j 4  # Runs all active scenarios
```

### Comparing Results

```python
import pypsa
import pandas as pd

scenarios = ['HT35_baseline', 'HT35_high_wind', 'HT35_low_nuclear']
results = []

for s in scenarios:
    n = pypsa.Network(f"resources/network/{s}_solved.nc")
    results.append({
        'scenario': s,
        'system_cost': n.objective / 1e9,
        'curtailment': (n.generators_t.p.sum().sum() / 
                       (n.generators_t.p_max_pu * n.generators.p_nom).sum().sum()),
        'emissions': n.generators.eval('p_nom * carrier_emissions').sum()
    })

df = pd.DataFrame(results)
print(df)
```

## Naming Conventions

Use consistent naming for clarity:

```
{FES_Pathway}{Year}_{Network}_{Variant}
```

Examples:
- `HT35` - Holistic Transition 2035, ETYS network
- `HT35_reduced` - Same but reduced network
- `EE40_high_wind` - Electric Engagement 2040, high wind variant
- `Historical_2022` - Historical validation

## Scenario Checklist

Before running a new scenario, verify:

- [ ] `modelled_year` makes sense for your analysis
- [ ] `renewables_year` has available cutout data
- [ ] `FES_scenario` is valid (for future years)
- [ ] `solve_period` is appropriate for your question
- [ ] Solver settings are configured
- [ ] Run `python scripts/validate_scenarios.py`
