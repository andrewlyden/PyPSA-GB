# Architecture

Overview of PyPSA-GB's software architecture and design decisions.

## High-Level Architecture

```{mermaid}
flowchart TB
    subgraph Config["Configuration Layer"]
        YAML["YAML Config Files"]
        SCENARIOS["Scenario Definitions"]
    end
    
    subgraph Workflow["Workflow Layer"]
        SNAKE["Snakemake"]
        RULES["Rule Definitions"]
    end
    
    subgraph Core["Core Layer"]
        SCRIPTS["Python Scripts"]
        PYPSA["PyPSA"]
    end
    
    subgraph Data["Data Layer"]
        RAW["Raw Data"]
        RESOURCES["Generated Resources"]
    end
    
    YAML --> SNAKE
    SCENARIOS --> SNAKE
    SNAKE --> RULES
    RULES --> SCRIPTS
    SCRIPTS --> PYPSA
    RAW --> SCRIPTS
    SCRIPTS --> RESOURCES
```

## Design Principles

### 1. Declarative Configuration

Users declare *what* they want, not *how* to build it:

```yaml
# User specifies desired outcome
HT35:
  modelled_year: 2035
  network_model: "ETYS"
  FES_scenario: "Holistic Transition"
```

Snakemake determines the execution path automatically.

### 2. Reproducibility

- All inputs are versioned or documented
- Configuration is explicit
- Random seeds are fixed where needed
- Logs capture execution details

### 3. Modularity

Each script does one thing well:

| Script | Single Responsibility |
|--------|----------------------|
| `build_network.py` | Create base network |
| `integrate_thermal_generators.py` | Add thermal capacity |
| `solve_network.py` | Run optimization |

### 4. Data Source Abstraction

The same workflow handles historical and future scenarios:

```{mermaid}
flowchart LR
    SCENARIO["Scenario Config"] --> DETECT["Scenario Detection"]
    DETECT -->|"≤2024"| HIST["Historical Sources"]
    DETECT -->|">2024"| FUTURE["FES Sources"]
    HIST --> INTEGRATE["Integration"]
    FUTURE --> INTEGRATE
```

## Component Architecture

### Configuration System

```
config/
├── config.yaml       # What to run
├── scenarios.yaml    # Scenario definitions
├── defaults.yaml     # Default values
└── config_loader.py  # Python interface
```

**Loading flow**:
1. Load `defaults.yaml`
2. Override with `scenarios.yaml` values
3. Override with `config.yaml` values
4. Override with command-line arguments

### Workflow System

```
Snakefile              # Main entry point
├── rules/
│   ├── network_build.smk
│   ├── generators.smk
│   ├── renewables.smk
│   ├── storage.smk
│   ├── solve.smk
│   └── analysis.smk
```

Rules define:
- Input/output file relationships
- Script to execute
- Parameters from config
- Log file locations

### Script Organization

```
scripts/
├── Core Modules
│   ├── solve_network.py
│   ├── build_network.py
│   └── scenario_detection.py
├── Integration Modules
│   ├── integrate_thermal_generators.py
│   ├── integrate_renewable_generators.py
│   └── add_storage.py
├── Utility Modules
│   ├── spatial_utils.py
│   ├── logging_config.py
│   └── carrier_definitions.py
└── Analysis Modules
    ├── analyze_results.py
    └── plotting.py
```

## Data Flow

### Network Building Pipeline

```{mermaid}
flowchart LR
    subgraph Build
        BASE["Base Network\n(buses, lines)"]
        DEMAND["+ Demand\n(loads)"]
        RENEW["+ Renewables\n(generators)"]
        THERMAL["+ Thermal\n(generators)"]
        STORAGE["+ Storage\n(storage_units)"]
        HYDRO["+ Hydrogen\n(electrolysis, H2)"]
        INTER["+ Interconnectors\n(links)"]
    end
    
    BASE --> DEMAND --> RENEW --> THERMAL --> STORAGE --> HYDRO --> INTER
```

Each step:
1. Loads the previous network state
2. Adds new components
3. Saves updated network

### File Naming Convention

```
{scenario}_network.nc                           # Base
{scenario}_network_demand.pkl                   # + demand
{scenario}_network_demand_renewables.pkl        # + renewables
{scenario}_..._thermal_generators.pkl           # + thermal
{scenario}_..._storage.pkl                      # + storage
{scenario}_..._hydrogen.pkl                     # + hydrogen
{scenario}_..._interconnectors.nc               # + interconnectors
{scenario}_solved.nc                            # Optimized
```

## PyPSA Integration

### Network Structure

PyPSA-GB uses standard PyPSA components:

| Component | Usage |
|-----------|-------|
| `Bus` | Substations/nodes |
| `Line` | Transmission circuits |
| `Transformer` | Voltage transformation |
| `Generator` | Power plants |
| `StorageUnit` | Batteries, pumped hydro |
| `Load` | Demand |
| `Link` | HVDC, interconnectors |

### Component Naming

```python
# Generators: {carrier}_{bus}_{index}
"wind_offshore_BEAU41_0"
"CCGT_PADI41_1"

# Storage: {carrier}_{bus}_{index}
"battery_LOND41_0"

# Lines: {bus0}_{bus1}_{circuit}
"BEAU41_DOUN41_1"
```

## Error Handling Strategy

### Levels of Handling

1. **Validation**: Catch errors before processing
   ```python
   validate_scenario_complete(scenario)
   ```

2. **Graceful Degradation**: Handle missing optional data
   ```python
   try:
       extra_data = load_optional_data()
   except FileNotFoundError:
       logger.warning("Optional data not found, using defaults")
       extra_data = defaults
   ```

3. **Fail Fast**: Stop on critical errors
   ```python
   if not network.buses.any():
       raise ValueError("Network has no buses!")
   ```

### Logging Strategy

```python
# DEBUG: Detailed diagnostic info
logger.debug(f"Processing bus {bus_name} with {n_gens} generators")

# INFO: Progress milestones
logger.info(f"Added {n_generators} generators to network")

# WARNING: Recoverable issues
logger.warning(f"Missing coordinates for {n_missing} sites, skipping")

# ERROR: Problems requiring attention
logger.error(f"Solver returned infeasible status")
```

## Extension Points

### Adding New Technologies

1. Define carrier in `carrier_definitions.py`
2. Add data processing in integration module
3. Update profile generation if needed
4. Add configuration options

### Adding New Data Sources

1. Place raw data in `data/`
2. Create processing script
3. Add Snakemake rule
4. Update scenario detection if needed

### Adding New Analysis

1. Create analysis script
2. Add rule in `analysis.smk`
3. Define output format (HTML, CSV, etc.)

## Performance Considerations

### Memory Management

- Large networks use ~8GB RAM
- Time series stored efficiently (NetCDF)
- Profile caching to avoid recomputation

### Computation Scaling

| Factor | Impact |
|--------|--------|
| More buses | O(n²) for LOPF |
| More timesteps | Linear |
| More generators | Sublinear (grouped) |

### Optimization

- Network clustering for faster solving
- Parallel rule execution
- Profile pre-generation
