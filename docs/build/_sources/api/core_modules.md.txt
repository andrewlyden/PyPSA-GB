# Core Modules

Documentation for core PyPSA-GB modules.

## solve_network

Network optimization using PyPSA's LOPF.

### Main Functions

#### `solve_network()`

Solve the optimal power flow for a network.

```python
def solve_network(
    network: pypsa.Network,
    solver_name: str = "gurobi",
    solver_options: dict = None,
    solve_mode: str = "LP"
) -> pypsa.Network:
    """
    Solve network optimal dispatch.
    
    Parameters
    ----------
    network : pypsa.Network
        Complete network with generators, loads, etc.
    solver_name : str
        Solver to use: 'gurobi', 'highs', 'cplex'
    solver_options : dict
        Solver-specific options
    solve_mode : str
        'LP' for linear, 'MILP' for integer
        
    Returns
    -------
    pypsa.Network
        Solved network with dispatch results
    """
```

### Usage Example

```python
import pypsa
from scripts.solve_network import solve_network

# Load prepared network
n = pypsa.Network("resources/network/HT35_finalized.nc")

# Solve
n = solve_network(
    n,
    solver_name="gurobi",
    solver_options={"method": 2, "threads": 4}
)

# Check result
print(f"Objective: £{n.objective/1e9:.2f}B")
print(f"Status: {n.optimization_status}")
```

---

## build_network

Base network construction from topology files.

### Main Functions

#### `build_base_network()`

```python
def build_base_network(
    network_model: str,
    year: int = None,
    logger: logging.Logger = None
) -> pypsa.Network:
    """
    Build a base PyPSA network from topology files.
    
    Parameters
    ----------
    network_model : str
        Network type: 'ETYS', 'Reduced', 'Zonal'
    year : int, optional
        Modelled year (for applying upgrades)
    logger : logging.Logger, optional
        Logger for output
        
    Returns
    -------
    pypsa.Network
        Base network with buses, lines, transformers
    """
```

---

## scenario_detection

Automatic routing based on modelled year.

### Main Functions

#### `is_historical_scenario()`

```python
def is_historical_scenario(scenario: dict) -> bool:
    """
    Check if scenario models a historical year (≤2024).
    
    Parameters
    ----------
    scenario : dict
        Scenario configuration dictionary
        
    Returns
    -------
    bool
        True if modelled_year ≤ 2024
    """
```

#### `auto_configure_scenario()`

```python
def auto_configure_scenario(scenario: dict) -> dict:
    """
    Add automatic configuration based on scenario type.
    
    Adds metadata for data source routing:
    - is_historical: bool
    - thermal_source: 'DUKES' or 'FES'
    - renewable_source: 'REPD' or 'FES'
    
    Parameters
    ----------
    scenario : dict
        Scenario configuration
        
    Returns
    -------
    dict
        Enhanced scenario with routing metadata
    """
```

### Usage Example

```python
from scripts.scenario_detection import is_historical_scenario, auto_configure_scenario

scenario = {
    'modelled_year': 2035,
    'FES_scenario': 'Holistic Transition'
}

# Check type
if is_historical_scenario(scenario):
    print("Using DUKES/REPD data")
else:
    print("Using FES projections")

# Add routing metadata
scenario = auto_configure_scenario(scenario)
print(f"Thermal source: {scenario['thermal_source']}")
```

---

## carrier_definitions

Technology/carrier definitions and attributes.

### Main Functions

#### `get_carrier_definitions()`

```python
def get_carrier_definitions() -> pd.DataFrame:
    """
    Get carrier definitions with colors and emissions.
    
    Returns
    -------
    pd.DataFrame
        Carrier definitions with columns:
        - nice_name: Display name
        - color: Hex color for plotting
        - co2_emissions: tCO2/MWh
        - renewable: bool
    """
```

### Defined Carriers

| Carrier | Nice Name | Color | Renewable |
|---------|-----------|-------|-----------|
| `wind_onshore` | Onshore Wind | `#3B6182` | Yes |
| `wind_offshore` | Offshore Wind | `#6895B8` | Yes |
| `solar_pv` | Solar PV | `#FFCC00` | Yes |
| `nuclear` | Nuclear | `#E63946` | No |
| `CCGT` | CCGT | `#FF7F50` | No |
| `battery` | Battery | `#9370DB` | No |
| `pumped_hydro` | Pumped Hydro | `#1E90FF` | No |

### Usage Example

```python
from scripts.carrier_definitions import get_carrier_definitions

carriers = get_carrier_definitions()

# Get color for plotting
wind_color = carriers.loc['wind_offshore', 'color']

# Get all renewable carriers
renewable_carriers = carriers[carriers.renewable].index.tolist()
```

---

## logging_config

Centralized logging configuration.

### Main Functions

#### `setup_logging()`

```python
def setup_logging(
    log_path: str = None,
    log_level: str = "INFO",
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for a script.
    
    Parameters
    ----------
    log_path : str
        Path to log file (from snakemake.log[0])
    log_level : str
        Level: DEBUG, INFO, WARNING, ERROR
    log_to_console : bool
        Also print to console
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
```

#### `get_log_path()`

```python
def get_log_path(fallback: str = None) -> str:
    """
    Get log path from Snakemake context.
    
    Parameters
    ----------
    fallback : str
        Fallback name if not in Snakemake
        
    Returns
    -------
    str
        Log file path
    """
```

### Usage Pattern

```python
from scripts.logging_config import setup_logging

# In Snakemake script
if __name__ == "__main__":
    snk = globals().get('snakemake')
    log_path = snk.log[0] if snk and snk.log else "my_script"
    logger = setup_logging(log_path)
    
    logger.info("Starting processing...")
```

---

## time_utils

Time series manipulation utilities.

### Main Functions

#### `align_time_series()`

```python
def align_time_series(
    series: pd.Series,
    target_index: pd.DatetimeIndex,
    method: str = 'ffill'
) -> pd.Series:
    """
    Align a time series to a target index.
    
    Parameters
    ----------
    series : pd.Series
        Input time series
    target_index : pd.DatetimeIndex
        Target time index
    method : str
        Interpolation method
        
    Returns
    -------
    pd.Series
        Aligned time series
    """
```

#### `resample_to_hourly()`

```python
def resample_to_hourly(
    df: pd.DataFrame,
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Resample half-hourly data to hourly.
    """
```

---

## validation_utils

Data validation functions.

### Main Functions

#### `validate_network_connectivity()`

```python
def validate_network_connectivity(
    network: pypsa.Network
) -> tuple[bool, list]:
    """
    Check that all buses are connected.
    
    Returns
    -------
    tuple[bool, list]
        (is_connected, isolated_buses)
    """
```

#### `validate_generator_data()`

```python
def validate_generator_data(
    generators: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate and clean generator data.
    
    Checks:
    - Valid coordinates
    - Positive capacity
    - Known carrier types
    
    Returns
    -------
    pd.DataFrame
        Validated generators (invalid rows removed)
    """
```
