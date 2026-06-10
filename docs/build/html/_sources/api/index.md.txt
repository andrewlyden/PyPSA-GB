# API Reference

Technical documentation for PyPSA-GB modules and functions.

```{toctree}
:maxdepth: 2

core_modules
integration_modules
```

## Overview

PyPSA-GB is organized into modular scripts:

### Core Modules

| Module | Purpose |
|--------|---------|
| `solve_network.py` | Network optimization |
| `scenario_detection.py` | Historical/future routing |
| `carrier_definitions.py` | Technology definitions |
| `logging_config.py` | Logging setup |

### Network Build Modules (`scripts/network_build/`)

| Module | Purpose |
|--------|--------|
| `ETYS_network.py` | ETYS network assembly from preprocessed data |
| `process_ETYS_data.py` | Raw ETYS Excel → intermediate CSVs |
| `ETYS_upgrades.py` | Network upgrade application (circuits, transformers, HVDC) |
| `etys_file_registry.py` | ETYS file/sheet name mapping and constants |
| `build_network.py` | Reduced/Zonal network builders |

### Integration Modules

| Module | Purpose |
|--------|---------|
| `integrate_thermal_generators.py` | Thermal generator integration |
| `integrate_renewable_generators.py` | Renewable generator integration |
| `add_storage.py` | Storage unit integration |
| `spatial_utils.py` | Geographic mapping utilities |
| `add_demand_flexibility.py` | Demand-side flexibility orchestration |
| `heat_pumps.py` | Heat pump disaggregation and flexibility |
| `electric_vehicles.py` | EV smart charging and V2G |
| `event_flex.py` | Event-based demand response |

### Market Modules (`scripts/market/`)

| Module | Purpose |
|--------|---------|
| `solve_wholesale.py` | Stage 1 copperplate wholesale dispatch and wholesale price export |
| `solve_balancing.py` | Stage 2 anchored balancing redispatch with full network constraints |
| `market_utils.py` | Bid/offer prices, ELEXON price loading, redispatch and congestion utilities |
| `elexon_data.py` | Historical ELEXON BMRS data retrieval and aggregation |
| `analyze_market.py` | Market dashboard and summary generation |
| `revenue_tracking.py` | CfD and ROC revenue accounting |
| `validate_bm.py` | Historical ELEXON BM validation |
| `validate_neso_constraints.py` | NESO constraint-cost and boundary-flow validation |

### Utility Modules

| Module | Purpose |
|--------|---------|
| `time_utils.py` | Time series handling |
| `coordinate_utils.py` | Coordinate transformations |
| `validation_utils.py` | Data validation |

## Using the API

### Direct Import

```python
from scripts.scenario_detection import is_historical_scenario, auto_configure_scenario
from scripts.carrier_definitions import get_carrier_definitions
from scripts.spatial_utils import map_sites_to_buses
```

### Within Snakemake

Scripts are typically invoked via Snakemake rules, but can also be run standalone:

```python
# Example: Build an ETYS network manually
import pypsa
from scripts.network_build.ETYS_network import create_network
from scripts.network_build.ETYS_upgrades import apply_etys_network_upgrades

# Or use the Snakemake rules (recommended):
# snakemake resources/network/ETYS_2024_base_network.nc --cores 1
```

## Type Hints

Most functions include type hints:

```python
def is_historical_scenario(scenario: dict) -> bool:
    """Check if scenario models a historical year."""
    ...
```

## Docstring Format

Functions use NumPy-style docstrings:

```python
def map_sites_to_buses(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    method: str = 'nearest'
) -> pd.DataFrame:
    """
    Map generator sites to network buses.
    
    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network with bus coordinates
    sites_df : pd.DataFrame
        Sites with 'lat', 'lon' columns
    method : str, optional
        Mapping method: 'nearest' or 'voronoi'
        
    Returns
    -------
    pd.DataFrame
        Sites with 'bus' column added
        
    Examples
    --------
    >>> sites = map_sites_to_buses(network, wind_sites)
    >>> sites.groupby('bus').capacity_mw.sum()
    """
```

## Error Handling

Modules use consistent error handling:

```python
from scripts.logging_config import setup_logging

logger = setup_logging("my_module")

try:
    result = process_data(input_data)
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise
except FileNotFoundError as e:
    logger.warning(f"Optional file not found: {e}")
    result = default_value
```

## Extending PyPSA-GB

### Adding a New Data Source

1. Create processing script in `scripts/`
2. Add Snakemake rule in `rules/`
3. Document the module here
4. Update configuration options

### Adding a New Technology

1. Add to `carrier_definitions.py`
2. Update integration module
3. Add test cases
4. Document the carrier

See {doc}`../development/contributing` for guidelines.
