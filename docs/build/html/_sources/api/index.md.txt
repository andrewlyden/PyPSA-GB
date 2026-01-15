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
| `build_network.py` | Base network construction |
| `scenario_detection.py` | Historical/future routing |
| `carrier_definitions.py` | Technology definitions |
| `logging_config.py` | Logging setup |

### Integration Modules

| Module | Purpose |
|--------|---------|
| `integrate_thermal_generators.py` | Thermal generator integration |
| `integrate_renewable_generators.py` | Renewable generator integration |
| `add_storage.py` | Storage unit integration |
| `spatial_utils.py` | Geographic mapping utilities |

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
# Example: Run network building manually
import pypsa
from scripts.build_network import build_base_network

network = build_base_network(
    network_model="ETYS",
    year=2035
)
```

## Function Documentation

Detailed documentation is available in the subpages:

- {doc}`core_modules` - Core functionality
- {doc}`integration_modules` - Data integration modules

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
