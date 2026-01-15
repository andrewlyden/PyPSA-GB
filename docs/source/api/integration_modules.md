# Integration Modules

Documentation for data integration modules that add components to the network.

## integrate_thermal_generators

Integrates thermal generators from DUKES (historical) or FES (future).

### Main Functions

#### `integrate_thermal_generators()`

```python
def integrate_thermal_generators(
    network: pypsa.Network,
    scenario: dict,
    logger: logging.Logger = None
) -> pypsa.Network:
    """
    Add thermal generators to network.
    
    Automatically routes to DUKES or FES based on modelled_year.
    
    Parameters
    ----------
    network : pypsa.Network
        Network to add generators to
    scenario : dict
        Scenario configuration
    logger : logging.Logger
        Logger for output
        
    Returns
    -------
    pypsa.Network
        Network with thermal generators added
    """
```

#### `load_dukes_generators()`

```python
def load_dukes_generators(
    dukes_file: str,
    year: int = None
) -> pd.DataFrame:
    """
    Load thermal generators from DUKES Excel file.
    
    Parameters
    ----------
    dukes_file : str
        Path to DUKES 5.11 Excel file
    year : int, optional
        Filter to stations operational in year
        
    Returns
    -------
    pd.DataFrame
        Thermal generators with columns:
        - name, carrier, p_nom, efficiency
        - lat, lon (coordinates)
    """
```

#### `load_fes_thermal()`

```python
def load_fes_thermal(
    fes_file: str,
    year: int,
    scenario: str
) -> pd.DataFrame:
    """
    Load thermal capacity from FES projections.
    
    Parameters
    ----------
    fes_file : str
        Path to processed FES CSV
    year : int
        Modelled year
    scenario : str
        FES scenario name
        
    Returns
    -------
    pd.DataFrame
        Thermal generators from FES
    """
```

### Carrier Mapping

| DUKES Fuel | PyPSA Carrier |
|------------|---------------|
| Combined Cycle Gas Turbine | `CCGT` |
| Open Cycle Gas Turbine | `OCGT` |
| Coal | `coal` |
| Nuclear | `nuclear` |
| Biomass | `biomass` |
| Oil | `oil` |

---

## integrate_renewable_generators

Integrates renewable generators with capacity and generation profiles.

### Main Functions

#### `integrate_renewable_generators()`

```python
def integrate_renewable_generators(
    network: pypsa.Network,
    scenario: dict,
    profiles_path: str,
    logger: logging.Logger = None
) -> pypsa.Network:
    """
    Add renewable generators with time series.
    
    Parameters
    ----------
    network : pypsa.Network
        Network to add generators to
    scenario : dict
        Scenario configuration
    profiles_path : str
        Path to renewable profiles CSV
    logger : logging.Logger
        Logger for output
        
    Returns
    -------
    pypsa.Network
        Network with renewables added
    """
```

#### `load_fes_renewable_generators()`

```python
def load_fes_renewable_generators(
    fes_file: str,
    year: int,
    scenario: str,
    network: pypsa.Network,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load renewable capacity from FES with spatial distribution.
    
    Handles:
    - GSP-level capacity (direct mapping)
    - Direct-connected capacity (REPD-distributed)
    - Unconnected capacity (geographic distribution)
    
    Parameters
    ----------
    fes_file : str
        Processed FES data path
    year : int
        Target year
    scenario : str
        FES scenario
    network : pypsa.Network
        Network for bus mapping
    logger : logging.Logger
        Logger
        
    Returns
    -------
    pd.DataFrame
        Renewable sites with bus assignments
    """
```

### Technology Mapping

| FES Technology | FES Detail | PyPSA Carrier |
|----------------|------------|---------------|
| Wind | Offshore Wind | `wind_offshore` |
| Wind | Onshore Wind | `wind_onshore` |
| Solar | Large Scale Solar | `solar_pv` |
| Solar | Small Scale Solar | `solar_pv` |
| Hydro | Large Hydro | `large_hydro` |
| Marine | Tidal Stream | `marine` |

---

## add_storage

Integrates storage units (batteries, pumped hydro).

### Main Functions

#### `add_storage()`

```python
def add_storage(
    network: pypsa.Network,
    scenario: dict,
    logger: logging.Logger = None
) -> pypsa.Network:
    """
    Add storage units to network.
    
    Parameters
    ----------
    network : pypsa.Network
        Network to modify
    scenario : dict
        Scenario configuration
    logger : logging.Logger
        Logger
        
    Returns
    -------
    pypsa.Network
        Network with storage added
    """
```

#### `load_fes_storage_data()`

```python
def load_fes_storage_data(
    fes_file: str,
    year: int,
    scenario: str,
    network: pypsa.Network,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load storage from FES with proper distribution.
    
    Handles:
    - GSP-connected storage (FES location)
    - Direct-connected (REPD distribution)
    - Pumped hydro (existing infrastructure)
    
    Returns
    -------
    pd.DataFrame
        Storage units with columns:
        - name, carrier, bus
        - p_nom (MW), max_hours
        - efficiency_store, efficiency_dispatch
    """
```

### Storage Types

| Carrier | Typical max_hours | Efficiency |
|---------|-------------------|------------|
| `battery` | 2-4 hours | 90% round-trip |
| `pumped_hydro` | 6-10 hours | 75% round-trip |

---

## spatial_utils

Geographic mapping utilities for assigning components to buses.

### Main Functions

#### `map_sites_to_buses()`

```python
def map_sites_to_buses(
    network: pypsa.Network,
    sites_df: pd.DataFrame,
    method: str = 'nearest',
    preserve_existing: bool = True
) -> pd.DataFrame:
    """
    Map sites with coordinates to network buses.
    
    Parameters
    ----------
    network : pypsa.Network
        Network with bus coordinates
    sites_df : pd.DataFrame
        Sites with 'lat', 'lon' or 'x', 'y' columns
    method : str
        'nearest' - nearest bus
        'voronoi' - Voronoi region
    preserve_existing : bool
        Keep pre-assigned 'bus' values
        
    Returns
    -------
    pd.DataFrame
        Sites with 'bus' column
    """
```

#### `detect_coordinate_system()`

```python
def detect_coordinate_system(
    coords: pd.DataFrame
) -> str:
    """
    Detect if coordinates are WGS84 or OSGB36.
    
    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame with x/y or lon/lat columns
        
    Returns
    -------
    str
        'WGS84' or 'OSGB36'
    """
```

#### `convert_coordinates()`

```python
def convert_coordinates(
    x: float, y: float,
    from_crs: str,
    to_crs: str
) -> tuple[float, float]:
    """
    Convert coordinates between systems.
    
    Parameters
    ----------
    x, y : float
        Input coordinates
    from_crs : str
        Source CRS ('WGS84' or 'OSGB36')
    to_crs : str
        Target CRS
        
    Returns
    -------
    tuple[float, float]
        Converted (x, y) coordinates
    """
```

### Coordinate Systems

| System | EPSG | X Range | Y Range | Units |
|--------|------|---------|---------|-------|
| WGS84 | 4326 | -180 to 180 | -90 to 90 | Degrees |
| OSGB36 | 27700 | 0-700k | 0-1200k | Meters |

### Usage Example

```python
from scripts.spatial_utils import map_sites_to_buses, convert_coordinates

# Map wind farm sites to buses
wind_farms = pd.DataFrame({
    'name': ['Farm1', 'Farm2'],
    'lat': [55.5, 56.2],
    'lon': [-3.1, -4.5],
    'capacity_mw': [100, 200]
})

wind_farms = map_sites_to_buses(network, wind_farms)
print(wind_farms[['name', 'bus', 'capacity_mw']])
```

---

## add_interconnectors

Adds cross-border interconnector links.

### Main Functions

#### `add_interconnectors()`

```python
def add_interconnectors(
    network: pypsa.Network,
    scenario: dict,
    logger: logging.Logger = None
) -> pypsa.Network:
    """
    Add interconnector links to network.
    
    Creates Link components for:
    - GB-France (IFA, IFA2, ElecLink)
    - GB-Netherlands (BritNed)
    - GB-Belgium (Nemo)
    - GB-Norway (NSL)
    - GB-Ireland (EWIC, Moyle)
    
    Parameters
    ----------
    network : pypsa.Network
        Network to modify
    scenario : dict
        Scenario config (for future capacity)
    logger : logging.Logger
        Logger
        
    Returns
    -------
    pypsa.Network
        Network with interconnectors
    """
```

### Interconnector Data

```python
# resources/interconnectors/interconnectors.csv
name,bus0,bus1,p_nom,efficiency
IFA,SELL41,FR,2000,0.97
IFA2,BRAW21,FR,1000,0.97
BritNed,GRAI41,NL,1000,0.97
...
```
