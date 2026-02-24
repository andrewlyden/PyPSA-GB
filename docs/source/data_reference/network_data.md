# Network Data

Detailed documentation of the transmission network data.

## ETYS Network Structure

The full ETYS network represents the GB transmission system at high resolution.

### Buses (Substations)

Each bus represents a substation in the transmission network.

**Key attributes**:

| Attribute | Description | Example |
|-----------|-------------|---------|
| `name` | Unique identifier | `BEAU41` |
| `v_nom` | Nominal voltage (kV) | 400, 275, 132 |
| `x` | Easting (OSGB36, m) | 267000 |
| `y` | Northing (OSGB36, m) | 843000 |
| `carrier` | Bus type | `AC` |
| `zone` | DNO region | `SSE-N` |

**Naming convention**:
- First 4 characters: Location code
- Last 2 characters: Voltage code
  - `41` = 400kV
  - `21` = 275kV
  - `11` = 132kV

Example: `BEAU41` = Beauly substation, 400kV

### Lines (Circuits)

Transmission lines connecting substations.

**Key attributes**:

| Attribute | Description | Units |
|-----------|-------------|-------|
| `bus0` | From bus | - |
| `bus1` | To bus | - |
| `s_nom` | Thermal rating | MVA |
| `r` | Resistance | p.u. |
| `x` | Reactance | p.u. |
| `b` | Susceptance | p.u. |
| `length` | Circuit length | km |
| `num_parallel` | Parallel circuits | - |

### Transformers

Connections between voltage levels.

**Key attributes**:

| Attribute | Description | Units |
|-----------|-------------|-------|
| `bus0` | High voltage bus | - |
| `bus1` | Low voltage bus | - |
| `s_nom` | Rating | MVA |
| `x` | Reactance | p.u. |
| `tap_ratio` | Tap position | - |

## Voltage Levels

The GB transmission network has three main voltage levels:

| Voltage | Usage | Operator |
|---------|-------|----------|
| 400kV | Supergrid backbone | NESO/TO |
| 275kV | Regional transmission | NESO/TO |
| 132kV | Sub-transmission (Scotland) | SHE-T, SPT |

```{note}
In England and Wales, 132kV is distribution (DNO-operated).
In Scotland, 132kV is transmission (TO-operated).
```

## Transmission Owners

| Operator | Region | Key Corridors |
|----------|--------|---------------|
| **SHE Transmission** | North Scotland | Beauly-Denny |
| **SP Transmission** | Central Scotland | Hunterston-Torness |
| **NGET** | England & Wales | North-South flows |
| **OFTO** | Offshore | Wind farm connections |

## Key Transmission Boundaries

Critical network boundaries that often constrain flows:

### B6 Boundary (Scotland-England)

The major constraint between Scotland and England.

**Circuits**:
- Harker-Strathaven (400kV)
- Eccles-Stella West (400kV)
- HVDC (Western Link, Eastern Link)

**Typical capacity**: 6-8 GW (depends on year)

### B4 Boundary (Central Scotland)

```{mermaid}
flowchart TB
    NORTH["North Scotland"] 
    CENTRAL["Central Belt"]
    ENGLAND["England"]
    
    NORTH -->|"B4"| CENTRAL
    CENTRAL -->|"B6"| ENGLAND
```

## Network Files

### Location

```
data/network/
├── ETYS/
│   ├── ETYS 2024 Appendix-B V1.xlsx    # ETYS 2024 circuits, transformers, HVDC
│   ├── ETYS Appendix B 2023.xlsx        # ETYS 2023 (alternative)
│   ├── ETYS Appendix B 2022.xlsx        # ETYS 2022 (alternative)
│   ├── GB_network.xlsx                  # Extra WF edges, BMU mappings, demand nodes
│   ├── substation_coordinates.csv       # Manually compiled bus coordinates
│   ├── Regional breakdown of FES24 data.xlsx  # FES regional capacity by GSP
│   └── ...                              # FES regional files for other years
├── reduced_network/
│   ├── buses.csv
│   └── lines.csv
└── zonal/
    ├── buses.csv
    └── links.csv
```

The ETYS publication year is selected via the `etys.year` configuration option (see {doc}`../user_guide/configuration`). The `etys_file_registry.py` module maps each publication year (2022, 2023, 2024) to the correct filenames and Excel sheet names.

### Loading Network Data

```python
import pypsa

# Load a built network
n = pypsa.Network("resources/network/HT35_network.nc")

# Inspect buses
print(n.buses.head())

# Inspect lines
print(n.lines.head())

# Inspect transformers
print(n.transformers.head())
```

## HVDC Links

High-voltage DC interconnections and internal links.

### Internal HVDC

| Link | Route | Capacity | Status |
|------|-------|----------|--------|
| Western Link | Hunterston-Connah's Quay | 2.2 GW | Operational |
| Eastern Link | Torness-Hawthorn Pit | 2.0 GW | 2029 |

### Interconnectors

| Link | Route | Capacity |
|------|-------|----------|
| IFA | GB-France | 2.0 GW |
| IFA2 | GB-France | 1.0 GW |
| BritNed | GB-Netherlands | 1.0 GW |
| Nemo | GB-Belgium | 1.0 GW |
| NSL | GB-Norway | 1.4 GW |
| ElecLink | GB-France | 1.0 GW |
| Viking | GB-Denmark | 1.4 GW |

## Network Upgrades

ETYS includes planned reinforcements through 2031+ (depending on ETYS publication year).

### Applying Upgrades

```yaml
# In scenario configuration
etys_upgrades:
  enabled: true
  upgrade_year: 2035  # Apply all upgrades through 2035 (null = use modelled_year)
```

Upgrades are read from the ETYS Appendix B Excel file selected by `etys.year`. The upgrade sheets are organized by transmission owner:

| Sheet | Content |
|-------|--------|
| B-2-2a | Circuit changes (SHE Transmission) |
| B-2-2b | Circuit changes (SP Transmission) |
| B-2-2c | Circuit changes (NGET) |
| B-2-2d | Circuit changes (OFTO) |
| B-3-2a | Transformer changes (SHE Transmission) |
| B-3-2b | Transformer changes (SP Transmission) |
| B-3-2c | Transformer changes (NGET) |
| B-3-2d | Transformer changes (OFTO) |

### Types of Upgrades

- **Circuit additions**: New transmission lines
- **Circuit removals**: Decommissioned lines
- **Circuit modifications**: Uprating existing circuits
- **Transformer additions**: New transformers between voltage levels
- **Transformer removals**: Decommissioned transformers
- **Transformer modifications**: Re-rated transformers
- **HVDC additions**: New high-voltage DC links (e.g., Eastern HVDC)

### Bus Placement for Upgrades

When upgrades reference buses not in the base network, a multi-pass strategy resolves coordinates:

1. **Strategy 0**: Lookup in `substation_coordinates.csv` (WGS84 → OSGB36 conversion)
2. **Strategy 1**: Copy coordinates from a same-site bus already in the network
3. **Strategy 2**: Copy from a bus added earlier in the same upgrade batch
4. **Strategy 3**: Estimate from a connected bus using circuit length as offset

After applying upgrades, orphan buses (disconnected from the network) are automatically removed.

## Coordinate System

### OSGB36 (British National Grid)

The network uses OSGB36 coordinates:
- **EPSG**: 27700
- **Units**: Meters
- **Origin**: Southwest of Cornwall

| Dimension | Range |
|-----------|-------|
| X (Easting) | 0 - 700,000 |
| Y (Northing) | 0 - 1,200,000 |

### Converting to WGS84

```python
from pyproj import Transformer

# Create transformer
transformer = Transformer.from_crs(
    "EPSG:27700",  # OSGB36
    "EPSG:4326",   # WGS84
    always_xy=True
)

# Convert (x, y) to (lon, lat)
lon, lat = transformer.transform(267000, 843000)
print(f"Beauly: {lat:.4f}°N, {lon:.4f}°W")
```

## Visualizing the Network

### Basic Plot

```python
import pypsa
import matplotlib.pyplot as plt

n = pypsa.Network("resources/network/HT35_network.nc")

fig, ax = plt.subplots(figsize=(8, 10))
n.plot(ax=ax, line_widths=0.5)
plt.title("GB Transmission Network")
plt.tight_layout()
plt.savefig("gb_network.png", dpi=150)
```

### With Generation

```python
# Size buses by generation capacity
bus_gen = n.generators.groupby('bus').p_nom.sum()

n.plot(
    bus_sizes=bus_gen / 500,  # Scale factor
    bus_colors=bus_gen,
    line_widths=1,
    line_colors='gray'
)
```

### Interactive Map

```python
import folium
from pyproj import Transformer

# Convert coordinates and create map
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

m = folium.Map(location=[54.5, -2], zoom_start=6)

for idx, bus in n.buses.iterrows():
    lon, lat = transformer.transform(bus.x, bus.y)
    folium.CircleMarker(
        [lat, lon],
        radius=3,
        popup=idx
    ).add_to(m)

m.save("network_map.html")
```

## Data Quality

### Validation Checks

The workflow validates network data:

1. **Connectivity**: All buses reachable
2. **Parameters**: R, X, B in reasonable ranges
3. **Coordinates**: All buses have valid locations
4. **Ratings**: All lines have positive s_nom

### Known Issues

| Issue | Description | Handling |
|-------|-------------|----------|
| Missing coordinates | Some ETYS buses lack x,y | Multi-tier resolution: GSP mapping → substation lookup → prefix fallback → distance-weighted guessing with land boundary validation |
| Zero impedance | Some short lines | Minimum impedance applied |
| Islanded buses | Disconnected substations | Connected via small impedance |
| Offshore buses | Wind farm connection buses outside GB landmass | Identified via OFTO data; `is_offshore` flag propagated through the pipeline |
| Land boundary | Guessed coordinates may fall in the sea | Coordinates validated against GSP region GeoJSON boundaries and moved to nearest land point |
