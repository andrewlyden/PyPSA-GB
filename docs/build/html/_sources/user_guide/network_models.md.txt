# Network Models

PyPSA-GB supports three network models with different levels of detail.

## Overview

| Model | Buses | Lines | Transformers | Typical Solve Time |
|-------|-------|-------|--------------|-------------------|
| **ETYS** | ~2000 | ~3000 | ~500 | 30-60 min/week |
| **Reduced** | 32 | 64 | 10 | 2-5 min/week |
| **Zonal** | 17 | ~30 | - | 1-2 min/week |

## ETYS Network

The full Electricity Ten Year Statement network from National Grid ESO.

```yaml
network_model: "ETYS"
```

### Characteristics

- Complete 400kV and 275kV transmission network
- All substations and switching points
- Detailed transformer ratings
- Planned reinforcements available

### Use Cases

- Production analysis requiring accurate constraints
- Locational marginal pricing studies
- Network constraint analysis
- Investment planning

### Data Source

Based on ETYS Appendix B data, including:
- Circuit parameters (R, X, B, rating)
- Transformer impedances and tap positions
- Bus coordinates (OSGB36)
- Planned upgrades timeline

## Reduced Network

A 32-bus equivalent capturing major flow paths.

```yaml
network_model: "Reduced"
```

### Topology

```{mermaid}
flowchart TB
    subgraph Scotland
        SHETL["SHETL Zone"]
        SPTL["SPTL Zone"]
    end
    
    subgraph England
        NORTH["North"]
        MIDLANDS["Midlands"]
        LONDON["London"]
        SOUTH["South"]
    end
    
    subgraph Wales
        WALES["Wales"]
    end
    
    SHETL --> SPTL
    SPTL --> NORTH
    NORTH --> MIDLANDS
    MIDLANDS --> LONDON
    MIDLANDS --> WALES
    LONDON --> SOUTH
    WALES --> SOUTH
```

### Use Cases

- Fast scenario testing
- Sensitivity analysis
- Educational purposes
- Debugging workflow issues

### Advantages

- 10-20x faster than ETYS
- Still captures major constraints (Scotland-England, etc.)
- Good enough for many policy questions

### Limitations

- Loses locational detail within zones
- Some constraint interactions missed
- Not suitable for detailed LMP analysis

## Zonal Network

Maximum aggregation to 17 zones.

```yaml
network_model: "Zonal"
```

### Zones

Aligned with DNO regions:
- SSE-N (North Scotland)
- SSE-S (South Scotland)  
- SPEN (SP Networks)
- NPG-NE (Northern Powergrid NE)
- NPG-Y (Northern Powergrid Yorkshire)
- ENWL (Electricity North West)
- WPD-EM (East Midlands)
- WPD-WM (West Midlands)
- WPD-SW (South West)
- WPD-W (Wales)
- UKPN-E (Eastern)
- UKPN-L (London)
- UKPN-SE (South East)
- SSEN-S (Southern)
- Northern Ireland (interconnected)

### Use Cases

- Quick screening studies
- Regional capacity analysis
- Very fast iteration

## Choosing a Network Model

```{mermaid}
flowchart TD
    Q1{Need locational\ndetail?}
    Q1 -->|Yes| Q2{Production\nrun?}
    Q1 -->|No| ZONAL[Zonal]
    
    Q2 -->|Yes| ETYS[ETYS]
    Q2 -->|No| Q3{Time\nconstrained?}
    
    Q3 -->|Yes| REDUCED[Reduced]
    Q3 -->|No| ETYS
```

### Decision Guide

| Your Need | Recommended Model |
|-----------|-------------------|
| Publication-quality results | ETYS |
| Constraint analysis | ETYS |
| Quick testing | Reduced |
| Parameter sweeps | Reduced |
| Regional aggregates | Zonal |
| Educational | Reduced or Zonal |

## Network Clustering

For intermediate detail, cluster the ETYS network:

```yaml
HT35_clustered:
  network_model: "ETYS"
  clustering:
    enabled: true
    n_clusters: 100
```

This reduces ETYS to ~100 buses while preserving:
- Major transmission corridors
- Generation locations (approximately)
- Regional balance

See {doc}`clustering` for details.

## Coordinate Systems

### ETYS Network

Uses **OSGB36 (British National Grid)**:
- X: Easting in meters (0-700,000)
- Y: Northing in meters (0-1,200,000)

### Conversion

If you need WGS84 (lat/lon):

```python
from pyproj import Transformer

# OSGB36 to WGS84
transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(x_osgb, y_osgb)
```

## ETYS Network Upgrades

For future years, ETYS planned upgrades can be applied:

```yaml
HT35_with_upgrades:
  modelled_year: 2035
  network_model: "ETYS"
  etys_upgrades:
    enabled: true
```

This includes:
- Eastern HVDC link
- Scottish reinforcements
- New transformers
- Circuit upratings

See the ETYS Appendix B documentation for details.

## Visualizing Networks

Load and plot any network:

```python
import pypsa
import matplotlib.pyplot as plt

n = pypsa.Network("resources/network/HT35_network.nc")

fig, ax = plt.subplots(figsize=(10, 12))
n.plot(
    ax=ax,
    bus_sizes=n.generators.groupby('bus').p_nom.sum() / 1000,
    line_widths=1,
    title="ETYS Network - Generation Capacity"
)
plt.tight_layout()
plt.savefig("network_map.png", dpi=150)
```

## Performance Comparison

Benchmarks for a typical 1-week solve:

| Model | Build Time | Solve Time | Memory |
|-------|------------|------------|--------|
| ETYS | 5 min | 45 min | 8 GB |
| Reduced | 1 min | 3 min | 2 GB |
| Zonal | 30 sec | 1 min | 1 GB |

*Times vary based on hardware and solver.*
