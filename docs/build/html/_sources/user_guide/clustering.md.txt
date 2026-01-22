# Network Clustering

Network clustering reduces the ETYS network to fewer buses while preserving key characteristics.

## Why Cluster?

The full ETYS network (~2000 buses) provides maximum detail but:
- Takes 30-60 minutes to solve per week
- Requires significant memory
- Slows sensitivity analysis

Clustering to 50-200 buses can:
- Reduce solve time by 5-10x
- Maintain most accuracy for aggregate results
- Enable faster iteration

## Clustering Methods (configurable)

These map to presets in `config/clustering.yaml` (resolved by `clustering: <preset>` in `scenarios.yaml`), or can be inlined with `clustering: { method: <...> }`.

### K-Means (`method: kmeans`)

```yaml
HT35_kmeans:
  network_model: "ETYS"
  clustering:
    method: "kmeans"
    n_clusters: 100
```

**Pros**: Simple, fast, good spatial distribution  
**Cons**: May split electrically-connected areas

### GSP Spatial (`preset: gsp_spatial`, `method: spatial`)

```yaml
HT35_gsp:
  network_model: "ETYS"
  clustering: gsp_spatial    # uses method: spatial + GSP boundaries
```

**Pros**: Aligns with FES GSP granularity  
**Cons**: Fixed cluster count (~300)

### Busmap / Regional (`method: busmap`)

```yaml
HT35_regional:
  network_model: "ETYS"
  clustering:
    method: "busmap"
    busmap_source: "data/zone/zonal_bus_mapping.csv"
```

**Pros**: Meaningful regional analysis, explicit control  
**Cons**: Fixed mapping; you must maintain the CSV

## Configuration

### Basic Clustering

```yaml
MyScenario:
  clustering:
    enabled: true
    n_clusters: 100
```

### Advanced Options

```yaml
MyScenario:
  clustering:
    method: "kmeans"
    n_clusters: 100
    # Optional: post-clustering component aggregation
    aggregate_components:
      enabled: true
      include_storage_units: true   # merge identical storage units
      include_stores: false         # merge Store components
```

## Running Clustered Scenarios

```bash
# Build and solve clustered network
snakemake resources/network/HT35_clustered_100_solved.nc -j 4
```

The clustering happens automatically in the workflow.

## How Clustering Works

### 1. Bus Aggregation

Buses are grouped into clusters based on the algorithm:

```{mermaid}
flowchart LR
    subgraph Original["Original (2000 buses)"]
        B1[Bus 1]
        B2[Bus 2]
        B3[Bus 3]
        B4[Bus 4]
    end
    
    subgraph Clustered["Clustered (100 buses)"]
        C1[Cluster 1]
        C2[Cluster 2]
    end
    
    B1 --> C1
    B2 --> C1
    B3 --> C2
    B4 --> C2
```

### 2. Generator & Storage Aggregation (optional)

If `aggregate_components.enabled: true`, identical generators and/or storage at each clustered bus are merged (capacities summed) when they share the same attributes and time series. Dispatch is unchanged, but asset count drops sharply (useful for memory).

### 3. Line Aggregation

Lines between clusters are combined:

```python
# Original: 3 lines between areas A and B
# Line 1: 1000 MW
# Line 2: 1000 MW
# Line 3: 500 MW

# After clustering:
# Equivalent line: 2500 MW (parallel)
```

### 4. Demand Aggregation

Loads are summed at cluster buses.

## Accuracy Considerations

### What's Preserved

- Total generation capacity by technology
- Total demand
- Major transmission constraints
- Regional balance

### What's Lost

- Intra-cluster congestion
- Precise locational prices
- Some line flow patterns
- Local voltage issues

### Accuracy vs Speed Trade-off

| Clusters | Accuracy | Solve Speed |
|----------|----------|-------------|
| 500 | ~98% | 3x faster |
| 200 | ~95% | 5x faster |
| 100 | ~90% | 10x faster |
| 50 | ~80% | 15x faster |

*Accuracy measured as correlation with full ETYS results for system cost.*

## Validation

Compare clustered results to full ETYS:

```python
import pypsa

# Full ETYS solve
n_full = pypsa.Network("resources/network/HT35_solved.nc")

# Clustered solve  
n_clust = pypsa.Network("resources/network/HT35_clustered_100_solved.nc")

# Compare key metrics
print(f"Full ETYS cost: £{n_full.objective/1e9:.2f}B")
print(f"Clustered cost: £{n_clust.objective/1e9:.2f}B")
print(f"Difference: {(n_clust.objective - n_full.objective)/n_full.objective*100:.1f}%")

# Generation mix comparison
full_gen = n_full.generators.groupby('carrier').p_nom.sum()
clust_gen = n_clust.generators.groupby('carrier').p_nom.sum()
print("\nCapacity preserved:", (clust_gen / full_gen).mean())
```

## Use Cases

### Sensitivity Analysis

Run many scenarios quickly:

```yaml
# Test 10 different configurations
scenarios:
  - HT35_clustered_100_base
  - HT35_clustered_100_high_wind
  - HT35_clustered_100_low_nuclear
  # ...
```

### Screening Studies

Identify interesting cases to run with full detail:

```bash
# Quick clustered runs
snakemake resources/network/HT35_clustered_100_solved.nc -j 4
snakemake resources/network/HT50_clustered_100_solved.nc -j 4

# Then full detail for most interesting
snakemake resources/network/HT35_solved.nc -j 4
```

### Educational

Faster iteration for learning:

```yaml
Tutorial_scenario:
  network_model: "ETYS"
  clustering:
    n_clusters: 50
  solve_period:
    start: "2035-01-01"
    end: "2035-01-02"  # Single day
```

## Troubleshooting

### Infeasible After Clustering

If the clustered network won't solve:

1. Check that major lines aren't eliminated
2. Increase `n_clusters` to preserve more detail
3. Use `preserve_buses` for critical nodes

### Results Don't Match Full ETYS

Expected differences are 5-15%. If larger:

1. Check cluster count (try higher)
2. Verify generators mapped correctly
3. Check line capacity aggregation

### Memory Issues

If clustering or solving uses too much memory:

```bash
# Reduce parallel jobs
snakemake resources/network/HT35_clustered.nc -j 1
```

- Enable `aggregate_components` to reduce asset count after clustering.
- Shorten `solve_period` or increase `timestep_minutes` to reduce time steps during solve.
