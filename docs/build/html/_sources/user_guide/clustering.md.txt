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

## Clustering Methods

### K-Means Clustering

Groups buses based on geographic location:

```yaml
HT35_kmeans:
  network_model: "ETYS"
  clustering:
    enabled: true
    algorithm: "kmeans"
    n_clusters: 100
```

**Pros**: Simple, fast, good spatial distribution  
**Cons**: May split electrically-connected areas

### GSP-Based Clustering

Groups buses by Grid Supply Point:

```yaml
HT35_gsp:
  network_model: "ETYS"
  clustering:
    enabled: true
    algorithm: "gsp"
```

**Pros**: Aligns with FES data granularity  
**Cons**: Fixed number of clusters (~300)

### Regional Clustering

Groups by DNO region:

```yaml
HT35_regional:
  network_model: "ETYS"
  clustering:
    enabled: true
    algorithm: "regional"
    regions: "dno"
```

**Pros**: Meaningful regional analysis  
**Cons**: Only ~15 clusters

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
    enabled: true
    algorithm: "kmeans"
    n_clusters: 100
    
    # Preserve specific buses
    preserve_buses:
      - "BEAU41"    # Beauly (major Scottish node)
      - "HARW41"    # Harwich (interconnector)
    
    # Weight by generation capacity
    weight_by: "generation"
    
    # Aggregate line parameters
    line_aggregation: "series"  # or "parallel"
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

### 2. Generator Aggregation

Generators at clustered buses are summed:

```python
# Original: 3 wind farms at different buses
# Bus A: 500 MW wind
# Bus B: 300 MW wind  
# Bus C: 200 MW wind

# After clustering (A, B, C → Cluster 1):
# Cluster 1: 1000 MW wind
```

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

If clustering itself uses too much memory:

```bash
# Reduce parallel jobs
snakemake resources/network/HT35_clustered.nc -j 1
```
