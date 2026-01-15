# Quickstart: Run Your First Model

This guide gets you running a PyPSA-GB scenario.

## Prerequisites

Ensure you've completed the {doc}`installation` steps and have the `pypsa-gb` environment activated:

```bash
conda activate pypsa-gb
```

## Step 1: Check Available Scenarios

PyPSA-GB comes with pre-configured scenarios. View them:

```bash
# List active scenarios
python -c "
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
print('Active scenarios:', config.get('scenarios', []))
"
```

Common scenarios include:

| Scenario | Description |
|----------|-------------|
| `HT35` | Holistic Transition 2035 (ETYS network) |
| `HT50` | Holistic Transition 2050 (ETYS network) |
| `Historical_2020` | Historical year 2020 |
| `HT35_reduced` | Holistic Transition 2035 (32-bus reduced) |

## Step 2: Run a Scenario

Run a single scenario using Snakemake:

```bash
# Run the HT35 scenario (2035 Holistic Transition)
snakemake resources/network/HT35_solved.nc -j 4
```

```{note}
The `-j 4` flag runs up to 4 jobs in parallel. Adjust based on your CPU cores.
```

### What Happens

The workflow executes these steps automatically:

```{mermaid}
flowchart LR
    A[Build Network] --> B[Add Demand]
    B --> C[Add Renewables]
    C --> D[Add Thermal]
    D --> E[Add Storage]
    E --> F[Add Hydrogen]
    F --> G[Add Interconnectors]
    G --> H[Solve Network]
```

## Step 3: View Results

Once complete, load and inspect the solved network:

```python
import pypsa

# Load the solved network
n = pypsa.Network("resources/network/HT35_solved.nc")

# Basic statistics
print(f"Buses: {len(n.buses)}")
print(f"Generators: {len(n.generators)}")
print(f"Storage units: {len(n.storage_units)}")
print(f"Lines: {len(n.lines)}")

# Generation capacity by technology
print("\nGeneration capacity (GW):")
print((n.generators.groupby('carrier')['p_nom'].sum() / 1000).round(1).sort_values(ascending=False))
```

## Step 4: Generate Analysis Outputs

Generate HTML analysis reports:

```bash
# Generate spatial analysis map
snakemake resources/analysis/HT35_spatial.html -j 1

# Generate summary dashboard
snakemake resources/analysis/HT35_dashboard.html -j 1
```

Open the generated HTML files in your browser.

## Quick Commands Reference

```bash
# Dry run (show what would execute)
snakemake resources/network/HT35_solved.nc -n -p

# Force re-run of a specific rule
snakemake -R solve_network -j 4

# Run with verbose logging
snakemake resources/network/HT35_solved.nc -j 4 --verbose

# Clean all outputs for a scenario
snakemake clean_HT35
```

## Faster Testing with Reduced Networks

For quick tests, use reduced network models:

```bash
# 32-bus reduced network (much faster)
snakemake resources/network/HT35_reduced_solved.nc -j 4

# Clustered to ~100 buses
snakemake resources/network/HT35_clustered_100_solved.nc -j 4
```

## Expected Run Times

| Network Type | Buses | Typical Solve Time |
|-------------|-------|-------------------|
| Zonal | 17 | ~1 minute |
| Reduced | 32 | ~2 minutes |
| Clustered (100) | ~100 | ~5 minutes |
| ETYS Full | 2000+ | ~30 minutes |

```{tip}
Start with reduced networks for testing, then scale up to full ETYS for production runs.
```

## Next Steps

- {doc}`first_scenario` - Create your own custom scenario
- {doc}`../user_guide/configuration` - Understand configuration options
- {doc}`../user_guide/workflow` - Deep dive into the Snakemake workflow
