# User Guide

This section provides comprehensive documentation for using PyPSA-GB.

```{toctree}
:maxdepth: 2

workflow
configuration
scenarios
network_models
clustering
```

## Overview

PyPSA-GB is designed around a **declarative workflow** where you:

1. **Define what you want** in configuration files
2. **Let Snakemake figure out how** to build it
3. **Analyze the results** using standard PyPSA tools

This approach ensures reproducibility and makes it easy to run multiple scenarios.

## Key Concepts

### Scenarios

A **scenario** is a complete specification of a model run, including:
- Target year (historical or future)
- Data sources (FES pathway, weather year)
- Network model (full ETYS, reduced, zonal)
- Solver settings

See {doc}`scenarios` for details.

### Workflow

The **Snakemake workflow** automatically:
- Builds the network from raw data
- Adds generation, storage, and demand
- Solves the optimal dispatch
- Generates analysis outputs

See {doc}`workflow` for the full pipeline.

### Network Models

Three levels of network detail are available:

| Model | Buses | Use Case |
|-------|-------|----------|
| ETYS | ~2000 | Production runs, detailed analysis |
| Reduced | 32 | Testing, quick studies |
| Zonal | 17 | Aggregate regional analysis |

See {doc}`network_models` for details.

## Common Tasks

### Run a Pre-configured Scenario

```bash
snakemake resources/network/HT35_solved.nc -j 4
```

### Create a Custom Scenario

1. Add to `config/scenarios.yaml`
2. Activate in `config/config.yaml`
3. Run: `snakemake resources/network/MyScenario_solved.nc -j 4`

### Compare Scenarios

```python
import pypsa

scenarios = ['HT35', 'HT50', 'EE35']
for s in scenarios:
    n = pypsa.Network(f"resources/network/{s}_solved.nc")
    print(f"{s}: {n.objective/1e9:.2f} B£ system cost")
```

### Generate Reports

```bash
snakemake resources/analysis/HT35_spatial.html -j 1
```

## Getting Help

- {doc}`../development/troubleshooting` - Common issues and solutions
- [GitHub Issues](https://github.com/andrewlyden/PyPSA-GB/issues) - Bug reports and questions
- [PyPSA Documentation](https://pypsa.readthedocs.io/) - Underlying framework docs
