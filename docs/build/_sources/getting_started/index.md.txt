# Getting Started

Welcome to PyPSA-GB! This section will help you get up and running quickly.

```{toctree}
:maxdepth: 2

installation
quickstart
first_scenario
```

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or later
- Conda package manager (via Anaconda or Miniconda)
- Git for cloning the repository
- ~10GB disk space for data and results
- Gurobi solver (free academic license) or HiGHS (open-source)

## Workflow

PyPSA-GB uses a **Snakemake workflow** to orchestrate the model:

1. **Configure** your scenario in YAML files
2. **Run** the Snakemake workflow
3. **Analyse** the results

If you encounter issues, check the {doc}`../development/troubleshooting` guide or open a [GitHub Issue](https://github.com/andrewlyden/PyPSA-GB/issues).
