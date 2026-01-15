# Getting Started

Welcome to PyPSA-GB! This section will help you get up and running quickly.

```{toctree}
:maxdepth: 2

installation
quickstart
first_scenario
```

## Overview

PyPSA-GB uses a **Snakemake workflow** to orchestrate the model. The basic process is:

1. **Configure** your scenario in YAML files
2. **Run** the Snakemake workflow
3. **Analyze** the results

## Prerequisites

Before you begin, ensure you have:

- Python 3.10 or later
- Conda package manager (via Anaconda or Miniconda)
- Git for cloning the repository
- ~10GB disk space for data and results
- Gurobi solver (free academic license) or HiGHS (open-source)

## Quick Links

- {doc}`installation` - Full setup instructions
- {doc}`quickstart` - Run your first model in 5 minutes
- {doc}`first_scenario` - Create and analyze a custom scenario

## Support

If you encounter issues:

1. Check the {doc}`../development/troubleshooting` guide
2. Search existing [GitHub Issues](https://github.com/andrewlyden/PyPSA-GB/issues)
3. Open a new issue with your error message and configuration
