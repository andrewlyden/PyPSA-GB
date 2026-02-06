# Tutorials

Interactive Jupyter notebook tutorials demonstrating PyPSA-GB's network-constrained optimal power flow (LOPF) capabilities across historical and future scenarios.

```{toctree}
:maxdepth: 1
:caption: Scenario Tutorials

1-historical-baseload-2015
2-historical-renewables-2023
3-future-holistic-transition-2035
4-future-electric-engagement-2050
```

```{toctree}
:maxdepth: 1
:caption: Component Tutorials

5-networks
6-demand
7-generators
8-marginal-costs
9-renewables
10-storage
11-interconnectors
12-hydrogen
13-heat-flexibility
```

## Tutorial Overview

These tutorials demonstrate PyPSA-GB's core capability: **network-constrained optimal power flow (LOPF)** modeling across different time periods and network resolutions.

### Scenario Tutorials (1-4)

Complete scenario walkthroughs from network building to results analysis:

| # | Notebook | Year | Network | Solve Time | Focus |
|---|----------|------|---------|------------|-------|
| **1** | Historical Baseload (2015) | 2015 | Reduced (32 buses) | 5-10 min | Traditional dispatch, coal/nuclear dominance |
| **2** | Historical Renewables (2023) | 2023 | Full ETYS (~2000 buses) | 30-60 min | High renewable penetration, network detail |
| **3** | Future Near-Term (2035) | 2035 | ETYS clustered (100 buses) | 10-15 min | Renewable dominance, storage, network upgrades |
| **4** | Future Long-Term (2050) | 2050 | ETYS clustered (100 buses) | 10-15 min | Near-100% renewables, stress test, adequacy |

### Component Tutorials (5-12)

Deep dives into specific aspects of power system modeling:

| # | Notebook | Topic | Description |
|---|----------|-------|-------------|
| **5** | Networks | Topology & Power Flow | Buses, lines, transformers, coordinate systems |
| **6** | Demand | Load Modeling | ESPENI/eload profiles, temporal patterns, spatial distribution |
| **7** | Generators | Dispatch Optimization | Thermal/renewable generators, merit order, capacity factors |
| **8** | Marginal Prices | Price Formation | LMPs, congestion pricing, generator revenues |
| **9** | Renewables | Variable Generation | Wind/solar profiles, curtailment, capacity factors |
| **10** | Storage | Flexibility Resources | Batteries, pumped hydro, state of charge, arbitrage |
| **11** | Interconnectors | Cross-border Trade | HVDC links, import/export flows, utilization |
| **12** | Hydrogen | Sector Coupling | Electrolysis, H2 storage, hydrogen-to-power |
| **13** | Heat Flexibility | Heat Pump Demand | TANK/COSY modes, COP, thermal storage, demand shifting |


## Running Tutorials

### Prerequisites

1. Complete the {doc}`../getting_started/installation`
2. Activate the environment:
   ```bash
   conda activate pypsa-gb
   ```
3. Generate the required scenario networks (see below)

### Generate Scenario Networks

Before running the tutorials, build the required networks:

```bash
# Notebook 1: Historical 2015 (Reduced network)
snakemake resources/network/Historical_2015_reduced_solved.nc -j 4

# Notebook 2: Historical 2023 (Full ETYS - takes longer)
snakemake resources/network/Historical_2023_etys_solved.nc -j 4

# Notebook 3: Future 2035 (Clustered ETYS)
snakemake resources/network/HT35_clustered_solved.nc -j 4

# Notebook 4: Future 2050 (Clustered ETYS)
snakemake resources/network/EE50_clustered_solved.nc -j 4
```


### Launch Jupyter

```bash
# Navigate to tutorials directory
cd docs/source/tutorials

# Start Jupyter
jupyter notebook
```

### First-Time Setup

Install the kernel if needed:
```bash
python -m ipykernel install --user --name=pypsa-gb
```

Then select the `pypsa-gb` kernel in Jupyter.

## Tutorial Requirements

### System Requirements

| Notebook | RAM | Solve Time | Disk Space |
|----------|-----|------------|------------|
| 1 - Historical 2015 | 4 GB | 5-10 min | ~500 MB |
| 2 - Historical 2023 | 16 GB | 30-60 min | ~2 GB |
| 3 - Future 2035 | 8 GB | 10-15 min | ~1 GB |
| 4 - Future 2050 | 8 GB | 10-15 min | ~1 GB |

### Solver Requirements

Tutorials require **Gurobi** solver (free academic license available) or **HiGHS** (open-source).

See {doc}`../getting_started/installation` for solver setup.

## Contributing Tutorials

We welcome new tutorial contributions:

1. Create a notebook in `docs/source/tutorials/`
2. Add to the toctree in this index (in appropriate section)
3. Test that it builds correctly

See {doc}`../development/contributing` for guidelines.
