# PyPSA-GB

PyPSA-GB is an open-source electricity system model for Great Britain, built on PyPSA (Python for Power Systems Analysis). It provides detailed power system optimization for both historical analysis and future scenario planning.

## Features

- **Multiple Network Models**: ETYS (2000+ buses), Reduced (32 buses), and Zonal (17 zones) representations
- **Historical & Future Scenarios**: Automatic data routing for years ≤2024 (historical) and >2024 (future projections)
- **Comprehensive Data Sources**: DUKES, REPD, NESO FES, ETYS, ESPENI demand profiles
- **Technology Coverage**: Thermal generation, wind/solar/marine renewables, storage (batteries, pumped hydro), interconnectors
- **Workflow Automation**: Snakemake-based pipeline for reproducible scenario generation
- **Network Planning**: ETYS infrastructure upgrades through 2031

## Quick Start

```bash
# Create environment
conda env create -f envs/pypsa-gb.yaml

# Activate environment 
conda activate pypsa-gb

# Run scenario workflow
snakemake -j 4
```

Configure scenarios in `config/scenarios.yaml` and active runs in `config/config.yaml`.

## Citation

If you use PyPSA-GB in your research, please cite:

**Lyden, A., Sun, W., Struthers, I., Franken, L., Hudson, S., Wang, Y. and Friedrich, D., 2024.** PyPSA-GB: An open-source model of Great Britain's power system for simulating future energy scenarios. *Energy Strategy Reviews*, 53, p.101375.

## Papers Using PyPSA-GB

- **Dergunova, T. and Lyden, A., 2024.** Great Britain's hydrogen infrastructure development—Investment priorities and locational flexibility. *Applied Energy*, 375, p.124017.

- **Desguers, T., Lyden, A. and Friedrich, D., 2024.** Integration of curtailed wind into flexible electrified heating networks with demand-side response and thermal storage: Practicalities and need for market mechanisms. *Energy Conversion and Management*, 304, p.118203.

- **Lyden, A., Alene, S., Connor, P., Renaldi, R. and Watson, S., 2024.** Impact of locational pricing on the roll out of heat pumps in the UK. *Energy Policy*, 187, p.114043.

- **Lyden, A., Sun, W., Friedrich, D. and Harrison, G., 2023.** Electricity system security of supply in Scotland. Study for the Scottish Government via ClimateXChange.

## Documentation

See [full documentation](https://pypsa-gb.readthedocs.io) for detailed usage.

![PyPSA-GB ETYS Clustered Network Model](pics/ETYS_clustered.jpg)
