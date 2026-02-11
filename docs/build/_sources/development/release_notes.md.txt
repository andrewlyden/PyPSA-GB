# Release Notes

Version history and changelog for PyPSA-GB.

## Version 2.0.0

### Major Features

- **Snakemake Workflow**: Complete redesign using Snakemake for reproducible execution
- **FES Support**: Updated to more easily incorporate new NESO Future Energy Scenarios
- **Scenario-Based Configuration**: YAML-based scenario definitions
- **Network ETYS and Clustering**: Support for clustering ETYS network to configurable sizes
- **Improved Documentation**: Comprehensive documentation

### New Capabilities

- Automatic historical/future scenario detection
- GSP-level FES data integration
- Network upgrade application for future years
- Centralised logging configuration
- Scenario validation tools

### Data Updates

- DUKES 2025 power station data
- REPD Q2 2025 renewable sites
- TEC Register July 2025
- ETYS 2023 network topology

---

## Version 1.0.0

### Original Release

- Initial PyPSA-GB implementation
- Jupyter notebook-based workflow
- Historical years 2010-2020 support
- FES scenario support (older FES versions)
- Reduced, and Zonal network models

---

<!-- ## Versioning Policy

PyPSA-GB follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible -->

## How to Cite

If you use PyPSA-GB in your research, please cite:

   **Lyden, A., Sun, W., Struthers, I., Franken, L., Hudson, S., Wang, Y. and Friedrich, D., 2024.**
   PyPSA-GB: An open-source model of Great Britain's power system for simulating future energy scenarios.
   *Energy Strategy Reviews*, 53, p.101375.
