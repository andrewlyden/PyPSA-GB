## `scripts/utilities/`

| Module | Summary |
|---|---|
| `scripts/utilities/build_contributors.py` | Write the docs' contributor grid from GitHub's contributor list. |
| `scripts/utilities/build_reference_tables.py` | Write the documentation's generated reference tables: the configuration keys and the module index. |
| `scripts/utilities/calendar_year.py` | Calendar helpers for putting weather-year and source-year series on a modelled year. |
| `scripts/utilities/carrier_definitions.py` | Define carrier attributes for PyPSA-GB generators. |
| `scripts/utilities/code_fingerprint.py` | Fingerprint of the project code an output was built with. |
| `scripts/utilities/conda_env.py` | Replicate the conda activation scripts for runs launched without activation. |
| `scripts/utilities/download_cutouts.py` | Download atlite cutouts with a tiered strategy: 1. Check data directory for cached copy (instant) 2. Try downloading from Zenodo repository (fast, ~minutes) 3. Fall back to atlite ERA5 download (slow, ~hours) |
| `scripts/utilities/elexon_cache.py` | ELEXON Calibration Data Cache — Pack and Restore ================================================== |
| `scripts/utilities/find_publications.py` | Find papers and reports that cite or mention PyPSA-GB, for the docs' Papers list. |
| `scripts/utilities/fresh.py` | Refuse to read workflow outputs that Snakemake would rebuild. |
| `scripts/utilities/gsp_nodes.py` | The network node where a location's own GSP takes its demand. |
| `scripts/utilities/logging_config.py` | Logging Configuration for PyPSA-GB |
| `scripts/utilities/network_io.py` | Fast Network I/O Utilities for PyPSA-GB |
| `scripts/utilities/plotting.py` | (no module docstring) |
| `scripts/utilities/postcode_geocoder.py` | UK Postcode Geocoding Utility with Caching |
| `scripts/utilities/prepare_cutouts.py` | Prepare atlite cutouts using a tiered acquisition strategy: |
| `scripts/utilities/retrieve_neso_document.py` | Download one NESO website document listed in data/neso_documents.yaml and check its SHA-256. |
| `scripts/utilities/scenario_detection.py` | Scenario Detection and Auto-Configuration Module |
| `scripts/utilities/spatial_utils.py` | Spatial Utilities for PyPSA-GB =============================== |
| `scripts/utilities/time_utils.py` | Time utilities for PyPSA-GB project. Handles time series resampling and alignment to config timesteps. |
| `scripts/utilities/timeseries_synchronization.py` | Timeseries Synchronization Utilities for PyPSA-GB |
