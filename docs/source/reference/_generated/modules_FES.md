## `scripts/FES/`

| Module | Summary |
|---|---|
| `scripts/FES/build_fes_inputs.py` | Snakemake driver for the FES 2025 targets and commodity prices (task Y1, ``rules/fes_long_range.smk``), any year. |
| `scripts/FES/extract_building_block_definitions.py` | Extract FES Building Block Definitions from NESO API. |
| `scripts/FES/extract_FES_prices.py` | Extract fuel and carbon prices from FES Data Workbooks (sheets CP1 and CP2). |
| `scripts/FES/fes2025_targets.py` | FES 2025 Data Workbook energy targets for a pathway and year (task Y1). |
| `scripts/FES/fes_commodity_prices.py` | FES commodity prices (sheets CP1 and CP2) read from the Data Workbook, with no fallback. |
| `scripts/FES/FES_data.py` | FES building blocks per edition: cached raw download, then normalisation to FES_{year}_data.csv. |
| `scripts/FES/fes_dispatch_validation.py` | FES long-range final-dispatch validation against FES 2025 ES1. |
| `scripts/FES/gsp_metadata.py` | Utilities for locating and generating FES GSP metadata files. |
