## `scripts/sources/`

Scenario sources: registries (pathways, GSP aliases, ES1 line maps), raw downloads and adapters.

| Module | Summary |
|---|---|
| `scripts/sources/build.py` | Write one source edition's contract tables: ``python -m scripts.sources.build --source FES --edition 2025``. |
| `scripts/sources/contract.py` | The scenario-source data contract: one directory of long tables per source edition. |
| `scripts/sources/cp30_adapter.py` | Clean Power 2030 (NESO, Dec 2024, workbook V003) to the scenario-source contract. |
| `scripts/sources/fes_adapter.py` | FES editions (2020-2025) to the scenario-source contract. |
| `scripts/sources/fes_raw.py` | Cached raw downloads of the NESO FES API tables, one file per edition and table. |
| `scripts/sources/fes_year_inputs.py` | A new long-range FES year's hand-made inputs, written by script. |
| `scripts/sources/new_fes_year.py` | Start a new long-range FES year: write its study folder and list what it still needs. |
| `scripts/sources/registry.py` | Scenario-source registries as data: pathways, GSP aliases and ES1 line maps. |
