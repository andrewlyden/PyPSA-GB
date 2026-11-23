## `scripts/interconnectors/`

| Module | Summary |
|---|---|
| `scripts/interconnectors/add_to_network.py` | Add Interconnectors to PyPSA Network ==================================== |
| `scripts/interconnectors/analyse_historical_market_coupling.py` | Compare market-coupled historical interconnector flows with ESPENI. |
| `scripts/interconnectors/availability.py` | Generate Interconnector Availability Profiles ============================================== |
| `scripts/interconnectors/build_cp30_foreign_prices.py` | Snakemake entry point for rule build_cp30_foreign_prices. |
| `scripts/interconnectors/build_price_shape.py` | Snakemake entry point for rule build_interconnector_price_shape (item IC1). |
| `scripts/interconnectors/build_tyndp_price_shape.py` | Snakemake entry point for rule build_tyndp_price_shape (task Y3, FES 2050 plan; any TYNDP year). |
| `scripts/interconnectors/clean_interconnectors.py` | Clean Interconnector Data ========================= |
| `scripts/interconnectors/combine_datasets.py` | Combine DUKES and NESO Interconnector Data ========================================== |
| `scripts/interconnectors/cp30_foreign_prices.py` | Foreign prices on CP30's fuel basis for the 2030 interconnectors. |
| `scripts/interconnectors/download_european_generation.py` | Download European electricity generation mix data from NESO FES API. |
| `scripts/interconnectors/enrich_locations.py` | Enhanced interconnector location enrichment using NESO data and external sources. |
| `scripts/interconnectors/extract_historical_flows.py` | Extract Historical Interconnector Flows from ESPENI Data ======================================================== |
| `scripts/interconnectors/extract_tyndp_hourly.py` | Extract the hourly TYNDP market data the 2050 foreign-price shape needs (task Y3). |
| `scripts/interconnectors/extract_tyndp_ntplus_mc.py` | Extract a TYNDP 2026 National Trends+ year's hourly marginal costs from the hourly dashboard archive. |
| `scripts/interconnectors/fetch_entsoe_prices.py` | Fetch ENTSO-E Day-Ahead Electricity Prices ============================================ |
| `scripts/interconnectors/foreign_price_override.py` | Solve-time foreign price sensitivity on a shared network. |
| `scripts/interconnectors/ingest_dukes.py` | Ingest DUKES Table 5.13 Interconnector Data =========================================== |
| `scripts/interconnectors/ingest_neso_register.py` | Ingest NESO Interconnector Register =================================== |
| `scripts/interconnectors/map_to_buses.py` | Map Interconnectors to Network Buses ==================================== |
| `scripts/interconnectors/market_coupling.py` | Shared constraints and helpers for directional interconnector links. |
| `scripts/interconnectors/pipeline_placeholder.py` | Create Interconnector Pipeline Placeholder ========================================== |
| `scripts/interconnectors/prepare_cp30_refit_inputs.py` | One-off data preparation for the 2013-weather refit of TYNDP prices. |
| `scripts/interconnectors/price_shape.py` | An ENTSO-E hourly price shape conditioned on the GB wind of the weather year (item IC1). |
| `scripts/interconnectors/process_european_generation.py` | Process European electricity generation mix data for interconnector modeling. |
| `scripts/interconnectors/tyndp_prices.py` | European foreign prices of a TYNDP year from ENTSO-E / ENTSOG TYNDP (task Y3 of the FES 2050 plan). |
| `scripts/interconnectors/validate_historical_flows.py` | Validate Historical Interconnector Flows ========================================= |
| `scripts/interconnectors/volume_tranches.py` | Volume tranches on the foreign price: GB's trade moves the price it trades at (item IC1). |
