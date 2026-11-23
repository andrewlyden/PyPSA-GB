# Data Sources

This page documents the external data sources used in PyPSA-GB. Licences and the attribution each source asks
for are on {doc}`/reference/licences`.

```{note}
Some NESO documents published on NESO's website (rather than on the NESO data portal) are not shipped in the
repository: the workflow downloads them when a rule first needs them. They are listed in `data/neso_documents.yaml`
with their address and SHA-256 checksum, and rule `retrieve_neso_document` fetches each one to its path and refuses a
file whose checksum differs. Today this covers the Clean Power 2030 data workbook
(`data/CP30/Clean Power 2030 Data Workbook_V003.xlsx`), the FES data workbooks 2020 to 2024, the FES 2024 regional
breakdown and the ETYS Appendix B workbooks 2022 to 2024. If NESO moves a document, download it by hand from the
landing page in that file and save it at the listed path.
```


## NESO Future Energy Scenarios (FES)

The primary source for future capacity projections.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | NESO (formerly National Grid ESO) |
| **Frequency** | Annual (typically July) |
| **Coverage** | GB electricity system 2025-2050 |
| **Granularity** | Technology × GSP × Year |
| **Access** | Open API + Excel files |
| **Available Releases** | FES 2020 to 2025 (data workbooks in `data/FES/`) |

### FES Releases and Scenarios

NESO publishes updated FES reports annually. You can model any FES release to compare how projections evolve.

### Data Fields

FES provides capacity by:
- **Technology**: Wind, Solar, Nuclear, CCGT, etc.
- **Technology Detail**: Offshore Wind, Onshore Wind, etc.
- **GSP**: Grid Supply Point (connection location)
- **Year**: 2025-2050

### Location

```
data/FES/
├── FES_api_urls.yaml          # NESO data portal endpoints, per release
├── *Data Workbook*.xlsx       # FES data workbooks, 2020-2025
├── fes_year_parameters.csv    # calibrated values by model year and pathway
└── gsp_aliases.csv            # GSP name clean-up per FES edition
```

GSP locations for each edition are in `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`.
Pathway names and codes for every edition are in `data/scenario_sources/pathways.csv`.

### Usage in Model

```python
# Processed FES data - example for 2024 release
resources/FES/FES_2024_data.csv
```

A scenario names the FES release and pathway in `targets`, and the build method in `method`
(see {doc}`/model/scenarios_configuration`):

```yaml
My_HT_2035:
  targets: {source: FES, edition: 2024, pathway: HT}   # FES 2024, Holistic Transition
  method: gsp_block
  modelled_year: 2035
```

For future years (`modelled_year > 2024`), PyPSA-GB uses the specified FES release to provide:
- Thermal generation capacity
- Renewable capacity by technology
- Storage capacity (GSP-level)
- Demand projections: ED1 consumer demand totals and Dem_BB003 GSP shares

**Demand accounting**: future demand does not use raw `Dem_BB003` as the national total. `Dem_BB003` is GSP-facing demand and is used as the spatial allocation across GSPs. The national annual target is read from the FES workbook `ED1` sheet as total consumer electricity demand, then allocated to GSPs using the `Dem_BB003` shares.

**Note**: Different FES releases may have different scenario names and capacity values for the same modelled year. Use this to explore sensitivity to FES updates.

---

## DUKES (Digest of UK Energy Statistics)

Historical power station data from UK Government.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | DESNZ (UK Government) |
| **Frequency** | Annual (March/July) |
| **Coverage** | All major UK power stations |
| **Granularity** | Individual power stations |
| **Access** | Open data (Excel) |

### Key Tables

| Table | Content |
|-------|---------|
| DUKES 5.11 | Power stations (capacity, fuel, location) |
| DUKES 5.10 | Plant efficiencies |
| DUKES 5.6 | Fuel input statistics |

### Data Fields

- Station name
- Installed capacity (MW)
- Fuel type
- Company/owner
- Commissioning year
- Grid reference (for mapping)

### Location

```
data/generators/
├── DUKES_5.11_2025.xlsx                  # DUKES Table 5.11 (power stations)
├── dukes_power_station_coordinates.csv   # resolved station locations
└── generator_data_by_fuel.csv            # technical parameters by fuel
```

### Usage in Model

For historical years (`modelled_year ≤ 2024`), DUKES provides:
- Thermal generator capacities
- Fuel types and efficiencies
- Approximate locations

---

## REPD (Renewable Energy Planning Database)

Site-level data for all renewable projects in the UK.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | DESNZ (UK Government) |
| **Frequency** | Quarterly |
| **Coverage** | All renewable projects (planned + operational) |
| **Granularity** | Individual sites |
| **Access** | Open data (CSV) |

### Data Fields

- Site name and reference
- Technology type
- Installed capacity (MW)
- Development status (Operational, Under Construction, etc.)
- Location (postcode, coordinates)
- Planning authority
- Connection voltage

### Location

```
data/renewables/
├── repd-q2-jul-2025.csv       # REPD, July 2025 extract (the default)
├── repd-q2-july-2024.csv      # earlier extracts kept for historical builds
├── hydro/                     # hydro inflow shapes
└── Marine/                    # tidal and wave inputs
```

### Usage in Model

- **Historical**: Direct use of operational sites
- **Future**: Distribution patterns for FES capacity

### Technology Mapping

| REPD Technology | PyPSA Carrier |
|-----------------|---------------|
| Wind Onshore | `wind_onshore` |
| Wind Offshore | `wind_offshore` |
| Solar Photovoltaics | `solar_pv` |
| Battery | `battery` |
| Hydro | `large_hydro` / `small_hydro` |

---

## ETYS (Electricity Ten Year Statement)

Transmission network data from NESO.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | NESO |
| **Frequency** | Annual (November) |
| **Coverage** | GB transmission network |
| **Granularity** | Individual circuits and transformers |
| **Access** | Open data (Excel appendices) |

### Key Appendices

| Appendix | Content |
|----------|---------|
| B-1 | Substation data (buses) |
| B-2-1a/b/c/d | Base circuit data (SHE-T / SPT / NGET / OFTO) |
| B-2-2a/b/c/d | Circuit upgrades (SHE-T / SPT / NGET / OFTO) |
| B-3-1a/b/c/d | Base transformer data (SHE-T / SPT / NGET / OFTO) |
| B-3-2a/b/c/d | Transformer upgrades (SHE-T / SPT / NGET / OFTO) |
| B-5-1 | HVDC data |

### Data Fields

**Substations (Buses)**:
- Name and identifier
- Voltage level (400kV, 275kV, 132kV)
- Coordinates (OSGB36)

**Circuits (Lines)**:
- From/To buses
- Resistance, Reactance, Susceptance
- Thermal rating (MVA)
- Length (km)

### Location

The ETYS publication year is selected by `network.etys.year` (default 2024, set in `config/defaults/base/network.yaml`;
2022, 2023 and 2024 are supported). Files are mapped by `scripts/network_build/etys_file_registry.py`.

```
data/network/
├── ETYS/
│   ├── ETYS 2024 Appendix-B V1.xlsx    # ETYS 2024 (default)
│   ├── ETYS Appendix B 2023.xlsx        # ETYS 2023
│   ├── ETYS Appendix B 2022.xlsx        # ETYS 2022
│   ├── GB_network.xlsx                  # Offshore WF edges, BMU mappings
│   ├── substation_coordinates.csv       # Bus coordinates
│   ├── bus_coordinate_overrides.csv     # Reviewed per-bus positions (each row sourced)
│   └── appendix_b_projects.csv          # Appendix B rows grouped into named projects
├── reduced_network/
│   ├── buses.csv
│   └── lines.csv
└── zonal/
    ├── buses.csv
    └── links.csv
```

---

## ERA5 Weather Data

Reanalysis weather data for renewable generation profiles.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | ECMWF / Copernicus |
| **Frequency** | Hourly |
| **Coverage** | Global (subset for GB) |
| **Granularity** | ~30km grid |
| **Access** | Zenodo (2010-2024) or CDS API |
| **Zenodo Record** | [10.5281/zenodo.18325225](https://zenodo.org/records/18325225) |

### Variables Used

- 100m wind speed (for wind power)
- 10m wind speed
- Surface solar radiation
- Temperature (for PV efficiency)
- Runoff (for hydro)

### Location

```
resources/atlite/cutouts/
├── uk-2019.nc                 # Weather cutout for 2019
├── uk-2020.nc
├── uk-2021.nc
└── ...
```

### Acquisition Strategy

PyPSA-GB uses a **tiered acquisition strategy** for weather cutouts:

1. **Data directory** - Check `data/atlite/cutouts/` for cached files
2. **Zenodo** - Download pre-built cutouts (~5-10 minutes per year)
3. **ERA5 API** - Generate from scratch via atlite (~2-4 hours per year)

#### Quick Start (Years 2010-2024)

For years 2010-2024, cutouts are automatically downloaded from Zenodo:

```bash
# Configure desired years in config/cutouts_config.yaml
snakemake -s Snakefile_cutouts --cores 1
```

**No CDS API credentials required** for these years!

#### Custom Years (Outside 2010-2024)

For other years, you'll need CDS API credentials:

```bash
# 1. Register at: https://cds.climate.copernicus.eu/user/register
# 2. Set up ~/.cdsapirc with your API key
# 3. Generate cutout
snakemake -s Snakefile_cutouts --cores 1
```

### Pre-built Cutouts (Zenodo)

Pre-built cutouts for years 2010-2024 are available on Zenodo:

- **Repository**: [PyPSA-GB Atlite Cutouts](https://zenodo.org/records/18325225)
- **License**: CC-BY-4.0
- **File size**: ~765 MB per year
- **Download time**: 5-10 minutes per year
- **MD5 verification**: Automatic

These cutouts are automatically used by the workflow and require no manual intervention.

---

## ESPENI (historical demand)

Half-hourly GB electricity demand, built from Elexon and NESO data.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | Wilson et al., University of Birmingham (Zenodo, concept DOI [10.5281/zenodo.3884858](https://doi.org/10.5281/zenodo.3884858)) |
| **Frequency** | Half-hourly |
| **Coverage** | GB total demand from November 2008 |
| **Granularity** | National total |
| **Access** | Open data (CC BY-NC 4.0) |

### Data Fields

- Timestamp
- Demand (MW)
- Settlement period

### Location

```
data/demand/
├── espeni.csv                   # ESPENI half-hourly demand
├── egy_7649_mmc1.xlsx           # published demand profiles (eLOAD, DESSTINEE)
├── cp30_named_load_nodes.csv    # CP30 transmission loads at their ETYS nodes
└── gsp_location_overrides.csv
```

### Usage

- **Historical years**: Direct use
- **Future years**: Profile shape only; annual totals come from FES ED1 consumer demand and are spatially allocated using Dem_BB003 GSP shares

---

## TEC Register

Transmission Entry Capacity from NESO.

### Overview

| Attribute | Details |
|-----------|---------|
| **Publisher** | NESO |
| **Frequency** | Monthly |
| **Coverage** | All transmission-connected generators |
| **Access** | Open data (CSV) |

### Data Fields

- Generator name (BMU ID)
- TEC capacity (MW)
- Connection point
- Fuel type
- Effective dates

### Location

```
data/generators/
├── tec-register-02-september-2025.csv   # read by the generator data preparation rules
└── tec-register-11-10-2024.csv          # earlier register, kept for historical builds
```

### Usage

Used to cross-reference DUKES data and validate capacities.

---

## Data Quality Notes

### Market And Validation Data

Market scenarios can use additional historical operational data:

| Source | Used For | Notes |
|--------|----------|-------|
| ELEXON BMRS BOD | Historical BM bid and offer prices | Used when `market.balancing.bid_offer_source: "elexon"` or when `auto` resolves to ELEXON for historical scenarios. |
| ELEXON BOAV/BOALF | Historical BM validation | Used by `validate_bm_results` to compare model redispatch with accepted bid/offer actions. |
| ELEXON MID | Wholesale price validation | Used by wholesale notebooks and validation plots for historical scenarios. |
| NESO thermal constraint data | Constraint-cost validation | Used by `validate_neso_constraints` to compare model constraint costs and boundary flows. |
| ESPENI generation by fuel | Physical dispatch validation | Used as an independent comparison for historical balancing dispatch. |

These files are cached under `data/market/`, `data/validation/`, or
`resources/market/` depending on whether they are persistent raw inputs,
validation caches, or per-scenario processed outputs. See
{doc}`/model/dispatch/index` for the market stages that read them.

### Known Issues

| Source | Issue | Mitigation |
|--------|-------|------------|
| DUKES | Missing coordinates for some stations | Manual geocoding |
| REPD | Duplicate entries | Deduplication script |
| ETYS | Some circuits under construction | Filter by status |
| FES | GSP names change between years | Mapping tables |

### Validation

The model validates data during build:

```bash
python scripts/validate_scenarios.py
```

This checks:
- All generators have valid coordinates
- Capacity totals match expected values
- No orphan buses in network

## Comprehensive Data Sources Table

| Type | Data | Data Processing | Source and License |
|------|------|------------------|--------------------|
| **Network** | Reduced network model | Matpower file converted to buses and lines | Bell and Tleis, Bukhsh et al. – GPL-3.0 |
| | Zonal model | Excel data converted to buses and links | NESO ETYS – NESO website document (see licences) |
| **Electrical Demand** | ESPENI | CSV converted to loads | Wilson et al. – CC BY-NC 4.0 |
| **Marginal Costs** | Fuel costs from FES | Excel data converted to marginal price with addition of EU-ETS and CPS | NESO FES – NESO Open Data Licence |
| | EU-ETS | Excel data converted to marginal price in addition to fuel costs and CPS | Ember Climate – CC-BY-4.0 |
| | Carbon Price Support (CPS) | Excel data converted to marginal price in addition to fuel costs and EU-ETS | UK Gov BEIS – Open Government Licence v3.0 |
| **Thermal Power Plants & Hydropower** | Historical data (location, fuel, type, capacity) | Conversion from Excel data to generators | DUKES dataset – Open Government Licence v3.0 |
| | Coordinates | Coordinate data converted to generator attributes | OpenStreetMap (ODbL), Global Energy Monitor (CC-BY-4.0), Google Maps |
| | Technical characteristics | Data from papers converted to generator attributes | Schröder et al., Angerer et al. |
| | Hydropower power output timeseries | Excel data converted to generator power timeseries | Elexon BMRS – BMRS Open Data Licence |
| | Future installed capacities and locations | Excel data converted to generator attributes | NESO FES – NESO Open Data Licence |
| **Renewable Power** | Renewable power timeseries | ERA5 weather data converted to generator power timeseries using Atlite | ERA5 (CC-BY-4.0), Atlite (MIT) |
| | Historical location, type, capacity | Excel data converted to generator attributes | REPD – Open Government Licence v3.0 |
| | Historical annual generation | Report data used to scale REPD for <150kW installations | UK Gov BEIS – Open Government Licence v3.0 |
| | Future installed capacities and locations | Excel data converted to generator attributes | NESO FES – NESO Open Data Licence |
| | Offshore wind (near-term) | Report data for near-term spatial distribution | REPD, Scottish Sectoral Marine Plan – OGL v3.0 |
| | Tidal lagoon and stream | Generator power timeseries using Thetis coastal ocean model | Thetis – MIT |
| | Wave power generation | ERA5 wave climate converted using power matrix | ERA5 (CC-BY-4.0), Power Matrix |
| **Storage** | Historical pumped hydro | Excel data converted to storage unit attributes | DUKES dataset – Open Government Licence v3.0 |
| | Future storage capacities and locations | Excel data converted to storage unit attributes | NESO FES – NESO Open Data Licence |
| | Efficiency and losses | Report data converted to storage attributes | Moseley and Garche |
| **Emissions** | Direct emissions factors | Data from various sources converted to carbon factors | Staffell, Schlömer et al. |

