# Data Reference

This section documents the data sources and data flow in PyPSA-GB.

```{toctree}
:maxdepth: 2

data_sources
network_data
maintenance
```

## Data Architecture

PyPSA-GB uses a layered data architecture:

```{mermaid}
flowchart TB
    subgraph External["External Sources"]
        FES["NESO FES API"]
        DUKES["DUKES Statistics"]
        REPD["REPD Database"]
        ETYS["ETYS Network Data"]
        ERA5["ERA5 Weather"]
        ESPENI["ESPENI Demand"]
    end
    
    subgraph Raw["data/ (Raw)"]
        FES_RAW["FES Excel/CSV"]
        DUKES_RAW["DUKES Excel"]
        REPD_RAW["REPD CSV"]
        ETYS_RAW["Network files"]
    end
    
    subgraph Processed["resources/ (Processed)"]
        FES_PROC["FES processed"]
        GEN_PROC["Generators"]
        RENEW_PROC["Renewables"]
        NETWORK["Network files"]
    end
    
    FES --> FES_RAW --> FES_PROC
    DUKES --> DUKES_RAW --> GEN_PROC
    REPD --> REPD_RAW --> RENEW_PROC
    ETYS --> ETYS_RAW --> NETWORK
```

## Directory Structure

```
data/                    # Raw input (versioned, rarely changes)
├── FES/                 # FES capacity projections
├── generators/          # DUKES thermal data
├── renewables/          # REPD site data
├── network/             # Network topology
├── demand/              # Demand profiles
├── storage/             # Storage parameters
└── interconnectors/     # Cross-border connections

resources/               # Generated outputs (recreated by workflow)
├── FES/                 # Processed FES data
├── generators/          # Processed generator files
├── renewable/           # Site data + generation profiles
├── network/             # Built PyPSA networks
├── marginal_costs/      # Fuel price time series
└── analysis/            # Output reports
```

## Key Data Files

| File | Description | Update Frequency |
|------|-------------|-----------------|
| `data/generators/DUKES_5.11_*.xlsx` | UK power station register | Annual (March) |
| `data/renewables/repd-*.csv` | Renewable sites | Quarterly |
| `data/generators/tec-register-*.csv` | Transmission entry capacity | Monthly |
| `data/FES/FES_api_urls.yaml` | NESO API endpoints | Per FES release |
| `data/demand/ESPENI/*.csv` | Historical demand | Annual |

## Scenario Data Routing

Data sources depend on the modelled year:

| Data Type | Historical (≤2024) | Future (>2024) |
|-----------|-------------------|----------------|
| Thermal | DUKES statistics | FES projections |
| Renewables | REPD site data | FES + REPD distribution |
| Storage | REPD site data | FES GSP-level |
| Demand | ESPENI profiles | FES totals + profile shape |
| Network | ETYS current | ETYS + planned upgrades |

## Quick Links

- {doc}`data_sources` - Detailed documentation of each data source
- {doc}`network_data` - Network topology and parameters
- {doc}`maintenance` - How to update data files
