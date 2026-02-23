# Missing Transmission Bus Coordinates - Research Summary

## Overview
This document summarizes the coordinate data found for 7 transmission substations (132kV, 275kV, 400kV) that are missing location data in the PyPSA-GB network model.

## Data Sources Checked

### 1. ETYS Excel Files
- **GB_network.xlsx**: Contains network topology (AC, HVDC sheets) with Node 1/Node 2 connections, but NO coordinate data
  - AC sheet: Lists transmission lines between nodes
  - Substation index sheets (B-1-1a, B-1-1b, B-1-1c, B-1-1d): Contain Site Code, Site Name, and Voltage only

- **ETYS Appendix B 2023.xlsx**: Contains substation names and voltages but NO coordinates
  - Sheet B-1-1a (SHE Transmission): TEAL, KINT, CASS, TUMM, ERRO, FOYE
  - Sheet B-1-1b (SPT): TORN

### 2. GSP Information Files
- **fes2021_regional_breakdown_gsp_info.csv**: Contains GSP (Grid Supply Point) coordinates
  - Found coordinates for KINT and CASS (both are GSPs)
  - Other target buses (TEAL, TUMM, ERRO, FOYE, TORN) are transmission substations, not GSPs

## Coordinates Found

### From GSP Files (fes2021_regional_breakdown_gsp_info.csv)

| Code | Name | Latitude | Longitude | Voltages (kV) | Operator |
|------|------|----------|-----------|---------------|----------|
| KINT | Kintore | 57.213 | -2.374 | 400, 275, 132 | SHE |
| CASS | Cassley | 56.542 | -4.517 | 132 | SHE |

### From Web Research

| Code | Name | Latitude | Longitude | Voltages (kV) | Operator | Source |
|------|------|----------|-----------|---------------|----------|--------|
| TEAL | Tealing | ~56.53 | ~-2.96 | 275, 132 | SHE | Village center approx. |
| TUMM | Tummel Bridge | ~56.72 | ~-3.94 | 400, 275, 132 | SHE | Power station area |
| ERRO | Errochty | See Grid Ref | See Grid Ref | 132 | SHE | OS Grid NN 7695 5909 |
| FOYE | Foyers | 57.2618 | -4.4835 | 275 | SHE | Pumped storage facility |
| TORN | Torness | 55.968 | -2.409 | 400, 132 | SPT | Nuclear power station |

## Detailed Findings

### TEAL - TEALING (SHE)
- **Status**: Approximate coordinates found
- **Location**: Angus, near Dundee
- **Coordinates**: 56.53357°N, 2.96305°W (village center)
- **Notes**: SSEN Transmission project site. Exact substation location needs verification.
- **Nodes in network**: TEAL1-, TEAL2J, TEAL2K
- **Web sources**:
  - Mapcarta: Tealing Substation mapping
  - SSEN Transmission: Tealing Substation Extension project page

### KINT - KINTORE (SHE)
- **Status**: ✓ Coordinates confirmed
- **Location**: Aberdeenshire
- **Coordinates**: 57.213°N, 2.374°W
- **Source**: FES 2021 GSP info file
- **Nodes in network**: KINT1-, KINT1B, KINT1P, KINT1R, KINT1T, KINT1U, KINT2J, KINT2K, KINT3-

### CASS - CASSLEY (SHE)
- **Status**: ✓ Coordinates confirmed
- **Location**: Scottish Highlands, Sutherland
- **Coordinates**: 56.542°N, 4.517°W
- **Source**: FES 2021 GSP info file
- **Nodes in network**: CASS1Q, CASS3-

### TUMM - TUMMEL (SHE)
- **Status**: Approximate coordinates found
- **Location**: Perthshire, near Tummel Bridge
- **Coordinates**: ~56.72°N, 3.94°W (power station area)
- **Notes**: Part of the Tummel hydro-electric power scheme. Tummel Bridge Power Station nearby.
- **Nodes in network**: TUMM1J, TUMM1K, TUMM2J, TUMM2K, TUMM4-
- **Web sources**:
  - Mapcarta: Tummel Substation mapping
  - SSE Renewables: Tummel hydro scheme
  - Wikipedia: Tummel hydro-electric power scheme

### ERRO - ERROCHTY (SHE)
- **Status**: Grid reference found, needs conversion
- **Location**: Perth and Kinross, part of Tummel hydro scheme
- **Grid Reference**: NN 7695 5909 (Errochty Switching Station)
- **Notes**: Errochty Dam (NN 71411 65622), Power Station (NN 7727 5929), Switching Station (NN 7695 5909)
- **Conversion needed**: OS National Grid → Lat/Lon
  - Approximate: 56.79°N, 4.07°W (needs verification)
- **Nodes in network**: ERRO1A, ERRO1B, ERRO1J, ERRO1K, ERRO1T, ERRO5J, ERRO5L
- **Web sources**:
  - Structurae: Errochty Dam details with OS grid references
  - Trove.scot: Errochty Power Station and Switching Station locations
  - Global Energy Monitor: Errochty hydroelectric plant

### FOYE - FOYERS (SHE)
- **Status**: ✓ Coordinates confirmed
- **Location**: Loch Ness, Highland
- **Coordinates**: 57.2618°N, 4.4835°W
- **Notes**: Foyers Pumped Storage Power Station, 300MW capacity
- **Nodes in network**: FOYE2-, FOYE2J
- **Web sources**:
  - Global Energy Observatory: Foyers Pumped Storage Power Station with coordinates
  - SSE Renewables: Foyers hydro scheme

### TORN - TORNESS (SPT)
- **Status**: ✓ Coordinates confirmed
- **Location**: East Lothian coast, near Dunbar
- **Coordinates**: 55.968°N, 2.409°W
- **Notes**: Torness Nuclear Power Station. Has 400kV and 132kV substations.
- **Nodes in network**: TORN1-, TORN4-
- **Web sources**:
  - Wikipedia: Torness nuclear power station with coordinates
  - Wikidata: Torness Nuclear Power Station
  - Trove.scot: Torness Power Station location

## OS Grid Reference Conversion

For ERRO (Errochty Switching Station): NN 7695 5909
- Easting: 276950
- Northing: 759090
- Approximate conversion to Lat/Lon:
  - Latitude: ~56.79°N
  - Longitude: ~-4.07°W
  - (Conversion needs verification using proper OS Grid → WGS84 conversion tool)

## Network Topology Analysis

All 7 substations have confirmed nodes in GB_network.xlsx AC sheet:
- TEAL: Connected to BIHI, DENS, FETT, GLRB, KINB, KINT, LUNA, MILC, WFIB (18 connections)
- KINT: Connected to 9 other buses
- CASS: Connected to 2 other buses
- TUMM: Connected to BONB, BRCW, ERRO, FAUG, MELG (9 connections)
- ERRO: Connected to CLUN, KIIN, TUMB, TUMM, WHIB (18 connections)
- FOYE: Connected to FARI (3 connections)
- TORN: Connected to CRYR, ECCL, INWI, SMEA (8 connections)

## Recommendations

1. **High Confidence (Use directly)**:
   - KINT: 57.213, -2.374 (from GSP file)
   - CASS: 56.542, -4.517 (from GSP file)
   - FOYE: 57.2618, -4.4835 (from power station coordinates)
   - TORN: 55.968, -2.409 (from nuclear station coordinates)

2. **Medium Confidence (Verify before use)**:
   - TEAL: 56.53, -2.96 (village center - substation may be offset)
   - TUMM: 56.72, -3.94 (power station area - substation may be offset)

3. **Requires Conversion**:
   - ERRO: Convert OS Grid NN 7695 5909 to WGS84 lat/lon
   - Expected: ~56.79, -4.07

4. **Alternative Approach**:
   - Can estimate coordinates by averaging connected substations with known locations
   - Use network topology from GB_network.xlsx AC sheet
   - Weight by inverse of line length (OHL Length + Cable Length)

## Next Steps

1. Convert ERRO grid reference to lat/lon using proper conversion tool
2. Verify TEAL and TUMM coordinates using:
   - SSEN Transmission network maps
   - Ordnance Survey maps
   - Google Maps satellite imagery
3. Create manual coordinate override file for PyPSA-GB
4. Update bus location data in network model

## Web Sources

- [SSEN Transmission - Tealing Substation Extension](https://www.ssen-transmission.co.uk/projects/project-map/tealing-substation-extension/)
- [Mapcarta - Tealing Substation](https://mapcarta.com/W111528318)
- [Mapcarta - Tummel Substation](https://mapcarta.com/W495512692)
- [Wikipedia - Tummel hydro-electric power scheme](https://en.wikipedia.org/wiki/Tummel_hydro-electric_power_scheme)
- [SSE Renewables - Tummel hydro scheme](https://www.sserenewables.com/hydro/tummel-valley/)
- [Structurae - Errochty Dam](https://structurae.net/en/structures/errochty-dam)
- [Trove.scot - Errochty Power Station](https://www.trove.scot/place/171377)
- [Global Energy Observatory - Foyers Pumped Storage](https://globalenergyobservatory.org/geoid/3219)
- [SSE Renewables - Foyers hydro scheme](https://www.sserenewables.com/hydro/foyers/)
- [Wikipedia - Torness nuclear power station](https://en.wikipedia.org/wiki/Torness_nuclear_power_station)
- [Wikidata - Torness Nuclear Power Station](https://www.wikidata.org/wiki/Q1466261)

---

**Generated**: 2026-02-12
**Working Directory**: c:\Users\alyden\OneDrive - University of Edinburgh\Python\PyPSA-GB v0.0.1
