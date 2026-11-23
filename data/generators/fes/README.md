# FES declared stations

`station_list.csv` names the sites that take the capacity FES puts at "Direct" transmission
connections with no site. Rule `build_fes_station_schedule` (`rules/fes_long_range.smk`,
script `scripts/generators/build_fes_station_schedule.py`) filters it by year and writes
`resources/FES/fes2025_<pathway>_<year>_station_schedule.csv`, which `scripts/generators/fes_supply.py`
reads when a scenario sets `fes_supply.thermal.station_schedule`. The 2050 schedule it writes is
byte-equal to the committed `fes2050_station_schedule.csv` it replaces.

Columns:

- `from_year`, `to_year`: the years a station is open (blank is open-ended). For named projects
  `from_year` is the first full calendar year at the NESO TEC register's cumulative capacity (MW
  Effective From of the last stage, 2 September 2025); `dates_source` gives the evidence.
- `allocation`: `fixed` (the station's `capacity_mw`), `share` (the FES total less the fixed rows,
  in proportion to `share_weight_mw`, else `capacity_mw`), `residual` (one row takes the rest) or
  `es1_capacity` (the year's ES1 capacity of `es1_carrier`, taken out of the Direct row: BECCS).
- `source` and `notes` may hold `{direct_mw}`, `{capacity_mw}`, `{remainder_mw}`, `{year}` and
  `{pathway_code}`, filled per year.
- `basis`: `named_project` (a project the TEC register or a published programme names),
  `declared_site` (a site chosen here, with its reason) or `site_share` (a declared set of sites
  sharing the residual by existing capacity or TEC capacity).

HT 2050 (FES 2025 Holistic Transition):

| Carrier | FES block | Licence area | MW | Rows |
| --- | --- | --- | ---: | --- |
| nuclear | Gen_BB020 | NGET | 14,200 | Hinkley Point C, Sizewell C (TEC register), Trawsfynydd SMR (TEC), five declared SMR sites |
| CCS Gas | Gen_BB009 ("CCGTs (non CHP)", gas CCS in 2050) | NGET | 18,885 | Net Zero Teesside Power, Stallingborough CCS (TEC), residual over 15 existing CCGT sites in CO2 clusters |
| CCS Gas | Gen_BB009 | SHETL | 3,320 | Peterhead Carbon Capture (TEC) plus Acorn units at Peterhead |
| H2 (turbines) | Gen_BB023 | NGET | 17,192 | Bute Hydrogen projects 1-6 and Ferrybridge D (TEC sites), shares of TEC MW |
| H2 (turbines) | Gen_BB023 | SHETL / SPTL | 1,300 / 605 | Peterhead; Grangemouth |
| CCS Biomass | Gen_BB010 (part) | NGET | 2,662 | Drax BECCS (ES1 row 174); the other 323 MW of BB010 Direct stays biomass |

Before 2050 Gen_BB009 also holds unabated CCGTs: the gas CCS total is the block's Direct capacity
times the year's ES1 CCS share (HT 2040: 19,625 of 25,033 MW). A year whose FES Direct capacity has
no open station (HT 2040: 20 MW of BB009 in SPTL) fails until a station is added here.

The build refuses a schedule whose rows for a (carrier, licence area) do not sum to the FES Direct
total (BECCS may not exceed it).

`data/storage/fes2050_pumped_hydro_sites.csv` gives the pumped-hydro sites over which FES Srg_BB003
Direct capacity is spread in each licence area (existing REPD sites plus declared pipeline sites;
capacities are share weights only).
