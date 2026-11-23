# Storage

Grid batteries, domestic batteries, pumped hydro and long-duration storage (liquid air, compressed air and CP30's
long-duration category). This page covers where storage comes from, where it connects and the parameters it runs
with. Consumer flexibility, also built from storage units, is shifted demand and has its own page ({doc}`consumer`).

## What it does

Rule `add_storage_to_network` (`rules/storage.smk`, script `scripts/storage/add_storage.py`) adds PyPSA
`StorageUnit` components with power `p_nom` (MW), duration `max_hours` (h, so energy is `p_nom x max_hours` MWh),
charge and discharge efficiencies, standing loss and a dispatch cost. Rule `extract_storage_from_repd`
(`scripts/storage/storage_from_repd.py`) prepares the REPD storage sites.

| Scenario type | Storage power and energy | Placement |
|---|---|---|
| Historical | REPD sites commissioned by the modelled year | each site's coordinates |
| FES GSP blocks | FES storage building blocks (Srg_BB001 grid batteries, Srg_BB002 domestic batteries, Srg_BB003 pumped hydro, Srg_BB004 other) by GSP and `Direct` | GSP rows at their GSP; `Direct` rows over REPD storage sites |
| CP30 brownfield | CP30 ES1 storage power and energy by type | existing pumped hydro and committed projects at their substations; the rest on FES proxy rows, split by ES5 region |
| FES long range | FES ES1 power and energy by type | as FES GSP blocks, `Direct` by licence area |

## Method and assumptions

### Historical storage

REPD lists storage projects with capacity, coordinates, status and dates. Only sites operational by the modelled year
are kept, so a 2020 run has 2020's fleet, not today's. Durations and efficiencies take technology defaults where REPD
gives none.

### Future storage: FES

FES gives storage power by building block and GSP. GSP rows are located by GSP id from the edition's own committed GSP
table (`data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`); a row matching no GSP raises
`GspLocationError` rather than being dropped, and a missing table fails the build. `Direct` rows are spread over REPD
storage sites; on the FES long-range chain they are split by licence area and placed on that licence's buses.
Pumped hydro takes FES's own projection (Srg_BB003) and is placed over today's storage sites.

On the FES long-range chain (`fleet.fes_supply.storage`) storage is built by type at FES ES1 power and energy, with
declared durations where ES1's are not physical (for example compressed air at 20 h) and pumped hydro at declared
sites.

### Future storage: CP30

CP30 ES1 gives national storage power (MW) and energy (GWh) per type. The model maps batteries to `Battery` and CP30's
long-duration storage (mostly existing pumped hydro) to carrier `LAES`.

1. **Existing pumped hydro.** Dinorwig, Cruachan, Ffestiniog and Foyers are placed at their own substations with their
   real power and energy (`renewables.hydro.pumped_hydro.existing_sites`). Only the remainder of CP30's long-duration
   storage is new build.
2. **Committed projects.** Named new-build long-duration projects each CP30 pathway counts sit at their TEC
   substations (`data/storage/cp30_committed_ldes_sites.csv`).
3. **Proxy new build.** The rest is spread over FES proxy rows at the residual duration, so CP30's national power and
   energy stay exact.
4. **Regional split.** With `fleet.spatial_proxy.storage_regional_allocation`, new-build rows are rescaled region by
   region to CP30 ES5's shares (batteries follow ES5 "Batteries", other storage "Storage" less "Batteries"), keeping
   fixed sites and each row's duration (`scripts/CP30/storage_regions.py`). The split is reconciled again on the region
   of each row's placed bus once placement is done.

```{figure} /_static/figures/supply/storage_2030_2050.png
:alt: Two panels of paired horizontal bars giving storage power in GW and energy in GWh by type for 2030 and 2050.
:width: 100%

Storage power and energy by type in the 2030 build (Clean Power 2030) and the 2050 build (FES 2025 Holistic Transition).
In 2030, CP30's long-duration category is split into today's pumped hydro and new build (committed projects and
proxies). Consumer flexibility is left out.
```

### Placement on ETYS networks

After nearest-bus matching:

- **GSP nodes.** A battery below its licence area's transmission threshold (10 MW in the north of Scotland, 30 MW in
  the south, 100 MW in England and Wales), and every FES GSP-level row, goes to the node that carries its GSP's demand
  (`scripts/utilities/gsp_nodes.py`).
- **Named sites.** A row with a site code (existing pumped hydro, committed projects) goes to the busbar of that
  substation with most headroom.
- **Branch capacity.** No node (bus sections and couplers counted as one) may hold more storage power than
  `storage.placement.max_loading` x the rating of the lines, transformers and links leaving it. The excess moves to
  the same substation's other busbars, else to the nearest bus of at least `min_target_voltage_kv` within
  `search_radius_km`. Storage stranded on a bus with no network of its own moves in full.
- **Weak-bus relief.** Storage proxies also get the generators' weak-bus relief
  (`network.spatial_mapping.weak_bus_relief`).

```{figure} /_static/figures/supply/storage_map_2030.png
:alt: Map of Great Britain with pink circles for grid batteries and dark blue circles for long-duration storage at each node, sized by power.
:width: 70%
:align: center

Grid batteries and long-duration storage at their nodes in the 2030 build, sized by power. Batteries are spread over GSP
demand nodes; long-duration storage sits at the pumped-hydro sites, the committed projects and the proxies.
```

### Parameters

| Parameter | Rule | Key |
|---|---|---|
| Round-trip efficiency $\eta_\text{rt}$ | each leg is $\sqrt{\eta_\text{rt}}$ | `storage.round_trip_efficiencies` |
| Standing loss | share of stored energy lost per hour | `storage.standing_losses` |
| Dispatch cost | GBP/MWh discharged: variable O&M, wear and a tie-break against charging and discharging at once | `storage.dispatch_costs` |
| Duration | from REPD, FES ES1 or CP30 energy / power | (data) |
| Pumped hydro state of charge | starts at `initial_soc_fraction` of capacity; floor `min_soc_fraction` (higher in the wholesale stage, `wholesale_min_soc_fraction`) | `renewables.hydro.pumped_hydro` |

Base round trips: grid and domestic batteries 0.8464 (0.92 per leg), pumped hydro 0.7569 (0.87 per leg), liquid air
0.55, CP30 long-duration 0.70. Units are not cyclic by default. In rolling market solves the state of charge carries
from one window to the next (`market.storage_soc`); see {doc}`../dispatch/wholesale`.

## Data

| Input | File |
|---|---|
| REPD | `data/renewables/repd-q2-jul-2025.csv` |
| FES building blocks | `resources/FES/FES_{edition}_data.csv` |
| FES GSP tables | `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv` |
| CP30 totals | `resources/CP30/CP30_ES1.csv`, `CP30_ES5_regional.csv` |
| CP30 committed long-duration projects | `data/storage/cp30_committed_ldes_sites.csv` |
| FES long-range pumped hydro | `data/storage/fes2050_pumped_hydro_sites.csv` |
| GSP polygons, `Dem_per_node` | `data/network/GSP/GSP_regions_27700_20250109.geojson`, `data/network/ETYS/GB_network.xlsx` |

See {doc}`/reference/data/data_sources`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `storage.round_trip_efficiencies` | see above | by carrier |
| `storage.dispatch_costs` | 1.0 GBP/MWh (pumped hydro 0.5, LAES 1.5) | by carrier; also a floor in the wholesale solve |
| `storage.standing_losses` | batteries 0.00005 per hour, others 0 | by carrier |
| `storage.placement` | on, `max_loading` 1.0, 40 km, 132 kV | capacity-aware placement |
| `renewables.hydro.pumped_hydro` | four existing sites; SoC fractions | pumped hydro sites and limits |
| `fleet.spatial_proxy.storage_regional_allocation` | CP30 study | ES5 regional split |
| `fleet.fes_supply.storage` | FES long-range studies | durations, pumped hydro sites |

Defaults are in `config/defaults/base/storage.yaml` and `base/renewables.yaml`.

## Validation

- CP30 builds compare storage power and energy with ES1, and the regional split with ES5 (rules
  `validate_CP30_network` and `validate_CP30_locational`, which also checks the long-duration sites).
- The fleet-location check compares storage by region and boundary side (rule `validate_CP30_fleet_location`).
- After the solve, storage throughput is reported against ES1 as information only, since a linear model's storage
  can cycle more than a unit-commitment model's.
- `scripts/market/diagnose_storage_degeneracy.py` diagnoses simultaneous charging and discharging.

## Limitations

- No degradation with cycling beyond a flat dispatch cost.
- One round-trip efficiency per carrier; no state-of-charge-dependent efficiency.
- Domestic batteries are modelled as grid storage at their GSP, dispatched by the system, not by household tariffs.
