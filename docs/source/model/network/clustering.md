# Network clustering

## What it does

Clustering reduces a full ETYS network to fewer buses while keeping the placement of every asset. The solve is then
smaller and faster, at the cost of the flows inside each cluster. It is the middle ground between the full ETYS model
and the fixed Reduced and Zonal models ({doc}`topology`).

Clustering happens **after** every component has been placed on the full network. Demand, generators, storage,
hydrogen and interconnectors are attached to their ETYS buses first, so placement uses the full detail; each then
moves to its bus's cluster.

```{mermaid}
flowchart TD
    A["{scenario}_network.nc<br/>(full ETYS)"] --> B["demand, renewables, thermal,<br/>storage, hydrogen, interconnectors"]
    B --> C["cluster_network<br/>(busmap + aggregation)"]
    C --> D["clustered network<br/>+ {scenario}_clustering_busmap.csv"]
    C --> F["validate_network_clustered<br/>(HTML report)"]
    D --> E["finalise and solve"]
```

The rules are in `rules/network_clustering.smk`; the code is `scripts/network_clustering/cluster_network.py`.

## Method and assumptions

Clustering has two parts: a **busmap**, which names the cluster of every bus, and **aggregation**, which merges the
components of each cluster. The busmap comes from one of five methods.

| `method` | Busmap | Typical use |
|---|---|---|
| `spatial` | each bus joins the polygon that contains it, from a GeoJSON file (for example GSP regions) | regional detail tied to a published geography |
| `kmeans` | k-means on bus coordinates, `n_clusters` clusters | quick reductions to a chosen size |
| `hierarchical` | agglomerative clustering on coordinates (`linkage`, `distance_metric`) | as k-means, deterministic |
| `busmap` | an explicit bus-to-cluster table (CSV with the bus as index and the cluster in the first column, or JSON) | a fixed, reviewed mapping |
| `custom` | a Python function `function_name(network, config)` in `function_path` returning the busmap | busmaps that need network logic (the boundary-respecting and NESO-zone maps below) |

In every method, buses that are not GB AC buses stay individual: foreign interconnector endpoints and carrier buses
(hydrogen, DC converter nodes) are never merged with GB AC buses. In the `spatial` method, a bus outside every
polygon (an offshore platform, a coastal bus) is assigned by `fallback`: `nearest_centroid` (the default; can send an
offshore bus to an inland region), `nearest_neighbor` (the cluster of the nearest assigned bus) or `default_cluster`.

### Aggregation

The busmap is applied with PyPSA's `get_clustering_from_busmap`:

| Component | What happens |
|---|---|
| Bus | one bus per cluster; `x`, `y` (and `lon`, `lat`) are the means of its members |
| Lines between the same pair of clusters | merged into one line: `s_nom` and the seasonal ratings `s_nom_winter/spring/summer/autumn` summed; `r` and `x` combined in parallel, $1/x = \sum_i 1/x_i$; `b` and `g` summed; `length` the mean; voltage the maximum |
| Lines inside one cluster | removed: their flow is no longer modelled |
| Transformers | not carried by the busmap clustering: convert them to lines first (below), or they are dropped |
| Links | kept one by one and re-attached to their clusters; a link with both ends in one cluster is dropped (this includes an internal HVDC link whose two converters fall in one cluster) |
| Generators, loads, storage units, stores | kept one by one, moved to their bus's cluster, time series unchanged |

Generators and storage are not merged by the clustering itself, so every unit keeps its own costs, profile and
support attributes. Merging is a separate step (component aggregation, below, and renewable aggregation,
{doc}`../generation/renewables`). The generator, load and storage entries in a strategy dictionary have no effect with
this clustering path.

Two details keep the reduced impedances right:

- **Per-unit aggregation.** Line impedances are combined in per unit on a 1 MVA base and converted back to ohm and
  siemens at the clustered line's `bus0` voltage. Combining raw ohms across voltage levels would make a 132 kV line
  merged with a 400 kV pair several times too stiff. The parallel rule for `r` and `x` and the sum for `b` and `g` are
  enforced whatever a strategy file says. A clustered line with zero or missing reactance gets the coupler value
  (0.01 % on 100 MVA).
- **Transformers as lines.** With `convert_transformers_to_lines: true` each transformer becomes a line with the same
  per-unit impedance and rating before clustering. Transformers inside a cluster then vanish with its internal lines,
  and those between clusters merge with the lines in parallel. Tap ratios and phase shifts are not represented.
  Set this on every ETYS clustering: without it the transformers, which join the voltage levels, are lost.

PyPSA computes clustered line lengths with the haversine formula, which needs degrees, so bus coordinates are
converted from OSGB36 to WGS84 for the clustering and back afterwards. The clustered network keeps OSGB36 `x` and `y`,
like every other network ({doc}`topology`).

A scenario can replace the bus or line strategy dictionary with `network.clustering.strategies`. The replacement is
per component: a `line:` entry replaces the whole default line dictionary, so it must list every line column that
differs within a cluster (ETYS networks carry seasonal ratings and provenance columns on lines, and PyPSA refuses to
aggregate a column without a rule). The FES long-range study templates show a complete line dictionary.
`config/clustering.yaml` also defines named strategy sets (`default`, `aggressive`, `economic`) for reference; the
clustering rule applies only a scenario's own `strategies`.

### Presets

`config/clustering.yaml` holds presets. A scenario sets `network.clustering` to a preset name, an inline
configuration, or a preset extended with its own keys:

```yaml
network:
  clustering: gsp_spatial          # a preset by name

network:
  clustering:                      # an inline configuration
    method: kmeans
    n_clusters: 100

network:
  clustering:                      # a preset extended
    preset: gsp_spatial
    fallback: nearest_neighbor
```

Naming a preset or a `method` switches clustering on; `network.clustering.enabled` (false in the base) matters only
for a block with neither. The loader that resolves this is `config_loader.resolve_clustering`.

| Preset | Method | Notes |
|---|---|---|
| `gsp_spatial` | `spatial` on GSP regions (`data/network/GSP/GSP_regions_27700_20250109.geojson`), transformers converted to lines | the recommended general preset: about 300 clusters at the granularity of FES regional data |
| `kmeans_10`, `kmeans_5` | `kmeans`, 10 or 5 clusters, fixed seed | fast tests |
| `hierarchical_15` | `hierarchical`, 15 clusters, Ward linkage | |
| `admin_regions` | `spatial` on administrative regions | expects `data/zone/uk_admin_regions.geojson`, which is not shipped |
| `custom_busmap`, `zonal_busmap` | `busmap` | expect busmap CSVs that are not shipped; supply your own |

### Busmaps for the secured boundaries

On an ETYS network the main transfer limits are the secured boundaries: sums of flow over the branches that cross
each ETYS boundary line ({doc}`constraints`). Clustering interacts with them: a cluster that straddles a boundary
turns crossing branches into internal ones, and the boundary constraint stops seeing that transfer. Two custom
busmaps avoid this.

**Boundary-respecting GSP busmap** (`scripts/network_clustering/boundary_busmap.py`).

1. Every GB AC bus joins the GSP region that contains it; a bus outside every region joins the nearest region by
   polygon distance (not by centroid, which can move an offshore bus inland).
2. Each boundary polyline is extended at both ends far outside GB, so it divides the plane. A bus's side is the
   parity of crossings between the bus and a fixed far-away point. A branch then crosses the boundary exactly when its
   two buses are on different sides, the same test the solve uses.
3. A GSP region whose buses are not all on one side of every boundary is split by side; the cluster is named
   `<GSP>__<boundary>-<side>`.
4. Optionally (`max_clusters`), the smallest cluster is merged into the neighbouring cluster with the same side
   signature and the largest connecting rating, until the count is at the target. Merging never joins different
   sides, so it cannot create a straddle.

The busmap committed as `data/clustering/fes2050_boundary_busmap.csv` (300 GB AC clusters, with a summary JSON
beside it) was built this way on an ETYS base network and records each bus's GSP region, side code and any merge.
It is a provenance and regression reference; it can also be used directly with `method: busmap`.

**NESO zone busmap** (`scripts/network_clustering/neso_zone_busmap.py`). Collapses GB AC buses onto NESO's 17
economic zones (the CP.08 zones of the CP30 workbook, `data/CP30/validation_references/neso_cp08_zones.geojson`),
zoned exactly as the CP30 balancing check zones a bus: by polygon, with offshore buses followed through the network
to the nearest on-land bus, then reconciled against the eleven ETYS boundaries that define the zones. Inter-zonal
transfer is then limited by those secured boundaries. The FES long-range studies use this busmap. Its settings are
under `network.clustering.neso_zone_busmap` (`remote_km`, `cutoff`), and it reads
`network.clustering.secured_capability.include` for internal HVDC links whose crossing must be declared (a straight
link path that meets a boundary line an even number of times although its ends are on opposite sides).

```{figure} /_static/figures/network/clustering_2050.png
:alt: Two maps of Great Britain: on the left the full ETYS network with the NESO zone borders dashed, on the right 17 numbered zone nodes joined by lines whose width shows their summed rating, with HVDC links dashed.
:width: 100%

The FES 2050 Holistic Transition network before and after clustering with the NESO zone busmap. The full ETYS
network (left) becomes one bus per NESO economic zone (right), placed at the mean position of its members; lines
between two zones are merged into one with the summed rating, and the internal HVDC links keep their own ratings.
```

On a clustered network the crossing branches carry the summed ratings of the circuits they replace, which may be
below the boundary capability NESO publishes for a future year (the capability assumes reinforcements that Appendix B
does not yet list). The optional `branch_scaling` of the secured boundaries raises the crossing ratings so that the
boundary, not the circuits, binds ({doc}`constraints`).

A third purpose-built map, `scripts/network_clustering/consumer_bill_14_zone.py`, keeps 14 zones around the
Scotland-England boundary and the main demand, generation and coastal areas. It is a screening network for bill
backtesting (scenario `Historical_2024_ETYS_14zone_market`), not a substitute for ETYS in constraint studies.

### Component aggregation

`network.component_aggregation` is separate from clustering and runs for every scenario when the finished network is
finalised (`scripts/solve/finalize_network.py`). It merges components at the same bus whose attributes and time series
are identical, summing their capacities. It is off in the base. Renewable generators have their own aggregation by bus
and carrier ({doc}`../generation/renewables`).

### Reduced and Zonal models

The Reduced (29-bus) and Zonal (17-zone) models are separate network models with their own data, not clusterings of
the current ETYS network ({doc}`topology`). Choose them with `network_model`. They do not carry seasonal ratings,
licensee or voltage information, so they cannot use the secured boundaries or the balancing-stage voltage floor.

## Data

| File | Content |
|---|---|
| `config/clustering.yaml` | presets, named aggregation strategy sets, method reference |
| `data/network/GSP/GSP_regions_27700_20250109.geojson` | GSP region polygons (OSGB36) |
| `data/network/ETYS/etys_boundaries_mar25_osgb36.geojson` | ETYS boundary lines (NESO boundary GIS, converted to OSGB36) |
| `data/CP30/validation_references/neso_cp08_zones.geojson` | NESO's 17 economic zones |
| `data/clustering/fes2050_boundary_busmap.csv` and `_summary.json` | committed boundary-respecting busmap |

Outputs (in `resources/network/`): `{scenario}_network_clustered_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc`
and `{scenario}_clustering_busmap.csv`; the validation report
`resources/validation/{scenario}_clustered_network_validation_report.html`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `network.clustering` | `{enabled: false}` | preset name, inline configuration, or `preset:` plus keys |
| `network.clustering.method` | | `spatial`, `kmeans`, `hierarchical`, `busmap` or `custom` |
| `network.clustering.n_clusters` | | `kmeans` and `hierarchical` |
| `network.clustering.boundaries_path`, `cluster_column`, `bus_crs`, `boundary_crs`, `fallback` | | `spatial` |
| `network.clustering.busmap_source` | | `busmap` |
| `network.clustering.function_path`, `function_name` | | `custom` |
| `network.clustering.convert_transformers_to_lines` | false | merge transformers as lines |
| `network.clustering.strategies` | built-in defaults | per-component aggregation rules |
| `network.clustering.neso_zone_busmap`, `network.clustering.secured_capability.include` | | NESO zone busmap settings |
| `network.component_aggregation.enabled` (and `include_*`) | false | merge identical components at a bus |

Example scenarios in `config/scenarios.yaml`: `HT35_clustered` (`gsp_spatial`), `EE50_clustered` (k-means, 100
clusters), `HT30_event_response` (`kmeans_10`), `Historical_2024_ETYS_14zone_market` (custom 14-zone map).

## Validation

Rule `validate_network_clustered` (`scripts/network_clustering/validate_network.py`) runs PyPSA's consistency check
on the clustered network and writes an HTML report of its contents (buses, lines, generators and so on, with a map).
The clustered network records the method, its configuration and the bus counts before and after in `network.meta`,
and the busmap is written beside it, so every original bus can be traced to its cluster.

To judge whether a clustering is fine enough for a question, solve one representative period on both the full and the
clustered network and compare the quantities that matter (system cost, boundary flows, curtailment by region).

## Limitations

- Flows, congestion and price differences inside a cluster are lost. Constraint costs are lower on a clustered
  network unless its binding limits (the secured boundaries) are kept intact, which is what the boundary-respecting
  busmaps are for.
- An internal HVDC link with both converters in one cluster is dropped with the internal lines. Check the busmap
  keeps the two ends of each link apart where the link matters.
- Clustered line ratings are sums of parallel ratings, which overstate the transfer of a corridor whose circuits do not
  share load evenly.
