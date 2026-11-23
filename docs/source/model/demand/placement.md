# Placement on the network

Every demand build ends with annual energy and an hourly series per Grid Supply Point (GSP), or for historical years
one GB series. This page covers how that demand reaches the network's buses, and the two adjustments made where a bus
cannot carry it. The same GSP-to-node rule places distribution-connected plant.

## Method and assumptions

### From GSP to network node

FES and ETYS name GSPs differently: FES uses GSP names ("Abham"), ETYS uses node ids. `scripts/demand/load.py` maps a
GSP name to its GSP id with the FES edition's GSP metadata
(`data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`), then to ETYS nodes with the `Dem_per_node` table of
`data/network/ETYS/GB_network.xlsx`, which gives each node's share of its GSP group's demand. A GSP without a usable
row is placed in this order:

1. a declared location in `data/demand/gsp_location_overrides.csv`;
2. a spelling variant of the name that matches the metadata;
3. the site of the GSP's metadata code;
4. the demand node nearest the GSP's centroid.

On Reduced and Zonal networks, GSP demand is summed onto the regional buses through their own mapping. Historical
years, which start from one GB series, use fixed weights instead ({doc}`historical`).

```{figure} /_static/figures/supply/demand_map_2030.png
:alt: Map of Great Britain with a blue circle at each network node, sized by its annual electricity demand in 2030.
:width: 70%
:align: center

Annual electricity demand at each node of the 2030 build (Clean Power 2030, ETYS network): fixed load plus the
consumption of the consumer-flexibility units, 314 TWh in all, electrolysis excluded.
```

### Buses in isolated sub-networks

A bus in a sub-network with no connection to the main grid could only meet its demand by shedding load. On the ETYS
network that demand is moved, before any other adjustment, to the nearest bus of the main network
(`redirect_demand_from_isolated_subnetworks`).

### Buses too small for their peak

Some low-voltage (132 kV and below) demand buses sit behind transformers or lines too small for their peak. Two rules
move the excess, chosen by `capacity_redistribution`. Both measure a bus's capacity as the smaller of the transformer
and the line capacity into it.

- **`pro_rata`** (the default): at the hour of national peak, a bus whose demand exceeds 0.9 of its capacity is capped
  there, and the excess goes to the other demand buses that can take it (275 kV and above, or with more than 10 MW of
  headroom) in proportion to their demand, repeated up to ten times. The bus's whole series is scaled, so the cap is a
  share of annual energy, not a limit hour by hour.
- **`same_site`**: a bus keeps demand up to `capacity_safety_margin` x its capacity in every hour; only the part above
  it moves, to a bus of the same substation at or above `min_supergrid_kv` (275 kV). One pass, lowest voltage first, so
  demand stays at its own site. Every move is logged with its energy and peak.

The CP30 build (`demand.build.capacity_redistribution`) and the FES long-range chain
(`demand.disaggregation.fes.capacity_redistribution.mode`) use `same_site`; other FES builds and historical ETYS builds use
`pro_rata`.

### The same rule for distribution-connected plant

Supply uses the same GSP-to-node rule. `scripts/utilities/gsp_nodes.py` joins a point to NESO's GSP region polygons,
maps the polygon's GSP ids through `Dem_per_node`, and resolves them on the network as the demand build does.
Distribution-connected plant (below its licence area's transmission threshold: 10 MW in the north of Scotland, 30 MW in
the south of Scotland, 100 MW in England and Wales) is placed on the node that carries its GSP's demand.

## Data

| Input | File | Used for |
|---|---|---|
| FES GSP metadata | `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv` | GSP name to id |
| `Dem_per_node` | `data/network/ETYS/GB_network.xlsx` | GSP id to ETYS node |
| GSP overrides | `data/demand/gsp_location_overrides.csv` | declared locations of unlisted GSPs |
| GSP polygons | `data/network/GSP/GSP_regions_27700_20250109.geojson` | GSP of a location (distribution-connected plant) |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `demand.build.capacity_redistribution` | `same_site` under `method/brownfield.yaml` | CP30 build: `same_site` or `pro_rata` |
| `demand.build.capacity_safety_margin` | `0.90` | share of a bus's capacity it keeps (`same_site`) |
| `demand.build.min_supergrid_kv` | `275.0` | lowest voltage a moved demand may go to (`same_site`) |
| `demand.disaggregation.fes.capacity_redistribution.mode` | set by the FES long-range studies | the same rule for the FES long-range chain (absent: `pro_rata`) |

## Validation

- Every placement step checks that total demand is conserved (`assert_demand_conserved` in `load.py`).
- The CP30 locational checks (rule `validate_CP30_locational`) test each node's peak load plus electrolysis against what
  its circuits can import at the ratings the balancing stage enforces ({doc}`cp30`).

## Limitations

- GSP shares are fixed for the modelled year. Where demand is not built from components (historical years and FES
  GSP-block builds), every node takes the same national hourly shape.
- `pro_rata` sizes its caps at the national peak hour, not at each bus's own peak, and can move demand far from its
  site; `same_site` avoids both.
