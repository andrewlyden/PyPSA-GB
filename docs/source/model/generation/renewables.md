# Renewable generation

Wind (onshore and offshore), solar PV, hydro (large reservoir and small run-of-river) and marine (tidal stream, wave,
tidal lagoon). This page covers which sites are in a scenario, where they connect, how their hourly output is made
and how the profiles are calibrated.

## What it does

Rule `add_renewables_to_network` (`rules/generators.smk`, script
`scripts/generators/integrate_renewable_generators.py`) adds renewable `Generator` components, each with an hourly
`p_max_pu` profile. The profiles come from rules in `rules/renewables.smk`:

| Rule | Script | Output |
|---|---|---|
| `prepare_renewable_site_data` | `scripts/renewables/prepare_renewable_site_data.py` | REPD site files by technology |
| `prepare_future_pipeline_sites` | `scripts/renewables/prepare_future_pipeline_sites.py` | named pipeline projects |
| `retrieve_cutout` | `scripts/utilities/prepare_cutouts.py` | ERA5 weather cutout `uk-{year}.nc` |
| `generate_renewable_profiles` | `scripts/renewables/map_renewable_profiles.py` | wind and solar output per site |
| `generate_marine_profiles` | `scripts/renewables/generate_marine_profiles.py` | tidal and wave profiles |
| `generate_hydro_profiles` | `scripts/renewables/generate_hydro_profiles.py` | hydro profiles |

Profiles are made once per weather year (`renewables_year`) and shared by every scenario with that weather year.

| Scenario type | Sites | Placement |
|---|---|---|
| Historical | REPD sites operational in the modelled year | each site at its own coordinates |
| FES GSP blocks | FES capacity by technology and GSP | GSP rows at their GSP; `Direct` rows spread over today's sites in their licence area |
| CP30 brownfield | named REPD projects, then FES proxy rows for the residual of CP30's regional totals | named projects at their connection; proxies at FES geography |
| FES long range | FES GSP blocks, with an offshore landing table | GSP rows on their GSP's demand node; offshore at declared landing buses |

## Method and assumptions

### Historical sites

REPD (the Renewable Energy Planning Database) lists projects with capacity, coordinates, status and operational date.
`prepare_renewable_site_data` keeps Great Britain's sites (Northern Ireland excluded), converts coordinates to WGS84
and splits them by technology; hydro is split at 10 MW into small (run-of-river) and large (reservoir). A historical
run keeps the sites operational by the end of the modelled year, or of the solve period when commissioning gating is
on, using evidence-backed date overrides where REPD is wrong.

REPD omits small embedded generation (rooftop solar, small wind) whose output ESPENI's demand series includes. With
`renewables.embedded.enabled`, the build reads ESPENI's embedded solar and wind, subtracts the modelled output of the
REPD solar and onshore wind already in the network, and adds `embedded_solar` and `embedded_wind` generators for the
residual. Demand is unchanged (see {doc}`../demand/historical`).

### Future sites: FES GSP blocks

FES gives capacity by technology and GSP. The mapping to carriers:

| FES technology or detail | Carrier |
|---|---|
| Offshore Wind | `wind_offshore` |
| Onshore Wind (>= 1 MW and < 1 MW) | `wind_onshore` |
| Solar Generation (large G99, small and domestic G98/G83) | `solar_pv` |
| Hydro, "Not pumped hydro" | `large_hydro` |
| Marine (tidal stream, wave, tidal lagoon) | `marine` |

A GSP row is located by its GSP id first, from the edition's own GSP table
(`data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`); island GSPs use
`data/network/GSP/gsp_declared_points.csv`; any other unplaced GSP stops the build. `Direct` rows (transmission
connected, no site) are spread over today's REPD sites of the same technology in the row's licence area, so new
capacity follows existing geography. Licence areas are latitude bands, or GSP polygons on the FES long-range chain.

On the FES long-range chain (`fleet.fes_supply`) GSP rows go on their GSP's demand node, and the FES `Direct` offshore
rows are replaced by a landing table for the year (`fleet.fes_supply.offshore_landing`): one row per landing bus,
each with the weather profile of a named project that lands there, else of the nearest profiled offshore site.

### Future sites: CP30 brownfield

CP30 says how much renewable capacity there is (ES1 nationally, ES5 by eight regions) but not which projects. The
brownfield build (`scripts/CP30/brownfield_fleet.py`) keeps CP30's totals and lets REPD decide what and where inside
them:

$$
\text{named}_{c,r} = \text{operational projects within their life} + \text{pipeline projects expected online},
\qquad
\text{residual}_{c,r} = \text{ES5}_{c,r} - \text{named}_{c,r}
$$

for each carrier $c$ (onshore wind, offshore wind, solar) and CP30 region $r$.

- **Named fleet.** Operational projects are retired at the end of their technical life
  (`fleet.brownfield.retirement_lifetimes_years`). Pipeline projects count when their REPD status is in `statuses`
  (by default Operational, Under Construction, Awaiting Construction). Declared exclusions
  (`fleet.brownfield.named_exclusions_file`) and projects whose CfD is terminated or absent are held out; their
  capacity passes to the residual.
- **Residual.** A positive residual is filled by FES proxy rows scaled to exactly the residual in that region. A
  negative residual is resolved by `overshoot_policy`: `trim_least_mature` removes pipeline projects, least mature
  first, never operational plant.
- **Offshore landing.** Named offshore farms are placed at their onshore landing bus from
  `offshore_landing_crosswalk`. A farm with no row falls back to the nearest bus and is flagged; large fallbacks move
  to a bus of at least 275 kV.
- **Small-scale solar.** With `fleet.spatial_proxy.brownfield.small_scale_solar_reservation`, FES small-scale
  (G98/G83) solar is held at its GSP node before the named pipeline fills the region, so rooftop solar keeps its
  GSP placement.
- **Distribution-size plant.** Named onshore wind and solar below their licence area's transmission threshold go to
  their own GSP's demand node (`scripts/utilities/gsp_nodes.py`).
- **Regional reconciliation.** After placement the totals are reconciled on the CP30 region of each row's placed bus,
  so per-region totals equal ES5 exactly.

Every decision (retirement, trimming, landing, exclusion) is written to an audit table.

```{figure} /_static/figures/supply/renewables_named_proxy_2030.png
:alt: Three panels of horizontal bars giving offshore wind, onshore wind and solar capacity in GW by CP30 region, split into named projects and proxy rows.
:width: 100%

Offshore wind, onshore wind and solar capacity of the 2030 build by CP30 region: named REPD projects (solid) and the FES
proxy rows that fill each region's residual (light). Each bar's total is CP30's ES5 figure for the region.
```

### Placement corrections (ETYS networks)

Nearest-bus matching can put a large site on a small busbar. Three corrections follow it:

| Correction | Key | What it does |
|---|---|---|
| Placement exclusions | `network.spatial_mapping.placement_exclusions` | onshore wind, solar and storage never go to DC hubs, offshore platforms or coordinate outliers |
| Weak-bus relief | `network.spatial_mapping.weak_bus_relief` | where placed capacity exceeds `max_loading` x the bus's branch capacity (MW per MVA), sites move, proxies first, to another busbar of the same substation, else the nearest bus of at least `min_target_voltage_kv` within `search_radius_km` |
| Export relief | `network.spatial_mapping.export_relief` | CP30 proxy onshore wind and solar move out of pockets that cannot export to the 275 kV grid over the branches the balancing stage constrains |

Rows placed on their GSP's demand node move only to another busbar of the same substation. Capacity, profiles and
regions do not change. On ETYS networks the BMU-to-node table also moves large named units to their transmission node.

### Weather profiles

Wind and solar output is computed with atlite from ERA5 reanalysis cutouts of the weather year, at each site's grid
cell:

| Carrier | atlite method |
|---|---|
| Onshore wind | `cutout.wind`, Vestas V112 3 MW power curve |
| Offshore wind | `cutout.wind`, NREL 5 MW reference turbine |
| Solar PV | `cutout.pv`, crystalline silicon panel, latitude-optimal orientation |

Profiles are written in MW per site and divided by capacity to give `p_max_pu`, then resampled to the model
timestep. Marine profiles are synthetic and cyclic (tidal harmonics, seasonal wave heights). Small hydro follows a
seasonal river-flow pattern.

A unit with no site profile of its own (an FES capacity block or a CP30 proxy) takes the capacity-weighted profile of
the sites of its carrier. With `renewables.proxy_profile_basis: regional`, used by CP30 and FES scenarios, the average
is over its own area: onshore wind and solar by region (eight GB regions, from the GSP polygons and
`data/network/ETYS/cp30_es5_region_by_gsp_group.csv`); offshore by sea basin, then region. It falls back to GB when
fewer than `proxy_profile_min_sites` sites or `proxy_profile_min_mw` MW are available. `proxy_profile_members` sets
which sites make up an area: `matched_units` (CP30) uses the build's own named units, placed in the network;
`all_sites` (FES) uses every site profile, located by the site's coordinates, since an FES scenario has no named units,
and gives each block the region of its own coordinates. With `proxy_profile_preserve_carrier_energy` one scalar per
carrier keeps the proxy fleet's energy equal to what the national profile gave, so the calibration below still holds.

On the national basis every block of a carrier follows one GB profile, so wind rises and falls together across GB.
Regional profiles keep the differences between places, which matter most for flows across the network's boundaries.
In the FES 2040 and 2050 builds (weather year 2013) the regional onshore wind profiles correlate between 0.49 (South
West) and 0.98 (South Scotland) with the GB average<!-- logs/integrate_renewable_generators_FES25_HT_2040.log: "Proxy wind_onshore profile group" lines -->,
the offshore basins between 0.66 and 0.90, and the solar regions between 0.88 and 0.99.

Leap years follow the calendar rule of {doc}`../demand/index` (29 February copied from 28 February when the weather year lacks
it).

### Performance factors

ERA5 and idealised power curves overstate real output. `p_max_pu` is multiplied by a performance factor per carrier,
the product of four parts:

$$
f_c = f^\text{ERA5 bias}_c \times f^\text{turbine curve}_c \times f^\text{wake}_c \times f^\text{availability}_c .
$$

The base values give 0.646 for offshore wind, 0.805 for onshore wind and 0.894 for solar (sources: Staffell and
Pfenninger 2016, doi:10.1016/j.energy.2016.08.004; Bloomfield et al. 2020, doi:10.1002/met.1858; Barthelmie et al.
2010, doi:10.1002/we.348). `renewables.performance_factors.factors` can set flat factors instead.

The factors must not include curtailment, which the balancing stage models: the uncurtailed (pre-balancing) output
should approximate metered output plus real curtailment. A registry
(`renewables.performance_factor_registry`, `data/market/support/renewable_performance_factor_registry.csv`) refines a
factor by resource region, year and month. A row may be pinned to the `weather_year` it was fitted on, since a factor
fitted on one weather year is wrong for another. CP30 builds calibrate their factors so that available energy matches
CP30's gross energy (ES1 generation plus curtailment). Factors describe the weather resource only; they never depend on
a site's support scheme.

```{figure} /_static/figures/supply/renewables_capacity_factors_2030.png
:alt: Two panels: monthly mean capacity factor in percent, and hourly duration curves of available output, for offshore wind, onshore wind and solar.
:width: 100%

Available output of the 2030 build's wind and solar fleets (capacity-weighted `p_max_pu`, performance factors applied,
before any curtailment), weather year 2013: monthly mean and the hourly duration curve.
```

### Large hydro

Large hydro is a reservoir (`renewables.hydro.large_hydro.reservoir_model`, enforced at solve time by
`scripts/solve/hydro_constraints.py`). Each unit has an hourly power ceiling `p_max_pu` and a water budget that
accrues `p_max_pu x p_nom x inflow_profile_factor` each hour, starts at `initial_storage_hours` of full output and is
capped at `reservoir_capacity_hours`. `month_end` sets the level at the end of each market run: at least the starting
level (`at_least_initial`), free, or on an annual path built from the inflow (`trajectory`).

Units with no site profile (CP30's hydro proxies) get a monthly inflow shape from observed transmission-metered hydro
output (ESPENI non-pumped hydro) in the weather year, scaled to CP30's hydro energy
(`scripts/renewables/large_hydro_inflow.py`). With `power_ceiling: availability` the inflow is kept in its own series
and `p_max_pu` is a flat availability. Missing pre-REPD stations can be filled from DUKES
(`dukes_legacy_supplement`).

### Aggregation by bus

With `renewables.aggregation.enabled`, generators of the listed carriers at the same bus are merged into one
(`scripts/generators/aggregate_renewable_generators.py`). `p_nom` is summed and the profile is the capacity-weighted
average, so capacity and available energy are conserved:

$$
p^\text{max,pu}_{b,c,t} = \frac{\sum_{g \in (b,c)} p^\text{nom}_g \, p^\text{max,pu}_{g,t}}{\sum_{g \in (b,c)} p^\text{nom}_g}.
$$

Support types in `preserve_support_types` stay as individual generators so their certificates and payments can be
settled station by station. Non-renewable generators are untouched. This is separate from network clustering
({doc}`../network/clustering`).

### Support labels

During integration every renewable generator is labelled `CfD`, `ROC` or `merchant` (`support.labels`):

- REPD sites use their CfD round and RO banding. A site operational before the RO closed to new accreditation
  (`roc_closure_date`) with no banding is taken as ROC-accredited at a default band per technology, for
  `roc_eligibility_years` from its operational date.
- FES rows are split CfD / merchant by `fes_cfd_fractions`.
- CP30 named projects take their contract from the CfD register.

The labels set dispatch costs and balancing bids, and drive the post-solve subsidy accounts; see
{doc}`/model/dispatch/bids_offers_support`.

## Data

| Input | File |
|---|---|
| REPD | `data/renewables/repd-q2-jul-2025.csv` |
| Offshore pipeline | `data/renewables/future_offshore_sites/offshore_pipeline.csv` |
| Site overrides | `data/renewables/historical_renewable_site_overrides.csv` |
| CfD round corrections | `data/market/lccc/cfd_round_corrections.csv` |
| ERA5 cutouts | `resources/atlite/cutouts/uk-{year}.nc` (downloaded by `retrieve_cutout`, or built with `Snakefile_cutouts`) |
| FES building blocks and GSP tables | `resources/FES/FES_{edition}_data.csv`, `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv` |
| CP30 totals | `resources/CP30/CP30_ES1.csv`, `CP30_ES5_regional.csv` |
| CP30 regions | `data/network/ETYS/cp30_es5_region_by_gsp_group.csv`, GSP polygons |
| Named fleet exclusions | `data/renewables/cp30_named_fleet_exclusions.csv` |
| Offshore landings | `data/network/ETYS/repd_offshore_landing_crosswalk.csv`; FES long range `fes{year}_offshore_landing.csv` |
| Performance factor registry | `data/market/support/renewable_performance_factor_registry.csv` |
| Hydro monthly shape | `data/renewables/hydro/espeni_npshyd_monthly_shape.csv` |

See {doc}`/reference/data/data_sources`.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `renewables_year` | (scenario) | weather year of the profiles |
| `renewables.performance_factors` | on, stacked factors | profile calibration |
| `renewables.performance_factor_registry.enabled` | `false` (CP30: `true`) | regional and weather-year factors |
| `renewables.embedded` | on | historical embedded solar and wind |
| `renewables.hydro.large_hydro` | reservoir model on | inflow factor, storage hours, `month_end` |
| `renewables.aggregation.enabled` | `false` | merge per (bus, carrier) |
| `renewables.proxy_profile_basis` | national (CP30 and FES: `regional`) | proxy profiles |
| `renewables.proxy_profile_members` | `matched_units` (FES: `all_sites`) | sites behind an area's profile |
| `fleet.spatial_proxy.brownfield` | CP30 study | named fleet on |
| `fleet.brownfield.*` | statuses, lifetimes, overshoot policy, landings | brownfield rules |
| `fleet.spatial_proxy.regional_allocation` | CP30 study | ES5 regions |
| `fleet.fes_supply.offshore_landing` | FES long-range studies | offshore landing table |
| `network.spatial_mapping.weak_bus_relief` | on, `max_loading` 1.5 | placement relief |
| `support.labels` | on | CfD / ROC / merchant labels |

Defaults are in `config/defaults/base/renewables.yaml`, `base/fleet.yaml`, `base/network.yaml` and
`base/support.yaml`; brownfield settings in `config/defaults/method/brownfield.yaml`.

## Validation

- CP30 builds compare available energy (before curtailment) by carrier with CP30 gross energy, and capacity with ES1
  and ES5 (rule `validate_CP30_network`); dispatched energy and curtailment are compared after the solve (rules
  `validate_CP30_stage1`, `validate_CP30_dispatch`).
- The CP30 locational checks (rule `validate_CP30_locational`) test the ES5 regional split, the offshore landings, that
  distributed plant sits at its own GSP's node and that no plant is stranded behind branches the balancing stage
  constrains; the fleet-location check
  (`validate_CP30_fleet_location`) compares capacity by region, boundary side and named offshore landing.
- `scripts/generators/calibrate_wind_profiles.py` compares ERA5 wind totals with ESPENI metered output.
- Historical output and curtailment are compared with Elexon and ESPENI in {doc}`/studies/validation_2020`.

## Limitations

- One turbine power curve per carrier; no hub-height, rotor or turbine-model detail per site.
- Proxy units share regional average profiles, so their output is smoother than real sites'.
- The FES `Direct` placement assumes new capacity follows today's geography.
- Marine and small hydro profiles are synthetic, not weather-driven.
