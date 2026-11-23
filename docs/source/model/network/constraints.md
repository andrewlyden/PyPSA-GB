# Network constraints

## What it does

This page describes how the transmission network limits power flow in a solve: which ratings apply, how they change
by season, which branches are relaxed, how boundary transfer limits are added, and how a delayed-network case is
built for the balancing stage. The network itself is described in {doc}`topology`.

| Limit | Acts on | Applies in |
|---|---|---|
| Thermal rating `s_nom` (winter rating) | each line and transformer | every network-constrained solve |
| Seasonal ratings (monthly factor) | each line with published seasonal ratings | every network-constrained solve |
| Voltage floor (relax branches below 220 kV) | lines and transformers with an end below the floor, except Scottish transmission | every network-constrained solve |
| Line rating overrides | named branches | every network-constrained solve |
| Secured boundary limits (ETYS capability, seasonal and outage factors) | signed sum of flows across each ETYS boundary | where `network.transmission.outage_schedule` selects them |
| NESO day-ahead boundary limits | signed sum of flows across each mapped NESO boundary | historical years, where selected |
| Network delivery case (works removed or reverted, boundary capability held back) | topology and boundary limits | the balancing stage, and a central-dispatch Stage 1 |

"Network-constrained solve" means a single least-cost solve on the network ({doc}`../dispatch/single_solve`), the
balancing stage of the two-stage market ({doc}`../dispatch/balancing`) and a central-dispatch Stage 1
({doc}`../dispatch/central_dispatch`). The wholesale stage of the market is a copperplate and sees none of these
limits ({doc}`../dispatch/wholesale`).

## Method and assumptions

All network-constrained solves use PyPSA's linearised (DC) load flow. A line or transformer carries flow
$|f_\ell(t)| \le s_{\text{max,pu},\ell}(t)\, s_{\text{nom},\ell}$; an internal HVDC link carries
$-p_\text{nom} \le p(t) \le p_\text{nom}$. Losses and reactive power are not modelled.

The preparation steps run in this order before the optimisation (`scripts/solve/solve_network.py`,
`scripts/market/solve_balancing.py`):

```{mermaid}
flowchart TD
    A["Branch units check<br/>(refuses a network not in PyPSA units)"] --> B["Network delivery case<br/>(balancing stage and central dispatch only)"]
    B --> C["Voltage floor, then line rating overrides"]
    C --> D["Outage schedule<br/>(per-branch s_max_pu, or boundary limits deferred to the model)"]
    D --> E["Optional floors and scaling of ratings"]
    E --> F["Solve period"]
    F --> G["Seasonal line ratings<br/>(monthly factor on s_max_pu)"]
    G --> H["Optimise, with the boundary constraints<br/>added to the linear model"]
```

The single solve applies the optional floors and scaling before the overrides; the order matters only if you use both.

### Thermal ratings

Every branch's `s_nom` is its published **winter** rating in MVA (ETYS Appendix B; for lines the build also keeps
`s_nom_winter`). Transformers have one rating all year. Couplers and bus sections are published at 9,999 MVA and never
bind. The Reduced network's lines carry a single rating each; the Zonal network's links carry their `p_nom`.

### Seasonal ratings

Overhead lines carry more current in cold weather, so ETYS publishes four ratings for most circuits. The build keeps
them on each line as `s_nom_winter`, `s_nom_spring`, `s_nom_summer` and `s_nom_autumn`
(`scripts/network_build/branch_attributes.py`), with `seasonal_rating_source` saying where they came from:

| Source | Lines | Seasonal ratings |
|---|---|---|
| `etys` | circuits in the SHE, SPT and NGET sheets | as published |
| `class_median_ratio` | OFTO circuits and other edges with no published seasons | `s_nom` x the median season/winter ratio of published lines at the same voltage |
| `relaxed` | couplers and other lines at or above 9,999 MVA | 9,999 MVA every season |
| `winter_only` | transformers | one rating |

At solve time `apply_seasonal_line_ratings` multiplies each line's `s_max_pu` in each hour by
$s_\text{nom,season}/s_\text{nom,winter}$, where the season is that of the hour's month. The months follow the
overhead-line seasons of ENA ER P27 / ACE 104 as tabulated in Northern Powergrid IMP/001/011 (Table 1):

| Season | Months |
|---|---|
| winter | November to February |
| spring | March, April |
| summer | May to August |
| autumn | September, October |

```{figure} /_static/figures/network/seasonal_line_ratings.png
:alt: Step chart of line rating as a share of the winter rating by month for 400, 275 and 132 kV lines, falling to about 0.95 in spring and autumn and about 0.87 to 0.89 in summer.
:width: 100%

The monthly factor in practice: the median ratio of each season's published rating to the winter rating, by
voltage, for the lines of the Clean Power 2030 build whose seasonal ratings come from ETYS, on the month calendar
above.
```

The factor multiplies any per-branch outage factor already in `s_max_pu`, and acts on the line's current `s_nom`, so a
rating override keeps the published seasonal shape. Lines relaxed to 9,999 MVA get no factor. Transformers keep their
winter rating all year. The secured boundary limits have their own seasonal factors (below); the two are separate
constraints and are not multiplied together. The one exception is the NESO `uniform_s_max_pu` mode, whose line limits
are observed day-ahead limits that already include the season, so those lines are not re-scaled.

### The voltage floor

Appendix B lists individual 132 kV (and some 66 kV and 33 kV) segments in England and Wales, but not the distribution
mesh and rings around them, because that network belongs to the distribution operators. A DC load flow over those
isolated segments saturates them at flows the real network spreads over paths the model does not have. NESO manages
these flows with operational tools the model does not represent, and its published boundary flows are at the 275 kV
and 400 kV level.

So every line and transformer with an end below `network.transmission.min_bm_constraint_voltage_kv` (220 kV by
default) has its rating set to 9,999 MVA before the solve (`solve_network.bm_voltage_floor_branches`). The branch stays
in the network and still carries flow by impedance; it just cannot bind. A transformer's impedance is preserved when
its rating is relaxed. 400 kV and 275 kV lines and 400/275 kV transformers keep their ratings. A floor of 0 keeps
published ratings at every voltage; 275 also relaxes 220 kV.

**Scottish exception.** In Scotland, 132 kV is transmission, owned by SHE Transmission and SP Transmission and
managed by NESO in the balancing mechanism. With `bm_voltage_floor.scottish_transmission_constrained` (on), a
branch in those licence areas whose lower voltage is at least `bm_voltage_floor.scottish_min_constrained_kv` (132 kV)
keeps its rating. A branch's licence area is the ETYS licensee of the Appendix B sheet it came from; branches from
other sources take the licensee most common at their substations, and failing that the side of the Scotland-England
border their buses lie on. Scottish 33 kV and 66 kV collector networks stay relaxed.

The floor is skipped for networks whose lines carry no voltage (Reduced and Zonal).

### Line rating overrides

`network.transmission.line_rating_overrides` sets the rating (MVA) of named lines or transformers, applied after the
voltage floor so it always wins. It is for documented data errors in Appendix B: for example circuits a later edition
publishes at a placeholder rating of 1 MVA where earlier editions and the change rows give the real rating, or 33 kV
artefacts that would trap a wind farm behind a small rating. Each entry in the base configuration carries its reason
and source in a comment; studies add their own. A name not found in the network is logged as a warning.

### Optional floors and scaling

`network.transmission.min_line_s_nom` and `min_transformer_s_nom` (MVA) raise any rating below them, and
`capacity_scale` multiplies every line and transformer rating. All are off by default (0, 0 and 1.0). They are
sensitivities, not part of any study.

### Boundary limits

A DC load flow with thermal ratings on an intact network does not see the security margin the system operator keeps
(the network must survive the loss of a circuit) or the circuits out for maintenance. Without them, modelled
constraint costs are far below NESO's. The model therefore adds boundary transfer limits: one constraint per boundary
on the sum of flows across it, as NESO itself represents the network in its constraint cost modelling.

The source is chosen by `network.transmission.outage_schedule.source` when `outage_schedule.enabled` is true:

| `source` | What it does | Typical use |
|---|---|---|
| `secured_capability` | ETYS secured boundary capability for the model year, with seasonal and outage factors, as sum-of-flow constraints | future years (CP30 2030, FES long-range) |
| `neso` | NESO's published day-ahead boundary limits for each half-hour | historical years with published limits |
| `csv` | a user file of per-branch outages (`component`, `component_id`, `start`, `end`, `s_max_pu`), read from `outage_schedule.file` | your own outage plan |
| `synthetic` | random maintenance and forced outages by voltage, weighted towards summer (`outage_schedule.synthetic`) | sensitivity |

The `csv` and `synthetic` sources derate individual branches through `s_max_pu`. The other two add constraints to the
linear model.

#### Secured boundary limits

`scripts/solve/secured_boundaries.py` builds one constraint per ETYS boundary from three published inputs:

- **Geometry**: NESO's ETYS boundary GIS, converted to OSGB36
  (`data/network/ETYS/etys_boundaries_mar25_osgb36.geojson`).
- **Capability**: the `Capability` rows of the ETYS boundary chart data by boundary, FES pathway and year
  (`data/network/etys_boundary_capability.csv`, from ETYS 2025). The secured capability is the largest pre-fault
  transfer across the boundary that meets the security standard (NETS SQSS), so it already includes the N-1 margin.
  It also includes the HVDC links that cross the boundary.
- **Seasonal factors**: NOA methodology Table 2.3, the fractions of winter capability used where the transmission
  owners give no seasonal capability.

```{figure} /_static/figures/network/etys_boundaries.png
:alt: Map of Great Britain with ETYS boundary lines, 22 of them in orange and labelled, beside a horizontal bar chart of their 2030 secured capability from 1.8 GW (NW1) to 15.3 GW (B9).
:width: 100%

The ETYS boundaries on the 2030 network. The 22 in orange have a 2030 secured capability in the CP30 2030 central
network case and are constrained; the bars give that capability, ordered north to south and shaded by the limit
type ETYS publishes. The dashed grey boundaries have no capability row for 2030 and are not constrained.
```

**Which branches cross.** A line, transformer or GB-internal DC link crosses a boundary when the straight segment
between its two buses cuts the boundary polyline an odd number of times (`find_boundary_crossings`). Each crossing is
signed by the side its `bus0` lies on, so the signed sum is the net transfer from one side to the other. Corrections
for individual branches go in `secured_capability.include` (with a sign) and `exclude`; an HVDC link whose straight
path meets the line an even number of times although its converters are on opposite sides must be declared this way.
Because the test uses bus positions, a misplaced bus can add or drop a crossing; the reviewed bus positions of
{doc}`topology` exist partly for this reason.

**The constraint.** For boundary $b$ with crossing branches $C_b$, signs $\sigma_\ell$ and flows $f_\ell(t)$:

$$
-L_b(t) \;\le\; \sum_{\ell \in C_b} \sigma_\ell\, f_\ell(t) \;\le\; L_b(t),
\qquad
L_b(t) = K_b(y, \text{pathway}) \times \phi_b(t)
$$

$K_b$ is the published capability for the capability year and FES pathway (MW), and $\phi_b(t)$ the hour's factor.
The limit is enforced in both directions: ETYS publishes one directional capability, and the reverse direction does
not bind in practice. Relaxed branches (the voltage floor) still count in the sum at their flow. Branch ratings stay
in force alongside the boundary constraint.

**Seasonal factor** (NOA methodology Table 2.3, by the limiting type in the capability table):

| Limit type | Winter (Nov-Feb) | Spring, autumn (Mar-Apr, Sep-Oct) | Summer (May) | Summer outage (Jun-Aug) |
|---|---|---|---|---|
| thermal | 1.00 | 0.85 | 0.80 | 0.70 |
| voltage | 1.00 | 1.00 | 1.00 | 0.90 |
| stability | 1.00 | 1.00 | 1.00 | 1.00 |

NOA gives the factors but not the calendar. The seasons are aligned with the line-rating calendar above, and June to
August is taken as the summer outage condition (the heart of the transmission outage season), with May at the plain
summer factor. That calendar is an assumption.

**Historic outage derate.** NESO's own constraint models scale boundary capability by outage durations taken from
historic transmission owner data. With `secured_capability.historic_derate.enabled`, a per-hour factor built from
NESO's published day-ahead limits replaces the NOA factor on the boundaries it covers. The table is built by
`scripts/solve/build_boundary_historic_derate.py` from the committed day-ahead limit history, for the boundaries
NESO maps to day-ahead constraint groups (B4, B6, B7a, B9 and LE1). The method used by the studies (`outage_generic`)
represents generic maintenance outages, as NESO's zonal model does:

- count the outage days of each boundary in each month of the reference years (days whose lowest day-ahead limit is
  well below the intact seasonal level);
- spread them evenly through the month, so the month's factor is the intact seasonal level less the month's outage
  share times the drop to the prior-outage level (NOA's summer-outage over summer factor: 0.875 of the season's level
  for thermal limits, 0.90 for voltage, 1.00 for stability);
- write rows `_default_thermal`, `_default_voltage` and `_default_stability` from the mean outage share of the mapped
  boundaries, which `historic_derate.unmapped: default` applies to every other boundary (`noa` leaves those on the NOA
  factors).

`historic_derate.mode: replay` uses the reference years' factors as they are, hour by hour (29 February takes 28
February's values). `mode: forward` keeps the part of the derate beyond the NOA factor but scales it by the ratio of
the target year's construction outage days to the reference year's (CP30 workbook sheet NA1), for a year with more
or less construction than the reference. The studies use the generic outage table for 2022 to 2025 in replay mode
(`data/network/neso_boundary_outage_generic_2022_25.csv`), set through the network delivery case (below).

```{figure} /_static/figures/network/boundary_limit_2030.png
:alt: Three step charts of the 2030 limit on boundaries B4, B6 and B7a, each falling from its winter capability to about 70 percent in summer, with the historic derate line below the NOA line in winter, spring and autumn and slightly above it in summer.
:width: 100%

The hourly limit $L_b(t)$ on three boundaries through 2030 in the CP30 2030 central network case: the secured
capability times the NOA seasonal factor alone (dashed), and times the generic outage derate replayed from NESO's
2022-25 day-ahead limits (solid), which is what the balancing stage enforces. Computed with
`scripts/solve/secured_boundaries.py` from the study's capability table.
```

**Other settings.** `capability_year` picks the capability year (default the modelled year) and
`capability_year_overrides` moves individual boundaries to another year; `fes_pathway` picks the pathway (HT by
default); `capability_file` picks the table; `boundaries` restricts the set (default every boundary with a capability
row). Two settings serve years after the last published capability (2045): `hold_last_year` takes each boundary's last
published value, and `branch_scaling` (with a `margin`) raises the ratings of the AC branches crossing a boundary whose
summed rating is below margin x capability, so the boundary, not circuits that the published capability assumes
reinforced, binds (`scripts/network_clustering/boundary_branch_scaling.py`; reinforcement is represented as parallel
circuits, so line impedance falls in proportion). The code also reads `capability_scale`, `seasonal_factors`,
`month_season`, `phase_in` (a boundary takes an earlier year's capability before a given date) and `soft_boundaries`
(a penalised slack per boundary); no shipped configuration sets them.

#### NESO day-ahead boundary limits

For historical years NESO publishes the day-ahead limit of each constraint boundary
(*Day-Ahead Constraint Flows and Limits*). These limits already include security margins and the outages that
actually happened. `data/network/neso_boundary_mapping.yaml` maps each NESO boundary to the ETYS lines that form it,
with flow signs (`data/network/neso_boundary_circuit_crosswalk.csv` is the circuit-level audit trail for the mapping).
The limits are read from a per-window cache, then from the committed history
`data/network/neso_day_ahead_constraint_limits.csv.gz`; the NESO API is queried only when `neso.allow_api_fetch` is
true. A mapped boundary with no data in the window raises an error rather than being skipped.

`neso.constraint_mode` selects how they enter:

- `aggregate_boundary` (default): line ratings are kept, and the signed sum of flows across each boundary is held
  within the NESO limit, the same constraint form as the secured boundaries. This is closest to what NESO publishes.
- `uniform_s_max_pu`: each boundary limit becomes one percentage derate on every mapped line (floored at
  `neso.min_s_max_pu`).

Gaps in the published series are filled by `neso.gap_fill_mode` (interpolate by default). `neso.soft_boundaries`
turns a boundary into a penalised soft limit, and a scenario can restrict the set with `neso.boundary_include`
(the 2020 validation study does both).

### Network delivery cases

NESO's constraint cost forecasts for 2030 differ by how much of the planned network is delivered. A network delivery
case reproduces such a case on the built network, in the balancing stage only
(`market.balancing.network_delivery`, `scripts/market/network_delivery.py`). The wholesale stage is a copperplate, so
the network does not move it: delivery cases share their parent's Stage 1 through `wholesale_from`
({doc}`../dispatch/balancing`). A central-dispatch Stage 1 schedules on the real network, so it applies the same case.

A case has a name (`case`), a `description` and any of:

**`remove_components`**: a list of `{component, name}` or `{component, prefix}` entries. Lines and transformers are
removed; links are kept with `p_nom = 0`, because the Stage 1 link schedule names them.

**`revert_appendix_b`**: undo Appendix B rows the build applied. A row is reverted if any selector picks it:

| Selector | Picks |
|---|---|
| `after_year: Y` | every row dated after Y, including later-snapshot rows pulled forward (CP30 dated works, HVDC terminal works) |
| `projects: [name, ...]` | the rows listed for each project in `data/network/ETYS/appendix_b_projects.csv` |
| `cp30_dated_works: true` | the rows the build pulled in as CP30 dated works |
| `rows: [id, ...]` | row ids as the upgrade audit's `appendix_b_row_id` gives them |

The revert works from the applied-row record the build stores in the network ({doc}`topology`). Each component a
chosen row touched is replayed through its recorded operations without the chosen rows, so the result is the network
the remaining rows alone would give, even when two rows changed the same circuit or a later row removed a circuit an
earlier one added. In order:

1. Branches the chosen rows removed are put back on their original buses, with any buses or islands the build pruned
   because of the removal.
2. Re-rated branches get their earlier rating, impedance and seasonal ratings.
3. Small pockets of new buses that the revert would strand (no load, store or live link, and at most
   `rehome_stranded_max_mw`, 50 MW by default, of generation and storage) are re-homed to the nearest surviving bus of
   the same voltage, and their circuit goes.
4. Branches the chosen rows added are removed, except where a generator, load, storage unit or live link would be cut
   off: then the fewest added branches that reach the main grid are kept (the lowest-rated among equals) and reported
   as connection assets. NESO's own delivery cases assume every planned generator connects.
5. Added links are kept with `p_nom = 0`; new buses left with no branch, or in an empty island, are dropped; and the
   class-median seasonal ratings are recomputed.

Connectivity for these checks counts only electrical (AC and DC) buses, so a pocket is not treated as connected
through a hydrogen bus that an electrolyser and a hydrogen turbine share. Each case matches a build made without the
chosen rows.

**`secured_capability`**: keys that replace those of `network.transmission.outage_schedule.secured_capability` for
this stage: `capability_year`, `capability_year_overrides`, `capability_file`, `boundaries`, `historic_derate` and
`phase_in`. A delayed project's boundaries are held at the capability of the year before it, as NESO's own
boundary-based studies do.

**`transmission`**: keys that replace `min_bm_constraint_voltage_kv` and `bm_voltage_floor` for this stage. A floor of
1000 kV with the Scottish exception off makes every branch non-binding, so transfer is limited by the boundaries
alone, as in NESO's zonal model.

The case runs before the voltage floor and the outage schedule, so restored branches take the floor and the boundary
constraints see the case's branches and capability. What was applied is logged and stored in
`network.meta["network_delivery"]`.

### Single solve and the two-stage market

| | Single solve | Stage 1 wholesale (copperplate) | Stage 1 central dispatch | Stage 2 balancing |
|---|---|---|---|---|
| Branch ratings, voltage floor, overrides | yes | no | yes | yes |
| Seasonal ratings | yes | no | yes | yes |
| Boundary limits | if configured | no | if configured | if configured |
| Network delivery case | no | no | yes | yes |
| Internal HVDC link ratings | yes | yes (island links can be lifted) | yes | yes |

The wholesale stage collapses the mainland synchronous network to one bus (`market.wholesale.copperplate_mode:
collapse`) so there is one GB price; islands, foreign buses and every link keep their identity and ratings. Its
results are expanded back onto the full network. The balancing stage then redispatches against all the limits above;
the cost of that redispatch is the model's constraint cost ({doc}`../dispatch/balancing`). With `copperplate_mode:
none`, Stage 1 is a central dispatch on the real network with the same preparation as the balancing stage
({doc}`../dispatch/central_dispatch`).

## Data

| File | Content |
|---|---|
| `data/network/ETYS/etys_boundaries_mar25_osgb36.geojson` | ETYS boundary lines (NESO ETYS boundary GIS), OSGB36 |
| `data/network/etys_boundary_capability.csv` | secured capability by boundary, FES pathway and year (ETYS 2025 boundary chart data), with limiting type and source |
| `data/network/etys2024_cp30_boundary_capability*.csv`, `etys_boundary_capability_etys2024_*.csv` | capability tables the CP30 2030 study uses (ETYS 2024 chart readings and delivery-case variants), each row with its source |
| `data/network/neso_boundary_outage_generic_2022_25.csv` (and `_2022_23`, `_2024_25`) | generic outage factors by boundary, month, day and hour |
| `data/network/neso_boundary_outage_days_*.csv`, `neso_boundary_historic_derate_*.csv` | the outage-day and quantile variants of the derate |
| `data/network/neso_day_ahead_constraint_limits.csv.gz` (+ `.json`) | committed NESO day-ahead boundary limits (built by `scripts/solve/build_neso_boundary_limit_cache.py`) |
| `data/network/neso_boundary_mapping.yaml`, `neso_boundary_mapping_ssesp_strict.yaml` | NESO boundary to ETYS lines, with signs |
| `data/network/neso_boundary_circuit_crosswalk.csv` | circuit-level evidence for the mapping |
| `data/network/ETYS/appendix_b_projects.csv` | named projects as sets of Appendix B rows |

## Configuration

Under `network.transmission` (defaults in `config/defaults/base/network.yaml`):

| Key | Default | Meaning |
|---|---|---|
| `min_bm_constraint_voltage_kv` | 220 | voltage floor (kV); 0 = published ratings everywhere |
| `bm_voltage_floor.scottish_transmission_constrained` | true | keep Scottish transmission branches constrained |
| `bm_voltage_floor.scottish_min_constrained_kv` | 132 | lowest Scottish voltage kept constrained (kV) |
| `seasonal_ratings.enabled` | true | apply seasonal line ratings |
| `seasonal_ratings.season_months` | ER P27 seasons | month numbers of each season |
| `line_rating_overrides` | documented corrections | `{branch_id: MVA}` |
| `min_line_s_nom`, `min_transformer_s_nom`, `capacity_scale` | 0, 0, 1.0 | optional floors and scaling |
| `outage_schedule.enabled` | false | add boundary limits or outages |
| `outage_schedule.source` | `neso` | `secured_capability`, `neso`, `csv` or `synthetic` |
| `outage_schedule.secured_capability.*` | set by the studies | `capability_file`, `capability_year`, `capability_year_overrides`, `fes_pathway`, `boundaries`, `include`, `exclude`, `historic_derate`, `hold_last_year`, `branch_scaling` |
| `outage_schedule.neso.*` | see base | `boundary_mapping`, `committed_cache`, `allow_api_fetch`, `constraint_mode`, `gap_fill_mode`, `min_s_max_pu`, `soft_boundaries` |
| `outage_schedule.synthetic.*` | see base | seed, forced outage rate by voltage, maintenance duration, monthly weights |

Under `market`:

| Key | Default | Meaning |
|---|---|---|
| `market.balancing.network_delivery` | null | the balancing stage's delivery case: `case`, `description`, `remove_components`, `revert_appendix_b`, `secured_capability`, `transmission`, `rehome_stranded_max_mw` |
| `market.wholesale.copperplate_mode` | `collapse` | `collapse`, `relax` (every branch at `transmission_relaxation` MVA) or `none` (central dispatch) |
| `market.wholesale.relax_island_hvdc` | true | lift island HVDC links in the copperplate |

The CP30 2030 study sets the secured boundaries in its `cp30_ffr_2030` template and the delivery cases in the
`network_*` templates of `config/studies/cp30_2030/templates.yaml`; the FES long-range studies set theirs in their own
templates. To see the resolved values for a scenario:

```bash
python config/config_loader.py --explain <scenario_id> --path network.transmission
python config/config_loader.py --explain <scenario_id> --path market.balancing.network_delivery
```

## Validation

- **Secured capability audit.** Rule `audit_secured_capability` (`scripts/solve/secured_capability_audit.py`,
  in `rules/CP30.smk`) reads only the configuration and writes `resources/analysis/secured_capability_by_family.csv`:
  for every market family in `run_scenarios`, per boundary, the capability its balancing stage will enforce, resolved
  exactly as the solve resolves it (capability year asked and used, MW, limit type, seasonal factors, historic derate,
  mean limit over the family's hours). It warns when a delay case's capability is above its central twin's on any
  boundary, when families read different capability files, or when a boundary has no capability or geometry. The
  central family it compares against is `secured_capability_audit_central` in the study settings.
- **Locational checks.** For CP30 builds, rule `validate_CP30_locational` (`scripts/CP30/locational_validation.py`)
  checks where capacity sits relative to the network: regional totals by the region of each unit's bus, offshore
  landings, electrolysers, long-duration storage and demand on each side of the main boundaries, radial pockets whose
  generation and storage exceed their connection, a boundary exposure summary, and each node's peak demand against
  what it can import at the balancing stage's ratings. A FAIL fails the rule. Settings are in
  `validation.locational` (CP30 source overlay). The fleet location check (`validate_CP30_fleet_location`) adds
  capacity on each side of the secured boundaries against CP30's regional figures.
- **NESO comparisons.** For historical market scenarios, rule `validate_neso_constraints`
  (`scripts/market/validate_neso_constraints.py`) compares model boundary flows and balancing costs with NESO's
  thermal constraint costs and day-ahead flows and limits, using the boundary mapping. For the CP30 2030 study, rule
  `validate_CP30_balancing` compares solved balancing families with NESO's published constraint costs, balancing
  volumes and zonal turn-down, including the delay cases ({doc}`/studies/cp30_2030`).
- **Network delivery.** Each case logs what it removed, restored, re-rated, re-homed and kept as a connection asset,
  and any replay conflict, and stores the record in the balancing network.

## Limitations

- The DC load flow ignores losses, reactive power and voltage limits; voltage- and stability-limited boundaries are
  represented only through their published capability.
- Boundary limits are fixed transfer capabilities, not the result of an N-1 contingency analysis on the modelled
  dispatch. They do not change with the generation pattern as NESO's real-time limits do.
- Which branches cross a boundary depends on straight lines between bus positions; declared `include` and `exclude`
  entries correct the known exceptions.
- The summer outage months and the reference years of the outage derate are assumptions; the derate replays past
  outage patterns, not a future outage plan.
- Relaxing branches below 220 kV in England and Wales removes local distribution constraints that do exist in
  practice; local curtailment behind them is not modelled.
- For years after the last published capability (2045), `hold_last_year` holds each boundary flat, and reinforcements
  the capability assumes but Appendix B does not list are represented only by scaling the crossing branches.
