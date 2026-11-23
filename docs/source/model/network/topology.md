# Network topology and upgrades

## What it does

The network step builds the electrical skeleton every other step attaches to: buses, AC lines, transformers and
GB-internal HVDC links. Demand, generation, storage and interconnectors are then placed on its buses
({doc}`../demand/index`, {doc}`../generation/index`, {doc}`../flexibility/storage`, {doc}`../flexibility/hydrogen`,
{doc}`../interconnectors`), and the solve limits power flow on its branches
({doc}`constraints`).

Three network models are available. A scenario picks one with the top-level key `network_model`.

| Model | Buses | Branches | Power flow | Built by |
|---|---|---|---|---|
| `ETYS` | about 2,400 (400 kV down to 11 kV) | about 1,900 lines and 1,600 transformers, plus internal HVDC links | linearised (DC) load flow on every line and transformer | `scripts/network_build/ETYS_network.py` |
| `Reduced` | 29 | 99 lines | linearised (DC) load flow | `scripts/network_build/build_network.py` |
| `Zonal` | 20 nodes in 17 zones | 31 inter-zone links | transport model (links only) | `scripts/network_build/build_network.py` |

The ETYS counts are for a 2030 build; they change with the year because planned works add and remove branches.
Interconnector cables and their foreign buses are not part of any of the three: they are added later by the
interconnector step ({doc}`../interconnectors`).

```{figure} /_static/figures/network/network_models.png
:alt: Three maps of Great Britain side by side: the dense ETYS network by voltage with HVDC links, the 29-bus Reduced network and the 20-node Zonal network with its zone outlines.
:width: 100%

The three network models. Left: the ETYS network of the Clean Power 2030 build, lines by voltage and internal
HVDC links dashed (transformers are not drawn). Middle and right: the Reduced and Zonal networks as shipped in `data/network/`,
with line width proportional to rating.
```

### Which model to use

| You want to | Use |
|---|---|
| Model transmission constraints, boundary flows, balancing (redispatch) costs or locational prices | `ETYS` |
| Reproduce the CP30 2030 or FES long-range studies | `ETYS` (the long-range studies then cluster it, {doc}`clustering`) |
| Test the workflow, a configuration change or a new data source quickly | `Reduced` |
| Screen regional energy balances where flows inside a zone do not matter | `Zonal` |
| Something between Reduced and full ETYS | `ETYS` with clustering ({doc}`clustering`) |

Only the ETYS model carries the information the network constraints need: published seasonal ratings, voltage
levels, the transmission licensee of each branch and real substation positions. The secured boundaries, the
seasonal ratings and the balancing-stage voltage floor of {doc}`constraints` act on ETYS networks (clustered or
not). The Reduced and Zonal models keep their fixed topology in every year: they get no planned upgrades.

## Method and assumptions

### The ETYS build

The ETYS model is built from NESO's *Electricity Ten Year Statement* Appendix B, the published list of every
transmission circuit, transformer and HVDC link with its terminals, impedance and rating. Two Snakemake rules in
`rules/network_build.smk` do the work:

```{mermaid}
flowchart TD
    A["ETYS Appendix B workbook<br/>GB_network.xlsx<br/>FES regional GSP data"] --> B["process_ETYS_data<br/>(one run per ETYS edition)"]
    B --> C["ETYS_{year}_components.csv<br/>ETYS_{year}_buses.csv"]
    C --> D["build_ETYS_base_network<br/>(one run per scenario)"]
    D --> E["{scenario}_network.nc<br/>+ upgrade and CP30 audits"]
```

1. **`process_ETYS_data`** (`scripts/network_build/process_ETYS_data.py`) reads the Appendix B workbook once per
   ETYS edition and writes two intermediate tables:
   - the components (lines, transformers, links) from the base sheets: circuits B-2-1a/b/c/d (SHE Transmission,
     SP Transmission, NGET, OFTOs), transformers B-3-1a/b/c/d and HVDC B-5-1;
   - the buses, with their voltage, a first position and an offshore flag.

   It adds the offshore wind farm and BMU connection edges of `data/network/ETYS/GB_network.xlsx`, with ratings
   taken from the matching Appendix B OFTO and NGET circuits where the workbook gives none. It also repairs islands
   caused by naming differences between the operators' sheets (for example one substation spelled two ways), using
   the consolidated AC sheet of `GB_network.xlsx`.
2. **`build_ETYS_base_network`** (`scripts/network_build/ETYS_network.py`) places the remaining buses, assembles the
   PyPSA network, applies the planned upgrades for the modelled year, adds the CP30 layer where a scenario asks for
   it, converts the branch impedances to PyPSA units and writes `resources/network/{scenario}_network.nc`.

`scripts/network_build/etys_file_registry.py` maps each ETYS edition (2022, 2023, 2024) to its file names and sheet
names, and holds the shared constants: voltage codes, sheet lists, electrical defaults and the named exceptions
described below. The edition is chosen with `network.etys.year` (2024 by default).

**Bus names.** ETYS node names have six characters. The first four are the substation (site) code, the fifth is the
voltage code and the sixth identifies the busbar or section:

| Fifth character | 4 | 2 | 1 | 6 | 3 | 5 |
|---|---|---|---|---|---|---|
| Voltage (kV) | 400 | 275 | 132 | 66 | 33 | 11 |

So `BEAU4-` is Beauly 400 kV and `COCK21` is busbar 1 of Cockenzie 275 kV. The site code is what the build uses to
recognise busbars of the same substation.

**Base topology.** The base sheets describe the network as it stood when the edition was published. The model has no
back-cast of earlier networks: a historical scenario for 2020 runs on the edition's existing network plus any
Appendix B rows dated up to 2020 (in practice none for ETYS 2024). Pick an earlier edition with `network.etys.year`
if that matters for your question.

**Duplicates with swapped terminals.** A few base and upgrade rows appear twice, once in each orientation, with the
same impedance and rating. They are built once (`ETYS_network.drop_reversed_duplicate_branches` for the base sheets,
`ETYS_upgrades.drop_reversed_duplicate_records` for Appendix B rows). Genuine parallel units are listed in the same
orientation and are kept.

**Internal links.** Every GB-internal HVDC link is bidirectional (`p_min_pu = -1`), lossless and has no marginal cost.
Identical parallel internal links (same buses, efficiency, cost and limits) are merged into one with the summed
rating (`scripts/network_build/parallel_links.py`): kept apart, a free lossless pair can carry an arbitrary loop flow
with no effect on cost, which makes the reported link flows meaningless.

### Bus coordinates

Two coordinate systems meet here:

| System | Units | Where it is used |
|---|---|---|
| OSGB36 / British National Grid (EPSG:27700) | metres; x about 0 to 700,000, y about 0 to 1,200,000 | bus `x` and `y` of every built network |
| WGS84 (EPSG:4326) | degrees of longitude and latitude | site data (REPD, TEC register, GSP points); kept on buses as `lon` and `lat` |

Positions are resolved in WGS84 and converted to OSGB36 once, when the network is assembled. Code that compares a
site with a bus must convert first; the helpers in `scripts/utilities/spatial_utils.py` detect which system a table uses from
its value range (`network.spatial_mapping.coordinate_detection`) and convert as needed. Comparing degrees with metres
silently puts every site on the bus nearest the origin.

Appendix B gives no coordinates, so each bus is placed in tiers:

1. **GSP mapping.** A bus named in the `Dem_per_node` sheet of `GB_network.xlsx` takes its Grid Supply Point's
   position from the FES regional GSP data.
2. **Substation lookup.** `data/network/ETYS/substation_coordinates.csv` gives a position per four-character site
   code.
3. **Prefix match.** The site code matched to a GSP name.
4. **Estimation.** Remaining buses are placed iteratively from their already-placed neighbours, using circuit lengths
   as distance weights.
5. **Land check.** Onshore buses that land in the sea are moved to the nearest point on land (the GSP region
   polygons, `data/network/GSP/GSP_regions_4326_20250109.geojson`). Offshore buses (OFTO platforms and wind farm
   collectors) are flagged `is_offshore` and left at sea.
6. **Reviewed overrides.** `data/network/ETYS/bus_coordinate_overrides.csv` sets the position (and the offshore flag)
   of buses that the steps above put in the wrong place. Each row must give its source. The rows are applied to the
   base buses, then again to the finished network, so buses created by Appendix B upgrades and the CP30 layer take
   them too. A row naming a bus that no stage of the build can create is an error, so stale rows cannot linger.

A misplaced bus matters more than it looks: the secured boundary constraints decide which branches cross a
boundary from the straight line between their two buses ({doc}`constraints`), and site placement
uses the nearest bus.

### Branch impedance units

ETYS publishes resistance, reactance and susceptance as "% on 100 MVA". PyPSA reads different units: a line's `r`
and `x` in ohm and `b` in siemens (at the `v_nom` of `bus0`), and a transformer's `r`, `x` and `b` per unit on its own
rating `s_nom`. Getting this wrong makes flows split on the wrong impedances, so the build follows one rule
(`scripts/network_build/branch_units.py`):

- Every branch source (the base sheets, Appendix B upgrade rows, `GB_network.xlsx` edges, documented bus sections,
  the CP30 layer and the type defaults) is carried through the build in **per unit on 100 MVA**
  ($x_{100} = \text{percent}/100$).
- The network is converted to PyPSA units **once**, at the end of `create_network`:

$$
\text{line: } x = x_{100}\,\frac{V^2}{100}\ \Omega,\quad b = b_{100}\,\frac{100}{V^2}\ \text{S}
\qquad\qquad
\text{transformer: } x = x_{100}\,\frac{s_\text{nom}}{100},\quad b = b_{100}\,\frac{100}{s_\text{nom}}
$$

  with $V$ the `bus0` voltage in kV and $s_\text{nom}$ in MVA. PyPSA then sees $x_\text{pu} = x_{100}/100$ for both
  (per unit on its 1 MVA base). A 400 kV overhead line comes out at roughly 0.3 ohm/km.
- `network.meta["branch_impedance_units"]` records which units the branch tables hold. A second conversion, or an
  upgrade applied in the wrong units, raises an error. Every solve refuses a network with lines or transformers that
  does not record PyPSA units (`require_pypsa_branch_units`). The Reduced network, whose MATPOWER source is also in
  per unit on 100 MVA, is converted the same way; the Zonal network has no impedances.
- **Zero and missing values** follow one policy (`resolve_source_impedance_pu100`). A published R = 0 or B = 0 is
  kept (series compensators, couplers and short cables are published that way). A published X = 0 on a circuit of
  type "Zero Length" or no longer than 1 km is a bus section and gets the coupler reactance ETYS itself publishes for
  its 9,999 MVA couplers, 0.01 % on 100 MVA. Any other missing value takes the type default in
  `ELECTRICAL_DEFAULTS` and is logged as an assumption.
- **Changing a transformer rating.** A transformer's stored impedance is relative to its `s_nom`, so changing `s_nom`
  alone changes its impedance. Every step that relaxes or rescales a transformer rating after the build (the
  balancing-stage voltage floor, rating overrides, capacity scaling, clustering) goes through
  `set_transformer_s_nom_preserving_impedance`, which rescales `x` and `r` with the rating.

### Planned upgrades from ETYS Appendix B

For a future year the base network is brought forward with the planned works Appendix B lists by year
(`scripts/network_build/ETYS_upgrades.py`, `apply_etys_network_upgrades`). Every row dated at or before the upgrade
year is applied; the upgrade year is `network.upgrades.upgrade_year`, or the scenario's `modelled_year` when that is
null.

| Sheet | Content | Operators |
|---|---|---|
| B-2-2a/b/c/d | circuit additions, removals and changes | SHE Transmission, SP Transmission, NGET, OFTO |
| B-3-2a/b/c/d | transformer additions, removals and changes | as above |
| B-5-1 | HVDC links | all |

Order of application:

1. **New buses.** Buses named by in-scope rows but absent from the network are created. Their position comes from the
   reviewed overrides, else `substation_coordinates.csv`, else another bus at the same site, else an offset from a
   connected bus by the circuit length.
2. **Circuits:** additions, then removals, then changes (re-ratings and re-impedancing; a change row changes every
   parallel circuit between its terminals). **Transformers:** the same.
3. **Documented bus sections.** Appendix B does not list switchgear inside substations. Where an upgrade splits a
   through-path that the transmission owner documents as continuous, a bus section is added, but only for the cases
   listed in `DOCUMENTED_INTERNAL_BUS_SECTIONS` with their evidence. Other possible split paths are reported in the
   log for review, never inferred.
4. **HVDC links** from B-5-1. Converter stations named by place rather than node code are resolved through
   `HVDC_PLACE_NAME_ALIASES`. Appendix B can date a link before the AC works that create its converter buses; for
   links named in `HVDC_TERMINAL_ENABLING_WORKS` the terminal substation works are pulled forward with the link.
   Any other link whose terminals arrive only in a later year stays out, and the audit says so.
5. **Clean-up.** Islands with nothing attached, left by removals or failed additions, are pruned
   (`network.upgrades.prune_dead_islands`); islands that carry generation, load or storage are kept and reported.
   Buses with no remaining branch are removed.

Two resolution rules help rows find their terminals. With `network.upgrades.busbar_alias_fallback` (on), a node
naming a specific busbar (`COCK21`) that the network carries only as the site's generic busbar (`COCK2-`) is joined to
the generic busbar, and the substitution is logged. **Documented topology corrections**
(`network.upgrades.documented_topology_corrections`) reconcile a handful of rows whose terminal names do not match
the base sheets, each with its source in the code comments; each can be switched off per scenario.

Rows that cannot be represented (missing terminals, ratings or impedances published as "TBC") are not guessed. They
are written to `resources/network/{scenario}_etys_upgrade_audit.csv` with the reason. The optional
`network.upgrades.engineering_fallbacks` (off) would add such rows at typical ratings by voltage; leave it off for a
strictly source-backed network.

**The applied-row record.** The build records what every applied Appendix B row did, in
`network.meta["appendix_b_works"]` (`scripts/network_build/appendix_b_record.py`): for each row its id, kind,
operation, year and terminals, and for each component it touched the operation in build order with the branch's
rating, impedance and seasonal ratings before and after. The upgrade audit's `appendix_b_row_id` column carries the
same ids. This record is what lets the balancing stage undo selected works for a delayed-network case
({doc}`constraints`).

### CP30 reinforcements and dated works

CP30 (NESO's *Clean Power 2030* advice) assumes some reinforcements are in service earlier than Appendix B dates
them, and Appendix B publishes only snapshot years (to 2029, then 2032 or 2033 depending on the operator). Where a
scenario enables `network.reinforcements` (the CP30 2030 study does), two things happen:

- **Dated works.** `data/network/ETYS/cp30_appendix_b_dated_works.csv` (set in the CP30 source overlay,
  `network.reinforcements.appendix_b_dated_works_file`) names later-snapshot Appendix B rows that CP30 requires by
  the modelled year. Each row is pulled into the earlier build unchanged, after checking that the workbook's R, X, B
  and winter rating equal the values pinned in the table: a changed workbook fails the build rather than drifting.
  Pulled rows are applied, audited and recorded like any other Appendix B row, and tagged `cp30_dated_work` with
  their CP30 project code.
- **The CP30 layer** (`scripts/CP30/reinforcement_layer.py`) adds the few CP30 accelerated projects listed in
  `network.reinforcements.components_file`, each with terminals and parameters from an Appendix B row (for example
  the Sea Link HVDC link). `scripts/CP30/reinforcements.py` then audits every CP30-required project due by the model
  year against what the network contains, so a scenario cannot be described as the full CP30 network while named
  projects are missing (`resources/network/{scenario}_cp30_reinforcement_audit.csv`).

```{figure} /_static/figures/network/appendix_b_works_2030.png
:alt: Map of Great Britain with the transmission works applied for 2030 drawn in blues by year and in orange for later works brought forward, concentrated in Scotland and along the east coast.
:width: 70%
:align: center

The Appendix B works applied by the Clean Power 2030 build (from its upgrade audit): circuits added or re-rated
(lines), transformer works (dots) and HVDC links (dashed), coloured by the year Appendix B dates them. Orange rows
are later-snapshot rows brought forward as CP30 dated works or HVDC enabling works. Removals are not drawn.
```

An optional register of Holistic Network Design offshore assets (`network.hnd_hybrid`, off in the shipped
configuration) adds individually evidenced HND buses and circuits in the same way.

### HVDC links inside GB

Internal HVDC links come from three places: the base B-5-1 sheet (the Western HVDC link and the Caithness-Moray-
Shetland system), the B-5-1 rows dated up to the upgrade year (for example the Eastern Green Links and the
Spittal-Peterhead link), and the CP30 layer. They are links with carrier `DC` whose two ends are GB buses.

How they are treated downstream:

- In the **secured boundary constraints** an internal link that crosses a boundary counts in the boundary's flow sum,
  because NESO's published boundary capability includes the HVDC links that cross it ({doc}`constraints`).
- In the **copperplate wholesale market** links keep their ratings, except that links to islands can be lifted
  (`market.wholesale.relax_island_hvdc`) so island wind is not curtailed in the day-ahead stage
  ({doc}`../dispatch/wholesale`).
- Being lossless and free, an internal link and the parallel AC network can form a cost-neutral tie. The two-stage
  market breaks it with a small charge on link flow ({doc}`../dispatch/balancing`).

### Reduced and Zonal networks

**Reduced.** A 29-bus, 99-line reduced model of the GB transmission system from the University of Strathclyde,
in MATPOWER format (`data/network/reduced_network/GBreducednetwork.m`; its header says how to cite it). The
build reads `buses.csv` (positions in OSGB36) and `lines.csv`, whose r, x and b are per unit on 100 MVA and are
converted as for ETYS. Lines carry a single rating. Demand is shared over the buses with
`reduced_bus_definitions.csv` and `bus_to_gsp.csv` ({doc}`../demand/placement`).

**Zonal.** Seventeen zones represented by 20 nodes (zone 1, northern Scotland, is split into four), joined by 31
links in `data/network/zonal/links.csv`. Links make it a transport model: flows are limited by each link's rating but
do not follow impedances. `zone_definitions.csv` maps GSPs to zones. Interconnector rows in `links.csv` are skipped by
the build and added by the interconnector step.

Neither model gets Appendix B upgrades, seasonal ratings or a balancing-stage voltage floor (their branches carry no
voltage or licensee). Use them for speed, not for constraint costs.

## Data

| File | Content |
|---|---|
| `data/network/ETYS/ETYS 2024 Appendix-B V1.xlsx` (and the 2022 and 2023 editions) | NESO ETYS Appendix B: circuits, transformers, HVDC links, planned works |
| `data/network/ETYS/GB_network.xlsx` | consolidated AC sheet (island repair), offshore wind farm and BMU edges, `Dem_per_node` (node to GSP) |
| `data/network/ETYS/substation_coordinates.csv` | position per site code |
| `data/network/ETYS/bus_coordinate_overrides.csv` | reviewed per-bus positions, each with its source |
| `data/network/ETYS/fes{edition}_regional_breakdown_gsp_info.csv`, `Regional breakdown of FES24 data.xlsx` | GSP positions |
| `data/network/GSP/GSP_regions_*.geojson` | GSP region polygons (land check, clustering) |
| `data/network/ETYS/cp30_appendix_b_dated_works.csv` | later-snapshot Appendix B rows CP30 needs by its year, values pinned |
| `data/network/ETYS/cp30_n1_accelerated_components.csv`, `cp30_n1_reinforcement_manifest.csv` | the CP30 layer and its project manifest |
| `data/network/ETYS/appendix_b_projects.csv` | named projects as sets of Appendix B rows (used to revert a project, {doc}`constraints`) |
| `data/network/reduced_network/` | Reduced model |
| `data/network/zonal/` | Zonal model |

Outputs (in `resources/network/`): `ETYS_{edition}_components.csv` and `ETYS_{edition}_buses.csv`;
`{scenario}_network.nc`; `{scenario}_etys_upgrade_audit.csv`; `{scenario}_cp30_reinforcement_audit.csv`;
`{scenario}_hnd_hybrid_audit.csv`. Field-level detail of the network tables is in {doc}`/reference/data/network_data`.

A scenario can solve on another scenario's network with `network_from: <parent>`: the parent's network is built once
and the child builds nothing. The CP30 2030 monthly scenarios all share one full-year network this way. The child may
differ from its parent only in solve-time keys.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `network_model` | (scenario) | `ETYS`, `Reduced` or `Zonal` |
| `network.etys.year` | 2024 | ETYS Appendix B edition: 2022, 2023 or 2024 |
| `network.upgrades.enabled` | true | apply Appendix B planned works (ETYS only) |
| `network.upgrades.upgrade_year` | null | apply works dated up to this year; null = `modelled_year` |
| `network.upgrades.documented_topology_corrections` | all true | named, sourced reconciliations of terminal names |
| `network.upgrades.topology_sensitivities` | `[]` | named non-default internal-topology sensitivities |
| `network.upgrades.busbar_alias_fallback` | true | join a busbar-specific node to the site's generic busbar when only that exists |
| `network.upgrades.prune_dead_islands` | true | remove empty islands left by removals |
| `network.upgrades.engineering_fallbacks.enabled` | false | add rows with unpublished ratings at typical values |
| `network.reinforcements.enabled` | false (true in the CP30 2030 study) | CP30 dated works, CP30 layer and audit |
| `network.reinforcements.appendix_b_dated_works_file` | set in `config/defaults/source/CP30.yaml` | the dated-works table |
| `network.reinforcements.components_file`, `manifest_file`, `reference_file`, `strict` | set by the CP30 study | CP30 layer and audit inputs |
| `network.hnd_hybrid.enabled` | false (set in `config/defaults/source/CP30.yaml`; off when unset) | HND asset register |
| `network.spatial_mapping.coordinate_detection` | see base | value ranges that tell OSGB36 from WGS84 |
| `network_from` | none | solve on another scenario's built network |

The defaults are in `config/defaults/base/network.yaml`. To see where a scenario's value comes from:

```bash
python config/config_loader.py --explain <scenario_id> --path network.upgrades
```

## Validation

Checks run on every build:

- **Topology checks** (`ETYS_network.validate_network_topology`, warnings only): disconnected sub-networks, self-loops,
  zero impedances, unusual X/R ratios, missing voltages or coordinates, zero ratings.
- **Upgrade audit.** Every in-scope Appendix B row has a status in `{scenario}_etys_upgrade_audit.csv`, and the build
  checks that every applied row is in the applied-row record.
- **Dated works.** A CP30 dated-works row whose workbook values differ from its pinned values fails the build.
- **Coordinate overrides.** A row without a source, a duplicated bus or a bus no stage can build fails the build.
- **Units.** The units flag must say PyPSA units before any solve.

For CP30 builds the rule `validate_CP30_fleet_location` (`scripts/CP30/fleet_location_validation.py`) also compares
bus positions with ETYS circuit lengths: a bus whose circuits are mostly longer in a straight line than their
published length allows (with a margin), or a transformer whose two ends are more than 2 km apart, is flagged as
suspect for review.

## Limitations

- Appendix B lists transmission assets only. In England and Wales the 132 kV and lower networks are distribution
  and appear only in part (individual 132 kV trunk segments without the surrounding mesh); see the voltage floor in
  {doc}`constraints`.
- Appendix B publishes snapshot years, not commissioning dates. Works are in service from 1 January of their row's
  year; within-year timing is not modelled.
- Substation internals (busbar couplers and switchgear) are not published. Sections are added only where documented.
- Bus positions outside the reviewed overrides are estimates. They are good enough for placing sites and drawing
  maps, but check a bus's position before drawing conclusions about a boundary it sits close to.
- The model is linearised (DC) load flow: no reactive power, voltage limits or losses on AC branches.
