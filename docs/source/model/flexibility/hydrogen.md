# Hydrogen

Future scenarios include a simple hydrogen system: electrolysers that turn electricity into hydrogen, a hydrogen
store, hydrogen demand outside the power sector, and hydrogen turbines that turn it back into electricity. This page
covers how it is built, where electrolysers connect and how the system behaves in the market stages.

## What it does

Rule `add_hydrogen_system` (`rules/hydrogen.smk`, script `scripts/hydrogen/add_hydrogen_system.py`) adds:

| Component | PyPSA type | Notes |
|---|---|---|
| `GB_H2` bus | `Bus`, carrier `H2_gas` | one GB-wide hydrogen bus: no hydrogen network |
| Electrolysers | `Link` electricity bus to `GB_H2`, carrier `electrolysis` | efficiency 0.70 (LHV) by default |
| Hydrogen store | `Store` on `GB_H2` | sized at 168 hours of turbine output by default |
| Hydrogen turbines | `Link` `GB_H2` (or their own fuel bus) to electricity bus, carrier `H2_turbine` | the hydrogen generators placed by the thermal stage, converted to links; efficiency 0.50 by default |
| Hydrogen offtake | `Load` on `GB_H2` | non-power hydrogen demand, flat |

Historical scenarios pass the network through unchanged.

Because turbines can only burn hydrogen that was made or bought, the system respects an energy balance: power to
hydrogen to power loses about two thirds of the electricity at the default efficiencies (0.70 x 0.50 = 0.35).

## Method and assumptions

### Capacities

| Scenario type | Electrolyser capacity | Turbine capacity |
|---|---|---|
| FES GSP blocks | FES electrolysis capacity for the pathway and year | FES hydrogen generation (thermal stage) |
| CP30 | CP30 ED1 grid-connected electrolysis; if missing, derived from ES1 electrolysis demand at `electrolysis_full_load_hours` | CP30 ES1 hydrogen generation |
| FES long range (`hydrogen.fes`) | FES networked electrolysis (Dem_BB009) at FES's sites | FES H2 generators from the declared station schedule |

### Electrolyser siting

`hydrogen.electrolyser_siting.method` chooses where electrolysers connect:

- `h2_turbine_pro_rata` (code default): pro rata to hydrogen-turbine capacity by bus.
- `fes_gsp` (brownfield builds): by FES networked electrolysis (Dem_BB009) by GSP node, for the FES edition and
  pathway the source overlay names; FES's `Direct` rows go pro rata to their licence area's GSP rows. The result is
  scaled to the scenario's electrolysis total, so the total is unchanged.

```{figure} /_static/figures/supply/hydrogen_map_2030_2050.png
:alt: Two maps of Great Britain side by side for 2030 and 2050, with rings for electrolysers and filled circles for hydrogen turbines, sized by capacity.
:width: 100%

Electrolysers (rings) and hydrogen turbines (filled) at their nodes in the 2030 build (Clean Power 2030, ETYS network)
and the 2050 build (FES 2025 Holistic Transition, clustered network). In 2030 the turbines are small and spread over
about 200 sites.
```

### Import-headroom relief

In the CP30 market runs electrolysers are held at their Stage 1 schedule in the balancing stage (see below). A node
whose peak load plus electrolysis exceeds what its circuits can import could then be balanced only by shedding load.
`relieve_electrolyser_import_headroom` prevents this at build time. For every node (a bus with its bus sections and
its substation's lower-voltage busbars) and every month:

$$
\text{peak load} + \text{electrolysis} \le \texttt{max\_loading} \times \text{import capacity},
$$

where import capacity is set by the first circuit to reach its rating when the node draws power from the
275 kV and 400 kV grid outside it (DC power-flow shares, at the ratings the balancing stage enforces). Storage
charging is not counted, since the balancing stage can stop it, and local generation gets no credit. `max_loading`
is 0.9 in the brownfield overlay (a margin for through-flow); 0 switches the relief off.

Excess electrolysis moves, lowest-voltage node first and by exactly the MW that clears the binding circuit, to:

1. a higher-voltage node of the same substation; else
2. the nearest node at or above `min_target_voltage_kv` in the same CP30 region and on the same side of each boundary
   in `hydrogen.electrolyser_siting.boundaries`, so the regional split behind those boundaries survives.

Moved parts are named `{site}_relief_{bus}`.

### Hydrogen demand and turbine fuel

- **Offtake.** With `hydrogen.exogenous_demand.enabled`, the pathway's electrolysis electricity demand (CP30 ES1)
  converted to hydrogen at the electrolyser efficiency is a flat load on `GB_H2`. Without it, electrolysis would run
  only to refuel the turbines. `market.hydrogen_offtake_scale` can rescale it at solve time.
- **Turbine fuel.** With `hydrogen.turbine_fuel.source: external_supply` (CP30), the turbines burn hydrogen bought
  from the wider hydrogen system at a fixed price (`price_gbp_per_mwh_h2`, GBP/MWh of hydrogen, LHV) from their own
  bus and store, so electrolysis serves the offtake alone. With `electrolysis` (code default) they burn from `GB_H2`.
- **Turbine VOM.** A link's marginal cost is per MWh of its input. `hydrogen.turbine_variable_om.basis:
  electricity_output` puts the VOM (GBP/MWh of electricity) on the electricity leg as VOM x efficiency;
  `hydrogen_input` puts it on the hydrogen input.
- **Turbine availability.** `hydrogen.turbine_availability` is a static `p_max_pu` on the turbine links.

### FES long-range hydrogen system

With `hydrogen.fes.enabled` (`scripts/hydrogen/fes_hydrogen.py`) the system follows FES 2025 for the modelled year:
electrolysers at FES's nodes and FES's efficiency, turbine links at the FES-implied efficiency, one store of FES's
hydrogen storage volume with no standing loss, non-power hydrogen demand (industry, transport, buildings, losses) as a
flat load, and gas reforming with CCS as a dispatchable link from a `GB_natural_gas` bus at the FES gas price, with
95% capture and a CO2 transport and storage fee (`hydrogen.fes.gas_reforming_ccs`).

The market solves run in windows, so an annual hydrogen volume cannot be imposed as one constraint. The offtake is a
flat hourly load instead; the store carries its level between windows, so the model still chooses when to make the
hydrogen.

### Hydrogen in the market stages

| Stage | Behaviour |
|---|---|
| Single solve and Stage 1 | electrolysers, turbines and the store are optimised with everything else. The store level carries between rolling windows (`market.hydrogen_store.carry_level`); hydrogen left at the end of a window is valued at `market.hydrogen_store.end_of_window_value_gbp_per_mwh_h2`, so an electrolyser runs when electricity costs less than efficiency x that value |
| Balancing | electrolysers and turbines are anchored to their Stage 1 position. Carriers in `market.balancing.hydrogen_links.hold_carriers` are held there; others move at prices derived from the Stage 1 hydrogen bus price. The base holds turbines; CP30 holds both turbines and electrolysers, so electrolysers are treated as demand fixed at the wholesale schedule |

When electrolysers may move, the store must end each balancing window within what the rest of the month can still
take back, and the month at Stage 1's level within `store_month_end_tolerance_pu`. See {doc}`../dispatch/balancing`.

## Data

| Input | File |
|---|---|
| FES building blocks (electrolysis, Dem_BB009) | `resources/FES/FES_{edition}_data.csv` |
| CP30 totals | `resources/CP30/CP30_ES1.csv`, `CP30_ED1.csv` |
| FES long-range targets and prices | `resources/FES/fes2025_*_targets.json`, `fes2025_future_costs_{year}.json` |
| Relief geography | GSP polygons, CP30 region map, `data/network/ETYS/etys_boundaries_mar25_osgb36.geojson` |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `hydrogen.electrolyser_siting.method` | `h2_turbine_pro_rata` (brownfield: `fes_gsp`) | electrolyser placement |
| `hydrogen.electrolyser_siting.max_loading` | 1.0 (brownfield: 0.9) | import-headroom relief; 0 = off |
| `hydrogen.electrolyser_siting.min_target_voltage_kv` | 275 | relief target voltage, kV |
| `hydrogen.electrolyser_siting.boundaries` | B4, B6, B8 | boundaries the relief must not cross |
| `hydrogen.exogenous_demand` | CP30: on | non-power hydrogen offtake |
| `hydrogen.turbine_fuel` | CP30: `external_supply` | turbine fuel source and price |
| `hydrogen.turbine_variable_om` | CP30: 5.0 GBP/MWh, `electricity_output` | turbine VOM |
| `hydrogen.fes` | FES long-range studies | FES hydrogen system |
| `market.hydrogen_store` | carry level, start half full | store handling across windows |
| `market.balancing.hydrogen_links.hold_carriers` | `[H2_turbine]` (CP30 adds `electrolysis`) | held in balancing |

There is no `hydrogen` block in the base defaults; the code defaults apply unless a method or source overlay, or a
study, sets the keys (`config/defaults/method/brownfield.yaml`, `config/defaults/source/CP30.yaml`, the FES study
templates).

## Validation

- CP30 checks compare electrolysis capacity and placement with CP30 (rule `validate_CP30_locational`) and electrolysis
  energy with ES1 after the solve (rules `validate_CP30_stage1`, `validate_CP30_dispatch`).
- The locational demand-headroom check applies the same import model as the relief to every node.
- FES long-range years are checked against FES ES1 (rule `validate_fes_dispatch`).

## Limitations

- One copperplate hydrogen bus: no pipelines, regional stores or hydrogen network constraints.
- Flat non-power hydrogen demand; no seasonal pattern.
- Electrolysers have a constant efficiency and no minimum load.
