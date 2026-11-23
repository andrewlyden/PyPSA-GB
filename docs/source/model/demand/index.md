# Demand

Electricity demand is the load every solve must meet, bus by bus and hour by hour. These pages cover where the annual
totals come from, how they are given an hourly shape and how they are placed on the network. The part of future
demand that can move in time is built as consumer flexibility ({doc}`../flexibility/consumer`).

## What it does

The demand stage turns a scenario's demand year and targets into PyPSA `Load` components on the network's buses, and,
where configured, into consumer-flexibility units that can shift consumption in time. How it does so depends on the
scenario:

| Scenario type | Annual energy | Hourly shape | Placement |
|---|---|---|---|
| {doc}`historical` (`modelled_year` <= 2024) | ESPENI outturn | ESPENI, half-hourly, for `demand_year` | fixed weights by GSP (FES 2023 regional breakdown) or by bus |
| {doc}`fes_gsp_blocks` (`method: gsp_block`) | FES consumer demand (ED1) | ESPENI (or eLOAD / DESSTINEE) of `demand_year` | FES GSP demand shares (Dem_BB003) |
| {doc}`cp30` (`method: brownfield`, ETYS) | CP30 ED1 with losses | component-built, calibrated to a peak target | FES 2024 GSP shares plus named transmission loads |
| {doc}`fes_long_range` (`demand.disaggregation.fes`) | FES ED1 lines, component by component, with losses | component-built on the weather year | GSP demand nodes, components at their own GSP shares |

{doc}`placement` covers how GSP demand reaches the buses. In the CP30 and FES long-range builds, the flexible part of
each component is carved out of these loads and built as consumer-flexibility units ({doc}`../flexibility/consumer`).

The rules are in `rules/demand.smk`: `build_base_demand` (script `scripts/demand/load.py`) builds the base load,
`extract_cp30_demand_inputs` prepares the CP30 named loads, `build_fes_demand_components` builds the FES long-range
components, and `finalize_demand` (script `scripts/demand/finalize_demand_integration.py`) adds disaggregated
components and flexibility. The output is `resources/network/{scenario}_network_demand.pkl`.

## Calendar alignment

A model year has 8,760 hours, or 8,784 in a leap year. When exactly one of the source year and the modelled year is a
leap year, series are matched by month, day and hour, not by day of year (`scripts/utilities/calendar_year.py`). A
29 February the source lacks takes 28 February's values; a source 29 February the modelled year lacks is dropped. The
same rule applies to the temperature for heat shapes, embedded profiles and interconnector availability.

One exception: at the hourly time step the half-hourly ESPENI demand shape is still matched by day of year, so in a
leap model year (2040) it reads the source day one later from 1 March ({doc}`fes_long_range`).

## Configuration shared by every build

| Key | Default | Meaning |
|---|---|---|
| `demand_year` | (scenario) | year of the ESPENI shape |
| `renewables_year` | (scenario) | weather year (temperature for heat shapes) |
| `timestep_minutes` | `60` | 60 or 30 |

The defaults are in `config/defaults/base/demand.yaml`, the brownfield build in `config/defaults/method/brownfield.yaml`
and the CP30 inputs in `config/defaults/source/CP30.yaml`. To see which file sets a value for a scenario:

```bash
python config/config_loader.py --explain <scenario_id> --path demand
```

See {doc}`/reference/data/data_sources` for each source's licence and version.

## Limitations shared by every build

- One weather year drives both the demand shape (through temperature) and renewable output; the ESPENI base shape is
  from `demand_year`, which studies set to the same year.
- Demand does not respond to price except through the flexibility units.

```{toctree}
:maxdepth: 1

historical
fes_gsp_blocks
cp30
fes_long_range
placement
```
