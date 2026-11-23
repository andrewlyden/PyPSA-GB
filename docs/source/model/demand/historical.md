# Historical years

A scenario with `modelled_year` up to 2024 (`method: historical`) takes its demand from outturn: one measured GB series,
spread over the network with fixed weights.

## Method and assumptions

### Energy and shape

Historical runs use ESPENI (Elexon Sum Plus Embedded Net Imports; Wilson et al. 2020, doi:10.5281/zenodo.3884858), the
half-hourly GB demand series `data/demand/espeni.csv`, for `demand_year`. The `TOTAL_ESPENI` column is used as measured,
in MW, with no rescaling: the year's energy and its hourly shape are both the outturn.

- **Embedded generation.** `TOTAL_ESPENI` already includes embedded (behind the meter) solar and wind, so demand is not
  reduced for them. Instead, the renewables stage adds `embedded_solar` and `embedded_wind` generators for the part of
  ESPENI's measured embedded output the REPD sites do not cover (`renewables.embedded`; see {doc}`../generation/renewables`).
- **Time step.** At the default 60-minute step, each pair of half-hours is averaged into one hour; at 30 minutes the
  series is used as it is.
- **Missing data.** If `demand_year` has fewer than 17,000 half-hours (about 97% of the year), a composite year is
  built instead: the mean of every day of year and half-hour over all ESPENI years.
- **Calendar.** A `demand_year` other than the modelled year is placed on it by the calendar rule
  ({doc}`index`).

### Spatial weights

ESPENI gives one GB series. It is spread over buses with fixed weights, so every bus takes the same hourly shape:

- **ETYS.** Each GSP's weight is its demand in the FES 2023 regional breakdown (ETYS 2023 Appendix E, sheet `Active`):
  Leading the Way in 2024, the mean of the morning and evening peak demand, summed over demand types. The GSP weights
  are summed to GSP groups and split to nodes by `Dem_per_node`'s share of each group's demand. Without the workbook,
  every GSP group gets an equal weight.
- **Reduced.** Each bus's `demand_weight` in `data/network/reduced_network/reduced_bus_definitions.csv`.
- **Zonal.** Each zone's summed `population_weight` in `data/network/zonal/zone_definitions.csv`.

On Reduced and Zonal networks, buses missing from the definitions file share the remaining weight equally. On ETYS, the
isolated sub-network and bus capacity steps then apply, with the `pro_rata` rule ({doc}`placement`).

## Data

| Input | File | Used for |
|---|---|---|
| ESPENI | `data/demand/espeni.csv` | demand and its shape; embedded solar and wind |
| FES 2023 regional breakdown | `data/network/ETYS/Regional breakdown of FES23 data (ETYS 2023 Appendix E).xlsb` | GSP weights (ETYS) |
| `Dem_per_node` | `data/network/ETYS/GB_network.xlsx` | GSP group to ETYS node |
| Reduced and Zonal definitions | `data/network/reduced_network/reduced_bus_definitions.csv`, `data/network/zonal/zone_definitions.csv` | bus and zone weights |

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `demand_year` | (scenario) | ESPENI year; studies set it to the modelled year |
| `timestep_minutes` | `60` | 60 (half-hours averaged) or 30 |
| `renewables.embedded.enabled` | `true` | add embedded solar and wind generators from ESPENI |

## Validation

- Every placement step checks that total demand is conserved (`assert_demand_conserved` in `load.py`).
- Historical demand and dispatch are validated over the whole of 2020 in {doc}`/studies/validation_2020`.

## Limitations

- One national shape at every bus: regional differences in timing (heating in Scotland, solar-heavy GSPs) are not
  represented.
- The ETYS weights are a 2024 projection from FES 2023, not outturn by GSP, and are used for every historical year.
