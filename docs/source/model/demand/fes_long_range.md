# FES long-range years

The FES long-range studies (FES 2025 Holistic Transition in 2040, 2045 and 2050) build demand as a non-electrified base
plus explicit components, each pinned to an FES 2025 line and given its own hourly shape on the weather year
(`scripts/demand/fes_demand.py`; rule `build_fes_demand_components`, joined to the network by `finalize_demand`). The
three studies use the same settings (`demand.disaggregation.fes`), with weather and demand-shape year 2013.

## Method and assumptions

### Annual energy

Every line is read from the FES 2025 data workbook for the pathway and model year, on a calendar-year basis:
calendar(Y) = 0.75 FY(Y) + 0.25 FY(Y-1) (`year_basis: calendar`).

| Component | FES line | Notes |
|---|---|---|
| heat pumps | Dem_BB004 + Dem_BB005 | |
| electric vehicles | Dem_BB006 + Dem_BB007 | split residential / commercial by ED1's EV rows, and depot (Dem_BB007, HGVs and buses) / public |
| district heat | ED1 residential district heat | |
| data centres | ED1 data-centre demand | inside commercial |
| base | ED1 consumer total less the four rows above | must be positive |
| DACCS | ES1 direct air capture electricity | outside ED1 |

The consumer total is residential + industrial (excluding electrolysis) + commercial + direct transmission customers;
electrolysis is excluded throughout and built in {doc}`../flexibility/hydrogen`. With `demand_basis: system` (the studies'
setting), the loads carry network losses: a single loss factor, ED1 system demand (excluding electrolysis) over the
consumer total, about 1.09 in HT 2050, multiplies every component and the base, so losses fall pro rata on every use.
DACCS is not uplifted. The FES at-peak figures used below are uplifted the same way.

### Where each component sits

Each component is spread over GSPs by its own FES building block, then onto buses through the same path as the base
loads ({doc}`placement`):

| Component | GSP shares |
|---|---|
| heat pumps, EV smart charging | Dem_BB004 + Dem_BB005; Dem_BB006 + Dem_BB007 |
| data centres | Dem_BB008b |
| district heat, storage heaters, residential demand response | Dem_BB008a |
| vehicle-to-grid | Srg_BB005 |
| industrial and commercial demand response | Srg_BB006a |
| DACCS | the AC bus nearest each declared CO2 cluster: Humber 0.35, Teesside 0.30, St Fergus 0.20, HyNet North West 0.15 |

**Carve-out.** The base loads (Dem_BB003 GSP shares, ESPENI shape, as {doc}`fes_gsp_blocks`) are first scaled to the
target total. At each bus the components allocated there, including the energy of the flexible units, are subtracted
from the bus's base energy (`residual_allocation: per_bus`), and the base series is scaled by a constant per bus, so it
keeps its hourly shape and only its level changes. Components are added as their own loads (carrier = component), and
flexible energy as `StorageUnit`s ({doc}`../flexibility/consumer`). Where a bus's components exceed its base energy, the base there
goes to zero and the clipped energy is logged; the total is rescaled to stay exact.

### Hourly shapes

**Temperature.** From the weather-year cutout (`resources/atlite/cutouts/uk-{renewables_year}.nc`): the cell nearest
each base-load bus, weighted by that bus's demand, so a demand-weighted GB temperature rather than a grid mean that
includes the sea.

**Heat pumps.** All treated as air source.

- Space heat follows heating degree hours, $\max(15.5 - T, 0)$, times a declared daily shape (overnight set-back, a
  morning peak and an evening plateau; the shape of the RHPP field-trial profiles, Love et al. 2017).
- Hot water is 20% of heat output (ECUK), on a declared draw-off shape (EST 2008) with a seasonal factor
  $1 + 0.10 \cos(2\pi(d - 15)/N)$, with $d$ the day of year and $N$ the days in the year, for colder mains water in
  January.
- Electricity is heat over the coefficient of performance, from the Staffell et al. (2012) air-source fit:

  $$
  \mathrm{COP} = 6.81 - 0.121\,\Delta T + 0.000630\,\Delta T^2, \qquad \Delta T = T_\mathrm{sink} - T,
  $$

  clipped to [1.5, 6.0], with a space-heating sink of $40 - T$ clipped to [30, 50] °C (Ruhnau et al. 2019) and hot
  water at 50 °C. The series is then scaled to the annual energy.

**District heat.** 70% daily-mean heating degree hours (heat networks buffer the daily swing) and 30% flat hot water; a
large heat pump at half the Carnot COP with a 70 °C sink, clipped to [1.5, 4.5].

**Electric vehicles.** Plug-in (unmanaged) charging shapes are declared mixtures of daily Gaussians on the model
calendar's weekdays:

| Use | Weekday | Weekend |
|---|---|---|
| home | 85% around 19:00 (sd 2 h), 15% around 08:00 | around 15:00 (sd 3.5 h) |
| public and workplace | 70% around 11:00, 30% around 14:00 | around 13:30 |
| depot (HGVs and buses) | around 01:00 (sd 2.5 h) | around 01:00, 0.6 of a weekday's energy |

Public charging on a weekend day is 0.8 of a weekday's energy. Home charging timing follows the My Electric Avenue and
Electric Nation trials.

**Data centres** are flat. **DACCS** is a constant duty that the optimiser schedules ({doc}`../flexibility/consumer`).

**Base.** ESPENI of `demand_year`, matched to the model year by day of year and hour, then scaled per GSP.

### Calibration at peak

ED1's peak is an average-cold-spell weekday, 17:00 to 18:00. The model's analogue is 17:00 on the ten coldest winter
(November to March) weekdays of the weather year. The residential smart-charging share of EVs is solved so that the
fixed EV load over those hours equals ED1's EVs at peak (capped at 0.95); the resulting smart-charging reduction lands on
FES's FLX1 figure without further tuning. Heat pumps and district heat are not calibrated at peak; their unmanaged
at-peak load is reported beside FES's.

## Data

| Input | File | Used for |
|---|---|---|
| FES 2025 data workbook (ED1, ES1, FLX1, building blocks) | `data/FES/` | every energy line, GSP shares, at-peak and flexibility figures |
| ESPENI | `data/demand/espeni.csv` | base shape |
| Weather cutout | `resources/atlite/cutouts/uk-{year}.nc` | temperature for heat shapes |
| GSP data | see {doc}`placement` | GSP to bus |

Outputs: `resources/demand/fes2050/{scenario}_profiles.csv`, `_allocation.csv` and `_spec.json` (the same folder for
every year).

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `demand.disaggregation.fes` | set by the FES long-range studies | the whole build; its defaults are `DEFAULT_CONFIG` in `fes_demand.py` |
| `...fes.demand_basis` | `system` | `system` (with losses) or `consumer` (at the meter) |
| `...fes.year_basis` | `calendar` | `calendar` or `fiscal` |
| `...fes.residual_allocation` | `per_bus` | carve per bus, or `flat` |
| `...fes.capacity_redistribution.mode` | `same_site` in the studies | bus capacity rule for the base loads ({doc}`placement`) |

## Validation

`fes_demand.demand_check` compares the network's energy with its target, line by line (total excluding DACCS, heat
pumps, EVs, district heat, data centres, base and DACCS): PASS within 0.5%, WARN otherwise. Peaks (the summed fixed
loads against ED1's peak, with and without the heat drains at their unmanaged times) and the calibration values are
reported. The table is written to the integration summary. The final dispatch is validated against FES ES1 (rule
`validate_fes_dispatch`).

## Limitations

- The base shape is a historical year's: matched by day of year, it does not keep weekdays (1 January 2013 was a
  Tuesday, 1 January 2040 a Sunday), while the EV shapes use the model calendar.
- In a leap model year at the hourly time step (2040), the base shape is matched by day of year, not by calendar date:
  from 1 March it reads the 2013 day one later than the temperature and weather, and 31 December takes the mean of that
  hour over the year.
- The components are spread onto buses with the `pro_rata` capacity rule, while the base loads use `same_site`; where a
  bus is capped, a component and the base it is carved from can land on different buses.
- Heat pumps are all air source, and the heat shapes are declared, not fitted to measured load.
