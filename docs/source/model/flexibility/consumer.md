# Consumer flexibility

Part of future demand can move in time: smart-charged and vehicle-to-grid EVs, pre-heating by heat pumps and heat
networks, storage heaters, demand response and direct air capture. This page covers how that flexibility is built,
how it behaves in the market stages, and the older optional modules other scenarios can switch on.

## Method and assumptions

### The units

In the CP30 and FES long-range builds, consumer flexibility is a set of PyPSA `StorageUnit` components on the AC buses
of the demand they shift (`scripts/demand/cp30_flexibility.py`, `scripts/demand/fes_demand.py`; carriers in
`scripts/demand/demand_carriers.py`). There is one unit per carrier per demand bus, placed with its component's GSP
shares ({doc}`../demand/cp30`, {doc}`../demand/fes_long_range`).

| Carrier | What it represents | Sized from |
|---|---|---|
| `EV_smart_charging` | smart-charged EVs | the smart share of residential EV energy |
| `V2G_fleet` | vehicle-to-grid | FES V2G at peak and maximum |
| `HP_flexibility` | heat-pump pre-heating | flexibility at peak (ED1 or FES FLX1) |
| `DH_flexibility` | district-heat pre-heating | flexibility at peak |
| `SH_flexibility` | storage heaters | direct-acting load at peak; charge windows |
| `IC_DSR`, `residential_DSR` | industrial, commercial and residential demand response | ED1 / FLX1 at peak |
| `DACCS` (FES long range) | direct air capture with storage | annual volume |

```{figure} /_static/figures/supply/consumer_flexibility.png
:alt: Paired horizontal bars of consumer-flexibility charging power in GW and energy buffer in GWh by type, for 2030 and 2050.
:width: 100%

Consumer-flexibility units by type in the 2030 build (Clean Power 2030) and the 2050 build (FES 2025 Holistic
Transition): charging power and energy buffer. Storage heaters hold the largest buffer; by 2050 smart charging and
vehicle-to-grid hold most of the power.
```

### How a unit works

- **Carve, then add.** The consumption the base shape already placed for that component (its reference) is removed
  from the base loads at the same buses, and the unit takes the same electricity through a fixed drain
  (`inflow` < 0): driving, space heat, a DACCS duty. Demand response is energy neutral, so nothing is carved for it.
  If a bus's load would fall below zero in any hour, that bus keeps part of its flexible energy as fixed load.
- **When to consume.** The unit charges whenever the optimiser chooses, within its plug-in or compressor limit and its
  energy buffer (`store_days` of driving for EVs; hours at peak for heat).
- **Windows.** Units start empty (`state_of_charge_initial = 0`, not cyclic). Each solve window must meet its own
  drain, so a unit can bring consumption forward but never push it later. Rolling market solves carry the state of
  charge between windows.
- **Holding costs.** Demand response pays a holding cost per MWh per hour held (activation cost divided by shift
  hours), so a shift lasts hours, not days. Drained units pay a small tie-breaking holding cost (GBP 0.005/MWh per hour)
  so flat or zero prices do not fill their stores with surplus that is never used.
- **Small units.** At solve time, consumer-flexibility units below 2 MW (`MIN_CONSUMER_FLEX_PNOM_MW` in
  `scripts/solve/solve_network.py`) are replaced by a fixed load at their own drain. Energy is conserved exactly; only
  the flexibility of those units is lost. This keeps the solver well conditioned.

### Sizing in the FES long-range build

The flexible share of each component is taken from FES and carved out of it; the rest stays a fixed load.

| Unit | Flexible share and drain | Power | Energy buffer |
|---|---|---|---|
| `HP_flexibility` | FLX1 heat-pump flexibility at peak / unmanaged heat-pump load at peak; drain = that share of the heat-pump load | 1.5 x peak drain | 2 h of peak drain; 2%/h standing loss |
| `DH_flexibility` | the same for district heat | 2.0 x peak drain | 12 h of peak drain; 0.5%/h loss |
| `EV_smart_charging` | the residential smart share (calibrated at peak, {doc}`../demand/fes_long_range`); drain on a driving shape, at 0.9 charge efficiency | 7 kW per flexible vehicle | 60 kWh x 0.6 per vehicle, capped at `store_days` (1.0) of mean driving |
| `V2G_fleet` | FLX1 V2G maximum / 7 kW vehicles, out of the flexible fleet | the larger of charging and FLX1's V2G maximum | as smart charging; 0.9 discharge efficiency, GBP 50/MWh degradation |
| `SH_flexibility` | FLX1 storage-heater load at peak, on the space-heat shape; carved from the base | coldest day's energy / 7 h | one coldest day; charging only 00:00-07:00 and 13:00-16:00 |
| `IC_DSR`, `residential_DSR` | FLX1 demand response at peak; no drain | the FLX1 figure | 2 h at power; GBP 30/MWh, GBP 15/MWh per hour held |
| `DACCS` | ES1 annual volume as a constant drain | mean load / 0.5 utilisation | 24 h of mean load |

Plug-in availability limits V2G by time of day: 0.9 x clip(1 - (1 - V2G at peak / V2G maximum) x driving / driving at
peak, 0.02, 1). The DACCS demand is met as a per-window pro-rata volume (drain x window hours), so its annual volume is
exact month by month while it chooses its hours within each window.

### In the market stages

In a two-stage market run (see {doc}`../dispatch/wholesale` and {doc}`../dispatch/balancing`):

- **Stage 1 (wholesale)** dispatches consumer flexibility like any other storage unit, against the copperplate price.
- **Balancing** holds every consumer-flexibility unit at its Stage 1 schedule
  (`market.balancing.participation.storage_units.hold_consumer_flexibility: true`). Households and businesses respond
  to their tariffs, not to balancing instructions. Setting the switch to `false` lets the units bid in balancing, as a
  sensitivity.
- **Accounting.** Outside the solve, the units count as demand: their net draw (`p_store - p_dispatch`) is added to the
  electrical load in the bill and market accounts, and they are left out of storage settlement, storage capacity
  checks and every ownership case.

```{figure} /_static/figures/supply/demand_winter_week_2030.png
:alt: Stacked hourly demand over a winter week, split into fixed load, heat flexibility and electric vehicles, above a line of the Stage 1 wholesale price.
:width: 100%

An example winter week (21 to 27 January) from the January Stage 1 solve of the 2030 study, weather year 2013: fixed
load and the consumption of the consumer-flexibility units as dispatched, with the Stage 1 price below. Heat flexibility
fills the overnight troughs; where the price is flat, as on most weekdays here, the timing is one of many equally cheap
choices.
```

### Optional flexibility modules (other scenarios)

Other future scenarios can switch on the earlier flexibility modules under `demand.disaggregation` and
`demand.flexibility` (`scripts/demand/add_demand_flexibility.py`). These follow Franken et al. (2025),
*Applied Energy* 377, 124522 (doi:10.1016/j.apenergy.2024.124522). They cannot be combined with the FES long-range
build.

| Module | Modes | PyPSA representation |
|---|---|---|
| Heat pumps | `TANK` (hot-water cylinder), `COSY` (building fabric), `MIXED` | thermal bus, `Store`, heat-pump `Link` at temperature-dependent COP, heat `Load` |
| Electric vehicles | `GO` (fixed cheap window), `INT` (smart charging), `V2G`, `MIXED` | EV battery bus, `Store`, charger `Link` with availability profile, driving `Load`; V2G adds a discharge `Link` with a degradation cost |
| Event response | `regular`, `winter`, `both` | `Generator` of demand reduction, available in event windows, at an incentive price |

Heat-pump and EV flexibility need the matching component in `demand.disaggregation.components`. `flex_share` sets the
share of the component's demand that is flexible; the rest stays a fixed load.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `demand.build.flexibility.enabled` | `true` under brownfield | CP30 consumer flexibility |
| `demand.disaggregation.fes` | set by the FES long-range studies | FES long-range components and flexibility (`electric_vehicles.store_days`, `residential_dsr`, `storage_heater_flexibility`, ...) |
| `demand.disaggregation.components` | heat pumps, EVs | optional disaggregation modules |
| `demand.flexibility.*` | off | optional heat-pump, EV and event-response modules |
| `market.balancing.participation.storage_units.hold_consumer_flexibility` | `true` | hold consumer flexibility at Stage 1 in balancing |

## Limitations

- Consumer flexibility has no explicit comfort or range constraint beyond its energy buffer and charging limits.
- The optional heat-pump and EV modules add their own buses and stores, which clustering keeps separate; the consumer
  flexibility units do not.
- Flexibility responds to the Stage 1 price with perfect foresight within each window.
