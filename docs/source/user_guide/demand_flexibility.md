# Demand-Side Flexibility

PyPSA-GB supports three integrated demand-side flexibility mechanisms that allow demand to shift in time or reduce during peak periods. These are modelled using PyPSA's native components (stores, links, generators) and can be enabled independently or together.

The methodology is built upon this work:

> Franken, L., Hackett, A., Lizana, J., Riepin, I., Jenkinson, R., Lyden, A., Yu, L. and Friedrich, D., 2025. Power system benefits of simultaneous domestic transport and heating demand flexibility in Great Britain's energy transition. *Applied Energy*, 377, p.124522. [doi:10.1016/j.apenergy.2024.124522](https://doi.org/10.1016/j.apenergy.2024.124522)

## Overview

| Flexibility Type | Mechanism | PyPSA Model | Duration |
|-----------------|-----------|-------------|----------|
| **Heat Pumps** | Pre-heat water tanks or building fabric | Store + Link + Load | Hours |
| **Electric Vehicles** | Smart charging and Vehicle-to-Grid | Store + Link + Load | Hours |
| **Event Response** | Saving Sessions style load reduction | Generator (negative load) | Minutes-hours |

All three are controlled via the `demand_flexibility` section of the configuration and orchestrated through the `finalize_demand` Snakemake rule.

## How It Works

```{mermaid}
flowchart TB
    subgraph Disaggregation["Demand Disaggregation"]
        HP_DIS["Disaggregate Heat Pumps"]
        EV_DIS["Disaggregate EVs"]
    end

    subgraph Profiles["Flexibility Profiles"]
        COP["COP Profiles (Atlite)"]
        EV_AVAIL["EV Availability"]
    end

    subgraph Integration["Flexibility Integration"]
        HP_FLEX["Heat Pump Flexibility"]
        EV_FLEX["EV Flexibility"]
        EVENT["Event Response"]
    end

    BASE["Base Demand"] --> HP_DIS
    BASE --> EV_DIS
    HP_DIS --> HP_FLEX
    COP --> HP_FLEX
    EV_DIS --> EV_FLEX
    EV_AVAIL --> EV_FLEX
    BASE --> EVENT

    HP_FLEX --> FINAL["Finalized Network"]
    EV_FLEX --> FINAL
    EVENT --> FINAL
```

### Flex Share

Each flexibility type has a `flex_share` parameter (0-1) controlling what fraction of disaggregated demand participates in flexibility:

- `flex_share: 0.2` means 20% uses smart flexibility (optimisable stores/links), 80% remains as rigid loads
- This prevents double-counting when disaggregation also models the same demand

## Heat Pump Flexibility

Heat pump flexibility exploits thermal storage to shift electrical demand. Two modes are available:

### TANK Mode (Hot Water Cylinder)

Pre-heats a hot water tank (~200L, 50-65 C) during low-price periods and draws from it during peaks.

**PyPSA components per bus:**
- 1 Bus (thermal carrier)
- 1 Store (hot water tank)
- 1 Link (heat pump: electricity to heat at COP efficiency)
- 1 Load (hot water demand)

**Key parameters:**
- Storage capacity: determined by tank volume and temperature range
- Standing loss: ~1% per hour (well-insulated cylinder)
- Efficiency: COP (temperature-dependent, typically 2-5)

### COSY Mode (Building Thermal Inertia)

Pre-heats building fabric during morning (07:00-09:00) and evening (17:00-19:00) windows, then allows temperature to drift down during peak demand.

**PyPSA components per bus:**
- 1 Bus (thermal inertia carrier)
- 1 Store (building thermal mass, state = temperature deviation)
- 1 Link (charging: pre-heating with efficiency boost during windows)
- 1 Link (discharging: using stored thermal energy)
- 1 Load (space heating demand)

**Key parameters:**
- Maximum temperature deviation: 2 C from setpoint
- Standing loss: ~5% per hour (building envelope losses)
- Pre-heat window boost: 50% extra capacity during morning/evening windows

### MIXED Mode

Splits heat pump demand between TANK and COSY using configurable shares (default 50/50).

### Spatial Allocation

Heat pump demand is distributed across buses using one of four methods:

| Method | Description |
|--------|-------------|
| `proportional` | Allocate by existing base demand |
| `uniform` | Equal across all buses |
| `urban_weighted` | Weighted toward high-demand areas |
| `fes_gsp` | Use FES GSP-level distribution |

## Electric Vehicle Flexibility

EV flexibility models smart charging across three tariff types:

### GO Tariff (Fixed Window)

Incentivises charging during a cheap 4-hour night window (00:00-04:00) using marginal cost signals. Charging outside the window is possible but expensive.

**PyPSA components per bus:**
- 1 Bus (EV battery carrier)
- 1 Store (fleet battery)
- 1 Link (charger with cost incentive)
- 1 Load (driving demand)

### INT Tariff (Intelligent Smart Charging)

Fully optimisable charging whenever vehicles are plugged in. The optimiser shifts charging to minimise system cost while meeting driving demand and minimum state-of-charge constraints.

**PyPSA components per bus:**
- 1 Bus (EV battery carrier)
- 1 Store (battery with time-varying e_min_pu)
- 1 Link (charger with availability profile p_max_pu)
- 1 Load (driving demand)

### V2G Tariff (Vehicle-to-Grid)

Bidirectional charging allows EVs to discharge back to the grid during peak demand. Includes a degradation cost to account for battery wear.

**PyPSA components per bus:**
- 1 Bus (EV battery carrier)
- 1 Store (battery with minimum SOC constraint)
- 1 Link (charger: grid to battery)
- 1 Link (V2G discharge: battery to grid, with degradation cost)
- 1 Load (driving demand)

**V2G capacity sources (priority order):**
1. FES Srg_BB005 (V2G MW availability) if `use_fes_capacity: true`
2. Calculated from fleet size and charger power

### MIXED Mode

Splits the EV fleet across GO, INT, and V2G tariffs. Shares can be calculated from FES data (`mode: "fes"`) or set manually.

**FES mode share calculation:**
- V2G share = FES V2G capacity / total EV capacity
- INT share = FES smart charging capacity - V2G share
- GO share = remainder

## Event Response (Demand-Side Response)

Models Saving Sessions style demand response events where households reduce load during specific windows.

### Event Schedule

Events are generated based on the configured mode:

| Mode | Description |
|------|-------------|
| `regular` | 2 events/week year-round |
| `winter` | 5 events/week Oct-Mar only |
| `both` | 2/week year-round + 5/week extra in winter |

Events only occur during the configured window (default 07:00-22:00) on weekdays.

### Capacity Calculation

Two methods:

1. **User-defined**: Set `dsr_capacity_mw` directly (e.g., 5000 MW), distributed proportionally to peak demand
2. **Calculated** (fallback): `base_demand x participation_rate x max_reduction_fraction`

### PyPSA Representation

Demand response is modelled as **generators** (negative loads):
- `p_nom`: maximum load reduction (MW)
- `p_max_pu`: event schedule (0/1 time series)
- `marginal_cost`: incentive payment (dispatched only when cost-effective)

## Configuration Reference

All demand flexibility settings live under `demand_flexibility` in `config/defaults.yaml`:

```yaml
demand_flexibility:
  enabled: true                       # Master switch

  heat_pumps:
    enabled: true
    mode: "MIXED"                     # TANK, COSY, or MIXED
    flex_share: 0.2                   # Fraction using smart flexibility
    mix:
      tank_share: 0.5
      cosy_share: 0.5
    peak_thermal_kw_per_dwelling: 10.0
    link_sizing_margin: 1.5
    cosy:
      morning_window: ["07:00", "09:00"]
      evening_window: ["17:00", "19:00"]
      max_temp_deviation_celsius: 2.0
      thermal_mass_hours: 0.3
      standing_loss_per_hour: 0.05
    tank:
      volume_liters: 200
      temp_range: [50, 65]
      standing_loss_per_hour: 0.01
      heater_power_kw: 3.0

  electric_vehicles:
    enabled: true
    tariff: "MIXED"                   # GO, INT, V2G, or MIXED
    flex_share: 0.2
    battery_capacity_kwh: 60.0
    charge_efficiency: 0.90
    flexibility_participation: 0.10
    urban_weight: 2.0
    energy_per_vehicle_kwh_per_day: 10.0
    charging_hours_per_day: 4.0
    go:
      window: ["00:00", "04:00"]
      window_cost: 0.0
      offpeak_cost: 100.0
    int:
      min_soc: 0.20
      target_departure_soc: 0.80
      charger_power_kw: 7.0
    v2g:
      participation_rate: 0.30
      discharge_efficiency: 0.90
      max_discharge_soc: 0.80
      degradation_cost_per_mwh: 50.0
      use_fes_capacity: true
    mixed:
      mode: "fes"                     # "fes" or "manual"
      go_share: 0.30
      int_share: 0.50
      v2g_share: 0.20

  event_response:
    enabled: true
    mode: "both"                      # "regular", "winter", or "both"
    events_per_week_regular: 2
    events_per_week_winter: 5
    event_window: ["07:00", "22:00"]
    dsr_capacity_mw: null             # Set directly or leave null for calculated
    participation_rate: 0.33
    max_reduction_fraction: 0.10
    marginal_cost: 100.0
    winter_months: [10, 11, 12, 1, 2, 3]
```

## PyPSA Component Summary

| Flexibility Type | Buses | Stores | Links | Loads | Generators |
|-----------------|-------|--------|-------|-------|------------|
| HP TANK | `{bus} heat` | Hot water tank | Heat pump (elec to heat) | Hot water demand | - |
| HP COSY | `{bus} thermal inertia` | Building thermal mass | Charge link + Discharge link | Space heating | - |
| EV GO | `{bus} EV battery` | Fleet battery | Charger (with cost) | Driving demand | - |
| EV INT | `{bus} EV battery` | Battery (with DSM) | Charger (with availability) | Driving demand | - |
| EV V2G | `{bus} EV battery` | Battery (with min SOC) | Charger + V2G discharge | Driving demand | - |
| Event Response | - | - | - | - | Demand response generator |

## Carriers

The following carriers are automatically added to the network:

**Heat-related:** `heat`, `hot water`, `hot water demand`, `heat pump`, `thermal inertia`, `space heating`, `thermal demand`

**EV-related:** `EV battery`, `EV charger`, `EV driving`, `EV driving demand`, `V2G`

**Demand response:** `demand response`

## Example Scenario

```yaml
# In config/scenarios.yaml
HT30_flex:
  description: "2030 Holistic Transition with demand flexibility"
  modelled_year: 2030
  FES_scenario: "Holistic Transition"
  network_model: "ETYS"

  demand_flexibility:
    enabled: true
    heat_pumps:
      enabled: true
      mode: "MIXED"
      flex_share: 0.2
    electric_vehicles:
      enabled: true
      tariff: "MIXED"
      flex_share: 0.2
    event_response:
      enabled: true
      mode: "both"
      dsr_capacity_mw: 5000
```

## Tutorials

For hands-on examples, see the tutorial notebooks:

- {doc}`../tutorials/13-heat-flexibility` - Heat pump flexibility (TANK/COSY modes, COP, thermal storage)
- {doc}`../tutorials/14-ev-flexibility` - EV flexibility (GO/INT/V2G tariffs, smart charging)
- {doc}`../tutorials/15-demand-side-response` - Event response (Saving Sessions, demand reduction)
