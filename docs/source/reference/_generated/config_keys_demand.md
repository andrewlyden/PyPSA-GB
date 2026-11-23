## `demand`

File: `config/defaults/base/demand.yaml`. Keys below are relative to `demand`.

| Key | Default | Description |
|---|---|---|
| `timeseries` | `ESPENI` | Demand time-series shape: "ESPENI"    - historical half-hourly GB demand (historical runs use it directly) "eload"     - eLOAD model hourly profile (2010 or 2050 via profile_year) "desstinee" - DESSTINEE model hourly profile (2010 or 2050 via profile_year) Future runs scale the shape to FES ED1 consumer demand and allocate it by Dem_BB003 GSP shares. For eload/desstinee, profile_year defaults to 2050 if modelled_year &gt;= 2040, else 2010. |
| `disaggregation` | (mapping) | Demand disaggregation (off by default): split components out of total demand, using FES fractions and GSP spatial allocation for future scenarios. |
| `disaggregation.enabled` | `false` |  |
| `disaggregation.components` | (list of 2 entries) |  |
| `flexibility` | (mapping) | Demand-side flexibility from heat pumps, EVs and demand response, each switched on separately. Heat-pump and EV flexibility need demand.disaggregation enabled with that component. |
| `flexibility.enabled` | `false` | Master switch - must be true for any flexibility to be modeled |
| `flexibility.heat_pumps` | (mapping) | Heat-pump flexibility (needs heat_pumps in demand.disaggregation.components) |
| `flexibility.heat_pumps.enabled` | `false` |  |
| `flexibility.heat_pumps.mode` | `MIXED` | TANK, COSY, or MIXED (use mix shares below) |
| `flexibility.heat_pumps.flex_share` | `0.2` | Fraction of HP demand that is flexible |
| `flexibility.heat_pumps.mix` | (mapping) |  |
| `flexibility.heat_pumps.mix.tank_share` | `0.5` | Fraction of flexible HP demand using TANK mode |
| `flexibility.heat_pumps.mix.cosy_share` | `0.5` | Fraction of flexible HP demand using COSY mode |
| `flexibility.heat_pumps.urban_weight` | `1.5` | Spatial allocation Power exponent for urban-weighted allocation (&gt;1 = more urban) |
| `flexibility.heat_pumps.peak_thermal_kw_per_dwelling` | `10.0` | Assumed peak thermal output per dwelling (kW) |
| `flexibility.heat_pumps.link_sizing_margin` | `1.5` | Margin factor for HP link capacity (1.5 = 50% headroom) |
| `flexibility.heat_pumps.cosy` | (mapping) | COSY mode: pre-heating using building thermal mass |
| `flexibility.heat_pumps.cosy.morning_window` | `['07:00', '09:00']` | Morning pre-heat window |
| `flexibility.heat_pumps.cosy.evening_window` | `['17:00', '19:00']` | Evening pre-heat window |
| `flexibility.heat_pumps.cosy.max_temp_deviation_celsius` | `2.0` | Max comfort deviation (C) |
| `flexibility.heat_pumps.cosy.thermal_mass_hours` | `0.3` | Hours of heating that can be stored in building fabric |
| `flexibility.heat_pumps.cosy.standing_loss_per_hour` | `0.05` | (~2.7 kWh per dwelling at ~10 kW peak thermal) Building thermal loss rate (5%/hour, faster than tank) |
| `flexibility.heat_pumps.tank` | (mapping) | TANK mode: hot water tank storage |
| `flexibility.heat_pumps.tank.volume_liters` | `200` | Average tank volume (litres) |
| `flexibility.heat_pumps.tank.temp_range` | `[50, 65]` | Min/max temperature (C) |
| `flexibility.heat_pumps.tank.standing_loss_per_hour` | `0.01` | Heat loss fraction per hour (1%) |
| `flexibility.heat_pumps.tank.heater_power_kw` | `3.0` | Immersion heater power (kW) |
| `flexibility.electric_vehicles` | (mapping) | Electric-vehicle flexibility (needs electric_vehicles in demand.disaggregation.components). FES building blocks: Dem_BB006 domestic and Dem_BB007 non-domestic EV charging demand (GWh), Srg_BB005 V2G availability (MW; V2G and MIXED), Srg_BB007a smart-charging availability (MW; MIXED). Without flexibility EVs are Load components with an evening-peak profile; with it, Store and Link components let charging be optimised. |
| `flexibility.electric_vehicles.enabled` | `false` |  |
| `flexibility.electric_vehicles.tariff` | `MIXED` | GO, INT, V2G, or MIXED (see modes below) |
| `flexibility.electric_vehicles.flex_share` | `0.2` | Share of EV demand that is flexible (smart charging); the rest is fixed load. |
| `flexibility.electric_vehicles.battery_capacity_kwh` | `60.0` | Fleet characteristics (affect storage/charger sizing) Average EV battery size (kWh) |
| `flexibility.electric_vehicles.charge_efficiency` | `0.9` | Grid-to-battery efficiency |
| `flexibility.electric_vehicles.flexibility_participation` | `0.1` | Fraction of fleet participating in smart charging |
| `flexibility.electric_vehicles.urban_weight` | `2.0` | Spatial allocation and fleet sizing Power exponent for urban-weighted allocation (&gt;1 = more urban) |
| `flexibility.electric_vehicles.energy_per_vehicle_kwh_per_day` | `10.0` | Assumed daily energy consumption per vehicle (kWh) |
| `flexibility.electric_vehicles.charging_hours_per_day` | `4.0` | Assumed charging hours/day (for FES share calculation) |
| `flexibility.electric_vehicles.min_share_threshold` | `0.001` | Minimum tariff share to create components |
| `flexibility.electric_vehicles.go` | (mapping) | GO: fixed cheap overnight window (Octopus Go style). The cost difference steers charging into the window; charging outside it remains possible at offpeak_cost. |
| `flexibility.electric_vehicles.go.window` | `['00:00', '04:00']` | Cheap charging window (4 hours) |
| `flexibility.electric_vehicles.go.window_cost` | `0.0` | Marginal cost during cheap window (GBP/MWh) |
| `flexibility.electric_vehicles.go.offpeak_cost` | `100.0` | Marginal cost outside window (GBP/MWh) |
| `flexibility.electric_vehicles.int` | (mapping) | INT: intelligent smart charging, shiftable across all plugged-in hours |
| `flexibility.electric_vehicles.int.min_soc` | `0.2` | Minimum SOC (range anxiety buffer) |
| `flexibility.electric_vehicles.int.target_departure_soc` | `0.8` | Target SOC at departure time (7am) |
| `flexibility.electric_vehicles.int.charger_power_kw` | `7.0` | Home charger power (kW) |
| `flexibility.electric_vehicles.v2g` | (mapping) | V2G: bidirectional. Capacity from FES Srg_BB005 (distributed by EV demand) if use_fes_capacity, else fleet x participation_rate x charger_power. |
| `flexibility.electric_vehicles.v2g.participation_rate` | `0.3` | Fraction of flexible fleet with V2G (if not using FES) |
| `flexibility.electric_vehicles.v2g.discharge_efficiency` | `0.9` | Battery-to-grid efficiency |
| `flexibility.electric_vehicles.v2g.max_discharge_soc` | `0.8` | Multiplies the V2G discharge link's p_max_pu (share of plugged-in power) |
| `flexibility.electric_vehicles.v2g.degradation_cost_per_mwh` | `50.0` | Battery wear cost (GBP/MWh) |
| `flexibility.electric_vehicles.v2g.use_fes_capacity` | `true` | Use FES Srg_BB005 V2G capacity |
| `flexibility.electric_vehicles.mixed` | (mapping) | MIXED: the fleet split across GO, INT and V2G |
| `flexibility.electric_vehicles.mixed.mode` | `fes` | "fes": shares from FES building blocks; "manual": the shares below |
| `flexibility.electric_vehicles.mixed.go_share` | `0.3` | Manual shares (mode "manual", or when FES data is unavailable); must sum to 1.0 30% on Octopus Go style (fixed window) |
| `flexibility.electric_vehicles.mixed.int_share` | `0.5` | 50% on intelligent smart charging |
| `flexibility.electric_vehicles.mixed.v2g_share` | `0.2` | 20% with V2G capability |
| `flexibility.electric_vehicles.mixed.fes_share_overrides` | `null` | In "fes" mode: V2G share = Srg_BB005 (V2G MW) / total EV charging capacity, INT share the remaining Srg_BB007a smart charging, GO the remainder. Option: fix a share here (e.g. go_share: 0.20) and the others adjust. |
| `flexibility.electric_vehicles.availability` | (mapping) | Availability profile (from GB traffic patterns): share of EVs plugged in |
| `flexibility.electric_vehicles.availability.avail_max` | `0.95` | Max fraction plugged in (overnight) |
| `flexibility.electric_vehicles.availability.avail_mean` | `0.8` | Mean fraction plugged in |
| `flexibility.electric_vehicles.availability.departure_hour` | `7` | Morning departure time for DSM constraint |
| `flexibility.event_response` | (mapping) | Domestic demand response (Saving Sessions style events): households reduce demand during events |
| `flexibility.event_response.enabled` | `false` |  |
| `flexibility.event_response.mode` | `both` | mode: "regular" - events_per_week_regular all year "winter"  - events_per_week_winter in winter_months only "both"    - events_per_week_winter in winter_months, events_per_week_regular otherwise |
| `flexibility.event_response.events_per_week_regular` | `2` | Events per week in regular/both modes |
| `flexibility.event_response.events_per_week_winter` | `5` | Events per week in winter/both modes |
| `flexibility.event_response.event_window` | `['07:00', '22:00']` | Event window [start, end]; e.g. ["17:00", "19:00"] evening peak, ["16:00", "20:00"] extended evening, ["07:00", "22:00"] full day Full day |
| `flexibility.event_response.dsr_capacity_mw` | `null` | Domestic DSR capacity (MW); null = participation_rate x max_reduction_fraction x demand |
| `flexibility.event_response.participation_rate` | `0.33` | Fallback: participation rate (33%) |
| `flexibility.event_response.max_reduction_fraction` | `0.1` | Fallback: max demand reduction per event (10%) |
| `flexibility.event_response.marginal_cost` | `100.0` | DSR incentive payment (GBP/MWh); high, so dispatched only when needed |
| `flexibility.event_response.winter_months` | `[10, 11, 12, 1, 2, 3]` | Oct-Mar for winter mode |
