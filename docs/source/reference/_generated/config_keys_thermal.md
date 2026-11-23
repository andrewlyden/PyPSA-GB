## `thermal`

File: `config/defaults/base/thermal.yaml`. Keys below are relative to `thermal`.

| Key | Default | Description |
|---|---|---|
| `nuclear_smoothing` | `hourly` | Smoothing of ESPENI-derived nuclear availability before it is applied as p_max_pu: "hourly" keeps the raw signal (individual plant outages), "daily" or "monthly" average it over days or months. "hourly" \| "daily" \| "monthly" |
| `availability_factors` | (mapping) | Thermal availability factors (share 0-1 of capacity): caps on thermal output for real-world availability. Biomass is limited by pellet supply and maintenance; small biofuel CHP plants (biogas, landfill and sewage gas) follow heat demand. Historical values are calibrated to ESPENI dispatch or Ofgem RO output. |
| `availability_factors.enabled` | `false` |  |
| `availability_factors.generator_monthly_file` | `null` | Optional station-by-month evidence for historical validation; named values replace (not multiply) the carrier factor. |
| `availability_factors.factors` | (mapping) |  |
| `availability_factors.factors.biomass` | `0.53` | 2252/4225 (ESPENI actual / installed) |
| `availability_factors.factors.biogas` | `0.15` | CHP heat-led, low electrical load factor |
| `availability_factors.factors.landfill_gas` | `0.527` | 2020 Ofgem ROC-eligible output / issuing-station DNC |
| `availability_factors.factors.sewage_gas` | `0.5593` | 2020 Ofgem ROC-eligible output / issuing-station DNC |
| `availability_factors.factors.advanced_biofuel` | `0.15` | CHP / small distributed |
| `availability_factors.factors.waste_to_energy` | `0.8` | More continuous operation |
| `availability_factors.min_output` | (mapping) | Must-run minimum output (p_min_pu, share of capacity) |
| `availability_factors.min_output.biomass` | `0.4` | ROC revenue protection; rarely below 40% |
| `availability_factors.min_output.waste_to_energy` | `0.5` | Continuous waste processing obligation |
| `availability_factors.min_output_energy_window_hours` | `{}` | {carrier: hours}. At solve time the listed carriers' hourly floor becomes an energy budget: each unit delivers at least the floor's energy in every block of that many hours of a Stage 1 optimisation, timing free up to p_max_pu, and the BM drops the hourly floor (scripts/market/market_utils.py). Empty = off. |
| `dispatch_anchors` | (mapping) | Option for historical validation: anchor the listed carriers to observed commitment decisions the LP does not represent. Never inherited by future scenarios unless set. |
| `dispatch_anchors.enabled` | `false` |  |
| `dispatch_anchors.carriers` | `[]` |  |
| `outages` | (mapping) | Reproducible planned maintenance and forced outages for thermal plant, applied to p_max_pu only (no unit commitment). Per carrier: planned outages per year, their duration (hours) and months, forced outage rate (share of hours) and mean forced-outage duration (hours). |
| `outages.enabled` | `true` |  |
| `outages.seed` | `42` |  |
| `outages.carriers` | (mapping) |  |
| `outages.carriers.CCGT` | (mapping) |  |
| `outages.carriers.CCGT.planned_outages_per_year` | `1` |  |
| `outages.carriers.CCGT.planned_outage_duration_hours` | `168` |  |
| `outages.carriers.CCGT.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `outages.carriers.CCGT.forced_outage_rate` | `0.06` |  |
| `outages.carriers.CCGT.forced_outage_mean_duration_hours` | `72` |  |
| `outages.carriers.OCGT` | (mapping) |  |
| `outages.carriers.OCGT.planned_outages_per_year` | `1` |  |
| `outages.carriers.OCGT.planned_outage_duration_hours` | `96` |  |
| `outages.carriers.OCGT.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `outages.carriers.OCGT.forced_outage_rate` | `0.04` |  |
| `outages.carriers.OCGT.forced_outage_mean_duration_hours` | `48` |  |
| `outages.carriers.gas_engine` | (mapping) |  |
| `outages.carriers.gas_engine.planned_outages_per_year` | `1` |  |
| `outages.carriers.gas_engine.planned_outage_duration_hours` | `96` |  |
| `outages.carriers.gas_engine.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `outages.carriers.gas_engine.forced_outage_rate` | `0.04` |  |
| `outages.carriers.gas_engine.forced_outage_mean_duration_hours` | `48` |  |
| `outages.carriers.CCS Gas` | (mapping) |  |
| `outages.carriers.CCS Gas.planned_outages_per_year` | `1` |  |
| `outages.carriers.CCS Gas.planned_outage_duration_hours` | `168` |  |
| `outages.carriers.CCS Gas.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `outages.carriers.CCS Gas.forced_outage_rate` | `0.08` |  |
| `outages.carriers.CCS Gas.forced_outage_mean_duration_hours` | `96` |  |
| `outages.carriers.CCS Biomass` | (mapping) |  |
| `outages.carriers.CCS Biomass.planned_outages_per_year` | `1` |  |
| `outages.carriers.CCS Biomass.planned_outage_duration_hours` | `168` |  |
| `outages.carriers.CCS Biomass.planned_outage_months` | `[4, 5, 6, 7, 8, 9]` |  |
| `outages.carriers.CCS Biomass.forced_outage_rate` | `0.1` |  |
| `outages.carriers.CCS Biomass.forced_outage_mean_duration_hours` | `96` |  |
| `ramping` | (mapping) | Per-unit ramp limits (share of p_nom per hour): linear LP constraints, not unit commitment. |
| `ramping.enabled` | `true` |  |
| `ramping.carriers` | (mapping) |  |
| `ramping.carriers.CCGT` | (mapping) |  |
| `ramping.carriers.CCGT.ramp_limit_up` | `0.5` |  |
| `ramping.carriers.CCGT.ramp_limit_down` | `0.5` |  |
| `ramping.carriers.OCGT` | (mapping) |  |
| `ramping.carriers.OCGT.ramp_limit_up` | `1.0` |  |
| `ramping.carriers.OCGT.ramp_limit_down` | `1.0` |  |
| `ramping.carriers.gas_engine` | (mapping) |  |
| `ramping.carriers.gas_engine.ramp_limit_up` | `1.0` |  |
| `ramping.carriers.gas_engine.ramp_limit_down` | `1.0` |  |
| `ramping.carriers.CCS Gas` | (mapping) |  |
| `ramping.carriers.CCS Gas.ramp_limit_up` | `0.35` |  |
| `ramping.carriers.CCS Gas.ramp_limit_down` | `0.35` |  |
| `ramping.carriers.Hydrogen` | (mapping) | CCS Biomass has no ramp limit: it is must-run at full availability, and a limit would make that floor infeasible at the end of an outage. |
| `ramping.carriers.Hydrogen.ramp_limit_up` | `0.5` |  |
| `ramping.carriers.Hydrogen.ramp_limit_down` | `0.5` |  |
