## `renewables`

File: `config/defaults/base/renewables.yaml`. Keys below are relative to `renewables`.

| Key | Default | Description |
|---|---|---|
| `aggregation` | (mapping) | Renewable aggregation (off by default): merges renewable generators of the same bus and carrier into one, with the capacity-weighted average p_max_pu profile (p_nom summed). Cuts the generator count without changing optimisation results. Independent of spatial clustering. |
| `aggregation.enabled` | `false` |  |
| `aggregation.preserve_support_types` | `[]` | Support types (e.g. named scheme cohorts) kept as individual generators for station-level certificate/payment accounting. |
| `aggregation.carriers` | (list of 8 items) | Carriers eligible for aggregation |
| `performance_factors` | (mapping) | Renewable performance factors: scale p_max_pu so ERA5 reanalysis and idealised power curves (NREL_5MW offshore, Vestas_V112_3MW onshore) give realistic uncurtailed availability. They must not include curtailment, which the BM stage models: the wholesale (pre-BM) generation should approximate ESPENI metered output plus real curtailment, not ESPENI alone. The factor applied to each carrier is the product of four components: era5_bias      ERA5 wind speed / irradiance overestimate turbine_curve  idealised power curve vs real fleet efficiency wake_losses    farm-scale wake interactions (absent from atlite) availability   maintenance, faults, grid connection downtime With the values below: wind_offshore 0.83 x 0.93 x 0.90 x 0.93 = 0.646, wind_onshore 0.91 x 0.95 x 0.98 x 0.95 = 0.805, solar_pv 0.95 x 0.97 x 1.00 x 0.97 = 0.894. References: Staffell & Pfenninger (2016) doi:10.1016/j.energy.2016.08.004; Bloomfield et al. (2020) doi:10.1002/met.1858; Barthelmie et al. (2010) doi:10.1002/we.348 (wake losses) Diagnostic: scripts/generators/calibrate_wind_profiles.py compares ERA5 totals with ESPENI metered output (see the script for the curtailment caveat). |
| `performance_factors.enabled` | `true` | Apply the factors during generator build |
| `performance_factors.era5_bias` | (mapping) | Stacked component factors (share 0-1 per carrier); their product is applied. ERA5 reanalysis wind speed / irradiance overestimate |
| `performance_factors.era5_bias.wind_offshore` | `0.83` | North Sea ~+15% speed, ~+20% power (1/1.20 = 0.83) |
| `performance_factors.era5_bias.wind_onshore` | `0.91` | UK onshore ~+7% speed, ~+10% power (1/1.10 = 0.91) |
| `performance_factors.era5_bias.solar_pv` | `0.95` | ERA5 SSRD small positive bias, ~+3-5% |
| `performance_factors.turbine_curve` | (mapping) | Idealised power curve vs real fleet efficiency |
| `performance_factors.turbine_curve.wind_offshore` | `0.93` | NREL_5MW ideal vs real fleet (5 to 7%) |
| `performance_factors.turbine_curve.wind_onshore` | `0.95` | Vestas_V112 ideal vs real fleet (~5%) |
| `performance_factors.turbine_curve.solar_pv` | `0.97` | Module temperature de-rating + inverter losses |
| `performance_factors.wake_losses` | (mapping) | Farm-scale wake interactions (not modelled in atlite) |
| `performance_factors.wake_losses.wind_offshore` | `0.9` | 5-15% depending on farm size/spacing |
| `performance_factors.wake_losses.wind_onshore` | `0.98` | 1-3% (wider spacing, less dense farms) |
| `performance_factors.wake_losses.solar_pv` | `1.0` | N/A for solar |
| `performance_factors.availability` | (mapping) | Maintenance, faults, grid connection downtime |
| `performance_factors.availability.wind_offshore` | `0.93` | 5-8% downtime (subsea cable, harsh environment) |
| `performance_factors.availability.wind_onshore` | `0.95` | 3-6% downtime |
| `performance_factors.availability.solar_pv` | `0.97` | 2-4% downtime (soiling, inverter faults) |
| `performance_factor_registry` | (mapping) | Option: flat effective factors per carrier. If 'factors' is set and non-empty it is used as-is and the stacked components above are ignored, e.g.: factors: wind_offshore: 0.82 wind_onshore:  0.80 solar_pv:      0.88 Optional registry of factors by carrier and physical resource region. Separate from support labels (CfD/ROC/merchant are the accounting regime, not the weather resource). |
| `performance_factor_registry.enabled` | `false` |  |
| `performance_factor_registry.file` | `data/market/support/renewable_performance_factor_registry.csv` |  |
| `embedded` | (mapping) | Embedded generation (historical scenarios only). REPD omits small-scale and rooftop solar (and embedded wind) below transmission, but TOTAL_ESPENI includes their output. When enabled, the build reads measured embedded solar (and wind) from the ESPENI NGEM columns for demand_year, writes the MW series to resources/demand/{scenario}_embedded_{carrier}.csv, subtracts the modelled output of the REPD generators already in the network (solar_pv, wind_onshore) and adds embedded_solar / embedded_wind generators for the residual. Demand is unchanged. |
| `embedded.enabled` | `true` |  |
| `embedded.solar` | `true` | Include embedded solar from ESPENI |
| `embedded.wind` | `true` | Include embedded wind from ESPENI |
| `embedded.wind_overlap_exclusion` | (mapping) |  |
| `embedded.wind_overlap_exclusion.enabled` | `false` | Validation option: exclude small REPD onshore wind from ELEX_WIND bucket |
| `embedded.wind_overlap_exclusion.max_capacity_mw` | `10.0` | Sites &lt;= this are represented by ESPENI embedded_wind instead |
| `embedded.wind_overlap_exclusion.preserve_explicit_support` | `true` | Keep source-recorded CfD/ROC projects individually auditable |
| `hydro` | (mapping) | Hydro and pumped hydro. Large hydro is reservoir-backed: an hourly power ceiling from the profile plus a cumulative water budget carried between rolling windows. |
| `hydro.enabled` | `true` |  |
| `hydro.large_hydro` | (mapping) |  |
| `hydro.large_hydro.reservoir_model` | `true` |  |
| `hydro.large_hydro.dukes_legacy_supplement` | `true` | Fill pre-REPD major stations from DUKES 5.11 |
| `hydro.large_hydro.roc_hydro_supplement_file` | `null` | Optional Ofgem-evidenced legacy-station supplement |
| `hydro.large_hydro.inflow_profile_factor` | `0.55` | Convert p_max_pu profile into effective inflow |
| `hydro.large_hydro.initial_storage_hours` | `48.0` | Water already in reservoir at period start |
| `hydro.large_hydro.reservoir_capacity_hours` | `168.0` | Approximate upper-reservoir energy capacity |
| `hydro.large_hydro.month_end` | `at_least_initial` | Reservoir level at the end of each market run (Stage 1 and BM; scripts/solve/hydro_constraints.py): at_least_initial - the store ends at or above initial_storage_hours, so output cannot exceed inflow; rolling windows end at or above the lowest level from which the rest of the run can still refill it at minimum output free             - no end condition trajectory       - start and end on an annual path built from the inflow (needs trajectory: {mean_level_hours, smoothing_days}) |
| `hydro.pumped_hydro` | (mapping) | Option (not set here; the CP30 2030 study sets it): power_ceiling "inflow_scaled" (code default: p_max_pu = inflow / inflow_profile_factor) or "availability" (p_max_pu = availability_pu, the inflow in its own series; scripts/renewables/large_hydro_reservoir.py). Changing it needs a rebuild. Pumped hydro: reservoir losses are negligible at hourly resolution (standing losses are in storage.standing_losses); a minimum SOC keeps flexibility for balancing. |
| `hydro.pumped_hydro.initial_soc_fraction` | `0.65` | Start with the upper reservoir partly full |
| `hydro.pumped_hydro.min_soc_fraction` | `0.15` | Preserve headroom for redispatch services |
| `hydro.pumped_hydro.wholesale_min_soc_fraction` | `0.4` | Higher floor in wholesale, preserving SoC for BM increases |
| `hydro.pumped_hydro.final_soc_fraction` | `null` | Optional end-of-horizon reserve target |
| `hydro.pumped_hydro.cyclic_state_of_charge` | `false` |  |
| `hydro.pumped_hydro.existing_sites` | (list of 4 entries) | Existing pumped hydro within CP30's LDES (carrier LAES), each at its own substation (site_code, the four-character ETYS code; the best-connected busbar, shared with its siblings if one cannot carry it) with its real power (MW) and energy (MWh). The rest of CP30's LDES power and energy is new build spread over the FES proxy at the residual duration, so CP30's LDES totals stay exact. Sources: en.wikipedia.org/wiki/Dinorwig_Power_Station; drax.com/about-us/our-sites-and-businesses/ cruachan-power-station; en.wikipedia.org/wiki/Cruachan_Power_Station. Keep public_ownership.existing_storage_sites (existing_mw) consistent. |
