## `fleet`

File: `config/defaults/base/fleet.yaml`. Keys below are relative to `fleet`.

| Key | Default | Description |
|---|---|---|
| `station_registry` | (mapping) | Public records used before BMU-to-station and support matching. The workflow fetches the Elexon, LCCC and NESO sources; the Ofgem RO accredited-stations extract is a local input (no stable public API). |
| `station_registry.enabled` | `false` | The CP30 2030 study enables it; other builds do not need the registry. |
| `station_registry.registry_id` | `gb_public_current` |  |
| `station_registry.repd_file` | `data/renewables/repd-q2-jul-2025.csv` |  |
| `station_registry.power_station_dictionary_ids_file` | `data/generators/power_station_dictionary_ids.csv` |  |
| `station_registry.ofgem_ro_accredited_stations_file` | `null` |  |
| `brownfield` | (mapping) | Brownfield fleet: future renewable capacity built from named REPD projects, with proxies for the residual only. The targets (CP30: ES1 nationally, ES5 by region) set how much; the named fleet sets what and where inside them. Enabled under fleet.spatial_proxy.brownfield, whose keys override these. |
| `brownfield.enabled` | `false` |  |
| `brownfield.named_exclusions_file` | `data/renewables/cp30_named_fleet_exclusions.csv` | Named REPD projects held out of the modelled-year fleet on TEC evidence (e.g. the Lewis wind farms, which connect on the post-2030 Western Isles link). The residual proxy backfills them. |
| `brownfield.statuses` | `[Operational, Under Construction, Awaiting Construction]` | REPD development statuses treated as delivering by the modelled year. Option: add "Application Submitted" (not by default: a submitted application has no consent). |
| `brownfield.require_expected_date` | `false` | false: a consented project with no stated date is assumed online by the modelled year (CP30's own assumption). true: keep only projects with an explicit expected operational date. |
| `brownfield.retirement_lifetimes_years` | (mapping) | Technical lifetimes (years) from the REPD operational date; REPD records no decommissioning. Source: DESNZ Electricity Generation Costs 2023. |
| `brownfield.retirement_lifetimes_years.wind_onshore` | `30` |  |
| `brownfield.retirement_lifetimes_years.wind_offshore` | `30` |  |
| `brownfield.retirement_lifetimes_years.solar_pv` | `35` |  |
| `brownfield.overshoot_policy` | `trim_least_mature` | When named capacity in a region exceeds the regional target: trim_least_mature  remove pipeline projects, least mature first (never operational plant); error              stop for review. |
| `brownfield.offshore_landing_policy` | `nearest_bus_fallback` | Named offshore projects with no crosswalk row: "nearest_bus_fallback" places them at the nearest bus to their REPD coordinates and flags them; "error" refuses to build. |
| `brownfield.offshore_landing_crosswalk` | `data/network/ETYS/repd_offshore_landing_crosswalk.csv` |  |
| `brownfield.offshore_fallback_min_voltage_kv` | `275` | Fallback-placed farms of at least offshore_fallback_min_voltage_capacity_mw (MW) move to the nearest bus of at least offshore_fallback_min_voltage_kv (kV), so a large farm does not land on a 132 kV busbar; smaller farms stay. Matches min_bus_voltage_kv on the proxy landing map. |
| `brownfield.offshore_fallback_min_voltage_capacity_mw` | `300` |  |
