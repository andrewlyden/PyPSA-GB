# Release notes

## Version 2.3.0 (November 2026)

### Highlights

- **Network constraints.** Secured boundary limits (the sum of flows across each NESO boundary held to its secured
  capability), ETYS seasonal circuit ratings, transmission outage schedules (NESO day-ahead boundary limits,
  a user file or synthetic outages), ETYS Appendix B upgrades applied by year, and network-delivery cases
  that revert selected upgrades in the balancing stage to model delayed reinforcement. Branch impedances are carried in
  per unit on 100 MVA and converted to PyPSA's units once. See {doc}`/model/network/constraints`.
- **Two-stage market.** A copperplate wholesale stage (single, rolling day-ahead or look-ahead) followed by a
  network-constrained balancing mechanism anchored to the wholesale positions. Bids and offers are derived from
  marginal costs and support schemes: CfD units bid their contract value, RO units their lost certificates, merchant
  plant its saved fuel. Interconnector re-trading prices cross-border counter-trades in the BM; a central-dispatch mode
  solves the system at cost with full network knowledge; wind turn-down is reported both as solved (cheapest first) and
  re-shared pro rata by zone. See {doc}`/model/dispatch/index`.
- **CP30 2030 study.** NESO's Clean Power 2030 (Further Flex and Renewables) built on the full ETYS network from named
  projects, solved month by month under several network-delivery cases, and validated against CP30's capacity, energy
  and prices and NESO's balancing-cost estimates. See {doc}`/studies/cp30_2030`.
- **Year-generic FES long-range chain.** One chain builds FES 2025 Holistic Transition for any long-range model year;
  studies for 2040, 2045 and 2050 are included. See {doc}`/studies/fes_long_range`.
- **Regional weather profiles for FES capacity blocks.** An FES block has no site of its own, so every onshore wind or
  solar block in GB used to follow one profile. Each block now takes the average profile of the existing sites in its
  own region (offshore wind: its sea basin), with each technology's national energy unchanged
  (`renewables.proxy_profile_members: all_sites`, set for every FES scenario; issue #51). In the 2040 and 2050
  studies the NESO-comparable redispatch cost rises from GBP 2,064 m to 2,324 m and from 2,642 m to 2,888 m, and wind
  turn-down in the balancing stage from 10.9 to 12.6 TWh and from 15.2 to 16.5 TWh. See
  {doc}`/model/generation/renewables`.
- **2020 validation.** Calendar 2020 solved month by month on the full ETYS network and compared with outturn: over
  the year the constraint cost is 1.29 times NESO's thermal constraint cost, with flows on the Scottish boundaries
  close to NESO's. Historical builds now place stations and offshore farms on their declared connection buses, take
  large-hydro inflow from metered output, and keep embedded wind and solar out of the balancing mechanism. See
  {doc}`/studies/validation_2020`.
- **Older FES editions.** Scenarios can be built on FES 2021 to 2025 (`targets: {source: FES, edition, pathway}`); FES
  2020 is refused, as NESO published no GSP regional breakdown for it.
- **Consumer bill (preliminary).** An accounting layer that projects the domestic electricity bill from the solved
  market.
  See {doc}`/model/after_the_solve/index`.
- **NESO documents fetched, not shipped.** NESO website documents that may not be redistributed (the CP30 data
  workbook, the FES data workbooks 2020-2024, the ETYS Appendix B workbooks 2022-2024) are downloaded by rule `retrieve_neso_document` and checked against a SHA-256 in
  `data/neso_documents.yaml`.
- **Documentation rebuilt.** A new structure (Overview, Getting Started, Model Description, Studies, Tutorials,
  Reference, Development), a chapter for every part of the model, study pages with their results, and figures
  throughout, drawn from the model by `scripts/analysis/docs_figures/`.
- **New tutorials.** Five notebooks follow one week, 10-16 February, through 2020, 2030 and 2040: a first run against
  the real week, the four network models, demand, FES 2025 Holistic Transition 2040, and the two-stage market on a
  Clean Power 2030 week. Each runs its own scenario (`config/studies/tutorials/`) and ends with changes to try.
- **AI assistants.** An Ask AI button on every documentation page, and `AGENTS.md` (with `CLAUDE.md`) to guide coding
  assistants in the repository; Getting Started > Using AI assistants shows how to use them.
- **Several scenarios at once.** `snakemake --config scenario=A,B,C` runs a list of scenarios in parallel within the
  `--cores` and `--resources` budget (previously one scenario).
- **A small default run.** `config/config.yaml` runs `Test_Market` only and loads just the tutorial scenarios; the
  larger studies are opt-in.
- **Tooling.** Pre-commit hooks (ruff and a 10 MB file guard), a GitHub Actions workflow (lint, unit tests, workflow dry
  run), and a generated reference of every configuration key ({doc}`/reference/configuration_keys`).

### Breaking changes and migrating from v2.2

1. **`config/defaults/` replaces `config/defaults.yaml`.** The defaults are split by domain: `base.yaml` and
   `base/**` hold the neutral defaults (the overlays add the keys only they set), and the file path is the key path (`base/market/balancing.yaml` holds
   `market.balancing`). Overlays in `method/`, `source/` and `dispatch/` apply by the scenario's declared axes. Move any
   edits you made to `defaults.yaml` into your scenario, or into `config/config.yaml` as a global override.
2. **Keys are nested by domain.** For example:

   | v2.2 key | v2.3 key |
   |---|---|
   | `solver` | `solve.solver` |
   | `voll` | `solve.voll` |
   | `etys` | `network.etys` |
   | `etys_upgrades` | `network.upgrades` |
   | `transmission` | `network.transmission` |
   | `clustering` | `network.clustering` |
   | `renewable_aggregation` | `renewables.aggregation` |
   | `demand_timeseries` | `demand.timeseries` |
   | `demand_flexibility` | `demand.flexibility` |
   | `subsidy_tracking` | `support.labels` |
   | `market.revenue_tracking` | `support.revenue` |

   An old path is refused with the new path named in the error, so a v2.2 scenario fails fast rather than silently
   ignoring a setting. The full list is `RENAMED_KEYS` in `config/config_loader.py`.
3. **`targets` and `method`.** Every future scenario now states where its numbers come from and how its fleet is
   built:

   ```yaml
   targets: {source: FES, edition: 2024, pathway: HT}   # source CP30 | FES | historical
   method: gsp_block                                     # brownfield | gsp_block | historical
   ```

   Scenario ids that existed before this change are listed in `data/scenario_sources/legacy_scenario_ids.txt` and may
   still derive both from `FES_year` and `FES_scenario`; a new scenario must write them. See
   {doc}`/model/scenarios_configuration`.
4. **Studies.** Large studies live in `config/studies/<study>/` and are loaded only when listed under `studies` in
   `config/config.yaml`. `run_scenarios` accepts a study's family name to run all twelve of its months.
5. **Scenario ids.** The study scenarios were renamed to a uniform pattern (for example
   `CP30_FFR_2030_<Mon>_<family>`, `FES25_HT_2050_<Mon>_<family>`). The canonical scenarios in
   `config/scenarios.yaml` (tutorials, examples, `Test_Market`) are unchanged.

## Version 2.2.1 (June 2026)

- Fixed future demand data.
- Zonal network: demand mapped to the correct buses (zone definitions re-keyed to the bus geometry).
- Interconnector pricing defaults to `derived`.
- Stable environment file; `nbsphinx-link` installed through pip.

## Version 2.2.0 (May 2026)

- Two-stage market dispatch: a copperplate wholesale stage and a network-constrained balancing stage, with ELEXON
  bid and offer data for historical years.
- Optional aggregation of renewable generators per bus and carrier.
- ETYS upgrades: HVDC links applied from Appendix B, HVDC place-name aliases, improved bus mapping.
- New default scenarios; market chapters added to the documentation.

## Version 2.1.0 (February 2026)

- Demand flexibility: heat pumps, electric vehicle smart charging and vehicle-to-grid, and event-based demand
  response.
- Weather cutouts for 2010-2024 downloaded from Zenodo; the cutout year follows the selected scenario.
- Aggregation of identical components to reduce memory use.
- Improved placement of thermal plant; DUKES carrier assignment fixed.
- Read the Docs configuration and an "Ask AI" search widget in the documentation.

## Version 2.0.0 (January 2026)

### Major features

- **Snakemake workflow**: a complete redesign for reproducible execution.
- **FES support**: new NESO Future Energy Scenarios can be added more easily.
- **Scenario-based configuration**: YAML scenario definitions.
- **ETYS network and clustering**: the ETYS network clustered to configurable sizes.

### New capabilities

- Automatic historical and future scenario routing.
- GSP-level FES data integration.
- Network upgrades applied for future years.
- Two-stage ETYS build (`process_ETYS_data`, then `build_ETYS_base_network`) and multi-year ETYS support (2022, 2023,
  2024).
- Land-boundary validation of bus coordinates and offshore bus identification.
- Transmission constraint settings (minimum line rating, capacity scaling).
- Optional two-stage market dispatch with ELEXON and NESO validation.
- Centralised logging and scenario validation tools.

### Data updates

- DUKES 2025 power stations, REPD Q2 2025, TEC register (2025), ETYS 2024 topology (default).

## Version 1.0.0 (October 2023)

- Initial PyPSA-GB implementation, with a Jupyter notebook workflow.
- Historical years 2010-2020 and FES scenarios (older FES editions).
- Reduced and Zonal network models.

## How to cite

If you use PyPSA-GB in your research, please cite:

**Lyden, A., Sun, W., Struthers, I., Franken, L., Hudson, S., Wang, Y. and Friedrich, D., 2024.**
PyPSA-GB: An open-source model of Great Britain's power system for simulating future energy scenarios.
*Energy Strategy Reviews*, 53, p.101375.
