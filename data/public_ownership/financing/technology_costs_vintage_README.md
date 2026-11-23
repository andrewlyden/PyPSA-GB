# Vintage technology costs, 2025 to 2050 (FES 2050 plan, task Y2)

Built 30 September 2026 by `scripts/public_ownership/build_vintage_technology_costs.py`. Run it by hand;
no Snakemake rule reads these files, and no 2030 rule is affected. Loader:
`scripts/public_ownership/vintage_costs.py`. Tests: `tests/unit/test_fes2050_vintage_costs.py`.

`technology_costs.csv` is unchanged. It remains the 2030 table, as the 2030 plan requires.

## Files

| File | What |
| --- | --- |
| `technology_costs_vintage.csv` | 34 technologies x build years 2025 to 2050, real 2024 GBP. It has the `technology_costs.csv` columns, plus `build_year`, the basis adjustment, the learning sensitivity and provenance. |
| `fes2025_hurdle_rates.csv` | Hurdle rates, pre-tax real, by build year: FES 2025 E.03 and ER.01 (annual), and DESNZ Annex A Low, Medium and High (2030, 2035, ..., 2050). Reference and sensitivity only: the finance layer's rates still come from the workbook snapshot. |
| `technology_economic_lives.csv` | The life the vintage table uses, beside the lives in `technology_costs.csv`, DESNZ Annex A and FES E.02. |

## Sources

| Source | File | Used for |
| --- | --- | --- |
| DESNZ Electricity Generation Costs 2025, Annex A (updated 18 March 2026), real 2024 GBP, sheet "Technical and Cost Assumptions", Medium | `data/fes_2050_sources/annexA_2025.xlsx` | Solar, onshore, fixed and floating offshore, CCGT, gas CCUS, hydrogen CCHT and OCHT, tidal stream |
| NESO FES 2025 Economics Inputs Data Workbook (December 2025), real 2025 GBP: sheets E.02, E.03, H.01, H.02, ER.01 and EA.01 | `data/fes_2050_sources/fes_econ_inputs.xlsx` | Storage, SMR, BECCS, electrolysers, salt caverns, nuclear, hydro, biomass and EfW ratios; pre-2030 shape; hurdle rates; deflator |
| Danish Energy Agency, Technology Data for Energy Storage (November 2018), Vanadium Redox Battery, p.125 | `data/fes_2050_sources/dea_storage_vanadium_redox_flow_battery.pdf` | Flow batteries (provisional) |
| DESNZ Hydrogen Production Costs 2021, p.14 and p.19 | `data/fes_2050_sources/desnz_hydrogen_production_costs_2021.pdf` | PEM life (30 years); stack replacement treated as an annual cost |
| FES 2025 Data Workbook V005, WS1 rows 466 and 593 | `data/FES/` | Electrolyser efficiency, 72.27 / 88.71 = 0.8147 (HHV) |
| CCC Seventh Carbon Budget Methodology Report (May 2025), p.294 Table 10.2 | research note | Learning rates |

## Method

**Full DESNZ basis (decision 26).** For DESNZ technologies:

- capex is pre-development + construction + infrastructure (Medium), with infrastructure divided by the
  reference plant size. Annex A note 5 says infrastructure includes the grid connection, except for
  offshore wind, where connection is excluded by construction.
- fixed O&M is fixed O&M + insurance. Insurance is also reported on its own in `insurance_gbp_per_kw_year`.
- connection and use-of-system charges are left out, because the finance layer charges TNUoS on its own
  line (item D4(a)).
- tidal stream is taken from Annex A: 4,430 GBP/kW in 2030, not 5,598.98.

**Anchor and ratio.**

- *Rows in `technology_costs.csv`:* the 2030 anchor is the corrected 2030 value, and a later year is the
  anchor x the source's year/2030 ratio. `basis_adjustment_*` is the corrected value minus the
  `technology_costs.csv` basis carried forward on the same ratio. The 2030 row minus the adjustment
  reproduces `technology_costs.csv` (tested to 1e-6).
- *Technologies absent from `technology_costs.csv`:* these take the source's own level, with a zero
  adjustment. The exception is the lithium battery durations (1, 2, 6, 8 and 9 h), which are anchored on
  the 4 h row x the FES duration and year shape, so that the duration curve is consistent. 8 h is linear
  between the FES 6 h and 9 h values.
- *`source_level_capex_power_gbp_per_kw`:* the source's own value, reported beside every row.

| Technology (2030) | Capex adjustment GBP/kW | Fixed O&M adjustment GBP/kW-yr | Reason |
| --- | --- | --- | --- |
| Solar PV | +117.31 | +1.60 | infrastructure 6.1m / 52 MW; insurance |
| Onshore wind | +315.89 | +3.80 | infrastructure 16.3m / 51.6 MW; insurance |
| Offshore wind | 0 | +8.90 | insurance (connection excluded by construction) |
| Gas generation - unabated | +14.65 | +2.50 | infrastructure 24.4m / 1,666 MW; insurance |
| Tidal stream | -1,168.98 | +56.28 | row replaced by Annex A (FOM 125.9 + insurance 89.5 = 215.4, against 159.12) |
| All other rows | 0 | 0 | not a DESNZ Annex A technology, or not in `technology_costs.csv` |

**Years.**

- DESNZ values are linear between 2030, 2035, 2040, 2045 and 2050.
- Before 2030, a DESNZ technology follows the FES shape of the same technology (`year_basis` =
  `fes_shape_before_first_desnz_year`).
- SMRs have no FES value before 2030, so the 2030 value is held back.
- Values are held flat after the last source year (decision 11). This affects salt caverns, which have
  no trajectory, and large nuclear, where FES is flat.

**Price base.**

- FES 2025 GBP x 0.97727, the FES EA.01 GDP deflator (2024 = 97.72656, 2025 = 100). This is the same
  GDP-deflator basis as Annex A. It is not the plan's 1.0384 CPI-type bill/finance factor.
- DEA EUR 2015 x 0.9514: the 2015 exchange rate from FES EA.01 (1.37802 EUR/GBP), then the GDP deflator
  2015 to 2024 (74.5377 to 97.72656).

**Electrolyser.** FES H.02 quotes costs per kW of hydrogen output (HHV). Per kW of electricity =
per kW H2 x 0.8147. FES variable OpEx (0.9% of capex a year, per kW) is folded into fixed O&M.

**Salt caverns.** 0.54432 GBP/kWh (2025 GBP) x 0.97727 = 0.532 GBP/kWh, flat. O&M is 4% of capex a
year, in `fixed_om_gbp_per_kwh_year`. The 30-year life is a declared assumption.

**Flow battery.** Two-part cost:

- power: the DEA capacity component, in `capex_power_gbp_per_kw`;
- energy: the energy component + other project costs, in `capex_energy_gbp_per_kwh`;
- fixed O&M: 2.0 to 1.5% of investment a year, split the same way;
- `duration_hours` is blank, so the finance layer does not rescale by duration.

**Learning sensitivity (`learning_capex_gbp_per_kw`).**

- Rates: CCC 7CB Table 10.2, per doubling of global deployment: offshore (fixed and floating) 12%,
  onshore 7%, solar 20%, lithium batteries 16%.
- The CCC applies them on the IEA NZE 2023 global deployment path. That series is behind the IEA
  licence and was not used here.
- Instead, each rate is applied as an annual-decline equivalent at a declared one global doubling per
  decade after 2030, on top of the central trajectory: 1.27, 0.72, 2.21 and 1.73% a year.
- By 2050 that is -22.6% (offshore), -13.5% (onshore), -36% (solar) and -29% (batteries) against
  central. This is the same order as the published DESNZ Low case.
- Other technologies: the column equals central.

## Cautions for the finance layer (Y4, Y10, Y11)

1. **Insurance would be counted twice.** `fixed_om_gbp_per_kw_year` includes DESNZ insurance, and the
   2030 finance layer adds 0.4% of capex (11.22 GBP/kW-yr offshore) on its own D4(a) line. Either switch
   the D4(a) insurance line off for 2050, or use `vintage_costs.vintage_cost_table(year, insurance="exclude")`.
2. **Some rows have no workbook WACC category.** `wacc_technology` gives the workbook category each row
   should take its rates from. Hydrogen CCHT and OCHT share "Hydrogen CCHT/OCHT - mature". Salt caverns
   have none: FES uses the 3.5% social rate.
3. **Battery levels.** The lithium rows keep the BEIS 2018 4 h level: 732.67 GBP/kW in 2030, against
   FES 639.1, so the anchored rows are 14.6% higher. The LDES row keeps the 2030 thermal-storage proxy. 2050 LAES assets
   should take the "Liquid air energy storage - N h" rows at the FES level.
4. **Lives differ by source.** For example, onshore is 35 years in DESNZ and 25 in FES, and large
   nuclear is 60 in the workbook and 40 in FES. See `technology_economic_lives.csv`.
