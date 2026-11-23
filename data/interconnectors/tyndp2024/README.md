# TYNDP hourly 2050 market data (task Y3, FES 2050 plan)

Inputs to rule `build_tyndp_price_shape` (`rules/interconnectors.smk`), method in
`scripts/interconnectors/tyndp_prices.py`. Extracted 30 September 2026 with
`scripts/interconnectors/extract_tyndp_hourly.py` from the files named below. 2050, plus one 2030 file for item F13 (6 Oct 2026).

## Licence and attribution

TYNDP scenario data are published by ENTSO-E and ENTSOG under the **Creative Commons Attribution
4.0 International licence (CC-BY 4.0)**. Attribution: "ENTSO-E and ENTSOG, TYNDP 2024 Scenarios,
market model outputs" and "ENTSO-E and ENTSOG, TYNDP 2026 Scenarios (draft, June 2026), NT+ hourly
dashboards". The files here are extracts (a subset of columns, sums of like columns); the values
are not otherwise changed.

## Files

| File | What | Source file | Status |
| --- | --- | --- | --- |
| `tyndp2024_DE2050CY{1995,2008,2009}_hourly.csv` | TYNDP 2024 Distributed Energy 2050, one climate year each, 8,736 h | `DE2050CY{year}.zip` -> `MMStandardOutputFile_DE2050_Plexos_CY{year}_v11_SoS.xlsb`, sheet `Hourly Market Data emarket` | final |
| `tyndp2024_GA2050CY{1995,2008,2009}_hourly.csv` | TYNDP 2024 Global Ambition 2050, same layout | `GA2050CY{year}.zip` -> `MMStandardOutputFile_GA2050_Plexos_CY{year}_v11_SoS.xlsb` | final |
| `tyndp2026_NTplus2050_mc.csv` | TYNDP 2026 draft National Trends+ 2050 hourly marginal cost, 8,760 h, columns `{node}_WS{091,092,106}` | `NT+_TimeSeriesDashboard.zip` -> `2050/Weather scenario NNN/{CC}_TimeSeriesDashboard_TY2050_WSNNN_SoS.xlsx`, sheet `E-Market`, column `Marginal Cost [EUR/MWh_e]` | draft |
| `tyndp2026_NTplus2050_WS{091,092,106}_UK00_wind.csv` | NT+ 2050 UK00 offshore wind generation, RES curtailment and marginal cost (a check against the price table) | `UK_TimeSeriesDashboard_TY2050_WSNNN_SoS.xlsx`, sheet `E-Market` | draft |
| `tyndp2024_NT2030CY2009_hourly.csv` | TYNDP 2024 National Trends 2030, climate year 2009, 8,736 h: `mc_{node}` (EUR/MWh) for UK00, UKNI, IE00, FR00, BE00, NL00, DE00, DKW1, NOS0; `UK00_wind_offshore_mw` (dispatched) and `UK00_dumped_mw` (UK00 dumped energy). Input to rule `build_cp30_foreign_prices` (item F13, the 2030 foreign prices on CP30's fuel basis; method in `scripts/interconnectors/cp30_foreign_prices.py`) | `MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx` (TYNDP 2024 scenarios download page, National Trends 2030 market model outputs), sheet `Hourly Market Data`; extracted 6 Oct 2026 with openpyxl | final |
| `tyndp2024_NT2030CY2009_zonal_hourly.csv` | TYNDP 2024 National Trends 2030, CY2009, 8,736 h, by European zone (TYNDP nodes summed as `cp30_foreign_prices.REFIT_ZONES`): `{zone}|demand` (Demand, losses included), `{zone}|onwind`, `{zone}|offwind`, `{zone}|solar` (generation plus the zone's dumped energy shared by annual generation, MW) and `price|{node}` (Marginal Cost, EUR). Input to the `refit_2013` method of rule `build_cp30_foreign_prices` (item F13) | same workbook as the row above; written by `scripts/interconnectors/prepare_cp30_refit_inputs.py` on 7 Oct 2026 | final |
| `renewables_ninja_2013_cf.csv` | 2013 hourly capacity factors (UTC) by the same zones: `{zone}|onwind` (current fleet), `{zone}|offwind` (near-term-future fleet, else current; PL from SE, PT and ES from FR) and `{zone}|solar` (SARAH), unscaled. The refit uses their timing only (each series is scaled to TYNDP's CY2009 annual energy) | renewables.ninja v1.3 country downloads, https://www.renewables.ninja/downloads (Ninja-Wind-v1.3, Ninja-PV-v1.3; MERRA-2, bias-corrected). **Licence CC BY-NC 4.0** (non-commercial; cite Staffell and Pfenninger 2016, Energy 114; Pfenninger and Staffell 2016, Energy 114) | final |

URLs:

- TYNDP 2024: `https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/{DE|GA}2050CY{1995|2008|2009}.zip`
  (download page `https://2024.entsos-tyndp-scenarios.eu/download/`; package 28 Jan 2025, zips last
  modified 16 Mar 2025). MD5 of the zips used: DE CY1995 `5969ef4b4a89b983a7a0a0046433059e`, CY2008
  `8d7e7ce09020cd0a3eb00d756ed45354`, CY2009 `88f0c575c5a1210a7bae13087a9ea3a7`; GA CY1995
  `9980d7dfc24d827ca4d5d309fb5fd893`, CY2008 `57d007c372e3506d6763010659b9ef55`, CY2009
  `5da3fd41099fc4c029e6a57bb22754c0`.
- TYNDP 2026 draft: `https://2026-data.entsos-tyndp-scenarios.eu/output/model-results/hourly-dashboards/NT+_TimeSeriesDashboard.zip`
  (1.78 GB, last modified 25 Jun 2026). MD5 of the UK dashboards: WS091 `25662ef40eadcb1d6945d107b1416a12`,
  WS092 `c7d70d5ef5dd8e217d49421f794851df`, WS106 `eb63c22513ec95e6287c5cb97232348d`.

## Columns of the TYNDP 2024 extracts

- `time`: TYNDP's own label, `01JAN00:00` to `30DEC23:00` (52 weeks, no year stamp, no 31 December).
- `mc_{node}`: `Marginal Cost` of the e-market node, EUR/MWh in **real 2022 euros** (TYNDP 2024
  Methodology Report, fuel and CO2 annex). Nodes: UK00, UKNI, IE00, FR00, BE00, NL00, DE00, DKW1, NOS0.
- `UK00_wind_offshore_potential_mw`: UK00 `Wind Offshore` generation potential (generation plus
  curtailment), MW, summed over the UK00 e-market node's offshore columns. Offshore-hub nodes are in a
  separate workbook and are not included; the series is used only standardised (an index of GB
  offshore weather), so the level does not matter.
- `UK00_wind_onshore_potential_mw`: the same for onshore.
- `demand_{UK00,FR00,DE00}_mw`: `Demand [MW]` (losses included), kept to check the calendar's weekday.

## Declared treatments (applied in `tyndp_prices.py`, not in these files)

- Price base: TYNDP 2024 is 2022 EUR, NT+ 2024 EUR. Both are brought to 2024 EUR by the euro-area
  (EA20) HICP annual average index (Eurostat `prc_hicp_aind`, INX_A_AVG, 2015 = 100: 2022 116.83,
  2024 126.08; ratio 1.0792; retrieved 30 Sep 2026), then to GBP at `interconnectors.pricing.eur_to_gbp`
  (0.86), the rate IC1's ENTSO-E 2024 prices use. Result: GBP 2024 per MWh, the model's money.
- The NT+ dashboards publish dispatched generation, not potential. The NT+ wind regressor is UK00
  offshore generation plus total UK00 RES curtailment.
- Southern Norway: TYNDP 2024 NOS0; NT+ splits it, and NOS2 (North Sea Link's Kvilldal end) is used.
- SEM: 0.669 IE00 + 0.331 UKNI, GB cable capacity to each side (East West 505 + Greenlink 504 MW
  against Moyle 500 MW).
