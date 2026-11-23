> **In this repository (23 Oct 2026, F27 item 2):** `technology_costs_historical.csv` and `technology_costs_historical_named_farms.csv` (the README's `named_farm_costs.csv`). The build script and the downloaded source copies named below were kept outside the repository; every row carries its source URL and page. Used by `scripts/public_ownership/declared_items.py` (the FIDeR return above cost).

# Historical (as-built) capital costs for GB renewable plant (F27)

Built 23 October 2026 for the public-ownership cost model. The goal is to cost pre-2025 plant (the FIDeR investment
contracts, CfD AR1-AR6 and RO plant from 2002 to 2017) at the cost of its own commissioning year, not at the 2030 build
costs in `data/public_ownership/financing/technology_costs.csv`. The repository was not edited.

| File | What it holds |
| --- | --- |
| `technology_costs_historical.csv` | One row per technology and commissioning year, in 2024 real GBP. Offshore, onshore and solar run 2002-2030. Biomass and hydro run 2012-2030, EfW 2016-2030. |
| `named_farm_costs.csv` | Reported capex for 29 named GB offshore wind farms, with OFTO transfer values and a generation-only GBP/kW in 2024 money. |
| `work/build_tables.py` | Builds both CSVs from the anchors typed into it. Run `python -I work/build_tables.py <this folder>`. |
| `work/named_farms_raw.csv` | The farm research (one quoted sentence and one URL per number). |
| `downloads/` | Copies of every source file read; the extracted text is in `work/*.txt`. |

## Conventions

- **Prices.** All values are 2024 real GBP. The `price_year` column is 2024 on every row; each source's own price
  year is given in the deflator section below.
- **Capex.** Capex is pre-development plus construction, Medium/central case. Infrastructure is left out, because the
  2030 table leaves it out. For example, DESNZ 2025 fixed offshore 2030 is 170 + 2,500 = 2,670, onshore 80 + 1,300 =
  1,380 and solar 30 + 500 = 530; these match `technology_costs.csv`.
  - Where a source gives only a total (Arup 2011), the pre-development and construction shares come from that
    source's own capex breakdown table: offshore 93%, onshore 91%, solar 98%, biomass 96%, hydro 98%.
- **Offshore transmission is excluded.**
  - The DECC 2013, BEIS 2016, BEIS 2020 and DESNZ 2023/2025 offshore rows put the OFTO in the connection and
    use-of-system charge. For example, DECC 2013 Round 2 pays GBP 45.9k/MW/yr, against GBP 4.5k for onshore.
  - Arup 2011 says it excludes OFTO costs (p.43).
  - Pre-2010 figures and the E&Y 2009 figure include the export system. They are multiplied by (1 - 0.244). 0.244 is
    the median OFTO-to-total share across 17 named farms (`named_farm_costs.csv`; range 0.168 West of Duddon Sands to
    0.308 Lincs).
- **O&M.** `fixed_om_gbp_per_kw_year` is the report's "Fixed O&M" line, without insurance or connection charges,
  where the report splits them. There are three exceptions:
  - DECC 2013 large-scale solar, EfW and hydro report insurance and connection charges "included in fixed O+M", so
    their fixed O&M is all-in.
  - The RO solar 2013-14 figure (GBP 24/kW, 2012 money) is an all-in annual O&M cost.
  - Before the first comparable O&M anchor (2016 for offshore and onshore, 2013 for solar), O&M is held at that
    anchor's value. The `basis` column says so.
  - DESNZ 2025 reports variable O&M of 0 for wind and solar because it puts fixed and variable together, so the
    2025-2030 rows run down to 0.
- **Commissioning year.**
  - DECC, BEIS and DESNZ rows are already by commissioning year.
  - Arup 2011 is at financial close 2010. It is mapped to commissioning by DECC 2013's construction periods
    (Table 19): offshore +3 years gives 2013, onshore +2 gives 2012, solar +1 gives 2011. Biomass and hydro +2 gives
    2012; that is an assumption.
  - E&Y 2009 is "for projects due to be completed in 2012".
  - The UKERC chart years are in-year installation (commissioning) years.
- **Which report vintage applies to a year.** Where two reports estimate the same year, the row uses the latest
  report published before that year:
  - 2016-17: DECC 2013.
  - 2018 and 2020: BEIS 2016 (Nov 2016) rather than DECC 2013.
  - 2025: DESNZ 2023 rather than BEIS 2020.
  - 2030: the repository's own 2030 row, so that the series meets `technology_costs.csv` exactly.
  - DESNZ 2025 has no commissioning year before 2030, so 2023-24 are interpolated between BEIS 2016's 2020 and DESNZ
    2023's 2025.
- **Offshore Round 2 or Round 3.** Round 2 is used for 2016-2018, the Round 2 era (Dudgeon, Race Bank, Galloper,
  Walney Extension). Round 3 is used for 2020, the era of Hornsea One, East Anglia One and Rampion. BEIS 2016's Round 2
  2020 value would be 2,888, against the Round 3 value of 3,235 (both 2024 GBP).
- **Rows between anchors** are linear in 2024 GBP and marked `basis=interpolated`.
- **Rows before the first source** are offshore 2002 (none), onshore 2002-2003 and solar 2002-2010. They hold the
  first value and are marked `basis=held_flat_before_first_source`, `status=no_year_specific_source`. They exist only
  because the brief asked for 2002-2024 coverage.
  - GB had almost no ground-mounted solar before 2011, so the solar rows before 2011 should cover no plant.
  - Biomass, hydro and EfW rows start at their first source and are not extended back.
- **Rows for 2025-2030** are included because AR4-AR6 plant commissions in those years. They are not historical:
  they interpolate between DESNZ 2023's 2025 value and the 2030 table.
  - The rise from 2025 to 2030 for offshore (1,902 to 2,670) is a change of report vintage. It reflects the cost
    inflation after 2022 that DESNZ 2025 records; it is not a modelled trend.

## Deflator

ONS series YBGB, the implied GDP deflator at market prices (seasonally adjusted), release of 30 September 2026, with
2024 = 100 (`downloads/ons_ybgb.csv`, from
https://www.ons.gov.uk/generator?format=csv&uri=/economy/grossdomesticproductgdp/timeseries/ybgb/qna, accessed
23 Oct 2026). The factor to 2024 GBP is 100 / index:

| Year | Factor | Year | Factor | Year | Factor |
| --- | --- | --- | --- | --- | --- |
| 2002 | 1.7452 | 2010 | 1.4388 | 2018 | 1.2626 |
| 2003 | 1.7065 | 2011 | 1.4045 | 2019 | 1.2330 |
| 2004 | 1.6611 | 2012 | 1.3850 | 2020 | 1.1779 |
| 2005 | 1.6129 | 2013 | 1.3569 | 2021 | 1.1669 |
| 2006 | 1.5649 | 2014 | 1.3369 | 2022 | 1.1038 |
| 2007 | 1.5361 | 2015 | 1.3298 | 2023 | 1.0384 |
| 2008 | 1.4859 | 2016 | 1.3072 | 2024 | 1.0000 |
| 2009 | 1.4556 | 2017 | 1.2870 | | |

Source price years:

| Source | Price year |
| --- | --- |
| UKERC offshore chart | 2009 |
| E&Y 2009 | Jan 2009 real |
| Arup 2011 | 2010 |
| UKERC onshore chart | 2011 (Jan-Nov average) |
| RO solar 2012 | 2012 |
| DECC 2013 | 2012 ("mid-2012 prices") |
| BEIS 2016 | 2014 |
| BEIS 2020 | 2018 |
| DESNZ 2023 | 2021 |
| DESNZ 2025 | 2024 |

The repository's 2030 rows for biomass, hydro and EfW imply a 2018 to 2024 factor of 1.26388 (for example, BEIS 2020
EfW: 8,740 to 11,046.29). This vintage gives 1.2626, 0.1% lower, presumably from an earlier ONS release. The 2030 rows
are copied from the repository unchanged.

## Sources (all accessed 23 Oct 2026)

| Source | URL | Used for |
| --- | --- | --- |
| UKERC (2013) WP/TPA/2013/007, offshore wind case study | https://ukerc.ac.uk/publications/cost-methodologies-project-offshore-wind-case-study/download | Offshore 2002-2009: Fig. 3 (p.5) in-year average actual capex, 2009 GBP, read from the chart (2002 1.58, 2003 1.67, 2004 1.67, 2005 1.58, 2006 1.69, 2007 1.84, 2008 2.46, 2009 2.77 GBP m/MW). The text on p.10 confirms about 1.7 for 2003 and about 2.5 for 2008. |
| UKERC (2010) Great Expectations (Greenacre, Gross, Heptonstall) | https://ukerc.ac.uk/publications/great-expectations-the-cost-of-offshore-wind-in-uk-waters-understanding-the-past-and-projecting-the-future/download | Offshore 2012: E&Y 2009 ">GBP 3.2m/MW in 2009 (for projects due to be completed in 2012)", p.42. Also Table 2.1 p.9 (North Hoyle 1.35, Scroby Sands 1.26 GBP m/MW). |
| Arup (2011) for DECC, Review of the generation costs and deployment potential of renewable electricity technologies | https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/42843/3237-cons-ro-banding-arup-report.pdf | Medians at financial close 2010, 2010 GBP: onshore >5 MW 1,524 (Table 4 p.19, breakdown Table 5 p.20); offshore >100 MW 2,722 (Table 11 p.44, Table 12 p.45, OFTO excluded p.43); solar >50 kW 2,710 (Tables 33-34 pp.104-105); hydro >5 MW 2,307 (Tables 21-22 p.62); dedicated biomass <50 MW 3,342 (Table 44 p.122, Table 42 p.121). |
| UKERC (2013) WP/TPA/2013/006, onshore wind case study | https://ukerc.ac.uk/publications/cost-methodologies-project-onshore-wind-case-study/download | Onshore 2004, 2006 and 2007: Fig. 2 (p.5), UK points in 2011 GBP read from the chart (0.96, 0.985 and 1.39 GBP m/MW, about +/-0.03). The appendix names the underlying studies: PB Power 2004 and 2006, DTI 2006, DTI/E&Y 2007. |
| DECC (Dec 2012) RO Banding Review, Government Response on solar PV | https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/66516/7328-renewables-obligation-banding-review-for-the-perio.pdf | Solar 2013: ground-mounted >5 MW, capex of a 2013-14 installation 1,140 GBP/kW (2012 prices), O&M 24, Table 1 p.17. |
| DECC (Dec 2013) Electricity Generation Costs | https://assets.publishing.service.gov.uk/media/5a7c71fde5274a5255bceb16/131217_Electricity_Generation_costs_report_December_2013_Final.pdf | Annex 3 Table 20 (pp.51-61), commissioning 2016, 2017 and 2020, 2012 prices. Onshore >5 MW 100+1,500 (p.54); Offshore R2 70+2,500/2,400 (p.54); large-scale solar 1,000 (p.60); dedicated biomass 5-50 MW 95+3,600/3,500 (p.53); hydro 5-16 MW 50+3,100 (p.57); EfW 4,900 (p.58). Block labels were checked against page renders. |
| BEIS (Nov 2016) Electricity Generation Costs, annexes 1-3 | https://assets.publishing.service.gov.uk/media/5ce55ca0ed915d247979f938/Generation_Costs_Report_2016_Annexes.xlsx (report https://assets.publishing.service.gov.uk/media/5a8155f2e5274a2e87dbd11b/BEIS_Electricity_Generation_Cost_Report.pdf) | Table 19 (report pdf pp.67-81), commissioning 2018, 2020 and 2025, 2014 prices (report pdf p.10: "All costs in this report are expressed in 2014 values"). |
| BEIS (2020) Electricity Generation Costs, key data | local copy `C:/tmp/beis_generation_costs_2020_key_data.xlsx` | Commissioning 2025-2040, 2018 GBP. Used as a cross-check only, and through the repository's 2030 rows. |
| DESNZ (2023) Electricity Generation Costs, Annex A | local copy `C:/tmp/desnz_generation_costs_2023_annex_a.xlsx` | 2025 anchor, 2021 GBP. |
| DESNZ (2025) Electricity Generation Costs, Annex A | local copy `C:/tmp/desnz_generation_costs_2025_annex_a.xlsx` | 2030 anchor (= `technology_costs.csv`), 2024 GBP. |
| Mott MacDonald (2010) UK Electricity Generation Costs Update | https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/65716/71-uk-electricity-generation-costs-update-.pdf | Cross-check only, 2009 prices. Onshore NOAK Medium 1,520 GBP/kW (Table A.7 p.83). Offshore FOAK Medium 3,110 GBP/kW (Table A.8 p.84), which appears to include the cable to shore: its connection and use-of-system charge is only 9k/MW/yr. |
| IRENA (2012) Renewable Energy Technologies Cost Analysis: Wind Power | https://www.irena.org/-/media/Files/IRENA/Agency/Publication/2012/RE_Technologies_Cost_Analysis-WIND_POWER.pdf | Not used in the table. Table 4.4 gives UK onshore installed costs 2004-2010 in 2010 USD/kW (879 to 1,734, from IEA Wind). The 2004-05 jump looks like currency and definition effects. |
| Named farms: company releases, trade press, Ofgem OFTO tender round pages and Cost Assessment Reports | URLs per row in `named_farm_costs.csv` | See below. |

Not found or not used:
- **Parsons Brinckerhoff 2004/2006 and DTI/E&Y 2007.** Not obtained directly; they enter only through the UKERC onshore
  chart.
- **Tidal stream.** No report gives a central capex for tidal plant commissioning before 2025: DECC 2013 and BEIS 2016
  have only 2025 FOAK rows. No tidal rows are given. MeyGen 1A (6 MW, 2017) was not researched.

## How the two tables compare (offshore, generation-only, 2024 GBP/kW)

| Commissioning | Report-based row | Named farms (generation-only) |
| --- | --- | --- |
| 2003-04 | 1,838 | North Hoyle 1,742, Scroby Sands 1,582. Not independent: the same 0.244 transmission share is applied. |
| 2012-2015 | 3,522-3,642 | Median 3,231 across eight farms: Greater Gabbard 2,646, London Array 2,920, Walney 1&2 3,003, Humber Gateway 3,215, Sheringham Shoal 3,246, Lincs 3,479, Gwynt y Môr 3,805, West of Duddon Sands 4,806 (its FID "up to GBP 1.6bn"). The reports run about 10% high. |
| 2017-2019 | 3,421 (2017), 3,021 (2018), 3,128 (2019) | Dudgeon 3,048, Rampion 3,393, Galloper 4,589, Beatrice 4,807 (median 3,991). The reports agree with Dudgeon and Rampion but sit 20-35% under Galloper and Beatrice. Galloper's and Beatrice's figures are financial-close totals and may include financing and contingency costs. |
| 2020-2023 | 3,235 (2020), 2,702 (2022), 2,435 (2023) | East Anglia One 2,982, Moray East 2,570, Triton Knoll 1,962, Seagreen 2,442. These agree within about 10%, except Triton Knoll at 27% under. |
| 2025 | 1,902 (DESNZ 2023) | Neart na Gaoithe 4,209 (2019 financial close, delayed project). Dogger Bank A 2,159 (2020 financial close, transmission of about GBP 800m stated). The report row is low against both. |

**Against the 2030 table.**
- **Offshore.** At 2,670, plant commissioned 2002-2007 is overstated (historical 1,700-2,000) and plant commissioned
  2009-2021 is understated (historical 2,900-3,650). The 2022-2024 cohort is roughly right.
- **Onshore and solar.** The 2030 figures understate every historical year: onshore 1,380 against 1,350-2,216, solar
  530 against 600-3,800 since 2011.

The biggest effect on "return above cost" is likely to be RO onshore wind and RO and FiT-era solar from 2011 to 2016.

## Named farms: reading the table

- **Stake prices.** For Walney Extension, Burbo Bank Extension, Race Bank, Hornsea One, Hornsea Two and Westermost
  Rough, only the price of a 50% stake is public. That price includes 50% build funding and the seller's gain, so no
  per-kW value is given; the notes give an upper bound of 2 x stake minus OFTO.
- **Reported totals are mostly at financial close or FID**, not audited outturn. Several come from search-result
  snippets, marked in the notes. Every total is treated as money of the year it was reported.
- **OFTO values.** Final transfer values come from Ofgem (rounds 5-6 tender pages; Moray East, Triton Knoll and
  Seagreen from the Cost Assessment Reports) and from the Wikipedia OFTO table for rounds 1-4 (matching Ofgem where
  checked). Neart na Gaoithe (GBP 500m) and Dogger Bank A (GBP 1bn) are Ofgem estimates; for Dogger Bank A the
  developer's stated transmission capex of about GBP 800m is used.
- **Generation-only** is the reported total minus the OFTO value, divided by MW and deflated from the year reported.
  The OFTO value is subtracted in nominal pounds; it is the cost of construction, roughly the same money as the build.
- **Floating demonstrators.** Kincardine (9,826) and Hywind Scotland (9,150) have no OFTO and are not comparable with
  fixed-bottom 'Offshore wind'.

## Confidence by technology

| Technology | Confidence | Why |
| --- | --- | --- |
| Offshore wind | Medium for 2012-2024, low-medium before 2010 | 2012-2024 rests on four report vintages and agrees with named farms within about 10-20%. Before 2010 the values are read from a chart of in-year averages that may include non-UK projects, with a uniform 24.4% transmission deduction; early Round 1 cables were short, so the deduction may be too large. |
| Onshore wind | Medium for 2012-2024, low for 2002-2011 | 2004-2007 are read from the chart (three UK points); 2008-2011 are interpolated; 2002-03 are held flat. DECC 2013's 2016-17 value (2,216) sits above both Arup and BEIS 2016, a vintage effect. |
| Solar PV | Medium for 2013-2024, low for 2011-2012, none before 2011 | 2011 is Arup's financial close 2010 value for >50 kW plant (2,656 GBP/kW in 2010 money), when GB had little ground-mount solar. 2012 is interpolated. 2002-2010 are placeholders. |
| Biomass - unabated | Low-medium | This is dedicated new-build 5-50 MW. The FIDeR and CfD biomass conversions (Drax units, Lynemouth) are not this technology: DECC 2013 puts conversion at 60+400 GBP/kW (2012 money, p.55) and BEIS 2016 at 80+200 (2014 money). Cost those separately. DECC 2013's 2016 value (5,118) against BEIS 2016's 2018 value (4,024) is a change of vintage. |
| Hydropower | Low-medium | 5-16 MW reference, as in the 2030 row. Arup's 2012 value covers plant >5 MW; DECC 2013's 2016-17 values are higher than BEIS 2016's. |
| Energy from waste | Low | DECC 2013 (2016-17: 6,787) and BEIS 2016 (2018: 11,270) differ by two-thirds. That is a change of method in the 2016 Arup update, not a cost change, so the 2016-17 rows are provisional; consider using the BEIS 2016 2018 value for all pre-2018 EfW. Most EfW sat outside the RO except for the CHP and biomass fraction. |
| Tidal stream | None given | No historical central capex source was found. |
