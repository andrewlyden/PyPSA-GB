# External benchmarks for GB 2050 (and 2035-2045): wholesale prices, constraint costs, curtailment

Compiled 25 Oct 2026 to sit beside PyPSA-GB FES 2024 Holistic Transition (HT) 2050. Provisional model figures:

- Demand-weighted mean Stage 1 price: GBP 31.19/MWh (2025 prices, 2013 weather).
- Thermal redispatch cost: about GBP 1.95 bn a year (offers 0.92 + bids 1.00 + interconnector re-trading 0.04), about GBP 3.1/MWh of demand.
- System demand: about 623 TWh.

Every number below was read from the primary document (the PDFs are in `downloads/`). "Read from chart" means the value was read off a figure by eye. Treat those as +/- about 5% of the axis span.

## 0. Conversions used (to 2025 money)

| From | Index | Factor to 2025 |
|---|---|---|
| 2022 prices (CPI) | ONS D7BT 121.7 -> 138.4 | x1.137 |
| 2022/23 prices (CPI, FY midpoint about 126) | | about x1.10 |
| 2023 prices (CPI) | 130.5 -> 138.4 | x1.061 |
| 2024 prices (CPI) | 133.9 -> 138.4 | x1.034 |
| 2024 prices (GDP deflator, used by DESNZ EEP) | ONS YBGB 100.0 -> 103.6 | x1.036 |

ONS CPI annual averages: 2021 111.6, 2022 121.7, 2023 130.5, 2024 133.9, 2025 138.4 (release of 21 Oct 2026). Convert with the deflator the source names. FTI and AFRY say "real 2024" without naming one, so CPI is assumed for them.

---

## 1. Headline comparison (2050, converted to 2025 GBP)

| Source | Scenario | 2050 wholesale price (GBP/MWh, 2025) | 2050 constraint cost (GBP bn, 2025) | Notes |
|---|---|---|---|---|
| **PyPSA-GB (ours)** | FES24 HT, 2013 weather | **31.2** (demand-weighted) | **1.95** (about 3.1/MWh) | Stage 1 copperplate; BM on ETYS / NESO 17-zone |
| FTI Consulting for Octopus Energy (2025) | CP30 to 2030, then FES24 HT / Beyond 2030; CY2009 | 49.5 time-weighted; about 58 load-weighted (chart) | about 3.67 (chart; text: 3.1-3.6 in 2024 money over 2030-50) | Closest scenario match; constrained-on is mostly interconnectors |
| AFRY (May 2025), "FES 24 HT Test" | FES24 HT + Beyond 2030, network flat after 2040 | about 42 (chart) | 2.38 (labelled 2.3 in 2024 money); variants 0.36-2.35 | Closest network-planning match to ours (capability held after 2040s) |
| LCP Delta / Grant Thornton for DESNZ (Mar 2024) | DESNZ Net Zero Higher Demand (about 700 TWh excl. electrolysis) | 48.9 (demand-weighted) | about 2.96 (chart); 6.2 without ICs in balancing | Only government-commissioned 2050 national price plus constraint series |
| Aurora for RenewableUK (Jun 2024) | Aurora "Net Zero 2035" | about 64 (baseload, chart) | not split out (BM cost about 3.4 bn/yr, 2036-50 average) | Commercial-model public summary |
| DESNZ EEP 2024-2050 (Feb 2026) | Reference (EEP-ready policies, NOT net zero power) | 118 (volume-weighted) | n/a | 119 TWh unabated gas in 2050 and carbon at GBP 238/t; an upper-bound context only |
| CCC Seventh Carbon Budget (Feb 2025) | Balanced Pathway | system cost about 89 (8.4 p/kWh, chart) | n/a | Full annualised system cost, not wholesale |

**Reading.** Our 2050 price (31) is 25-45% below every net-zero comparator (42-50 in 2025 money). It is similar to their 2035-2040 troughs: FTI-Octopus 26-29, AFRY about 29-31, FTI-for-NESO 20-22.

The comparators all show prices rising from 2040 to 2050, because:

- demand catches up with renewables build;
- hydrogen- and CCS-gas-fired plant set the price more often;
- the carbon price rises.

Our constraint cost (1.95 bn) sits inside the published 2050 range (0.4-3.7 bn) and just below the central studies (2.4-3.7 bn). Per MWh it is close to LCP/GT (about 4.2/MWh) and AFRY (about 3.6/MWh).

---

## 2. Wholesale price benchmarks (best comparators first)

### 2.1 FTI Consulting for Octopus Energy: *Impact of a Potential Zonal Market Design in Great Britain* (Feb 2025; PDF dated May 2025)

- **URL:** https://octoenergy-production-media.s3.amazonaws.com/documents/FTI_Impact_of_Zonal_Market_Design_-_May_2025-min.pdf (press release: https://octopus.energy/press/zonal-pricing-to-slash-more-than-55bn-off-energy-bills-major-new-report-finds/)
- **Location:** slide 27 (national generation mix and time-weighted wholesale price, values labelled); slide 32 (load-weighted wholesale cost per MWh, chart); slide 15 (assumptions).
- **Model and scenario:**
  - PLEXOS, about 1,200-node GB, 12 zones, linked pan-European model.
  - GB follows NESO CP30 for 2030, then FES 2024 HT plus Beyond 2030 network for 2035-2050. EU follows TYNDP 2024 Distributed Energy.
  - Modelled years 2030, 2035, 2040, 2045, 2050.
  - **Climate year CY2009.** Real 2024 GBP (EUR 1.20 = GBP 1).
- **Metric:** national annual average day-ahead price, **time-weighted** (stated). Slide 32 gives a load-weighted wholesale cost per MWh.
- **Values (GBP/MWh, 2024), time-weighted national:**
  - 2030 45.9, 2035 25.6, 2040 28.2, 2045 36.0, 2050 47.9 (labelled).
  - Wholesale + CfD top-up: 77.5, 80.1, 79.6, 68.9, 63.0.
  - 2050 load-weighted wholesale cost about 56 (read from chart).
- **Price duration:** zero-priced hours about 41% of hours in 2030 and about 54% in 2035 (slide 27 text). Not given for 2050.
- **How to compare:**
  - Use our **time-weighted** Stage 1 mean against 47.9 x1.034 = 49.5, and our demand-weighted 31.19 against about 58 (load-weighted).
  - Same FES vintage and pathway as ours, but CY2009 rather than 2013.
  - Their 2045-2050 rise is attributed to demand growth catching up with renewables roll-out.
  - Check our CCS-gas and H2 fuel and carbon assumptions and our flexible demand (electrolysis), which is the likely source of the gap.

### 2.2 LCP Delta and Grant Thornton for DESNZ: *System benefits from efficient locational signals* (REMA technical research, DESNZ 2023/057, published 12 Mar 2024)

- **URL:** https://assets.publishing.service.gov.uk/media/65e3a3dc3f69450263035fc3/9-system-benefits-from-efficient-locational-signals.pdf (collection: https://www.gov.uk/government/publications/review-of-electricity-market-arrangements-rema-technical-research-supporting-consultation)
- **Location:** section 4.5 (p.67), Figure 27 (p.68) and **Figure 29 (p.69, national price printed per year)**.
- **Scenario:**
  - DESNZ Net Zero Higher Demand: about 700 TWh excl. electrolysis by 2050; an alternative Net Zero Lower case reaches 525 TWh.
  - Network: NOA7 Refresh (boundary capacity tripled by 2040).
  - LCP's EnVision model, 12 zones, modelled years 2030, 2035, 2040, 2045, 2050.
  - Weather: "historical shapes" (year not stated).
- **Metric:** national-pricing counterfactual, **demand-weighted** wholesale price.
- **Values (GBP/MWh, real 2022):**
  - National: 2030 31, 2035 30, 2040 36, 2045 40, 2050 43.
  - Zonal range in 2050: 38-51.
- **Drivers given:** carbon and hydrogen prices rise and hydrogen sets the price more often, while renewables penetration grows slowly after 2035.
- **How to compare:** like for like with our demand-weighted 31.19. Convert 43 x1.137 = **48.9**. The DESNZ scenario has about 12% more demand than FES24 HT and its own generation mix.

### 2.3 AFRY: *Sensitivity analysis on the modelled benefits of zonal electricity markets in Great Britain* (Summary report, May 2025)

- **URL:** https://afry.com/sites/default/files/2025-05/2025_gb_zonalmarketsensitivities_summaryreport_v100.pdf
- **Sponsors:** SSE, RWE, ScottishPower, NGV, Orsted, Greencoat, TRIG.
- **Location:** Annex A.2 slide 46 (wholesale prices, chart); slide 13 (scenario inputs).
- **Scenario "FES 24 HT Test":**
  - CP30 "Further Flex and RES" to 2030, then FES24 HT demand and capacity (with its locational split).
  - FES 2024 commodity prices.
  - CP30 network in 2030, then Beyond 2030, **held flat after 2040**.
  - BID3 national and zonal (11 zones). Weather year not stated.
- **Metric:** annual average national wholesale price. Weighting not stated.
- **Values (GBP/MWh, real 2024; read from chart):** 2030 about 44, 2035 about 28, 2040 about 30, 2050 about 41.
  - Zonal spread in 2050: about 39 (Scotland, East) to about 60 (Thames).
- **How to compare:** FES24 HT inputs and a network frozen after the 2040s, as ours is. 41 x1.034 = about **42**.

### 2.4 FTI Consulting for NESO: *Quantitative assessment of central scheduling - FES 24 Update* (Apr 2025)

- **URL:** https://www.neso.energy/document/377391/download
- **Location:** slide 8 (annual average wholesale price, core vs updated, chart).
- **Scenario "Updated":** CP30 "Further Flex and RES" for 2030, then FES24 HT with the CP30 / Beyond 2030 network. **CY2013**, real 2024. Modelled years 2030, 2035, 2040 only.
- **Values (national, GBP/MWh, chart):** 2030 about 39.5, 2035 about 19, 2040 about 21.
  - Their FES22 core run gives about 37-38 for 2035-2040.
- **How to compare:** **same weather year and FES pathway as ours, but stops at 2040.** It shows that FES24 HT dispatch in the late 2030s gives very low prices, about 20 (about 22 in 2025 money). This report extends to 2040; I did not locate the Feb 2026 REMA S&D report's horizon.

### 2.5 FTI Consulting and Energy Systems Catapult for Ofgem: *Assessment of locational wholesale electricity market design options in GB* (Oct 2023)

- **URLs:**
  - Final report: https://www.ofgem.gov.uk/sites/default/files/2023-10/FINAL%20FTI%20Assessment%20of%20locational%20wholesale%20electricity%20market%20design%20options%20-%2027%20Oct%202023%205.pdf
  - Key findings deck: https://www.ofgem.gov.uk/sites/default/files/2023-10/Key%20findings%20from%20FTI%20Consulting%20presentation%20June%202023%20.pdf
- **Location:** key findings slides 23 and 28 (values labelled). Final report: 2022 prices (para 5.x, "all in 2022 prices"); **2013 climate year** (line about para 5.7x).
- **Scenario:** FES21 Leading the Way (LtW) with NOA7 network, LtW with HND, and System Transformation (SysTr). **Horizon 2025-2040.** Yes, it extends beyond 2035, but only to 2040.
- **Metric:** national **load-weighted** annual average wholesale price.
- **Values (GBP/MWh, 2022):**

  | Scenario | 2035 | 2040 |
  |---|---|---|
  | LtW (NOA7) | 34.10 | 50.90 |
  | LtW (HND) | 34.10 | 50.90 |
  | SysTr | 29.90 | 30.20 |

- **How to compare:** an old FES vintage, but the same weather year, and the metric matches our demand-weighted figure. 2040 in 2025 money: LtW 57.9, SysTr 34.3.

### 2.6 Aurora Energy Research for RenewableUK: *Is offshore wind still good value for billpayers?* (Jun 2024)

- **URL:** https://www.renewableuk.com/media/ufngftob/aurora-energy-research_is-offshore-wind-still-good-value-for-billpayers_june-2024.pdf
- **Location:** p.19 (wholesale market price chart, base case shown beside the scenario), with the same base on pp.22, 25 and 28.
- **Scenario:** Aurora "Net Zero 2035" base case (power decarbonised by 2035). Real 2022.
- **Metric:** baseload annual price, plus 5th and 95th percentile hourly prices.
- **Values (read from chart):**
  - Baseload: 2035 about 67, 2040 about 62, 2045 about 58, 2050 about 56.
  - 5th percentile about 4 in the late 2040s and 2050 (so at least 5% of hours near zero).
  - 95th percentile about 150.
- **How to compare:** time-weighted (baseload); 56 x1.137 = about **64**. It is a commercial model with gas still marginal in many hours. Use it as the high end of the net-zero range. The percentiles are a rough price duration curve check.

### 2.7 DESNZ: *Energy and emissions projections: 2024 to 2050* (EEP, 3 Feb 2026), Annex M "Growth assumptions and prices"

- **URLs:**
  - Annex M: https://assets.publishing.service.gov.uk/media/6981d725f2ea511998158f75/Annex_M_assumptions_growth_price.ods
  - Publication page: https://www.gov.uk/government/publications/energy-and-emissions-projections-2024-to-2050
- **Location:** Annex M, sheet "Reference" (and FFP_Low/High, GDP_Low/High, Existing), rows "Price - Wholesale, Electricity (volume weighted / baseload)", note 22.
- **Definition:** short-run marginal cost of generation including UK ETS and Carbon Price Support. Dynamic Dispatch Model. Real 2024 (GDP deflator).
- **Values (Reference, p/kWh 2024; x10 for GBP/MWh):**

  | Series | 2030 | 2035 | 2040 | 2045 | 2050 |
  |---|---|---|---|---|---|
  | Volume-weighted | 6.21 | 6.67 | 8.18 | 10.12 | 11.39 |
  | Baseload | 6.03 | 6.47 | 7.89 | 9.72 | 10.97 |

  - Low fossil prices, 2050: 10.02 (volume-weighted).
  - Carbon price for the electricity sector: GBP 64/t (2030), 144/t (2040), 238/t (2050).
- **Comparability:** **not a net-zero power system.** Annex J Reference has 119 TWh unabated gas generation in 2050 of 437 TWh supplied, so gas plus a rising carbon price sets the price. Use it only to show what happens if gas stays marginal. In 2025 money: 2050 about GBP 118/MWh.
- The 2023-2050 edition (Apr 2025) is superseded.

### 2.8 Climate Change Committee: *The Seventh Carbon Budget* (Feb 2025), Figure 7.5.8

- **URL:** https://www.theccc.org.uk/wp-content/uploads/2025/02/The-Seventh-Carbon-Budget.pdf, p.220.
- **Metric:** **annualised system cost per unit of electricity**: annualised capex + opex of generation, storage and networks, including UK ETS carbon costs, per unit supplied. Explicitly not the wholesale or retail price.
- **Modelling and scenario:** AFRY BID3, 11 zones. Balanced Pathway: gross demand 692 TWh in 2050, average weather year. 2023 prices.
- **Values (p/kWh, read from chart):**
  - Balanced Pathway: 2035 about 10.6, 2040 about 9.8, 2045 about 8.7, 2050 about 8.4.
  - Baseline: about 12.8-13.0 throughout.
- **How to compare:** a full-cost ceiling. Our wholesale price plus CfD, capacity, network and balancing per MWh should come out near 8.4 x1.061 = about **GBP 89/MWh** in 2025 money. The CCC publishes no wholesale price.
- The 2026 supplementary analysis (Box 2.5) confirms the series is robust to its sensitivities and gives no wholesale figure.

### 2.9 Sources checked that give no 2050 wholesale price

- **NESO FES 2025 Economics Annex (Dec 2025)** (https://www.neso.energy/document/374246/download): costs only as a share of GDP; no GBP/MWh. It notes that heavy 2030s renewables build raises curtailment.
- **NESO FES 2024 and 2025 data workbooks:** no price variable.
- **TYNDP 2024:** GB (UK00) 2040/2050 marginal prices are only in ENTSO-E market-model data downloads or open-tyndp. Not verified here.
- **Cornwall Insight and LCP Delta long-run curves:** public pages stop about 2030-2035; 2050 values are behind paywalls.
- **NIC 2020 (Aurora) annex:** page now 404.

---

## 3. Constraint / balancing cost benchmarks (best comparators first)

### 3.1 AFRY (May 2025): FES 24 HT Test redispatch costs, plus five better-optimised variants

- **URL:** as in 2.3.
- **Location:**
  - Annex A.2 slide 45 (redispatch volumes and costs by technology, national BAU; **values labelled**).
  - Annex B.2 slide 55 (transmission constraint costs 2018-2050 for all variants; chart).
  - Slide 14 (wind curtailment).
- **Definition:** transmission redispatch (BM) cost under national pricing, GBP bn real 2024. It excludes voltage and other local constraints. Historical comparator: MBSS constraint costs minus RoCoF and intertrips.
- **Values:**

  | Year | Cost, GBP bn (2024) | Volume, TWh |
  |---|---|---|
  | 2030 | 2.5 | 27 |
  | 2035 | 2.4 | 40 |
  | 2040 | 2.2 | 49 |
  | 2050 | 2.3 | 35 |

  - **2050 is dominated by interconnector redispatch** (red bar, about 1.7 bn).
- **Variants, 2050 (read from chart):**
  - "FES 24 HT Adjusted (HVDC & AC)": about 0.68.
  - "Adjust network (HVDC only)": about 1.28.
  - "Relocate offshore wind": about 2.05.
  - "Partially relocate ICs" and "More central offshore": about 1.9.
  - "AFRY (Feb 25)" own view: about 0.35.
- **How to compare:**
  - The best structural match for our 2050 setup: FES24 HT, a network frozen after the Beyond 2030 / 2040s plan, national pricing.
  - 2.3 x1.034 = 2.38 bn, or about GBP 3.6/MWh on FES24 HT's 667 TWh (approximate; AFRY's own demand not extracted).
  - Our 1.95 bn sits between AFRY's Test case and its re-optimised cases.
  - **Check the interconnector share:** AFRY (and FTI below) resolve most 2040s-2050 constraints by redispatching interconnectors. Our IC re-trading is only 0.04 bn, so our offers (0.92 bn) are probably doing work their models give to ICs.

### 3.2 LCP Delta and Grant Thornton for DESNZ (Mar 2024): national-pricing counterfactual constraint costs to 2050

- **URL:** as in 2.2.
- **Location:**
  - **Figure 34, p.80** (constraint costs, core NZH and "no interconnectors in locational balancing").
  - Figure 42, p.90 (pay-as-clear / pay-as-bid / pay-as-bid with uplift, no-IC limited-storage case).
- **Definition:** BM constraint costs = constrained volume x bid/offer prices. Core: bids and offers up to the marginal unit's cost. GBP real 2022.
- **Values (read from chart, GBP bn):**

  | Case | 2030 | 2035 | 2040 | 2045 | 2050 |
  |---|---|---|---|---|---|
  | Core NZH | 0.52 | 1.02 | 1.6 | 2.32 | 2.6 |
  | No ICs in balancing | 1.1 | 2.6 | 3.8 | 5.42 | 5.47 |

  - Figure 42, 2050: pay-as-clear about 5.2, pay-as-bid about 4.2, uplift about 5.4.
- **How to compare:** 2.6 x1.137 = **2.96 bn**, about **GBP 4.2/MWh** on about 700 TWh. This is the only government-commissioned 2050 constraint series. It is close to our 1.95 bn and 3.1/MWh once demand is scaled. Constraint cost rises steadily to 2050 because demand and capacity outgrow the network after 2040, the same mechanism as our held-at-2045 network.

### 3.3 FTI Consulting for Octopus Energy (2025): national constraint costs 2030-2050

- **URL:** as in 2.1.
- **Location:** slide 29 (volumes and costs by technology, chart), slide 32 (national vs zonal).
- **Definition:** GB-wide redispatch cost under national pricing. BM bids and offers from FTI's assumptions; CfD and merit-order rules on slide 29. GBP real 2024.
- **Values:**
  - Text: "annual constraint costs remain between GBP 3.1 bn and GBP 3.6 bn across the modelled period".
  - Read from chart: 2030 3.1, 2035 about 3.5, 2040 about 3.35, 2045 about 3.45, 2050 about 3.55.
  - Constraint volumes about 53 TWh on each side in 2050 (chart).
  - Constrained-on is mostly interconnectors from 2035; CCS gas in the North has negative constraint cost after 2040.
  - Constraint cost per non-electrolyser MWh about 7 in 2050 (read from slide 32).
- **How to compare:** 3.55 x1.034 = **about 3.67 bn**, the high end of the 2050 range. Same scenario family as ours. Note that FTI values unserved energy in redispatch at a DSR price (GBP 280-340/MWh).
- **Related:** FTI's July 2025 Octopus report ("Helping Hercules", https://octoenergy-production-media.s3.amazonaws.com/documents/2025_FTI_Transmission_Report_-_Helping_Hercules.pdf) restates GBP 2-5 bn a year from 2030 onwards.

### 3.4 FTI Consulting for NESO (Apr 2025), FES 24 Update: BM constraint cost to 2040

- **URL:** https://www.neso.energy/document/377391/download, slide 13.
- **Values (national self-scheduling, GBP bn real 2024):**
  - 2030 about 2.5, **2035 3.2 (text)**, **2040 about 3.95 (chart)**.
  - FES22 core run: 5.6 (2035) and about 5.9 (2040).
  - Constrained-on is dominated by interconnectors.
- **How to compare:** same weather year (2013) and FES24 HT as ours, to 2040 only. In 2025 money: 2040 about 4.1 bn.

### 3.5 FTI Consulting for Ofgem (Oct 2023): national constraint costs to 2040

- **URL:** as in 2.5. Key findings slides 21-22; final report Figure 10-3 (2022 prices, undiscounted).
- **Values (read from chart, GBP bn 2022):**

  | Case | 2035 | 2040 |
  |---|---|---|
  | LtW (NOA7) | about 4.6 | about 5.1 |
  | LtW (HND) | about 3.2 | about 2.7 |
  | SysTr | about 2.0 | about 2.85 |

  - Text: "could exceed GBP 4bn by 2035", "just under GBP 3.5bn" with HND, SysTr "GBP 2-3bn p.a.".
- **Comparability:** FES21 vintage, 2013 climate year. In 2025 money, 2040 is 3.1-5.8 bn.

### 3.6 NESO (ESO): *Modelled Constraint Costs - NOA 2021/22 Refresh* (Aug 2022)

- **URL:** https://www.neso.energy/document/266576/download, Figure 1 (p.3).
- **Values (read from chart):** Leading the Way Plus (NOA7 Refresh, HND network), GBP bn 2022/23 prices: 2030 about 0.95, 2035 about 3.1, 2038 about 2.15, 2040 about 2.85, **2041 about 3.1**.
- **Note:** the last published NESO thermal constraint series that runs beyond 2035. It is pre-energy-crisis (FES21) data. In 2025 money: 2040-41 about 3.1-3.4 bn.

### 3.7 NESO (ESO): *Balancing Costs: Annual Report and Future Projections - Technical Report* (May 2024, first ABC report)

- **URL:** https://www.neso.energy/document/318666/download, **Figure 8, p.17** (also Figures 9-10).
- **Definition:** **total** balancing costs, not thermal only:
  - thermal constraints from NOA7r (pre-2030) and Beyond 2030 Leading the Way (post-2030);
  - voltage, stability, reserve and response costs rolled forward at FY23/24.
- Price base not stated (presumably 2023/24 money). Financial years to 2041/42.
- **Values (read from chart, GBP bn):**

  | Scenario | 2034/35 | 2039/40 | 2041/42 |
  |---|---|---|---|
  | Leading the Way | about 3.35 | about 2.9 | about 2.85 |
  | Consumer Transformation | | | about 4.5 (peak about 5.2 in 2040/41) |
  | System Transformation | | | about 2.25 |
  | Falling Short | | | about 2.4 |

- **How to compare:** subtract roughly GBP 1-1.5 bn of non-thermal costs to compare with our thermal figure. **This 2024 edition extends to 2041/42; the 2025 edition's Fig. 30 (the one already in use) stops at 2035.** LCP's March 2026 report reproduces NESO's FES24 HT series to 2035 only (about 2.85 bn in 2035).

### 3.8 NESO: *Analysis of the impact of Zonal Pricing on Transmission Network Investment* (Jun 2025)

- **URL:** https://www.neso.energy/document/364271/download, Key message 3 (slide 6).
- **Value:** "Even if the Beyond 2030 network is built in full, under national pricing, we would still expect **up to GBP 2bn of thermal constraint costs per year in the late 2030s**." FES 2023 Customer Transformation; price base not stated.
- Same document, slide 7: FES 2024 HT has about 85 GW offshore wind by 2035 and over 100 GW by 2050. Demand rises from 285 TWh today to 419 TWh (2035) and **667 TWh (2050)**.
- **How to compare:** the best NESO statement for late 2030s thermal constraints with the planned network. Our 1.95 bn in 2050 is at this level.

### 3.9 Aurora for RenewableUK (Jun 2024): balancing market cost

- p.19 system cost breakdown: "Balancing Market" about GBP 3 bn a year averaged over 2036-2050, base case Net Zero 2035, real 2022 (read from chart).
- Aurora's BM cost covers more than thermal constraints. In 2025 money: about 3.4 bn.

### 3.10 Path-to-2035 only (no 2040+ number)

- **LCP Delta, *Reformed National Pricing measures on GB grid constraint costs* (Mar 2026):** https://insights.lcp.com/rs/032-PAO-331/images/LCP-Delta-Reformed-National-Pricing-Measures-on-GB-Grid-Constraint-Costs-March-2026.pdf, p.21.
  - FES25 HT (LCP), 2035: about 4.4 bn (GBP 2025 real, chart). NESO FES24 HT as plotted: about 2.85 bn.
  - 2033 spike: about 5.9 bn for both.
- **LCP Delta, *Differences in regional wholesale electricity prices* (Jun 2025):** 2035, DESNZ scenario. National constraint costs at least 3.2 bn (removed under zonal; GBP 2024).
- **Baringa for ESO, Constraints Collaboration Project (Jul 2024):** https://www.neso.energy/document/322331/download. To 2035; "nearly GBP 5bn in 2034" (FES23 LtW, flagged as unrealistic).
- **NESO Demand for Constraints Assessment (Mar 2025):** https://www.neso.energy/document/358366/download. To 2035; B6 peak GBP 701 m in 2034.
- **NESO, *Beyond 2030 - Electricity Transmission Update* (30 Jun 2026):** https://www.neso.energy/document/384126/download (technical report: https://www.neso.energy/document/383386/download).
  - Qualitative only: without reinforcement, constraint costs "could increase by three times between 2031 and 2035".
  - The 43 recommended options give net transmission benefits of GBP 58-82 bn (NPV, including constraint savings).
  - No annual series; the horizon is the mid-2030s.
  - The original Beyond 2030 (Mar 2024, https://www.neso.energy/document/304756/download) likewise gives no annual constraint cost beyond 2030.
- **NAO, *Upgrading the electricity transmission network* (HC 589, 11 Sep 2026):** https://www.nao.org.uk/reports/upgrading-the-electricity-transmission-network/. 2030 only (GBP 6.6-7.8 bn without acceleration, 12.7 bn with no upgrades, 2024 prices). It notes that NESO does not recompute constraint costs per delay.

---

## 4. Curtailment and turn-down benchmarks

### 4.1 NESO FES 2024 data table ES1: curtailment (TWh), Holistic Transition

- **URL:** https://api.neso.energy/dataset/549b0667-b533-4748-95bd-f6e13933a47d/resource/8c8a436d-408a-441b-8c7a-84249805772c/download/fes2024_es1_v002.csv
- **Location:** variable "Curtailment (TWh)", by connection and type.
- **Values (wind + solar, HT):**

  | Year | Total TWh | Of which offshore (transmission) | Onshore | Solar |
  |---|---|---|---|---|
  | 2035 | 58.4 | | | |
  | 2040 | 64.3 | | | |
  | 2045 | 51.0 | | | |
  | 2050 | 42.9 | 34.7 | 7.0 | 1.2 |

  - Separate variable "Curtailment including DACCS (TWh)": 31.4 (2030), 58.4 (2035), 63.6 (2040), 33.8 (2045), 12.6 (2050).
  - HT total generation in 2050: 768.7 TWh.
- **How to compare:** FES curtailment is surplus energy in NESO's unconstrained (national) dispatch, not network turn-down. Compare it with our **Stage 1 (copperplate) curtailment**. Same pathway and vintage as ours, so it is the most direct volume check. The DACCS variable's definition is not documented in the CSV.

### 4.2 AFRY (May 2025) FES 24 HT Test: wind market curtailment and network turn-down

- **Location:** slide 14 (labelled values).
- **Wind available resource:** 307 (2030), 491 (2035), 547 (2040), 576 (2050) TWh.
- **Market curtailment before transmission constraints:** 47, 97, 74, **24** TWh. That is 20% of wind in 2035.
- **Wind net constrained-off in the BM:**
  - Offshore: 4.0, 15.0, 17.3, 6.5 TWh.
  - Onshore: 3.9, 5.8, 7.9, 12.5 TWh.
  - 2050 total: **19.0 TWh**.
- **How to compare:** Stage 1 curtailment against 24 TWh; BM wind turn-down against 19 TWh.

### 4.3 FTI for Octopus (2025): constraint volumes

- Slide 29 (chart): total constrained-off (and matching constrained-on) volume about 45 TWh (2040), about 48 TWh (2045), **about 53 TWh (2050)**, mostly offshore wind off and interconnectors on.
- The 2030 volume of 22.6 TWh matches NESO CP30's 23.3 TWh.

### 4.4 FTI for NESO (Apr 2025): constraint volumes

- Slide 8 (chart): constrained-on volume, updated FES24 run, about 34 TWh (2035) and about 45 TWh (2040).

### 4.5 FTI for Octopus (2025) and Aurora (2024): price-duration indicators

- FTI: zero-priced hours about 41% (2030) and about 54% (2035).
- Aurora: 5th-percentile hourly price about GBP 4/MWh by 2050.
- No public source gives a 2050 GB price duration curve or share of negative-price hours. Capture prices by technology for 2050 are not published either. FTI's "wholesale + CfD top-up" (63.0 in 2050) gives an implied CfD gap, not capture prices.

---

## 5. How to compare: our figure for each definition

| Their metric | Our figure to use |
|---|---|
| Demand / load / volume-weighted national price (LCP/GT, FTI-Ofgem, FTI-Octopus slide 32, EEP volume-weighted) | Demand-weighted Stage 1 mean (31.19) |
| Time-weighted / baseload (FTI-Octopus slide 27, Aurora, EEP baseload, probably AFRY) | Unweighted hourly mean of the Stage 1 GB price (compute it) |
| Thermal / transmission redispatch cost (AFRY, LCP/GT, FTI, NOA, NESO 2025 note) | 1.95 bn (offers + bids + IC re-trading); per MWh 3.1 |
| Total balancing cost (NESO ABC 2024, Aurora BM) | Our thermal cost + about 1-1.5 bn of non-thermal (reserve, response, voltage, stability) |
| Market / economic curtailment (FES ES1, AFRY market curtailment) | Stage 1 curtailment |
| Network turn-down (AFRY constrained-off, FTI volumes) | BM wind turn-down (and total bid volume) |
| System cost per MWh (CCC) | Full bill-relevant cost per MWh, not the wholesale price |

**Points to check that the benchmarks raise:**

1. Every net-zero comparator's price rises from 2040 to 2050, by GBP 10-20/MWh. Ours at 31 looks like their late-2030s level. Likely levers:
   - CCS-gas and hydrogen fuel cost and the carbon price on CCS residuals;
   - hydrogen turbines as price setters;
   - how flexible demand (electrolysers, storage) bids;
   - the interconnector foreign price in 2050.
2. Interconnectors resolve most 2040s constraints in AFRY and FTI. Our IC re-trading is small, so check whether IC counter-trading is active in 2050.
3. Weather: FTI-Ofgem and FTI-NESO use 2013 (as we do), FTI-Octopus uses 2009, CCC uses an average year, LCP/GT uses "historical shapes".

## 6. Gaps

- **No NESO (or DESNZ) published thermal constraint cost for 2045-2050.** Best proxies:
  - LCP/GT for DESNZ Figure 34 (2050);
  - AFRY's FES24 HT runs (2050);
  - FTI-Octopus (2050).
  - The NESO series stop at 2041 (NOA 2021/22 Refresh) or 2041/42 (ABC 2024 total balancing).
- **No official 2050 net-zero wholesale price.** DESNZ EEP is a policy-baseline projection with gas at the margin. Best net-zero proxies: LCP/GT for DESNZ (demand-weighted), FTI-Octopus (FES24 HT, time-weighted), AFRY (FES24 HT).
- **No public 2050 price duration curve, negative-price share or capture prices by technology for GB.** Proxies: Aurora percentiles, FTI zero-price shares (to 2035).
- **TYNDP 2024 GB 2040/2050 marginal prices:** not extracted. They are available in ENTSO-E market-model data or the open-tyndp project if wanted.
