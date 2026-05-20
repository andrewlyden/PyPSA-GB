# Validation_Jan2020 — Deep Dive Review
_Generated: 2026-04-25 15:04_
_Scenario: Validation_Jan2020 · period: 2020-01-01 → 2020-02-01 (744h) · two-stage copperplate→constrained dispatch_
Reference data sources: Elexon BMRS (MID, SBP/SSP, BOALF, DISBSAD), ESPENI (fuel-type fleet generation), NESO (thermal constraint costs, DA boundary flows).

---
## 1. Headline numbers
| Metric | Model | Reference | Source | Ratio |
|---|---:|---:|---|---:|
| Total demand served (MWh) | 27,695,657 | 27,247,170 | ESPENI | 1.016 |
| Mean wholesale price (£/MWh) | 35.88 | 34.67 | Elexon MID (N2EX) | 1.035 |
| Mean SBP (£/MWh) | 35.88 | 35.21 | Elexon SBP | 1.019 |
| BM cost (period total, £) | £51,270,104 | £70,521,804 | NESO thermal Jan 2020 | 0.727 |
| BM cost annualised (£/yr) | £603,664,124 | £1.4B | NESO published BSUoS | 0.431 |
| Total BOALF increase volume (MWh) | 904,422 | 1,308,052 | Elexon BOALF (all flags) | 0.691 |
| Total BOALF decrease volume (MWh) | 789,212 | 1,459,310 | Elexon BOALF (all flags) | 0.541 |
| Total net imports (MWh) | 1,380,402 | 1,382,312 | ESPENI | 0.999 |

## 2. Wholesale price (Stage 1) vs market indices
Model SMP is the LP shadow price of the **copperplate** Stage-1 wholesale solve (zero spread between buses).

|                       |   Mean (£/MWh) |   Median |   Std |   Min |    Max |
|:----------------------|---------------:|---------:|------:|------:|-------:|
| Model wholesale (SMP) |          35.88 |    35.2  |  5.4  | 10    |  45.44 |
| Elexon MID (N2EX)     |          34.67 |    34.94 | 10.46 |  5.1  |  86.26 |
| Elexon SBP            |          35.21 |    35.31 | 20.3  | -4.36 | 150    |
| Elexon SSP            |          35.21 |    35.31 | 20.3  | -4.36 | 150    |
| Model BM mean nodal   |          26.41 |    27.36 | 14.21 |  3.76 | 117.59 |

- Hourly correlation **model vs MID**: r = 0.565  ·  MAE £6.43/MWh  ·  RMSE £8.71/MWh
- Hourly correlation **model vs SBP**: r = 0.361  ·  MAE £15.47/MWh  ·  RMSE £19.03/MWh
- Mean **model BM nodal** price (demand buses): £26.41/MWh vs SBP £35.21/MWh → ratio 0.75.
- Nodal spread (max-min within hour): mean £58.22/MWh, 66.3% of hours > £50, 9.9% of hours < £5 (copperplate-like).

## 3. Carrier dispatch — wholesale and BM stages vs ESPENI fuel-type fleet
ESPENI/Elexon ELEC_POWER_* aggregates are the only available fuel-type fleet truth.
`r` = hourly Pearson correlation over the 744-hour window.

| carrier                              | model_ws_mwh   | model_bm_mwh   | espeni_mwh   |   ws_ratio |   bm_ratio | r_ws_hourly   | r_bm_hourly   |
|:-------------------------------------|:---------------|:---------------|:-------------|-----------:|-----------:|:--------------|:--------------|
| CCGT                                 | 7,931,325      | 8,834,175      | 8,292,180    |      0.956 |      1.065 | 0.978         | 0.987         |
| nuclear                              | 4,966,653      | 4,966,653      | 4,972,170    |      0.999 |      0.999 | 0.946         | 0.946         |
| coal                                 | 0              | 0              | 1,489,079    |      0     |      0     | 0.129         | 0.102         |
| biomass                              | 1,666,076      | 1,666,075      | 1,752,804    |      0.951 |      0.951 | -0.131        | 0.019         |
| OCGT                                 | 59,963         | 39,015         | 8,738        |      6.862 |      4.465 | 0.264         | 0.275         |
| large_hydro                          | 432,377        | 209,961        | 517,145      |      0.836 |      0.406 | 0.374         | 0.585         |
| embedded_wind                        | 1,960,451      | 1,959,633      | 1,961,651    |      0.999 |      0.999 | 0.999         | 0.999         |
| embedded_solar                       | 73,322         | 73,310         | 294,884      |      0.249 |      0.249 | 0.786         | 0.786         |
| wind_onshore+offshore (vs ELEX_WIND) | 7,555,363      | 7,020,222      | 6,328,930    |      1.194 |      1.109 | 0.965         | 0.985         |
| wind total (incl embedded vs NGEM)   | 9,515,814      | 8,979,855      | 8,290,581    |      1.148 |      1.083 | —             | —             |
| pumped_hydro (discharge)             | 31,716         | 18,390         | 139,272      |      0.228 |      0.132 | 0.271         | 0.265         |
| pumped_hydro (charging)              | 34,005         | 111,503        | 180,001      |      0.189 |      0.619 | —             | —             |

**Reading guide**
- `ws_ratio` = model wholesale ÷ ESPENI. >1 means the copperplate over-dispatches the carrier; the BM should bring it back toward 1.0 if the network constraint is the binding mechanism.
- `bm_ratio` close to 1.0 with high `r_bm_hourly` is the dispatch target.
- Embedded wind/solar are forced to ESPENI profiles in the model so r ≈ 1.0 by construction.

## 4. Balancing mechanism volumes by carrier — model vs Elexon BOALF
Elexon volumes are aggregated from `BOALF` acceptance levels (all flags) for January 2020.

|                                 | model_increase_mwh   | model_decrease_mwh   | elexon_increase_mwh   | elexon_decrease_mwh   | inc_ratio   | dec_ratio   | model_net_mwh   | elexon_net_mwh   |
|:--------------------------------|:---------------------|:---------------------|:----------------------|:----------------------|:------------|:------------|:----------------|:-----------------|
| CCGT                            | 903,309              | 460                  | 595,556               | 952,090               | 1.517       | 0.000       | 902,850         | -356,534         |
| unknown                         | 0                    | 0                    | 252,850               | 4,002                 | 0.000       | 0.000       | 0               | 248,848          |
| Pumped Storage Hydroelectricity | 0                    | 0                    | 139,436               | 164,610               | 0.000       | 0.000       | 0               | -25,174          |
| wind_onshore                    | 2                    | 518,477              | 116,390               | 767                   | 0.000       | 675.980     | -518,475        | 115,622          |
| coal                            | 0                    | 0                    | 112,877               | 269,740               | 0.000       | 0.000       | 0               | -156,863         |
| large_hydro                     | 394                  | 222,811              | 38,348                | 2,641                 | 0.010       | 84.366      | -222,416        | 35,707           |
| biomass                         | 1                    | 1                    | 32,902                | 65,286                | 0.000       | 0.000       | -0              | -32,384          |
| nuclear                         | 0                    | 0                    | 13,794                | 126                   | 0.000       | 0.001       | -0              | 13,669           |
| wind_offshore                   | 0                    | 16,666               | 5,900                 | 50                    | 0.000       | 333.320     | -16,666         | 5,850            |
| advanced_biofuel                | 0                    | 0                    | 0                     | 0                     | —           | —           | -0              | 0                |
| shoreline_wave                  | 0                    | 7,565                | 0                     | 0                     | —           | —           | -7,565          | 0                |
| waste_to_energy                 | 703                  | 1                    | 0                     | 0                     | —           | —           | 702             | 0                |
| OCGT                            | 0                    | 20,948               | 0                     | 0                     | —           | —           | -20,947         | 0                |
| tidal_stream                    | 0                    | 1,437                | 0                     | 0                     | —           | —           | -1,437          | 0                |
| solar_pv                        | 0                    | 1                    | 0                     | 0                     | —           | —           | -0              | 0                |
| small_hydro                     | 0                    | 8                    | 0                     | 0                     | —           | —           | -7              | 0                |
| oil                             | 0                    | 0                    | 0                     | 0                     | —           | —           | -0              | 0                |
| sewage_gas                      | 0                    | 0                    | 0                     | 0                     | —           | —           | -0              | 0                |
| biogas                          | 2                    | 1                    | 0                     | 0                     | —           | —           | 1               | 0                |
| EU_import                       | 0                    | 0                    | 0                     | 0                     | —           | —           | 0               | 0                |
| landfill_gas                    | 3                    | 2                    | 0                     | 0                     | —           | —           | 1               | 0                |
| embedded_wind                   | 5                    | 822                  | 0                     | 0                     | —           | —           | -817            | 0                |
| embedded_solar                  | 1                    | 12                   | 0                     | 0                     | —           | —           | -12             | 0                |
| load_shedding                   | 0                    | 0                    | 0                     | 0                     | —           | —           | 0               | 0                |

**Notes**
- `unknown` in the Elexon column is largely interconnectors and unmapped BMUs — model has none.
- Where `inc_ratio` << 1 the model is *not using that carrier* in the BM (mostly because wholesale has already dispatched it at full availability — see §3).
- CCGT inc_ratio = 1.52 reflects the structural copperplate penalty: Scottish wind is dumped south in Stage 1 and English CCGT must turn up in Stage 2 — exaggerated because wind in Stage 1 is 1.19x ELEX_WIND.
- Wind decrease in the BM (~518 GWh onshore + 16.7 GWh offshore) is the model's constraint-driven curtailment; Elexon shows almost no wind BOALF turn-down because real-world curtailment happens via offer prices < SBP (commercial), not via STOR/STOR-RR.
- Large hydro `dec_ratio` ≈ 84 and PSH `inc_ratio` = 0 are artefacts of the model treating reservoir hydro as a cheap energy carrier in Stage 1 and using it for turn-down in Stage 2; the real BM uses PSH for turn-up and large_hydro is largely unflagged.

> **Headline numbers caveat:** the §1 BOALF totals (904 / 789 GWh inc/dec) come from the *net difference* between BM and wholesale dispatch summed by carrier. The `Validation_Jan2020_bm_validation.csv` headline of 1,612,887 MWh inc/dec is the *sum of all signed dispatch changes per timestep* (i.e. it counts a reservoir hydro unit going down then back up as twice the volume) and is therefore ~1.7-2.0x higher.

## 5. BM cost decomposition by carrier (model)
Total model BM net cost: **£51,270,104** (£603,664,124/yr).

| carrier                         |       offer_cost |          bid_cost |          net_cost |   increase_MWh |   decrease_MWh |   pct_total |
|:--------------------------------|-----------------:|------------------:|------------------:|---------------:|---------------:|------------:|
| CCGT                            |      6.40493e+07 |      -2.01057e+07 |       4.39436e+07 |    1.60699e+06 |         704144 |          86 |
| wind_onshore                    |    733           |       1.04569e+07 |       1.04577e+07 |    2           |         518477 |          20 |
| Battery                         |  22847           |       1.22883e+06 |       1.25168e+06 |  190           |          24577 |           2 |
| Pumped Storage Hydroelectricity | 437096           |   30133           |  467229           | 4367           |          95190 |           1 |
| waste_to_energy                 |  84302           |      60           |   84362           |  703           |              1 |           0 |
| embedded_wind                   |      0           |   32887           |   32887           |    5           |            822 |           0 |
| load_shedding                   |   2666           |     498           |    3164           |    0           |              0 |           0 |
| landfill_gas                    |    308           |     145           |     453           |    3           |              2 |           0 |
| embedded_solar                  |      0           |     368           |     368           |    1           |             12 |           0 |
| nuclear                         |    318           |      26           |     344           |    0           |              0 |           0 |
| biogas                          |    244           |      84           |     329           |    2           |              1 |           0 |
| small_hydro                     |      0           |     227           |     227           |    0           |              8 |           0 |
| biomass                         |     80           |      86           |     166           |    1           |              1 |           0 |
| wind_offshore                   |     37           |      72           |     110           |    0           |          16666 |           0 |
| advanced_biofuel                |     28           |      31           |      58           |    0           |              0 |           0 |
| solar_pv                        |      0           |      31           |      31           |    0           |              1 |           0 |
| coal                            |     10           |      12           |      22           |    0           |              0 |           0 |
| sewage_gas                      |     11           |       7           |      18           |    0           |              0 |           0 |
| EU_import                       |      0           |       0           |       0           |   13           |             13 |           0 |
| oil                             |     33           |     -40           |      -7           |    0           |              0 |          -0 |
| tidal_stream                    |     -0           | -207303           | -207303           |    0           |           1437 |          -0 |
| OCGT                            |  12761           | -940778           | -928018           |   95           |          21042 |          -2 |
| shoreline_wave                  |      0           |      -1.70384e+06 |      -1.70384e+06 |    0           |           7565 |          -3 |
| large_hydro                     |  24518           |      -2.15792e+06 |      -2.13341e+06 |  511           |         222927 |          -4 |

- `offer_cost` = paid for turn-up; `bid_cost` = paid (or revenue if negative) for turn-down.
- The model has no equivalent of Elexon system price (SBP/SSP) per carrier; comparison must be done at total-cost level (§1) or via NESO thermal cost (§6).

## 6. Constraint costs vs NESO thermal benchmark
- **Model BM cost (Jan 2020):** £51,270,104
- **NESO thermal cost (Jan 2020):** £70,521,804
- **Ratio model/NESO:** 0.727 (target ≥ 0.80 for in-month accuracy; structural copperplate gap explains some of the shortfall).

**NESO cost by boundary (Jan 2020)**

| boundary   | value       | share   |
|:-----------|:------------|:--------|
| SSE-SP     | £34,768,885 | 49.3%   |
| SCOTEX     | £22,704,409 | 32.2%   |
| SSHARN     | £12,764,182 | 18.1%   |
| SEIMP      | £284,328    | 0.4%    |
| ESTEX      | £0          | 0.0%    |
| SWALEX     | £0          | 0.0%    |

**DA boundary flows — model vs NESO (Jan 2020 means)**

| boundary   |   model_mean_flow_mw |   model_max_flow_mw |   neso_mean_flow_mw |   neso_max_flow_mw |   neso_mean_limit_mw |   neso_mean_utilisation |   neso_pct_above_90 |
|:-----------|---------------------:|--------------------:|--------------------:|-------------------:|---------------------:|------------------------:|--------------------:|
| ESTEX      |              2196.54 |             3141.06 |             2656.05 |            6282    |              8488    |                    0.34 |                0.92 |
| SCOTEX     |              3315.86 |             4541.1  |             3193.69 |            5180.19 |              4441.37 |                    0.72 |               23.33 |
| SEIMP      |              3466.07 |             5222.08 |             3627.47 |            7684.84 |              7427.86 |                    0.5  |                2.34 |
| SSE-SP     |              2474.26 |             7104.69 |             2375.86 |            2705    |             14250.4  |                    0.84 |               70.83 |
| SSHARN     |              6114.71 |             8146.29 |             5723.21 |            8750    |              7433.99 |                    0.77 |               36.5  |
| SWALEX     |               943.2  |             1996.87 |              912.21 |            3074    |             99999    |                    0.01 |                0    |

## 7. Non-BOALF balancing actions (DISBSAD) — Elexon only
These are Disposal/Bilateral Service Adjustment Data — settlement-period actions outside BOALF (e.g. STOR, fast reserve, disconnection). The model does not represent ancillary services, so this is a context number for the *unmodelled* portion of BSUoS.

| Metric                 | Jan 2020     |
|:-----------------------|:-------------|
| DISBSAD records        | 10,162.00    |
| Abs volume MWh         | 409,465.82   |
| Net volume MWh         | -142,425.40  |
| Total cost £           | 4,455,735.75 |
| SO-flagged share (vol) | 87.10        |

DISBSAD net cost (£4,455,736) is roughly **8.7%** of the model's BM cost — i.e. there is a structural ~5-10% of January 2020 balancing that the model cannot represent by design.

## 8. Interconnectors (sanity — fixed to ESPENI in Stage 1)

| interconnector   | model_net_import_mwh   | espeni_net_import_mwh   |   ratio |   r_hourly |
|:-----------------|:-----------------------|:------------------------|--------:|-----------:|
| IFA              | 746,506                | 748,798                 |   0.997 |      1     |
| Britned          | 358,634                | 363,220                 |   0.987 |      0.999 |
| Nemo             | 355,860                | 355,821                 |   1     |      1     |
| IRL              | -94,748                | -95,563                 |   0.991 |      0.993 |
| EastWest         | 14,150                 | 10,036                  |   1.41  |      0.997 |

Total net import — model 1,380,402 MWh  vs ESPENI 1,382,312 MWh (ratio 0.999).

## 9. Top constrained network elements

| component       | type   |   s_nom_MVA |   hours_congested |   max_loading_fraction |   mean_loading_fraction |
|:----------------|:-------|------------:|------------------:|-----------------------:|------------------------:|
| NORT41_OSBA42_0 | line   |        2009 |               420 |                  1     |                   0.759 |
| WADW21_WACW21_0 | line   |         220 |               380 |                  0.966 |                   0.684 |
| DRAX41_EGGB42_0 | line   |        2089 |               304 |                  1     |                   0.845 |
| CONQ41_FLIB41_1 | line   |        1438 |               106 |                  0.977 |                   0.676 |
| CRUA2Q_DALL2-_0 | line   |         283 |                86 |                  1     |                   0.251 |
| GRAI41_TILB41_0 | line   |        2003 |                73 |                  1     |                   0.567 |
| COTT42_STAY41_0 | line   |        2212 |                72 |                  1     |                   0.677 |
| KEAP41_KEAD41_0 | line   |         500 |                57 |                  1     |                   0.166 |
| CRYR4-_TORN4-_0 | line   |         935 |                 7 |                  1     |                   0.511 |
| SMEA4Q_STHA4B_0 | line   |         900 |                 5 |                  0.997 |                   0.371 |
| SMEA4Q_TORN4-_0 | line   |         900 |                 5 |                  0.997 |                   0.371 |
| AUCH2-_MAHI2-_0 | line   |         570 |                 4 |                  1     |                   0.559 |

## 10. Summary verdict

| Dimension | Result | Comment |
|---|---|---|
| Wholesale price level | **1.03x MID** | Within 5% of N2EX. |
| Wholesale price shape (hourly r) | **r = 0.57 vs MID** | Strong if >0.7. |
| BM total cost vs NESO | **0.73x** | 0.66–0.73x — structural copperplate gap. |
| BM volume (BOALF inc) | **0.69x** inc / **0.54x** dec | Net carrier-level diff; matches BOALF inc within ~30% but mix is wrong (see §4). |
| BM carrier mix | CCGT inc 1.52x; PSH/coal/biomass/nuclear inc ~0 | Wholesale has already dispatched PSH/coal/biomass/nuclear at full availability, leaving zero BM headroom. |
| Carrier dispatch (BM, vs ESPENI) | CCGT 1.07x, wind 1.11x, large_hydro 0.41x | Coal 0.00x (none modelled in Jan-2020 fleet), biomass 0.95x, nuclear hourly r = 0.95. |
| Interconnectors | 1.00x ESPENI total | Fixed to actuals. |

**Key gaps to close (in priority)**

1. **Coal absent from Jan-2020 fleet (model 0 vs ESPENI 1.49 TWh)** — coal-fired plant is in the generators table but availability/MC pushes it out of merit; this is the largest single carrier-level energy gap and feeds directly into the CCGT over-dispatch.
2. **PSH discharge 0.13x ESPENI / inc_ratio = 0** — PSH provides no Stage-2 turn-up; wholesale already exhausts its energy budget. Likely needs reservoir/cycle constraints or higher Stage-1 marginal cost so headroom remains for BM.
3. **Biomass and coal BM turn-up ≈ 0** — same diagnosis as PSH: Stage 1 dispatches to full availability. Add min-stable / must-run or Stage-1 MC uplift to free BM headroom.
4. **CCGT BM turn-up 1.52x BOALF, dispatch 1.07x ESPENI** — structural copperplate symptom; English CCGT covers Scottish wind that would be self-curtailed at day-ahead in reality. Partial relaxation of wholesale s_nom or zonal Stage-1 pricing would close it.
5. **Wind +19% in Stage 1, +11% in BM vs ELEX_WIND** — performance factors (0.80/0.82) over-shoot after curtailment; revisit factor calibration on a longer window (Jan-only is volatile).
6. **Wholesale price hourly r = 0.57 vs MID (mean within 5%)** — level OK, shape correlation is mediocre; investigate diurnal merit-order ordering and gas/coal MC spread.
7. **DISBSAD £4.5M (8.7% of model BM cost)** — no model analogue (STOR/fast-reserve/disconnection); document as structural omission.
8. **NESO thermal coverage 0.73x** — outage schedule + voltage constraints would close the remaining 25-30% gap.
