# Household electricity bills: 2030, 2040 and 2050

All money is GBP in real 2025 prices, for the bill's reference
household: 3,100 kWh a year, single-rate meter, direct debit, GB average.

## Question

What would a household's electricity bill be on the power systems solved in the {doc}`cp30_2030` and
{doc}`fes_long_range` studies, and which parts of the bill move with the network and the year?

## Setup

The bill is built line by line from the solved market years and from published charges, in Ofgem's price-cap
structure: wholesale energy and its hedging, the network charges (TNUoS, DUoS and BSUoS, with the constraint costs
the balancing stage gives), the policy schemes (CfD, RO, Capacity Market and the rest), and suppliers' operating costs,
EBIT and headroom, plus 5% VAT. How each line is built is in {doc}`/model/after_the_solve/consumer_bill/index`.

| Bill | Solved year | Network |
| --- | --- | --- |
| 2030 LOWER | Clean Power 2030, Further Flex and Renewables | the network CP30 recommends |
| 2030 CENTRAL | as above | CP30's network without its three essential accelerations |
| 2030 UPPER | as above | no reinforcements after 2025 |
| 2040 | FES 2025 Holistic Transition | the FES network as built for the year |
| 2050 | FES 2025 Holistic Transition | the FES network as built for the year |

## How to reproduce

The bill reads a solved market year: solve the study's twelve monthly market months first ({doc}`cp30_2030`,
{doc}`fes_long_range`). Each bill is a forecast case in `consumer_bill.forecasts.cases`, named as in the table below
(`CP30_2030_cp30_network`, `CP30_2030_no_acceleration`, `CP30_2030_no_upgrades`, `FES25_HT_2040`, `FES25_HT_2050`),
and its main outputs are:

| Output | What it holds |
| --- | --- |
| `resources/analysis/<case>_consumer_bill_future_summary.json` | The bill's total, VAT and groupings |
| `resources/analysis/<case>_consumer_bill_future_components.csv` | Every bill line, with its Ofgem group and source |

## Results

GBP a year for the reference household:

| | 2030 LOWER | 2030 CENTRAL | 2030 UPPER | 2040 | 2050 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wholesale | 419.31 | 399.75 | 390.63 | 387.34 | 384.39 |
| Network costs | 317.72 | 366.88 | 396.73 | 371.65 | 400.29 |
| Policy costs | 122.78 | 119.12 | 116.43 | 31.98 | 23.89 |
| Operating costs, EBIT and headroom | 197.28 | 197.32 | 197.41 | 194.94 | 194.94 |
| **Bill excluding VAT** | **1,057.10** | **1,083.06** | **1,101.20** | **985.92** | **1,003.51** |
| VAT (5%) | 52.85 | 54.15 | 55.06 | 49.30 | 50.18 |
| **Bill including VAT** | **1,109.95** | **1,137.21** | **1,156.26** | **1,035.21** | **1,053.68** |

<!-- resources/analysis/<case>_consumer_bill_future_components.csv: annual_gbp_per_3100kwh_ex_vat summed by
ofgem_group; resources/analysis/<case>_consumer_bill_future_summary.json: ex_vat_gbp, vat_gbp; cases
CP30_2030_cp30_network, CP30_2030_no_acceleration, CP30_2030_no_upgrades, FES25_HT_2040, FES25_HT_2050 -->

```{figure} /_static/figures/studies/household_bill.png
:alt: Stacked bars of the household electricity bill by Ofgem price-cap group for 2030 on three network cases, 2040 and 2050, rising from 1,110 to 1,156 GBP across the 2030 network cases and falling to 1,035 GBP in 2040 and 1,054 GBP in 2050.
:width: 100%

The household bill by Ofgem price-cap group, GBP a year including VAT, for 2030 on the three network-delivery cases
and for 2040 and 2050.
```

What the bills show:

- **The network delivered by 2030 moves the bill by about GBP 46 a year.** From the recommended network (LOWER) to no
  reinforcements after 2025 (UPPER), network costs rise by GBP 79 as constraint costs grow, while the wholesale line
  falls by GBP 29.
- **Policy costs fall away after 2030**, from about GBP 120 to 32 in 2040 and 24 in 2050, as the Renewables
  Obligation closes and the earlier CfDs expire.
- **Network costs overtake wholesale energy** as the largest line in 2050, and already in 2030 on the least
  reinforced network (UPPER).

## Limitations

- **A preliminary research layer.** The bill combines model results with published charges carried forward; it is a
  scenario projection, not a forecast of Ofgem's cap.
- **One weather year and one pathway per year**, as in the studies it reads; each year is a snapshot, not a solved
  pathway between the years.
- **Supplier costs** are held flat in real terms; they scale with customers, not with the power system.
- **The reference household** uses 3,100 kWh a year; a household with a heat pump or an electric car uses more and
  pays a different bill.
