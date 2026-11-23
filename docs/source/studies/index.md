# Studies

Applications of PyPSA-GB. Each study states its question, its setup, how to reproduce it, its results and its
limitations.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: pgb-gallery

:::{grid-item-card} Clean Power 2030: network constraints and balancing costs
:img-top: /_static/figures/studies/cp30_wind_turn_down_map.png
:img-alt: Three maps of Great Britain showing where wind is turned down in 2030 under three network-delivery cases
:link: cp30_2030
:link-type: doc

What managing the network costs in 2030 on NESO's Clean Power plan, for three network-delivery cases, against
NESO's own estimates.
:::

:::{grid-item-card} FES long-range years: 2040 and 2050
:img-top: /_static/figures/studies/fes_wind_turn_down_map.png
:img-alt: Two maps of Great Britain showing where wind is turned down in 2040 and 2050
:link: fes_long_range
:link-type: doc

The FES 2025 Holistic Transition system in 2040 and 2050: dispatch, curtailment and balancing costs, checked
against FES.
:::

:::{grid-item-card} Household electricity bills: 2030, 2040 and 2050
:img-top: /_static/figures/studies/household_bill.png
:img-alt: Stacked bars of the household electricity bill by Ofgem price-cap group for 2030, 2040 and 2050
:link: household_bills
:link-type: doc

A household's bill on the solved systems, line by line in Ofgem's price-cap groups, and how it moves with the network
and the year.
:::

:::{grid-item-card} Historical validation: 2020
:img-top: /_static/figures/studies/validation_2020_monthly.png
:img-alt: Monthly charts of the model's constraint cost, constraint volume and wholesale price against NESO and ELEXON outturn in 2020
:link: validation_2020
:link-type: doc

The model's 2020 against what happened: constraint cost and volume, Scottish boundary flows, dispatch and prices, month by month.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

cp30_2030
fes_long_range
household_bills
validation_2020
```
