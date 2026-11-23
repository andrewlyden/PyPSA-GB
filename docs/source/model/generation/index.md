# Generation

Generators that meet demand: thermal plant, which burns a fuel or splits atoms and can be dispatched at will, and
renewables, whose output each hour is set by the weather. Storage, hydrogen and the interconnectors, which also supply
power at times, have chapters of their own ({doc}`../flexibility/index`, {doc}`../interconnectors`); where each step sits
in the build is on {doc}`../index`.

| Page | Rules | What it adds |
|---|---|---|
| {doc}`renewables` | `add_renewables_to_network`, with the profile rules in `rules/renewables.smk` | wind, solar, hydro and marine `Generator`s with hourly availability (`p_max_pu`) |
| {doc}`thermal` | `add_thermal_generators_to_network`, `apply_marginal_costs_to_network` | gas, nuclear, coal, oil, biomass, waste and hydrogen-turbine `Generator`s, their availability and every generator's marginal cost |

Renewables are added before thermal plant, and marginal costs are set once both are in, for every generator. The
finalise step after thermal (`finalize_generator_integration`) also adds a load-shedding generator at every demand bus,
sized at 1.2 x the bus's peak load and priced at the value of lost load (`solve.voll`, GBP/MWh;
{doc}`../dispatch/single_solve`). How support schemes (CfD, RO) set renewable bids and offers is in
{doc}`../dispatch/bids_offers_support`.

Each page follows the same order: what the step does, its method, data, configuration and validation.

```{toctree}
:maxdepth: 2

renewables
thermal
```
