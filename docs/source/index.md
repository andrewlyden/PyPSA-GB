# PyPSA-GB: Great Britain Energy System Model

```{raw} html
<div class="pgb-hero">
  <p class="pgb-tagline">Open-source modelling of Great Britain's energy system</p>
  <p class="pgb-lead">PyPSA-GB is a model of Great Britain's energy system, centred on its electricity network,
  built on <a href="https://pypsa.org">PyPSA</a>. It represents the transmission network in detail, places demand
  and every generator, storage unit, hydrogen asset and interconnector on it, and dispatches it either in a single
  network-constrained solve or as a two-stage market: wholesale trading followed by the balancing mechanism.
  Scenarios cover historical years and NESO's Clean Power 2030 and Future Energy Scenarios.</p>
  <div class="pgb-badges">
    <a href="https://github.com/andrewlyden/PyPSA-GB"><img src="https://img.shields.io/badge/GitHub-PyPSA--GB-3f6f59?style=flat-square&logo=github" alt="GitHub"></a>
    <a href="development/release_notes.html"><img src="https://img.shields.io/badge/version-2.3.0-5c8b75?style=flat-square" alt="Version 2.3.0"></a>
    <a href="https://doi.org/10.1016/j.esr.2024.101375"><img src="https://img.shields.io/badge/DOI-10.1016%2Fj.esr.2024.101375-89b6d4?style=flat-square" alt="DOI 10.1016/j.esr.2024.101375"></a>
    <img src="https://img.shields.io/badge/Python-3.11+-4f8a9e?style=flat-square&logo=python&logoColor=white" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/License-MIT-c98a5a?style=flat-square" alt="License: MIT">
  </div>
</div>
```

## Explore the documentation

::::{grid} 1 2 3 3
:gutter: 4
:class-container: pgb-cards

:::{grid-item-card} Getting Started
:link: getting_started/index
:link-type: doc
:class-card: sd-shadow-md

**Install and run.** From installation to your first run, your own scenario and its results.
:::

:::{grid-item-card} Model Description
:link: model/index
:link-type: doc
:class-card: sd-shadow-md

**How it works.** Network, demand, supply and dispatch: method, data, settings and validation, one page per part.
:::

:::{grid-item-card} Studies
:link: studies/index
:link-type: doc
:class-card: sd-shadow-md

**What it has been used for.** Clean Power 2030 constraints, FES 2040 and 2050, household bills and the 2020
validation.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
:class-card: sd-shadow-md

**Learn by example.** Notebooks that walk through runs and their results.
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc
:class-card: sd-shadow-md

**Look it up.** Configuration keys, data sources and licences, outputs, publications, API and glossary.
:::

:::{grid-item-card} Development
:link: development/index
:link-type: doc
:class-card: sd-shadow-md

**Contribute.** Architecture, contributing, tests and release notes.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

getting_started/index
model/index
studies/index
tutorials/index
reference/index
development/index
```

## Great Britain from 2030 to 2050

::::{grid} 1 3 3 3
:gutter: 3
:class-container: pgb-maps

:::{grid-item}
```{image} _static/landing/constraints_2030.png
:alt: Map of Great Britain in 2030 showing where the balancing mechanism turned wind down and other generation up
:target: studies/cp30_2030.html
```
**2030: where the network binds.** Wind turned down in Scotland and East Anglia, plant and imports turned up
further south. {doc}`More <studies/cp30_2030>`
:::

:::{grid-item}
```{image} _static/landing/generation_2040.png
:alt: Map of Great Britain in 2040 with a pie of installed generation by technology in each of NESO's 17 zones
:target: model/generation/index.html
```
**2040: what gets built.** Installed generation in each of NESO's 17 zones: offshore wind nearly doubles from 2030.
{doc}`More <model/generation/index>`
:::

:::{grid-item}
```{image} _static/landing/flexibility_2050.png
:alt: Map of Great Britain in 2050 with bars of grid storage, electrolysers and hydrogen turbines in each zone
:target: model/flexibility/index.html
```
**2050: flexibility at net zero.** Grid storage, electrolysers and hydrogen turbines in each zone.
{doc}`More <model/flexibility/index>`
:::
::::

## What it is for

| Question | What the model gives |
|---|---|
| How do transmission limits shape dispatch, and what does managing them cost? | Hourly flows on every line and boundary, the redispatch needed to make a market schedule feasible, its volume and cost, and curtailment by technology and location. |
| What does a future GB system look like hour by hour? | Dispatch, prices, storage, hydrogen and interconnector flows for NESO's Clean Power 2030 pathways and Future Energy Scenarios, on the network those pathways assume. |
| How close is the model to what happened? | Historical years built from published plant and demand data, compared with outturn generation, prices and balancing actions. |

## The model at a glance

```{raw} html
:file: _includes/model_at_a_glance.html
```

Each scenario is built once into a PyPSA network, then solved either as one network-constrained dispatch or as a
two-stage market, in which Stage 2 redispatches the Stage 1 market schedule until it fits the network; the
difference is the cost of network constraints. {doc}`model/index` explains every part.

## How to cite

If you use PyPSA-GB in your research, please cite:

> Lyden, A., Sun, W., Struthers, I., Franken, L., Hudson, S., Wang, Y. and Friedrich, D., 2024.
> PyPSA-GB: An open-source model of Great Britain's power system for simulating future energy scenarios.
> *Energy Strategy Reviews*, 53, p.101375. <https://doi.org/10.1016/j.esr.2024.101375>

````{dropdown} BibTeX
```bibtex
@article{lyden2024pypsagb,
  author  = {Lyden, A. and Sun, W. and Struthers, I. and Franken, L. and Hudson, S. and Wang, Y. and Friedrich, D.},
  title   = {{PyPSA-GB}: An open-source model of {Great Britain}'s power system for simulating future energy scenarios},
  journal = {Energy Strategy Reviews},
  volume  = {53},
  pages   = {101375},
  year    = {2024},
  doi     = {10.1016/j.esr.2024.101375}
}
```
````

Please also cite PyPSA and the data sources your study relies on ({doc}`reference/licences`). Papers and reports
that use PyPSA-GB are listed under {doc}`reference/publications`.

## Contributors

Developed at the Institute for Energy Systems, University of Edinburgh, originally as part of the INTEGRATE project led by
Prof Daniel Friedrich.

```{raw} html
:file: _includes/contributors.html
```

## Licence

MIT. The data PyPSA-GB reads keep their own licences: {doc}`reference/licences`.
