# Model Description

How PyPSA-GB builds and solves a model of Great Britain's electricity system. Each page covers what that part of the model does, the method and assumptions, the data it uses, the settings that control it, and how it is validated.

## Scenarios and networks

Every scenario states where its numbers come from (`targets`) and how its fleet is built (`method`); see
{doc}`scenarios_configuration`.

| Type | `targets.source` | `method` | Years | Main data |
|---|---|---|---|---|
| Historical | `historical` | `historical` | `modelled_year` up to 2024 | DUKES thermal plant, REPD renewable and storage sites, ESPENI demand profiles |
| NESO Clean Power 2030 | `CP30` | `brownfield` | 2030 (pathways `FFR`, Further Flex and Renewables, and `ND`, New Dispatch) | CP30 data workbook capacity and energy targets, met with named REPD and TEC projects on the ETYS network |
| NESO Future Energy Scenarios | `FES` | `gsp_block` | any future year (editions 2021 to 2025) | FES building blocks by Grid Supply Point (GSP) and technology; annual demand from FES, shaped by a historical demand year |

The weather year (`renewables_year`) and the demand-shape year (`demand_year`) are chosen separately from the
modelled year, so a 2035 system can be run on 2010 or 2019 weather.

| `network_model` | Size | Use |
|---|---|---|
| `ETYS` | about 2,000 buses, from NESO's Electricity Ten Year Statement, with planned upgrades by year | Constraint and boundary studies; the CP30 and FES studies |
| `Reduced` | 32 buses | Fast tests and teaching; the default quickstart |
| `Zonal` | 17 zones | Fast zonal studies |
| `ETYS` + `network.clustering` | as few buses as you choose (GSP regions, k-means, hierarchical) | A middle ground between detail and run time |

See {doc}`network/topology` and {doc}`network/clustering`.

## How a network is built

The network is built in a fixed order, each step adding one kind of component to the network of the step before:

```{mermaid}
flowchart TD
    subgraph r1[" "]
        direction LR
        D["Demand"] --> R["Renewables"] --> T["Thermal"]
    end
    subgraph r2[" "]
        direction LR
        C["Marginal costs"] --> S["Storage"] --> H["Hydrogen"]
    end
    subgraph r3[" "]
        direction LR
        I["Interconnectors"] --> N["Clustering (optional),<br/>then the solve"]
    end
    r1 --> r2 --> r3
```

The finished network is `resources/network/{scenario}_network_demand_renewables_thermal_generators_storage_hydrogen_interconnectors.nc`.
The finalise step after thermal (`finalize_generator_integration`) also adds a load-shedding generator at every demand
bus, sized at 1.2 x the bus's peak load and priced at the value of lost load (`solve.voll`, GBP/MWh).

How much generation and storage capacity there is, and where, depends on the scenario's `targets` and `method`:

| Method | Capacity source | Placement |
|---|---|---|
| `historical` | DUKES (thermal), REPD (renewables and storage), the interconnector register | each site at its own location |
| `gsp_block` | FES building blocks by GSP and technology | GSP rows at their GSP; `Direct` rows spread within their transmission licence area |
| `brownfield` | a publication's national and regional totals (CP30 ES1 and ES5) | named projects and stations first, then FES rows as proxies for the residual |

```{figure} /_static/figures/supply/capacity_2030_2050.png
:alt: Two panels of horizontal bars giving installed capacity in GW by technology group for 2030 and 2050.
:width: 100%

Installed capacity by technology group in the 2030 build (Clean Power 2030, Further Flex and Renewables pathway) and the
2050 build (FES 2025 Holistic Transition). Gas includes gas with CCS (hatched); hydrogen turbines have their own bar; storage
leaves out consumer flexibility; imports are the interconnector ratings.
```

## Chapters

```{toctree}
:maxdepth: 2

scenarios_configuration
network/index
demand/index
generation/index
flexibility/index
interconnectors
dispatch/index
after_the_solve/index
```
