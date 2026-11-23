# Consumer bill

## What it does

The bill layer turns a solved market year into an annual household electricity bill. Some lines come from the
model's dispatch and prices; others apply a published rule to a modelled quantity; the rest are published charges
taken as they stand. Every line keeps either a model derivation or a named published source, never a blend that
cannot be traced.

The model does not set regulated charges or supplier costs. It supplies operational evidence: hourly prices, dispatch
and redispatch, and the plant that earns support.

There are two products:

| Product | What it does | Boundary |
| --- | --- | --- |
| Future projection | Combines twelve solved months with declared published charges | A scenario projection, not a statutory tariff forecast |
| Historical tariff replay | Replays pinned Ofgem tariffs by cap period and charge-restriction region | A source-accounting result, not a reconstructed model bill |

## Method and assumptions

### The household

| | |
| --- | --- |
| Consumption | 3,100 kWh a year, single-rate meter |
| Payment | Direct debit (Ofgem's "other payment method" allowances) |
| Location | GB average |
| Period | The modelled calendar year |
| Prices | GBP, real 2025 |
| VAT | 5%, applied once to the ex-VAT total |

A standard-credit customer pays more, through Ofgem's higher operating-cost and debt allowances; the bill does not
model them. The projector also reports archetype bills at other consumption levels.

### From the model to the bill

```{mermaid}
flowchart LR
    S1["Stage 1: wholesale market<br/>hourly GB price"]
    S2["Stage 2: balancing mechanism<br/>physical dispatch, accepted bids and offers"]
    REG["Registers and contract books<br/>CfD rounds, ROC stations, CM agreements"]
    PUB["Published charges<br/>Ofgem, NESO, DESNZ, OBR"]
    REC["Recovery<br/>volumes, losses, price year"]
    BILL["Household bill<br/>about 30 lines + VAT"]
    S1 --> REC
    S2 --> REC
    REG --> REC
    PUB --> REC
    REC --> BILL
```

1. **Prices.** Stage 1 is a copperplate market for the whole of GB, solved in rolling day-ahead windows for each
   month. Its hourly price drives the wholesale energy lines and is the reference price CfDs settle against. See
   {doc}`wholesale`.
2. **Dispatch.** Stage 2 redispatches plant around the grid's limits. Its physical output is what CfDs and ROCs are
   paid on, and its accepted bids and offers are the constraint cost. See {doc}`cfd` and {doc}`networks`.
3. **Contracts and registers.** Allocation rounds, ROC stations and Capacity Market agreements come from LCCC, Ofgem
   and NESO, matched to the modelled fleet. See {doc}`capacity_market` and {doc}`policy_levies`.
4. **Published charges.** Networks, supplier costs and the smaller levies come from Ofgem, NESO and government
   figures, each stating how its own period maps onto the bill's calendar year. See {doc}`networks` and
   {doc}`supplier`.
5. **Recovery.** Each system cost is spread over the volume it is levied on and multiplied by the household's
   3.1 MWh. See {doc}`recovery`.

### Where each line comes from

Each line is one of three kinds, declared in `data/market/consumer_bill_benchmarks/bill_line_sources.csv`: modelled
(from the solve), model-informed (a published rule applied to a modelled quantity) or published.

```{figure} /_static/figures/dispatch/bill_sources.png
:alt: Four horizontal stacked bars, one per Ofgem price-cap group, split by where each line comes from: wholesale (about 400 GBP) is mostly modelled, network costs (367) mostly published with a modelled constraint line, operating costs (197) all published, policy costs (119) model-informed and published.
:width: 100%

A worked example: the 2030 household bill of the central case of the CP30 2030 study by Ofgem price-cap group, each line coloured by its kind (before VAT). The model drives the wholesale lines and the constraint part of BSUoS; most network and supplier costs are published charges.
```

### How the lines are grouped

How a line is grouped changes how the bill is shown, never what it adds up to. Groupings are data:
`data/market/consumer_bill_benchmarks/bill_groupings.csv` assigns every line to a group in each view, with the reason,
and `bill_grouping_views.csv` declares the vintage in use. Read them only through
`scripts/consumer_bill/bill_groupings.py`. A line with no group is refused.

**Ofgem's price-cap categories.** The headline bill is grouped as Ofgem groups the price cap, at a named cap release
(the current vintage is Oct-Dec 2026: cap model v1.31, Annex 2 v1.30, Annex 4 v1.25):

| Group | Contains | Ofgem source |
| --- | --- | --- |
| Wholesale | Energy, shaping, procurement allowances, losses, CfD, Capacity Market | Direct fuel (which carries the CfD, shaping, hedging and losses) and the Capacity Market |
| Policy costs | RO, FiT, Warm Home Discount, Sizewell C RAB, Network Charging Compensation, AAHEDC, ECO | Annex 4 |
| Network costs | TNUoS, DUoS and BSUoS (constraints, other services, voltage and stability) | Annex 3 |
| Operating costs, EBIT and headroom | Supplier costs and margin, smart-meter net cost change | Cap operating costs, EBIT and headroom |

Lines outside the cap are placed by the levy that would recover them: the Power BECCS CfD and the Net Zero Teesside
Power DPA with the CfD; the LDES and interconnector cap and floor with network costs (both settle through TNUoS); the
battery revenue top-up, a declared line with no levy to follow, with policy costs. When Ofgem moves a line between
categories, a new vintage is added and old vintages stay as they were.

**What each cost pays for.** A second, functional view does not follow Ofgem: energy and procurement; low-carbon
support schemes; capacity and flexibility; system operation; networks; social and industry levies; and retail.

## Data

| Input | File |
| --- | --- |
| Rules and published charges, by year | `data/market/consumer_bill_benchmarks/<year>_pypsa_informed_assumptions*.json` (`model_linked_rules`, `sourced_externals`) |
| Recovery volumes | `data/market/consumer_bill_forecast/<year>_recovery_demand_assumptions.json` |
| RO scheme parameters | `data/market/consumer_bill_forecast/2030_ro_scheme_parameters.json` |
| Line sources and groupings | `data/market/consumer_bill_benchmarks/bill_line_sources.csv`, `bill_groupings.csv`, `bill_grouping_views.csv` |
| CfD strikes | `data/market/lccc/cfd_strike_prices_by_round.csv` |
| Ofgem cap releases (historical) | `data/market/ofgem_price_cap/` with source manifests in `data/market/consumer_bill_sources/` |

Each page of this chapter names its own sources.

## Configuration

Future cases are declared in `consumer_bill.forecasts.cases` (`config/defaults/base/consumer_bill.yaml`). Each case
names twelve monthly solved outputs (`source_scenario_pattern`), its assumptions file, and files it cannot run without:

- `recovery_demand_assumptions`: every recovery volume, so no line invents its own. The projector checks the file's
  system demand against the run.
- `annual_aggregate_study`: the twelve-month aggregate (a `monthly_aggregates` entry of the study) that owns the
  annual accounting, so the bill and the NESO comparison cannot drift apart.
- `ro_scheme_parameters` and `ro_relief_case`: the Renewables Obligation. There is no default relief case.
- `cfd_recovery.method`: `eligible_supplier_sales`, the CfD levy base.

The CP.32 price-level mapping (`consumer_bill.price_level_mapping`) maps each Stage 1 hour onto CP30's published price
duration curve for the price-level sensitivity. Historical replays are configured per scenario under
`consumer_bill.price_cap` (below).

## How to run

### A future projection

The projection reads saved outputs and re-solves nothing:

```sh
# Once per case: sum the twelve monthly market summaries
snakemake resources/analysis/CP30_2030_no_acceleration_annual_summary.json \
  --allowed-rules aggregate_CP30_months --cores 1

# Constraint costs, the RO scheme cost, then the bill
snakemake consumer_bill_future_reports \
  --allowed-rules extract_future_balancing_costs build_future_ro_scheme_cost \
                  project_future_consumer_bill consumer_bill_future_reports \
  --cores 1
```

Leave out `--allowed-rules` only when rebuilding the scenarios is intended. The model rules take the merged
configuration as a parameter, so a configuration edit outside the bill blocks can re-trigger solves; add
`--rerun-triggers mtime` when only bill inputs have changed.

Outputs in `resources/analysis/`:

| File | Content |
| --- | --- |
| `<case>_consumer_bill_future_components.csv` | One row per bill line, ex VAT: value, method text, source, `ofgem_group`, `functional_group` |
| `<case>_consumer_bill_future_summary.json` | Totals, VAT, group totals, recovery volumes and `line_recovery`, CfD and RO provenance, `diagnostics`, `limitations` |
| `<case>_consumer_bill_future.html` | A readable report |
| `<case>_consumer_bill_future_hourly_inputs.csv` | The hourly price, demand and household weight behind the wholesale lines |
| `<case>_future_balancing_costs_*` | Constraint costs by month and carrier |
| `<case>_ro_*` | The ROC-eligible fleet and the scheme cost |

The summary's `diagnostics` block holds model quantities that are reported but never charged (constraint costs
against NESO outturn, CfD turn-down, zero-price hours, interconnector congestion rent, feasibility slack, the wholesale
sensitivities), so every model quantity is either a charge with a stated recovery route or a diagnostic.

The projector fails rather than degrading quietly when:

- the balancing-cost extraction or the RO scheme-cost extraction is absent;
- the recovery-demand file is absent, disagrees with the run, or its volumes do not reconcile;
- a published line does not declare how its period maps onto the bill's calendar year (`period_basis`);
- `ro_relief_case` names an undeclared case;
- the strike table's indexation does not land on the model price year;
- a `model_linked_rules` entry is declared that no code applies;
- the CfD rule differs from the one the balancing bids used;
- a bill line has no group in `bill_groupings.csv`.

### Adding or changing a bill line

1. **The value.** A published charge goes in `sourced_externals` of the assumptions file, with its unit
   (`gbp_per_household_year` or `gbp_per_mwh_final_demand`), price year and index, `period_basis`, source URL and
   method. A model-linked line needs code in `scripts/consumer_bill/project_future_bill.py` and a
   `model_linked_rules` entry.
2. **Its recovery.** Name the volume it is levied on, its loss treatment and any allocation factor; the projector
   writes them to `line_recovery` in the summary (see {doc}`recovery`).
3. **The source label.** Add a row to `bill_line_sources.csv`.
4. **The groups.** Add a row in each view of `bill_groupings.csv`. `tests/unit/test_bill_groupings.py` collects every
   line id the code can emit and fails on a missing one.

### A historical tariff replay

Historical scenarios declare an annual bundle and a pinned source manifest:

```yaml
consumer_bill:
  price_cap:
    enabled: true
    input_mode: historical_backtest
    data_bundle: data/market/ofgem_price_cap/2024_annual
    source_manifest: data/market/consumer_bill_sources/2024.yaml
    payment_methods: [direct_debit]
    regions: [GB_average]
    annual_consumption_kwh: [3100]
    cost_stack:
      enabled: true
```

The manifest pins source workbooks, their delivery periods and checksums. The bundle builder writes the normalised
tariff, component ledger and provenance under `data/market/ofgem_price_cap/<bundle>/`, and fails if a source changes
checksum or layout: update the manifest rather than overwriting a historical source.

```sh
snakemake consumer_bill_historical_sources --cores 1   # download and verify sources, build the bundles
snakemake consumer_bill_historical_replay --cores 1    # source-only tariff replays
snakemake consumer_bill_historical_reports --cores 1   # HTML reports and the shared index
```

For an already-solved scenario, `snakemake -s Snakefile_consumer_bill --cores 1` avoids network construction and
optimisation. Reports are in `resources/consumer_bill/<scenario>/`.

### Tests

`tests/unit/test_future_consumer_bill.py`, `test_bill_groupings.py`, `test_cap_rule_quarter_weights.py`,
`test_historical_consumer_bill_bundle.py`, `test_historical_component_ledger.py`,
`test_historical_model_comparison.py` and `test_annual_support_to_bill_bridge.py`.

## Validation

See {doc}`validation`: the line-by-line audit against Ofgem's current cap, the 2020 backtest against Ofgem's 2020
cap, and the reconciliations of CfD settlement against LCCC.

## Worked examples on these pages

The tables included on these pages are generated from the bill's outputs by `scripts/analysis/build_bill_docs_values.py`;
each table's first line names the files it read. They illustrate the method on the CP30 2030 bill (the private bill,
central network case); the bill for every network case and year is in {doc}`/studies/household_bills`.

The private 2030 bill, GBP a year, real 2025, excluding VAT:

```{include} _generated/bill_lines.md
```

By Ofgem group:

```{include} _generated/group_totals.md
```

By kind of source:

```{include} _generated/source_kinds.md
```

## Pages

```{toctree}
:maxdepth: 1

wholesale
cfd
capacity_market
policy_levies
networks
supplier
recovery
validation
```
