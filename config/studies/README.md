# config/studies

A study is a set of related scenarios solved together, usually month by month. Each is a folder, loaded when
`config/config.yaml` lists it under `studies`:

| Study | What | Scenario ids |
|-------|------|--------------|
| `cp30_2030` | NESO Clean Power 2030, Further Flex and Renewables, 2030 | `CP30_FFR_2030_<Mon>_<family>`; build `CP30_FFR_2030` |
| `fes2050` | FES 2025 Holistic Transition, 2050 | `FES25_HT_2050_<Mon>_<family>`; build `FES25_HT_2050` |
| `validation_2020` | the 2020 validation year | `Validation_<Mon>2020`, `ConsumerBill_<Mon>2020_Fast` |

## A study's files

```
<study>/
├── study.yaml       label, year, family_id (the id pattern), monthly_aggregates and other study settings
├── templates.yaml   named building blocks, composed by inherits
├── families.yaml    month families: one entry becomes twelve scenarios, one per calendar month
└── scenarios.yaml   single-period scenarios (the shared network build, one-off runs)
```

- **Templates and `inherits`.** A template or scenario may say `inherits: [a, b, ...]`. The parents are merged left to
  right, then the entry's own settings last. Two parents may not set the same setting to different values unless the
  entry sets it itself, so a composition never depends on the order of its list.
- **Families.** A family `<name>: {inherits: [...], <settings>}` becomes one scenario per month, named by the study's
  `family_id` (`{Mon}`, `{year}`, `{family}`) or the family's own `id`. Each month gets its own `solve_period`.
  Optional keys: `months` (a subset), `stage1: <family>` (take the wholesale result from that family's same month:
  sets `wholesale_from`), `network_from` (the shared network build), `month_overrides: {Mar: {...}}` (one month's
  exceptions), `description`.
- **Running a family.** A family's name in `run_scenarios` runs its twelve months.
- **Annual aggregates.** `study.yaml`'s `monthly_aggregates` names families whose twelve solved months cover one
  calendar year; rule `aggregate_CP30_months` sums them into `resources/analysis/<name>_annual_summary.json`, and the
  bill layer's forecast cases read them by name.
- New scenario ids write `targets` and `method` (`config/README.md`); in a study they usually sit in its base template.

## The CP30 FFR 2030 study

Templates follow the modelling axes: the scenario (`cp30_ffr_2030`, `cp30_ffr_2030_month`), the market
(`market_stage1`, `market_balancing`, `market_central_dispatch` and their parts), the network case (`network_cp30`,
`network_no_acceleration`, `network_seven_late`, `network_no_upgrades`) and the bid stack (`public_bids_p2`,
`public_bids_p3`). Families are named network case first:

| Family | |
|--------|--|
| `stage1` | the wholesale market (Stage 1) every balancing family reads |
| `no_acceleration` | balancing mechanism, CENTRAL network: CP30's three accelerations not delivered |
| `cp30_network` | balancing mechanism, LOWER network: CP30's recommended network |
| `seven_late`, `no_upgrades` | NESO's seven most critical projects late; UPPER: no upgrades after 2025 |
| `<network>_p2_bids`, `<network>_p3_bids` | public ownership: state-owned plant bidding at cost |
| `<network>_central_dispatch` | central dispatch at resource cost |
| `stage1_neso_consistent`, `neso_consistent_ic_caps`, `cp30_network_ic_premium_0` | calibration variants |

## The archive

`archive/<study>/` holds retired families and scenarios with their ids unchanged, written fully resolved, so that
earlier results stay reproducible. They are not loaded by default: `config_loader.load_config(include_archive=True)`
loads them (tests that check retired scenarios do this). `archive/canonical/` holds the scenarios retired from
`config/scenarios.yaml`.
