# Configuration keys

Every setting a scenario can change, with its default and the comment that documents it in the defaults files.
The tables are generated from `config/defaults/base.yaml` and `config/defaults/base/**` by
`scripts/utilities/build_reference_tables.py`, so they match the files they come from.

## How to read the tables

- **File path = key path.** `config/defaults/base/market/balancing.yaml` holds the keys under `market.balancing`;
  `base/market.yaml` holds the other keys under `market`; `base.yaml` holds the top-level keys. Each section below is
  one file, and its keys are written relative to the section heading: under `market.balancing`, the row
  `turn_down_allocation.mode` is the key `market.balancing.turn_down_allocation.mode`.
- **Default** is the value in the base. `(mapping)` marks a key that holds further keys (listed below it);
  long lists are shown as a count.
- **Description** is the file's own comment for that key: what it does, its units and its source. A comment above
  a block of keys is shown on the first key of the block. Rationale and longer explanations are in the
  {doc}`model chapters </model/index>`.
- The base holds the neutral defaults; keys that only the overlays' scenarios carry are in the overlay files (below).
  A key path that was renamed in v2.3.0 is refused with its new path named (see {doc}`/development/release_notes`).

To see which file sets a value for one scenario, ask the loader:

```bash
python config/config_loader.py --explain <scenario_id> --path market.balancing
```

The full guide to scenarios and layers is {doc}`/model/scenarios_configuration`.

## Overlays

The base is neutral. Three overlays change it for the scenarios that declare the matching axis. They are not listed
key by key here: they reset keys of the base and also introduce keys of their own that only their scenarios carry
(the CP30 build and data keys such as `fleet.spatial_proxy`, `demand.build` and `network.reinforcements`, and the
`hydrogen`, `source_data` and `validation` blocks). Read the overlay files directly.

| Axis | Value | File | What it sets |
|---|---|---|---|
| `method` | `brownfield` | `config/defaults/method/brownfield.yaml` | named-fleet placement, demand build, electrolyser siting, proxy profiles |
| `method` | `gsp_block` | `config/defaults/method/gsp_block.yaml` | nothing (the FES GSP-block build uses the base) |
| `method` | `historical` | `config/defaults/method/historical.yaml` | nothing |
| `targets.source` | `CP30` | `config/defaults/source/CP30.yaml` | the CP30 publication's files, editions, prices, cables and checks |
| `targets.source` | `FES` | `config/defaults/source/FES.yaml` | nothing |
| `targets.source` | `historical` | `config/defaults/source/historical.yaml` | nothing |
| `market.dispatch` | `central` | `config/defaults/dispatch/central.yaml` | central dispatch at cost: copperplate off, at-cost pricing, no scarcity rent or support-based bids |

Merge order: base, global overrides in `config/config.yaml`, method, source, dispatch, then the scenario's entry
(its `inherits` templates, then its own settings).

## Regenerating

After changing a defaults file, run:

```bash
python scripts/utilities/build_reference_tables.py --only config
python scripts/utilities/build_reference_tables.py --check   # fails if a table is out of date
```

The generator leaves out parentheticals and sentences of a comment that are internal working notes (dated notes,
references to planning documents); the files themselves keep them.

## Top level

```{include} _generated/config_keys_top_level.md
:heading-offset: 1
```

## Network, demand and supply

```{include} _generated/config_keys_network.md
:heading-offset: 1
```

```{include} _generated/config_keys_demand.md
:heading-offset: 1
```

```{include} _generated/config_keys_renewables.md
:heading-offset: 1
```

```{include} _generated/config_keys_thermal.md
:heading-offset: 1
```

```{include} _generated/config_keys_storage.md
:heading-offset: 1
```

```{include} _generated/config_keys_interconnectors.md
:heading-offset: 1
```

```{include} _generated/config_keys_fleet.md
:heading-offset: 1
```

```{include} _generated/config_keys_marginal_costs.md
:heading-offset: 1
```

## Solving and the market

```{include} _generated/config_keys_solve.md
:heading-offset: 1
```

```{include} _generated/config_keys_market.md
:heading-offset: 1
```

```{include} _generated/config_keys_market_wholesale.md
:heading-offset: 1
```

```{include} _generated/config_keys_market_balancing.md
:heading-offset: 1
```

```{include} _generated/config_keys_support.md
:heading-offset: 1
```

## After the solve

These layers are preliminary (see {doc}`/model/after_the_solve/index`).

```{include} _generated/config_keys_consumer_bill.md
:heading-offset: 1
```

```{include} _generated/config_keys_public_ownership.md
:heading-offset: 1
```

```{include} _generated/config_keys_public_ownership_acquisition.md
:heading-offset: 1
```

```{include} _generated/config_keys_public_ownership_consumer_bill.md
:heading-offset: 1
```

```{include} _generated/config_keys_public_ownership_reference_run_groups.md
:heading-offset: 1
```

```{include} _generated/config_keys_public_ownership_sensitivities.md
:heading-offset: 1
```
