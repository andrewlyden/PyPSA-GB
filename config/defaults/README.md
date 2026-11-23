# config/defaults

The defaults every scenario starts from, and the overlays that adjust them by what a scenario declares.

## Merge order

A scenario's config is merged in this order, each layer deep-merged over the one before:

1. **The base**: `base.yaml` and every file under `base/`. The neutral defaults: every setting that applies to
   every scenario exists here, at the value a scenario gets unless something below changes it. The overlays also
   introduce keys of their own that only their scenarios carry (the CP30 build and data keys, `hydrogen`,
   `source_data`, `validation`); those are listed in the overlay files. File path = key path:
   `base/market/balancing.yaml` holds the contents of `market.balancing`; `base/market.yaml` holds `market`'s
   other keys; `base.yaml` holds the top-level keys.
2. **`config/config.yaml`'s global overrides** (normally none).
3. **The method overlay**, `method/<method>.yaml`, picked by the scenario's `method`: how the fleet is built.
4. **The source overlay**, `source/<targets.source>.yaml`, picked by the scenario's `targets.source`: what the
   publication the scenario follows gives (its files, editions, prices, targets and checks).
5. **The dispatch overlay**, `dispatch/<market.dispatch>.yaml`, picked by the scenario's `market.dispatch`:
   `central` (the system operator dispatching at cost) has a file, `self` (the market) has none.
   A mode switch, so it may override method and source keys; a central scenario may not change the keys it
   fixes (`config_loader.CENTRAL_DISPATCH_FIXED`).
6. **The scenario's entry**: its `inherits` templates, then its own settings
   (`config/scenarios.yaml`, or a study in `config/studies/<study>/`).
7. **The loader's derivations**: `targets` and `method` themselves, year parameters, clustering presets.

The overlays attach by the declared axes, never by scenario name. Every axis value has a file; most are empty on
purpose, because only CP30 differs from the base today:

| Axis | Value | File | Holds |
|---|---|---|---|
| `method` | `brownfield` | `method/brownfield.yaml` | named-fleet placement, demand build, electrolyser siting, proxy profiles |
| `method` | `gsp_block` | `method/gsp_block.yaml` | nothing (the FES GSP-block build uses the base) |
| `method` | `historical` | `method/historical.yaml` | nothing (historical years take the base) |
| `targets.source` | `CP30` | `source/CP30.yaml` | CP30's files, editions, prices, hydrogen demand, cables, checks |
| `targets.source` | `FES` | `source/FES.yaml` | nothing |
| `targets.source` | `historical` | `source/historical.yaml` | nothing |
| `market.dispatch` | `central` | `dispatch/central.yaml` | copperplate off, at-cost pricing, no scarcity rent or support bids, central solver |
| `market.dispatch` | `self` | (no file) | the market as the base and layers set it |

## Rules (each held by a test, `tests/unit/test_config_layers.py`)

- A key goes in an overlay only when every scenario with that axis value needs it and it differs from the base.
  Settings of one study or family belong in that study's file.
- The method overlay holds no source data: no publication's files, editions or targets
  (a brownfield build on another source must not inherit CP30's).
- The method and source overlays never set the same key; the dispatch overlay, a mode switch merged after them,
  may override either, and touches only `market`, `solve` and `support`.
- Code defaults equal the base (project rule).
- An old key path is refused with its new one named (`config_loader.RENAMED_KEYS`).

## Seeing what a scenario gets

```bash
python config/config_loader.py --explain <scenario id>                 # the layers, and which file set each value
python config/config_loader.py --explain <scenario id> --path market.wholesale
python config/config_loader.py --dump-merged FILE                     # every scenario's merged config (JSON lines)
```

Read the defaults in code only through `config_loader.load_defaults()`, `load_layer(layer, name)` and
`read_config(path)`; Snakemake rules declare `DEFAULTS_FILES` as inputs.
