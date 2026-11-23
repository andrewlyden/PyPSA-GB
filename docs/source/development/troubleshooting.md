# Troubleshooting

Common problems and what to do about them. When a rule fails, Snakemake prints the path of its log file under
`logs/`; read that first.

## Installation

### The conda environment does not resolve

1. Update conda, or use mamba, which resolves faster:

   ```bash
   conda install -n base mamba
   mamba env create -f envs/pypsa-gb.yaml
   ```

2. Clear the package cache and retry: `conda clean --all`.

`envs/pypsa-gb-stable.yaml` pins every package to an exact version. The pins were taken on Windows, and the file does
not currently install on Linux or macOS; use `envs/pypsa-gb.yaml` there.

### Gurobi licence not found

`GurobiError: No Gurobi license found` means Gurobi cannot see a licence file (`gurobi.lic` in your home folder by
default). Academic licences are free from Gurobi. Without one, use HiGHS, the open-source solver, in the scenario:

```yaml
solve:
  solver:
    name: highs
```

HiGHS is slower on the large ETYS networks; see {doc}`/model/dispatch/solvers`.

### `snakemake: command not found`

Activate the environment (`conda activate pypsa-gb`) and check with `snakemake --version`.

## Configuration

### A key is refused with "renamed" in the message

Settings were nested by domain in v2.3.0 (for example `solver` is now `solve.solver`, `etys` is `network.etys`,
`transmission` is `network.transmission`). The loader refuses the old path and names the new one: move the setting.
See {doc}`release_notes` and {doc}`/reference/configuration_keys`.

### A new scenario fails for missing `targets` or `method`

A new future scenario must declare both, for example:

```yaml
targets: {source: FES, edition: 2024, pathway: HT}
method: gsp_block
```

Only the ids in `data/scenario_sources/legacy_scenario_ids.txt` may still derive them from the older keys. See
{doc}`/model/scenarios_configuration`.

### Where does this value come from?

```bash
python config/config_loader.py --explain <scenario_id> --path market.balancing
```

shows which layer (base, overlay, template or scenario) sets each value.

## Workflow

### Missing input files

1. Check the scenario id is spelled as in `config/scenarios.yaml` or the study, and that the study is listed under
   `studies` in `config/config.yaml`.
2. Ask Snakemake to build the file: `snakemake <path> --cores 1`. If no rule makes it, the file is a raw input that
   must be under `data/`.
3. A NESO website document (for example the CP30 data workbook) is downloaded by rule `retrieve_neso_document`. If
   the download fails, fetch it by hand from the landing page in `data/neso_documents.yaml` and save it at the listed
   path.

### "No rule to produce" a target

The file name does not match any rule's output pattern. Compare it with {doc}`/reference/outputs`, and check that the
scenario is in `run_scenarios` or named with `--config scenario=<id>`.

### A rule re-runs when nothing seems to have changed

Snakemake re-runs a rule when its inputs, code or parameters change. Run `snakemake -n` to see the reason for each
job. Build and solve rules ignore changes to reporting-only blocks of the configuration.

## Network

### All generators land on one bus

Usually WGS84 coordinates (degrees) compared with OSGB36 bus coordinates (metres). Map sites through
`scripts/utilities/spatial_utils.py` (`map_sites_to_buses`), which converts between the two;
`detect_coordinate_system(x_values, y_values)` tells you which system an array is in.

### Isolated buses

```python
import pypsa

n = pypsa.Network("resources/network/<scenario>.nc")
n.determine_network_topology()
print(n.buses.sub_network.value_counts())   # more than one sub-network: parts are not connected
```

## Solving

### Infeasible solve

1. **Demand above supply.** Compare capacity with peak demand:

   ```python
   print(f"Generation: {n.generators.p_nom.sum() / 1e3:.1f} GW")
   print(f"Peak demand: {n.loads_t.p_set.sum(axis=1).max() / 1e3:.1f} GW")
   ```

2. **A disconnected part of the network** (see above).
3. **Lines with zero rating:** `n.lines[n.lines.s_nom == 0]`.

Load-shedding generators at the value of lost load (`solve.voll`) normally keep a solve feasible, so an infeasible
solve usually points at the network or a hard constraint rather than at the generation mix.

### The solve takes too long

- Solve a shorter window with `solve_period` (`enabled: true`, `start`, `end`).
- Use the Reduced network or clustering for tests.
- Keep the barrier settings in `solve.solver` (`method: 2`, `crossover: 0`), the defaults.

### Out of memory

Run fewer jobs at once (`snakemake --cores 1`), cluster the network, or shorten the solve period. Full ETYS networks
need 8 GB of memory or more.

## Market

### ELEXON prices requested for a future year

ELEXON bid-offer data is historical. For a future year use derived bids and offers:

```yaml
market:
  balancing:
    bid_offer_source: derived
```

The default, `auto`, uses ELEXON only for historical years where the data is available.

### The balancing stage is infeasible

The wholesale stage solves, but `solve_balancing_mechanism` fails. Common causes: participation settings that hold
too many units fixed, or interconnector and outage settings that over-constrain the network. Start from the default
participation settings (`market.balancing.participation`: every generator and storage unit in, non-participants
priced out rather than fixed) and tighten from there. `market.balancing.diagnostics.compute_iis: true` logs a Gurobi
irreducible infeasible set for a failed window.

### The wholesale price spread looks too high

Averaging `buses_t.marginal_price` over every bus includes hydrogen and internal DC-link buses, which keep directional
shadow prices. Use `resources/market/{scenario}_wholesale_price.csv`, which reports the price over demand buses only.

## Data

### Weather cutout missing

Cutouts are built by a separate workflow:

```bash
snakemake -s Snakefile_cutouts --configfile config/cutouts_config.yaml --cores 4
```

For 2010-2024 it downloads pre-built cutouts from Zenodo (no credentials needed). Other years are built from ERA5 and
need Copernicus Climate Data Store credentials (`~/.cdsapirc`). Cutouts go to `resources/atlite/cutouts/uk-<year>.nc`.
Alternatively, point the scenario's `renewables_year` at a year you have.

### FES data not found

Check the edition is registered (see {doc}`/reference/data/maintenance`) and rebuild the processed table:
`snakemake resources/FES/FES_<edition>_data.csv --cores 1`.

## Getting help

Search the [GitHub issues](https://github.com/andrewlyden/PyPSA-GB/issues) first. A new issue should include:

1. the environment (`python --version`, `conda list | grep -E "pypsa|snakemake|pandas|linopy"`);
2. the command you ran;
3. the full error and the rule's log file;
4. the scenario entry (or the `--explain` output for the relevant keys).
