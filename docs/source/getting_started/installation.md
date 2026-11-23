# Installation

This page takes you from nothing to a working environment. Run every command from the repository root.

## 1. Clone the repository

```bash
git clone --filter=blob:none https://github.com/andrewlyden/PyPSA-GB.git
cd PyPSA-GB
```

`--filter=blob:none` makes a "blobless" clone: you get every file of the current version, but the contents of
old versions are fetched only if you ask for them (for example with `git log -p`). The repository's history is
large, so this saves a long download and several GB of disk. Everything else in Git works as usual.

## 2. Create the conda environment

Two environment files are provided:

| File | Environment name | Use |
|---|---|---|
| `envs/pypsa-gb-stable.yaml` | `pypsa-gb-stable` | **Recommended.** Every package pinned to the versions the model is tested with. |
| `envs/pypsa-gb.yaml` | `pypsa-gb` | Version ranges rather than pins. It is also the environment the rules name if you run Snakemake with `--use-conda`. |

```bash
mamba env create -f envs/pypsa-gb-stable.yaml     # or: conda env create -f envs/pypsa-gb-stable.yaml
conda activate pypsa-gb-stable
```

Check the installation:

```bash
python -c "import pypsa, linopy; print(pypsa.__version__, linopy.__version__)"
snakemake --version
```

## 3. Set up a solver

The model's problems are linear programmes. The solver is set by `solve.solver.name` in
`config/defaults/base/solve.yaml`, which is `gurobi`.

### Gurobi (default)

The environment installs the `gurobipy` package, but its built-in licence is limited to small models and will not
solve a PyPSA-GB network. You need a full licence; academic licences are free.

1. Register at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/) and request a named
   academic licence.
2. Run the `grbgetkey` command Gurobi gives you; it writes `gurobi.lic` to your home folder.
3. Check: `python -c "import gurobipy; gurobipy.Model()"` prints the licence it found and no error.

The default options in `solve.yaml` (barrier method without crossover, relaxed tolerances, two threads) are
tuned for these models. The market stages retry a failed window on more conservative Gurobi settings, then on
dual simplex, before they give up; this fallback exists only for Gurobi.

### HiGHS (open source)

HiGHS is installed with the environment (`highspy`) and needs no licence. Select it for one scenario:

```yaml
My_Scenario:
  # ...
  solve:
    solver:
      name: highs
```

or for every scenario, by adding the same `solve:` block to `config/config.yaml` (any key there other than
`studies` and `run_scenarios` overrides the defaults for all scenarios). HiGHS receives only the thread count;
the Gurobi-specific options are not passed to it. It is slower than Gurobi on the larger networks, so start with
the Reduced or Zonal network or a clustered ETYS network. Every setting, and what to choose for runs of different
sizes, is in {doc}`/model/dispatch/solvers`.

## 4. Weather cutouts

Renewable output is computed from ERA5 weather data with [atlite](https://github.com/PyPSA/atlite). The data for
one weather year is a "cutout", `resources/atlite/cutouts/uk-<year>.nc`.

You do not need to fetch cutouts by hand. When a scenario needs the cutout of its `renewables_year`, rule
`retrieve_cutout` gets it, trying in order:

1. a copy you have placed in `data/atlite/cutouts/`;
2. the published PyPSA-GB cutouts on Zenodo ([10.5281/zenodo.18325225](https://doi.org/10.5281/zenodo.18325225)):
   years 2010 to 2024, about 700 to 765 MB each, a few minutes per year, no account needed;
3. a fresh ERA5 download through atlite: two to four hours per year, and it needs a
   [Copernicus CDS](https://cds.climate.copernicus.eu/how-to-api) API key in `~/.cdsapirc`.

To fetch several years ahead of time, list them under `years_config.years_to_generate` in
`config/cutouts_config.yaml` and run the separate cutout workflow:

```bash
snakemake -s Snakefile_cutouts --cores 1
```

## 5. Data the workflow downloads

Most input data is in `data/`. A few sources are fetched when a scenario first needs them, so the first run of a
scenario needs an internet connection:

- **NESO Future Energy Scenarios** tables, from the NESO data portal (`data/FES/FES_api_urls.yaml`). The raw
  tables are cached under `resources/sources/raw/`.
- **NESO website documents** that the repository may not redistribute. They are listed, with their download URL,
  landing page and SHA-256 checksum, in `data/neso_documents.yaml`; today these are the Clean Power 2030 data
  workbook, the FES data workbooks 2020 to 2024, the FES 2024 regional breakdown and the ETYS Appendix B
  workbooks 2022 to 2024. Rule `retrieve_neso_document` downloads each one to its listed path
  when it is missing and refuses a file whose checksum differs. If NESO moves a document, download it by hand
  from its landing page and save it at the listed path.

## 6. Optional API tokens

Nothing here is needed for the quickstart. Copy `.env.example` to `.env` (which Git ignores) and fill in what you
need:

| Variable | Used by |
|---|---|
| `ENTSOE_API_TOKEN` | `scripts/interconnectors/fetch_entsoe_prices.py`, for hourly day-ahead prices from the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/). The repository ships the prices for 2015 to 2024 (`data/interconnectors/entsoe_day_ahead/`), so a token is needed only for another year; without one, that year stops with a message saying how to add it. |
| `ZENODO_TOKEN` | Maintainers only: uploading the input data bundle. |

## 7. Check the workflow

A dry run lists the jobs Snakemake would run, without running them:

```bash
snakemake -n
```

It should end with a job count and "This was a dry-run". If it fails while parsing, the error names the
configuration problem (an unknown scenario id, a renamed key, a missing file).

```{tip}
On Windows consoles that use the cp1252 code page, `python config/config_loader.py --validate` can fail while
printing its tick mark. Set `PYTHONIOENCODING=utf-8` first (`$env:PYTHONIOENCODING = "utf-8"` in PowerShell).
```

## Next

{doc}`quickstart`: run your first scenario.
